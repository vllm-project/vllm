# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Warmup kernels used during model execution.
This is useful specifically for JIT'ed kernels as we don't want JIT'ing to
happen during model execution.
"""

from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np
import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.model_executor.warmup.deep_gemm_warmup import deep_gemm_warmup
from vllm.model_executor.warmup.deepseek_v4_mhc_warmup import (
    deepseek_v4_mhc_warmup,
)
from vllm.model_executor.warmup.flashinfer_autotune_cache import (
    resolve_flashinfer_autotune_file,
    write_flashinfer_autotune_cache,
)
from vllm.model_executor.warmup.flashinfer_sparse_mla_warmup import (
    deepseek_v4_sparse_mla_attention_warmup,
    flashinfer_sparse_mla_decode_autotune_warmup,
)
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import is_deep_gemm_supported
from vllm.utils.flashinfer import has_flashinfer
from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
from vllm.v1.structured_output.utils import apply_grammar_bitmask

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    from vllm.v1.worker.gpu_worker import Worker

logger = init_logger(__name__)

_DEEPSEEK_V4_SPARSE_MLA_BACKENDS = frozenset(
    {
        "V4_FLASHMLA_SPARSE",
        "DEEPSEEK_SPARSE_SWA",
    }
)
_DEEPSEEK_V4_SPARSE_MLA_MIXED_WARMUP_TOKENS = 16
# Cap warmup at the largest single-chunk prefill the scheduler will ever
# issue (max_num_batched_tokens). 8192 covers the canonical SM12x serve
# (max_num_batched_tokens=8192); larger scheduler caps clamp to this
# value via _clamp_warmup_tokens at the call site, smaller caps clamp
# down naturally.
_DEEPSEEK_V4_SPARSE_MLA_PREFILL_WARMUP_TOKENS = 8192
# Steady-state MTP decode shapes to warm. Keep this bounded to high-concurrency
# SM12x gates while still avoiding the scheduler's raw max_num_seqs (often 1024),
# which can consume multiple GiB of temporary workspace on long-context serves
# before the first request.
_DEEPSEEK_V4_MTP_UNIFORM_DECODE_WARMUP_REQUESTS = (1, 2, 4, 8, 16, 24, 32)
_DEEPSEEK_V4_MTP_UNIFORM_DECODE_MAX_WARMUP_REQUESTS = 256
_DEEPSEEK_V4_SLOT_MAPPING_WARMUP_TOKENS = tuple(range(1, 17)) + (
    32,
    64,
    128,
    256,
    512,
)


def _attention_backend_name(backend: object) -> str | None:
    get_name = getattr(backend, "get_name", None)
    if get_name is None:
        return None
    try:
        return get_name()
    except NotImplementedError:
        return None


def _has_deepseek_v4_sparse_mla_backend(runner: "GPUModelRunner") -> bool:
    for groups in getattr(runner, "attn_groups", []) or ():
        for group in groups:
            name = _attention_backend_name(getattr(group, "backend", None))
            if name in _DEEPSEEK_V4_SPARSE_MLA_BACKENDS:
                return True
    return False


def _clamp_warmup_tokens(num_tokens: int, max_tokens: int) -> int:
    return max(0, min(num_tokens, max_tokens))


def _is_deepseek_v4_mtp_spec_decode(runner: "GPUModelRunner") -> bool:
    spec_config = getattr(runner, "speculative_config", None)
    return (
        getattr(spec_config, "method", None) == "mtp"
        and getattr(runner, "num_spec_tokens", 0) > 0
    )


def _deepseek_v4_mtp_uniform_decode_warmup_requests(
    runner: "GPUModelRunner",
    max_tokens: int,
    max_reqs: int,
) -> tuple[int, ...]:
    if not _is_deepseek_v4_mtp_spec_decode(runner):
        return ()

    query_len = getattr(
        runner,
        "uniform_decode_query_len",
        1 + getattr(runner, "num_spec_tokens", 0),
    )
    if query_len <= 0:
        return ()

    max_warmup_reqs = min(
        max_reqs,
        max_tokens // query_len,
        _DEEPSEEK_V4_MTP_UNIFORM_DECODE_MAX_WARMUP_REQUESTS,
    )
    candidates = sorted(
        set(_DEEPSEEK_V4_MTP_UNIFORM_DECODE_WARMUP_REQUESTS) | {max_warmup_reqs}
    )
    return tuple(reqs for reqs in candidates if reqs <= max_warmup_reqs)


def _deepseek_v4_slot_mapping_warmup(runner: "GPUModelRunner") -> None:
    max_tokens = getattr(runner, "max_num_tokens", 1)
    block_table = runner.input_batch.block_table

    # Snapshot the runner buffers we mutate so warmup never leaks state into
    # the first real request.
    saved_query_start_loc_np: np.ndarray | None = None
    saved_query_start_loc_gpu: torch.Tensor | None = None
    if hasattr(runner, "query_start_loc"):
        saved_query_start_loc_np = runner.query_start_loc.np[:2].copy()
        saved_query_start_loc_gpu = runner.query_start_loc.gpu[:2].clone()

    try:
        for requested_tokens in _DEEPSEEK_V4_SLOT_MAPPING_WARMUP_TOKENS:
            num_tokens = _clamp_warmup_tokens(requested_tokens, max_tokens)
            if num_tokens <= 0:
                continue

            positions_source = torch.arange(
                num_tokens, dtype=torch.int64, device=runner.device
            )
            if hasattr(runner, "query_start_loc"):
                runner.query_start_loc.np[0] = 0
                runner.query_start_loc.np[1] = num_tokens
                runner.query_start_loc.copy_to_gpu(2)
                query_start_loc = runner.query_start_loc.gpu[:2]
            else:
                query_start_loc = torch.tensor(
                    [0, num_tokens], dtype=torch.int32, device=runner.device
                )

            if hasattr(runner, "positions"):
                saved_positions: torch.Tensor | None = runner.positions[
                    :num_tokens
                ].clone()
                runner.positions[:num_tokens].copy_(positions_source)
                positions = runner.positions[:num_tokens]
            else:
                saved_positions = None
                positions = positions_source

            try:
                block_table.commit_block_table(1)
                block_table.compute_slot_mapping(1, query_start_loc, positions)
            finally:
                if saved_positions is not None:
                    runner.positions[:num_tokens].copy_(saved_positions)
    finally:
        if saved_query_start_loc_np is not None:
            runner.query_start_loc.np[:2] = saved_query_start_loc_np
            assert saved_query_start_loc_gpu is not None
            runner.query_start_loc.gpu[:2].copy_(saved_query_start_loc_gpu)


def _deepseek_v4_structured_output_bitmask_warmup(
    runner: "GPUModelRunner",
) -> None:
    vocab_size = runner.model_config.get_vocab_size()
    if vocab_size <= 0:
        return

    dtypes = [torch.float32]
    model_dtype = getattr(runner.model_config, "dtype", None)
    if isinstance(model_dtype, torch.dtype) and model_dtype not in dtypes:
        dtypes.append(model_dtype)

    bitmask_width = (vocab_size + 31) // 32
    req_id = "_deepseek_v4_warmup_"
    grammar_bitmask = np.full((1, bitmask_width), fill_value=-1, dtype=np.int32)
    grammar_output = GrammarOutput(
        structured_output_request_ids=[req_id], grammar_bitmask=grammar_bitmask
    )

    for dtype in dtypes:
        for req_ids in ([req_id], [req_id, "_deepseek_v4_warmup_unmasked_"]):
            logits = torch.zeros(
                (len(req_ids), vocab_size), dtype=dtype, device=runner.device
            )
            input_batch = SimpleNamespace(req_ids=req_ids)
            apply_grammar_bitmask(
                SchedulerOutput.make_empty(),
                grammar_output,
                input_batch,  # type: ignore[arg-type]
                logits,
            )


@torch.inference_mode()
def _deepseek_v4_request_prep_warmup(worker: "Worker") -> None:
    if not envs.VLLM_ENABLE_DEEPSEEK_V4_SPARSE_MLA_WARMUP:
        return

    runner = worker.model_runner
    if runner.is_pooling_model or not _has_deepseek_v4_sparse_mla_backend(runner):
        return
    if not current_platform.is_cuda_alike():
        return

    logger.info("Warming up DeepSeek V4 request preparation kernels.")
    _deepseek_v4_slot_mapping_warmup(runner)

    if getattr(runner, "is_last_pp_rank", True):
        try:
            _deepseek_v4_structured_output_bitmask_warmup(runner)
        except ImportError:
            logger.debug(
                "Skipping DeepSeek V4 structured output bitmask warmup because "
                "xgrammar is unavailable."
            )

    torch.accelerator.synchronize()


def _run_deepseek_v4_mtp_spec_decode_warmup_kernels(
    *,
    device: torch.device,
    num_reqs: int,
    num_spec_tokens: int,
    vocab_size: int,
    block_size: int,
    max_model_len: int,
    hidden_size: int,
) -> None:
    from vllm.v1.sample.logits_processor import LogitsProcessors
    from vllm.v1.sample.metadata import SamplingMetadata
    from vllm.v1.sample.rejection_sampler import rejection_sample
    from vllm.v1.spec_decode.utils import (
        eagle_prepare_inputs_padded_kernel,
        eagle_prepare_next_token_padded_kernel,
        eagle_step_update_slot_mapping_and_metadata,
        next_power_of_2,
    )

    num_sampled_tokens = num_spec_tokens + 1
    sampled_token_ids = torch.arange(
        num_reqs * num_sampled_tokens, dtype=torch.int32, device=device
    ).reshape(num_reqs, num_sampled_tokens)
    sampled_token_ids.remainder_(vocab_size)
    discard_request_mask = torch.zeros(num_reqs, dtype=torch.bool, device=device)
    backup_next_token_ids = torch.zeros(num_reqs, dtype=torch.int32, device=device)
    next_token_ids = torch.empty(num_reqs, dtype=torch.int32, device=device)
    valid_sampled_tokens_count = torch.empty(num_reqs, dtype=torch.int32, device=device)
    eagle_prepare_next_token_padded_kernel[(num_reqs,)](
        sampled_token_ids,
        discard_request_mask,
        backup_next_token_ids,
        next_token_ids,
        valid_sampled_tokens_count,
        vocab_size,
        num_sampled_tokens,
        num_reqs,
        sampled_token_ids.stride(0),
        BLOCK_SIZE_TOKENS=next_power_of_2(num_sampled_tokens),
    )

    cu_num_draft_tokens = torch.arange(
        num_spec_tokens,
        num_reqs * num_spec_tokens + 1,
        num_spec_tokens,
        dtype=torch.int32,
        device=device,
    )
    query_start_loc = torch.arange(
        0,
        (num_reqs + 1) * num_sampled_tokens,
        num_sampled_tokens,
        dtype=torch.int32,
        device=device,
    )
    token_indices_to_sample = torch.empty(num_reqs, dtype=torch.int32, device=device)
    num_rejected_tokens = torch.empty(num_reqs, dtype=torch.int32, device=device)
    eagle_prepare_inputs_padded_kernel[(num_reqs,)](
        cu_num_draft_tokens,
        valid_sampled_tokens_count,
        query_start_loc,
        token_indices_to_sample,
        num_rejected_tokens,
        num_reqs,
    )

    positions = torch.arange(num_reqs, dtype=torch.int64, device=device)
    block_table_tensor = torch.zeros((num_reqs, 1), dtype=torch.int32, device=device)
    seq_lens = torch.ones(num_reqs, dtype=torch.int32, device=device)
    out_clamped_positions = torch.empty_like(positions)
    out_slot_mapping = torch.empty(num_reqs, dtype=torch.int64, device=device)
    eagle_step_update_slot_mapping_and_metadata(
        positions,
        block_table_tensor,
        seq_lens,
        block_size,
        max_model_len,
        out_clamped_positions,
        out_slot_mapping,
        input_batch_size=num_reqs,
    )

    total_draft_tokens = num_reqs * num_spec_tokens
    draft_token_ids = torch.arange(total_draft_tokens, dtype=torch.int32, device=device)
    draft_token_ids.remainder_(vocab_size)
    draft_probs = torch.rand(
        total_draft_tokens, vocab_size, dtype=torch.float32, device=device
    )
    draft_probs = draft_probs / draft_probs.sum(dim=-1, keepdim=True)
    target_logits = torch.randn(
        total_draft_tokens, vocab_size, dtype=torch.float32, device=device
    )
    bonus_token_ids = torch.zeros((num_reqs, 1), dtype=torch.int32, device=device)
    sampling_metadata = SamplingMetadata(
        temperature=torch.full((num_reqs,), 0.7, dtype=torch.float32, device=device),
        all_greedy=False,
        all_random=True,
        top_p=None,
        top_k=None,
        generators={},
        max_num_logprobs=None,
        no_penalties=True,
        prompt_token_ids=None,
        frequency_penalties=torch.empty(0, device=device),
        presence_penalties=torch.empty(0, device=device),
        repetition_penalties=torch.empty(0, device=device),
        output_token_ids=[[] for _ in range(num_reqs)],
        allowed_token_ids_mask=None,
        bad_words_token_ids={},
        logitsprocs=LogitsProcessors(),
        logprob_token_ids=None,
        spec_token_ids=[[] for _ in range(num_reqs)],
    )
    rejection_sample(
        draft_token_ids=draft_token_ids,
        num_draft_tokens=[num_spec_tokens] * num_reqs,
        max_spec_len=num_spec_tokens,
        cu_num_draft_tokens=cu_num_draft_tokens,
        draft_probs=draft_probs,
        target_logits=target_logits,
        bonus_token_ids=bonus_token_ids,
        sampling_metadata=sampling_metadata,
    )

    # rejection_greedy_sample_kernel: the metadata above is all_random=True, so
    # rejection_sample skips its greedy branch and that kernel stays JIT-cold. Run
    # a second pass with greedy metadata (fresh instance, not a mutation) so the
    # greedy kernel compiles here instead of on the first greedy request.
    try:
        import dataclasses

        greedy_metadata = dataclasses.replace(
            sampling_metadata,
            all_greedy=True,
            all_random=False,
            temperature=torch.zeros(num_reqs, dtype=torch.float32, device=device),
        )
        rejection_sample(
            draft_token_ids=draft_token_ids,
            num_draft_tokens=[num_spec_tokens] * num_reqs,
            max_spec_len=num_spec_tokens,
            cu_num_draft_tokens=cu_num_draft_tokens,
            draft_probs=draft_probs,
            target_logits=target_logits,
            bonus_token_ids=bonus_token_ids,
            sampling_metadata=greedy_metadata,
        )
    except Exception as exc:  # noqa: BLE001 - warmup must never break startup
        logger.warning(
            "DeepSeek V4 MTP greedy rejection-sample warmup skipped: %s", exc
        )

    # _mtp_shared_head_rmsnorm_kernel: the MTP shared-head RMSNorm is not driven by
    # any dummy run, so it JITs on the first MTP step. Direct-launch it (its only
    # compile key is hidden_size, so one call covers the model).
    try:
        from vllm.models.deepseek_v4.common.ops.fused_mtp_input_rmsnorm import (
            mtp_shared_head_rmsnorm,
        )

        hs = torch.randn(
            num_reqs * num_sampled_tokens,
            hidden_size,
            dtype=torch.bfloat16,
            device=device,
        )
        norm_w = torch.ones(hidden_size, dtype=torch.bfloat16, device=device)
        mtp_shared_head_rmsnorm(hs, norm_w, 1e-6)
    except Exception as exc:  # noqa: BLE001 - warmup must never break startup
        logger.warning(
            "DeepSeek V4 MTP shared-head RMSNorm warmup skipped: %s", exc
        )


def _deepseek_v4_indexed_d512_split_prefill_warmup(runner: "GPUModelRunner") -> None:
    """Force-compile the DeepSeek-V4 D512-split sparse-MLA prefill kernels.

    The split path (``_use_indexed_d512_split_prefill`` ->
    ``accumulate_indexed_d512_split_sparse_mla_attention``) bottoms out in three
    plain ``@triton.jit`` kernels whose compile key is the constexpr set --
    chiefly ``num_candidates`` (= the per-chunk ``combined_topk``) plus the
    workspace buffer strides. ``combined_topk`` is 128-aligned
    (``_SPARSE_PREFILL_TOPK_ALIGNMENT``) and the split path is gated to
    ``[256, 1152]`` (``_is_indexed_d512_split_topk``), so the complete
    specialization set is the eight widths {256, 384, ..., 1152}. The kernels
    never see ``compress_ratio``, so one warm per width covers cr=4 and cr=128.

    Without this, the first long-prefill request JIT-compiles these kernels
    inside the engine step (~20s), parking EngineCore in shm_broadcast and
    surfacing as a "sample_tokens RPC timed out" wedge (PR #41834).

    Triton compilation is data-independent, so synthetic zero tensors compile
    the same cubin a real request uses -- provided every constexpr matches. Two
    non-obvious constexprs (verified against the live jit_monitor compile key):
    the per-chunk ``scores``/``indices`` workspaces are sized to that chunk's own
    ``combined_topk`` (contiguous at width C, so ``stride_scores_h == C`` and
    ``stride_indices_t == C`` -- NOT a slice of a wider buffer), and the prefill
    ``q`` buffer is padded to the FP8-decode head count (``padded_heads``), so
    ``stride_q_t == padded_heads * head_dim`` even though the kernel reads only
    ``n_local_heads``. The synthetic tensors mirror both.

    Scope: only the split path (``combined_topk <= 1152``) is warmed. DeepSeek-V4
    -Flash caps ``combined_topk`` at ``sparse_prefill_combined_topk_size(
    index_topk=512, 128) = 640`` for every context length, so that is complete
    coverage. A variant whose ``combined_topk`` can exceed 1152 routes onto the
    chunked path (extra split-stride and merge kernels) which is not pre-warmed
    here; that case is warned at startup rather than left as a silent gap.
    """
    if not (
        envs.VLLM_DEEPSEEK_V4_INDEXED_D512_SPLIT_PREFILL_WARMUP
        and envs.VLLM_DEEPSEEK_V4_INDEXED_D512_SPLIT_PREFILL
    ):
        return

    try:
        from vllm.models.deepseek_v4.common.ops.cache_utils import (
            sparse_prefill_combined_topk_size,
        )
        from vllm.models.deepseek_v4.nvidia.flashmla import (
            _INDEXED_D512_SPLIT_PREFILL_MAX_TOPK,
            _INDEXED_D512_SPLIT_PREFILL_MIN_TOPK,
            DeepseekV4FlashMLAAttention,
        )
        from vllm.v1.attention.backends.mla.sparse_mla_env import (
            is_triton_sparse_mla_enabled_for_platform,
            triton_sparse_mla_query_chunk_size,
        )
        from vllm.v1.attention.backends.mla.sparse_mla_kernels import (
            accumulate_indexed_d512_split_sparse_mla_attention,
        )
    except ImportError as exc:
        # The early gate above already confirmed the warmup is requested, so a
        # failed import here is not a benign "kernels unavailable" case — it is
        # usually a renamed symbol (it silently disabled this warmup for weeks).
        # Surface it at WARNING so a future rename does not no-op the warmup.
        logger.warning(
            "Skipping DeepSeek V4 D512-split prefill warmup: a required symbol "
            "failed to import (%s). The split kernels are likely present but a "
            "helper was renamed; the first long prefill will JIT them mid-inference.",
            exc,
        )
        return

    try:
        if not is_triton_sparse_mla_enabled_for_platform():
            return
        if (
            getattr(runner, "max_model_len", 0)
            < envs.VLLM_DEEPSEEK_V4_INDEXED_D512_SPLIT_PREFILL_MIN_TOKENS
        ):
            return

        # The split kernel never sees compress_ratio, so any cr in (4, 128)
        # layer yields identical strides; the first one is representative.
        layer = None
        for module in runner.get_model().modules():
            if (
                isinstance(module, DeepseekV4FlashMLAAttention)
                and module.compress_ratio in (4, 128)
            ):
                layer = module
                break
        if layer is None:
            return

        head_dim = int(layer.head_dim)
        if head_dim != 512:
            return
        num_heads = int(layer.n_local_heads)
        window_size = max(1, int(layer.window_size))
        device = layer.attn_sink.device

        # The width fed to the split kernels at runtime is the per-request
        # combined_topk (combined_indices.shape[-1]), and it is NOT bounded by the
        # static `topk_bound + window_size`. For the C4 indexer layers
        # (compress_ratio=4) that expression IS the width (~640 for DSv4-Flash:
        # indexer top-k 512 + window 128). But the C128A layer (compress_ratio=128)
        # uses a context-dependent `effective_topk` (_c128a_effective_topk_width):
        # a 128-aligned ceiling of `max_pos // compress_ratio` that GROWS with the
        # request's context length up to the split ceiling. So a long-context
        # request sweeps widths 768/896/1024/1152, not just <=640 (observed: all 8
        # widths 256..1152 launched on a 60k-token / mnbt=512 request). The old
        # `min(ceiling, topk_bound+window)` cap (640) therefore left 768..1152 to
        # JIT on the first long request (PR #23 / lennytinkeredapps,
        # max_model_len=1M + mnbt=512). Warm the WHOLE split range so no split-path
        # width can JIT in production; the runtime workspace already accommodates
        # the full range (a 60k request at width 1152 runs correctly). The extra
        # widths cost a few seconds of one-time startup compile — the warmup's
        # purpose.
        c4_static_combined_topk = sparse_prefill_combined_topk_size(
            DeepseekV4FlashMLAAttention._prefill_workspace_topk_bound(layer),
            window_size,
        )
        # Variants whose static C4 width alone already exceeds the split ceiling
        # never use the split path (they route to the chunked path, which is not
        # pre-warmed here); the C128A layer can also exceed it at extreme context.
        if c4_static_combined_topk > _INDEXED_D512_SPLIT_PREFILL_MAX_TOPK:
            logger.warning(
                "DeepSeek V4 D512 prefill: static C4 combined_topk is %d (> %d); "
                "this config routes to the chunked-prefill path, whose kernels are "
                "NOT pre-warmed and may JIT on the first long prefill.",
                c4_static_combined_topk,
                _INDEXED_D512_SPLIT_PREFILL_MAX_TOPK,
            )
        max_topk = _INDEXED_D512_SPLIT_PREFILL_MAX_TOPK
        topk_widths = list(
            range(_INDEXED_D512_SPLIT_PREFILL_MIN_TOPK, max_topk + 1, 128)
        )
        if not topk_widths:
            return

        # The real prefill q buffer is padded to the FP8-decode head count; the
        # split kernel reads only n_local_heads, but stride_q_t (a constexpr in
        # the compile key) reflects the padded width, so match it.
        padded_heads = int(
            getattr(layer, "padded_heads", 0)
            or DeepseekV4FlashMLAAttention.get_padded_num_q_heads(num_heads)
        )
        # T sizes only the launch grid -- the cubin is T-independent -- so keep
        # it small to bound the transient footprint.
        num_tokens = max(1, min(triton_sparse_mla_query_chunk_size(), 32))

        logger.info(
            "Warming up DeepSeek V4 D512-split sparse-MLA prefill kernels for "
            "combined_topk widths=%s (heads=%d, padded_q_heads=%d).",
            topk_widths,
            num_heads,
            padded_heads,
        )

        # Throwaway tensors -- never the shared workspace, so warmup can't grow
        # or leak steady-state memory. q/kv/state are width-independent; scores
        # and indices are contiguous at each per-chunk width so their constexpr
        # strides (stride_scores_h == width, stride_indices_t == width) match the
        # runtime per-chunk workspace exactly.
        q = torch.zeros(
            (num_tokens, padded_heads, head_dim), dtype=torch.bfloat16, device=device
        )
        kv_flat = torch.zeros(
            (max_topk, head_dim), dtype=torch.bfloat16, device=device
        )
        max_score = torch.zeros(
            (num_tokens, num_heads), dtype=torch.float32, device=device
        )
        denom = torch.zeros(
            (num_tokens, num_heads), dtype=torch.float32, device=device
        )
        acc = torch.zeros(
            (num_tokens, num_heads, head_dim), dtype=torch.float32, device=device
        )
        lens = torch.zeros((num_tokens,), dtype=torch.int32, device=device)

        for width in topk_widths:
            # indices=0 (valid row) + lens=width keep every candidate active so
            # the full kernel body, including the tl.dot MMA, compiles rather
            # than an early-return stub.
            indices = torch.zeros(
                (num_tokens, width), dtype=torch.int32, device=device
            )
            scores = torch.zeros(
                (num_tokens, num_heads, width), dtype=torch.float32, device=device
            )
            lens.fill_(width)
            accumulate_indexed_d512_split_sparse_mla_attention(
                q=q,
                kv_flat=kv_flat,
                indices=indices,
                lens=lens,
                scale=layer.scale,
                scores=scores,
                max_score=max_score,
                denom=denom,
                acc=acc,
            )
        torch.accelerator.synchronize()
    except Exception as exc:  # noqa: BLE001 - warmup must never break startup
        # Warn (not debug): a swallowed failure here silently leaves the split
        # kernels uncompiled, so the first long prefill pays the JIT stall again.
        logger.warning(
            "DeepSeek V4 D512-split prefill warmup skipped after error "
            "(first long prefill may JIT in-inference): %s",
            exc,
        )


def _deepseek_v4_sparse_mla_attention_warmup(worker: "Worker") -> None:
    if not envs.VLLM_ENABLE_DEEPSEEK_V4_SPARSE_MLA_WARMUP:
        return

    runner = worker.model_runner
    if runner.is_pooling_model or not _has_deepseek_v4_sparse_mla_backend(runner):
        return

    max_tokens = worker.scheduler_config.max_num_batched_tokens
    mixed_tokens = _clamp_warmup_tokens(
        _DEEPSEEK_V4_SPARSE_MLA_MIXED_WARMUP_TOKENS, max_tokens
    )
    prefill_tokens = _clamp_warmup_tokens(
        _DEEPSEEK_V4_SPARSE_MLA_PREFILL_WARMUP_TOKENS, max_tokens
    )
    uniform_decode_reqs = _deepseek_v4_mtp_uniform_decode_warmup_requests(
        runner,
        max_tokens=max_tokens,
        max_reqs=worker.scheduler_config.max_num_seqs,
    )
    if mixed_tokens <= 0 and prefill_tokens <= 0 and not uniform_decode_reqs:
        return

    logger.info(
        "Warming up DeepSeek V4 sparse MLA attention "
        "for mixed tokens=%s, prefill tokens=%s, and MTP uniform decode "
        "requests=%s.",
        mixed_tokens,
        prefill_tokens,
        list(uniform_decode_reqs),
    )
    if mixed_tokens > 0:
        runner._dummy_run(
            num_tokens=mixed_tokens,
            skip_eplb=True,
            is_profile=True,
            force_attention=True,
            create_mixed_batch=True,
        )
    if prefill_tokens > 0:
        runner._dummy_run(
            num_tokens=prefill_tokens,
            skip_eplb=True,
            is_profile=True,
            force_attention=True,
            create_single_prefill=True,
        )
        # Simulate the second-and-later chunk of a chunked prefill so
        # `_build_prefill_chunk_metadata_kernel` and the alt-shape
        # `_w8a8_triton_block_scaled_mm` configs that fire when the
        # indexer sees prior context get JIT-compiled here, not on the
        # first user request that exceeds `max_num_batched_tokens`.
        runner._dummy_run(
            num_tokens=prefill_tokens,
            skip_eplb=True,
            is_profile=True,
            force_attention=True,
            create_single_prefill=True,
            profile_seq_lens=prefill_tokens * 2,
        )
        # Do not synthesize multi-request prefill here: that dummy shape
        # overflows the CUTeDSL KV-gather workspace on SM12x. Revisit only
        # with a real buffer-sizing fix for that warmup path.

    # The prefill dummies above never drive the C128A indexer, so the
    # D512-split prefill kernels stay uncompiled until the first long request
    # (PR #41834 wedge). Compile them directly with synthetic inputs.
    _deepseek_v4_indexed_d512_split_prefill_warmup(runner)

    query_len = getattr(runner, "uniform_decode_query_len", 0)
    for num_reqs in uniform_decode_reqs:
        runner._dummy_run(
            num_tokens=num_reqs * query_len,
            skip_eplb=True,
            is_profile=True,
            force_attention=True,
            uniform_decode=True,
        )

    if uniform_decode_reqs and current_platform.is_cuda_alike():
        vocab_size = runner.model_config.get_vocab_size()
        block_size = getattr(runner.cache_config, "block_size", None) or 16
        logger.info(
            "Warming up DeepSeek V4 MTP spec-decode kernels for request "
            "counts=%s and %d draft tokens.",
            list(uniform_decode_reqs),
            runner.num_spec_tokens,
        )
        for num_reqs in uniform_decode_reqs:
            _run_deepseek_v4_mtp_spec_decode_warmup_kernels(
                device=runner.device,
                num_reqs=num_reqs,
                num_spec_tokens=runner.num_spec_tokens,
                vocab_size=vocab_size,
                block_size=block_size,
                max_model_len=runner.max_model_len,
                hidden_size=runner.model_config.get_hidden_size(),
            )
        torch.accelerator.synchronize()


def kernel_warmup(worker: "Worker"):
    from vllm.model_executor.warmup.minimax_m3_msa_warmup import (
        minimax_m3_msa_warmup,
    )

    # DSv4 mHC TileLang kernels (hc_pre/hc_post/hc_head_op) run every decoder
    # layer per token; warm them across token sizes first so the first real
    # request doesn't pay JIT cost. No-op for non-DSv4 models (gated inside).
    deepseek_v4_mhc_warmup(
        worker.get_model(),
        max_tokens=worker.scheduler_config.max_num_batched_tokens,
        cudagraph_capture_sizes=(
            worker.vllm_config.compilation_config.cudagraph_capture_sizes or []
        ),
    )

    # Run next so input-prep kernels JIT against pristine runner state.
    flashinfer_sparse_mla_decode_autotune_warmup(worker)
    deepseek_v4_sparse_mla_attention_warmup(worker)

    # Deep GEMM warmup
    do_deep_gemm_warmup = (
        envs.VLLM_USE_DEEP_GEMM
        and is_deep_gemm_supported()
        and envs.VLLM_DEEP_GEMM_WARMUP != "skip"
    )
    if do_deep_gemm_warmup:
        model = worker.get_model()
        max_tokens = worker.scheduler_config.max_num_batched_tokens
        deep_gemm_warmup(model, max_tokens)

    minimax_m3_msa_warmup(worker)

    _deepseek_v4_sparse_mla_attention_warmup(worker)
    _deepseek_v4_request_prep_warmup(worker)

    enable_flashinfer_autotune = (
        worker.vllm_config.kernel_config.enable_flashinfer_autotune
    )
    # FlashInfer autotune for Hopper (SM 9.0) and Blackwell (SM 10.0) GPUs
    if enable_flashinfer_autotune is False:
        logger.info("Skipping FlashInfer autotune because it is disabled.")
    elif has_flashinfer() and current_platform.has_device_capability(90):
        flashinfer_autotune(worker.model_runner)

    # FlashInfer attention warmup
    # Only warmup if the model has FlashInfer attention groups
    # and is not a pooling model
    def _is_flashinfer_backend(backend):
        try:
            return backend.get_name() == "FLASHINFER"
        except NotImplementedError:
            return False

    if (
        not worker.model_runner.is_pooling_model
        and worker.model_runner.attn_groups
        # NOTE: This should be `any` instead of `all` but other hybrid attention
        # backends don't support this dummy run. Once we remove
        # `build_for_cudagraph_capture`, we can change it to `any`.
        and all(
            _is_flashinfer_backend(group.backend)
            for groups in worker.model_runner.attn_groups
            for group in groups
        )
    ):
        logger.info("Warming up FlashInfer attention.")
        # Warmup with mixed batch containing both prefill and decode tokens
        # This is to warm up both prefill and decode attention kernels
        worker.model_runner._dummy_run(
            num_tokens=16,
            skip_eplb=True,
            is_profile=True,
            force_attention=True,
            create_mixed_batch=True,
        )


# TODO: remove once FlashInfer upstream fixes the persistent file cache
# to resolve collisions like `use_8x4_sf_layout=True/False`, which causes
# invalid tactics to be chosen
_FLASHINFER_USE_PERSISTENT_CACHE = False


def flashinfer_autotune(runner: "GPUModelRunner") -> None:
    """
    Autotune FlashInfer operations.
    FlashInfer have many implementations for the same operation,
    autotuning runs benchmarks for each implementation and stores
    the results. The results are cached transparently and
    future calls to FlashInfer will use the best implementation.
    Without autotuning, FlashInfer will rely on heuristics, which may
    be significantly slower.

    Tuning is performed only on rank 0. The resulting cache is broadcast
    to every rank so all ranks dispatch the same kernel tactic.
    """
    import vllm.utils.flashinfer as fi_utils
    from vllm.distributed.parallel_state import get_world_group

    if not _FLASHINFER_USE_PERSISTENT_CACHE:
        with torch.inference_mode(), fi_utils.autotune():
            runner._dummy_run(
                num_tokens=runner.scheduler_config.max_num_batched_tokens,
                skip_eplb=True,
                is_profile=True,
            )
        get_world_group().barrier()
        return

    world = get_world_group()
    is_leader = world.rank_in_group == 0

    cache_path = resolve_flashinfer_autotune_file(runner)
    if is_leader:
        logger.info("Using FlashInfer autotune cache file: %s", cache_path)

    # We skip EPLB here since we don't want to record dummy metrics.
    # When autotuning with number of tokens m, flashinfer will autotune
    # operations for all number of tokens up to m, so we only need to
    # run with the max number of tokens.
    dummy_run_kwargs = dict(
        num_tokens=runner.scheduler_config.max_num_batched_tokens,
        skip_eplb=True,
        is_profile=True,
    )

    with torch.inference_mode():
        if is_leader:
            with fi_utils.autotune(tune_mode=True, cache=str(cache_path)):
                runner._dummy_run(**dummy_run_kwargs)
        else:
            runner._dummy_run(**dummy_run_kwargs)

    # Broadcast autotune cache from rank 0 to all other ranks so every
    # rank loads the same set of chosen tactics.
    tune_results: bytes | None = None
    if is_leader and cache_path.exists():
        with open(cache_path, "rb") as f:
            tune_results = f.read()

    tune_results = world.broadcast_object(tune_results, src=0)

    if tune_results is None:
        logger.warning(
            "No FlashInfer autotune cache entries found."
            "Falling back to default tactics."
        )
    else:
        write_flashinfer_autotune_cache(cache_path, tune_results)
        world.barrier()
        from flashinfer.autotuner import AutoTuner

        AutoTuner.get().load_configs(str(cache_path))
        logger.info(
            "FlashInfer autotune cache loaded on rank %d from %s.",
            world.rank_in_group,
            cache_path,
        )
