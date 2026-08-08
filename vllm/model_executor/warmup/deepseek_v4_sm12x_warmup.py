# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM12x DeepSeek-V4 warmup passes carried by this fork.

Upstream splits warmup into one module per concern; these are the DSv4 passes
that upstream does not have (or has only in a narrower form), kept here so
``kernel_warmup.py`` stays close to upstream and future upstream warmup work
does not collide with ours.

- MTP spec-decode draft/verify kernels and uniform-decode shapes.
- Slot-mapping prep (V1 and V2 runners) and the structured-output bitmask.
- Indexed-D512 split prefill across the full effective-topk range.
- SM12x paged-MQA rowwise decode logits (JITs mid-inference otherwise).
- Sparse-MLA attention: a superset of upstream's mixed-token pass, adding the
  prefill and MTP uniform-decode shapes.
"""

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.model_executor.warmup.flashinfer_sparse_mla_warmup import (
    _attention_backend_name,
    _clamp_warmup_tokens,
    _has_deepseek_v4_sparse_mla_backend,
)
from vllm.platforms import current_platform
from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
from vllm.v1.structured_output.utils import apply_grammar_bitmask

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    from vllm.v1.worker.gpu_worker import Worker

logger = init_logger(__name__)


# Backend names that mark a DSv4 sparse-MLA attn group as live. Shared with
# flashinfer_sparse_mla_warmup so the two warmup gates cannot drift on a
# backend rename (the old local "V4_FLASHMLA_SPARSE" was renamed
# "FLASHMLA_SPARSE_DSV4" upstream and only kept matching via DEEPSEEK_SPARSE_SWA).
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
    # The DeepSeek-V4 slot-mapping warmup runs on BOTH GPU model runners. The V2
    # runner exposes ``block_tables`` + ``input_buffers``; the V1 runner exposes
    # ``input_batch.block_table`` + ``query_start_loc``/``positions``. Dispatch on
    # whichever interface the runner provides (V1 is our production default) so
    # the warmup never silently no-ops and reintroduces first-request JIT.
    block_tables = getattr(runner, "block_tables", None)
    if block_tables is not None:
        _deepseek_v4_slot_mapping_warmup_v2(runner, block_tables)
    elif getattr(runner, "input_batch", None) is not None:
        _deepseek_v4_slot_mapping_warmup_v1(runner)


def _deepseek_v4_slot_mapping_warmup_v2(
    runner: "GPUModelRunner", block_tables: Any
) -> None:
    max_tokens = getattr(runner, "max_num_tokens", 1)
    input_buffers = getattr(runner, "input_buffers", None)
    idx_mapping = torch.zeros(1, dtype=torch.int32, device=runner.device)

    # Snapshot the runner buffers we mutate so warmup never leaks state into
    # the first real request.
    saved_query_start_loc_gpu: torch.Tensor | None = None
    query_start_loc_buf = None
    if input_buffers is not None:
        query_start_loc_buf = input_buffers.query_start_loc
        saved_query_start_loc_gpu = query_start_loc_buf[:2].clone()

    try:
        for requested_tokens in _DEEPSEEK_V4_SLOT_MAPPING_WARMUP_TOKENS:
            num_tokens = _clamp_warmup_tokens(requested_tokens, max_tokens)
            if num_tokens <= 0:
                continue

            positions_source = torch.arange(
                num_tokens, dtype=torch.int64, device=runner.device
            )
            if query_start_loc_buf is not None:
                query_start_loc_buf[:2].copy_(
                    torch.tensor(
                        [0, num_tokens], dtype=torch.int32, device=runner.device
                    )
                )
                query_start_loc = query_start_loc_buf[:2]
            else:
                query_start_loc = torch.tensor(
                    [0, num_tokens], dtype=torch.int32, device=runner.device
                )

            positions_buf = (
                None if input_buffers is None else input_buffers.positions[:num_tokens]
            )
            if positions_buf is not None:
                saved_positions: torch.Tensor | None = positions_buf.clone()
                positions_buf.copy_(positions_source)
                positions = positions_buf
            else:
                saved_positions = None
                positions = positions_source

            try:
                block_tables.compute_slot_mappings(
                    idx_mapping,
                    query_start_loc,
                    positions,
                    num_tokens_padded=num_tokens,
                )
            finally:
                if saved_positions is not None:
                    assert positions_buf is not None
                    positions_buf.copy_(saved_positions)
    finally:
        if saved_query_start_loc_gpu is not None:
            assert query_start_loc_buf is not None
            query_start_loc_buf[:2].copy_(saved_query_start_loc_gpu)


def _deepseek_v4_slot_mapping_warmup_v1(runner: "GPUModelRunner") -> None:
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
        logger.warning("DeepSeek V4 MTP shared-head RMSNorm warmup skipped: %s", exc)


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
            if isinstance(
                module, DeepseekV4FlashMLAAttention
            ) and module.compress_ratio in (4, 128):
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
        kv_flat = torch.zeros((max_topk, head_dim), dtype=torch.bfloat16, device=device)
        max_score = torch.zeros(
            (num_tokens, num_heads), dtype=torch.float32, device=device
        )
        denom = torch.zeros((num_tokens, num_heads), dtype=torch.float32, device=device)
        acc = torch.zeros(
            (num_tokens, num_heads, head_dim), dtype=torch.float32, device=device
        )
        lens = torch.zeros((num_tokens,), dtype=torch.int32, device=device)

        for width in topk_widths:
            # indices=0 (valid row) + lens=width keep every candidate active so
            # the full kernel body, including the tl.dot MMA, compiles rather
            # than an early-return stub.
            indices = torch.zeros((num_tokens, width), dtype=torch.int32, device=device)
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


def _deepseek_v4_paged_mqa_rowwise_decode_warmup(runner: "GPUModelRunner") -> None:
    """Force-compile ``_fp8_paged_mqa_logits_rowwise_kernel`` (SM12x decode).

    The decode dummy runs set ``seq_lens = max_query_len`` (3 for MTP2,
    gpu_model_runner.py:6044), so ``DeepseekV4Indexer.forward`` short-circuits
    on ``max_seq_len // compress_ratio <= topk_tokens`` (attention.py:912) and
    this kernel is never launched during warmup; it JITs inside the first
    request whose context exceeds ``compress_ratio * index_topk``.

    Compile-variant accounting (verified against the kernel signature at
    sm12x_mqa.py:296-331 and the launch at :464-500): every stride argument is
    a ``tl.constexpr`` except ``stride_lm``, and all constexprs are pinned by
    the config, so the only free axes are the four runtime ints, which Triton
    specializes on ``== 1`` and ``% 16 == 0``:

      * ``token_start`` is always 0 in the reachable path
        (``_fp8_paged_mqa_logits_sm12x`` calls the wrapper with defaults).
        The chunked direct-topk path (sm12x_deep_gemm_fallbacks.py:638) needs
        ``num_padded_tokens * logits_width * 4 > 256 MB``; the budget here tops
        out at 192 * 12288 * 4 = 9.4 MB, and even if it were hit its
        token_starts are 8192-multiples, i.e. the same specialization. 1 value.
      * ``num_rows = batch * next_n``: three classes (== 1, % 16 == 0, neither),
        all reachable. Covered by iterating BATCH counts (1, 3, 16) rather than
        row counts, so the classes stay covered for next_n > 1 as well, where
        ``num_rows`` is always a multiple of next_n.
      * ``logits_width`` (== ``token_count``) and ``stride_lm``
        (== ``logits.stride(0)`` == ``token_count``) always move together, so
        they contribute one factor of two: 16-aligned (e.g. the 12288 ceiling)
        vs not (``logits_width`` floors at topk_tokens=512 but otherwise tracks
        the batch's ``max_seq_len``, an arbitrary integer). Never == 1.

    6 launches per reachable indexer-KV geometry. On DSv4-Flash at
    max_model_len=49152 only the compress_ratio=4 geometry is reachable
    (49152 // 128 = 384 <= index_topk 512 makes every C128A layer take the
    short-context short-circuit), so 6 compiles total.

    Fidelity notes: ``stride_btb`` is a constexpr, so the block-table row width
    is taken from the live ``DeepseekV32IndexerMetadataBuilder`` buffer rather
    than recomputed (the builder's width carries ``get_kv_cache_shard_count()``
    under DCP; a formula that drops it would bake a stride production never
    uses and silently warm the wrong cubin). The KV strides likewise come from
    the bound cache tensor, which is a PADDED strided view (page stride 8640,
    not 64*132 = 8448), so a synthetic contiguous cache would miss.
    """
    try:
        from vllm.model_executor.layers.sparse_attn_indexer import (
            kv_cache_as_quant_view,
        )
        from vllm.models.deepseek_v4.attention import DeepseekV4Indexer
        from vllm.models.deepseek_v4.nvidia.ops.sm12x_mqa import (
            fp8_paged_mqa_logits_triton,
        )
        from vllm.utils.math_utils import cdiv
        from vllm.v1.attention.backends.mla.indexer import (
            DeepseekV32IndexerMetadataBuilder,
        )
        from vllm.v1.worker.cp_utils import get_kv_cache_shard_count
    except ImportError as exc:
        # A failed import here is a renamed symbol, not a benign "kernels
        # unavailable" case; surface it so a rename cannot silently no-op the
        # warmup (see _deepseek_v4_indexed_d512_split_prefill_warmup).
        logger.warning(
            "Skipping SM12x paged-MQA rowwise decode warmup: a required symbol "
            "failed to import (%s); the first long-context decode will JIT it "
            "mid-inference.",
            exc,
        )
        return

    # Only SM12x routes paged-MQA logits onto the Triton fallback
    # (vllm/utils/deep_gemm.py: fp8_fp4_paged_mqa_logits).
    if not (
        current_platform.is_cuda()
        and current_platform.is_device_capability_family(120)
    ):
        return

    try:
        num_spec = int(getattr(runner, "num_spec_tokens", 0) or 0)
        next_n_decode = 1 + num_spec
        # DeepseekV32IndexerMetadataBuilder.use_flattening (indexer.py:537):
        # outside the SM100 family every next_n not in (1, 2) is flattened to
        # one row per decode token, so the kernel only ever sees next_n == 1
        # (MTP2 included). For next_n in (1, 2) the native layout survives and
        # next_n is the batch's max_decode_len, which can be 1 or 2.
        if next_n_decode in (1, 2):
            next_n_values = tuple(sorted({1, next_n_decode}))
        else:
            next_n_values = (1,)

        # stride_btb is a constexpr. Prefer the live builder buffer (exact
        # production stride and alignment); keep the shard-aware formula as a
        # fallback, and warm both if they ever disagree.
        bt_widths: set[int] = set()
        for group_list in getattr(runner, "attn_groups", []):
            for group in group_list:
                for builder in getattr(group, "metadata_builders", []):
                    if isinstance(builder, DeepseekV32IndexerMetadataBuilder):
                        bt_widths.add(
                            int(builder.expanded_block_table_buffer.stride(0))
                        )
        bt_widths.add(
            cdiv(
                runner.max_model_len,
                runner.cache_config.block_size * get_kv_cache_shard_count(),
            )
        )
        bt_widths = {w for w in bt_widths if w > 0}
        if not bt_widths:
            return

        fp8_dtype = current_platform.fp8_dtype()
        seen: set[tuple[int, int, int, int]] = set()
        warmed = False

        for module in runner.get_model().modules():
            if not isinstance(module, DeepseekV4Indexer) or module.use_fp4_kv:
                continue
            # module.max_model_len is already max_model_len // compress_ratio.
            # When it cannot exceed index_topk the layer always takes the
            # short-context path and never reaches paged MQA (DSv4-Flash C128A:
            # 49152 // 128 = 384 <= 512).
            if module.max_model_len <= module.topk_tokens:
                continue
            kv_cache = getattr(module.k_cache, "kv_cache", None)
            if kv_cache is None or kv_cache.numel() == 0:
                continue

            num_heads = int(module.n_head)
            head_dim = int(module.head_dim)
            key = (
                int(kv_cache.shape[1]),
                int(kv_cache.stride(0)),
                num_heads,
                head_dim,
            )
            if key in seen:
                continue
            seen.add(key)

            device = kv_cache.device
            # Exactly what sparse_attn_indexer feeds the kernel.
            kv_view = kv_cache_as_quant_view(kv_cache, head_dim, False)

            logger.info(
                "Warming up SM12x paged-MQA rowwise decode logits "
                "(heads=%d, head_dim=%d, kv_block=%d, page_stride=%d, "
                "block_table_widths=%s).",
                num_heads,
                head_dim,
                key[0],
                key[1],
                sorted(bt_widths),
            )

            for bt_width in sorted(bt_widths):
                for next_n in next_n_values:
                    # num_rows = batch * next_n. Batches (1, 3, 16) give the
                    # == 1 / non-16-aligned / 16-aligned classes at next_n == 1
                    # and the non-16-aligned / 16-aligned classes at next_n == 2
                    # (num_rows == 1 is unreachable there).
                    for batch in (1, 3, 16):
                        num_rows = batch * next_n
                        q = torch.zeros(
                            (batch, next_n, num_heads, head_dim),
                            dtype=fp8_dtype,
                            device=device,
                        )
                        weights = torch.zeros(
                            (num_rows, num_heads),
                            dtype=torch.float32,
                            device=device,
                        )
                        block_tables = torch.zeros(
                            (batch, bt_width), dtype=torch.int32, device=device
                        )
                        # logits_width classes: 16-aligned / not. The value only
                        # sizes the grid; the cubin depends on the class alone.
                        for width in (512, 513):
                            context_lens = torch.full(
                                (batch, next_n),
                                width,
                                dtype=torch.int32,
                                device=device,
                            )
                            fp8_paged_mqa_logits_triton(
                                q,
                                kv_view,
                                weights,
                                context_lens,
                                block_tables,
                                width,  # max_model_len == logits_width
                            )  # token_start=0, token_count=None
                            warmed = True

        if warmed:
            torch.accelerator.synchronize()
    except Exception as exc:  # noqa: BLE001 - warmup must never break startup
        logger.warning(
            "SM12x paged-MQA rowwise decode warmup skipped after error "
            "(first long-context decode may JIT in-inference): %s",
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

    # Same class of gap on the decode side: the decode dummies below run with
    # seq_lens == max_query_len, so the indexer short-circuits and never
    # reaches the paged-MQA logits kernel.
    _deepseek_v4_paged_mqa_rowwise_decode_warmup(runner)

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


_LL_BF16_WARMUP_MODEL_SHAPES: tuple[tuple[int, int], ...] = (
    (6144, 264),  # Inkling
    (7168, 256),  # DSV3
    (7168, 384),  # DSV4-Pro
    (14400, 256),  # DSV4-Flash
)
_LL_BF16_WARMUP_M_RANGE = range(1, 17)


