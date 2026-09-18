# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Uno shared-model parallel drafting for Model Runner V2."""

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.config.compilation import CUDAGraphMode
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.platforms import current_platform
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig
from vllm.v1.spec_decode.uno_noise import fill_uno_noise
from vllm.v1.worker.gpu.attn_utils import build_slot_mappings_by_layer
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.cudagraph_utils import (
    BatchExecutionDescriptor,
    CudaGraphManager,
    prepare_inputs_to_capture,
)
from vllm.v1.worker.gpu.dp_utils import DPSyncState
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.model_states.interface import ModelState
from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator
from vllm.v1.worker.gpu.spec_decode.uno_prepare import prepare_uno_inputs_fused
from vllm.v1.worker.utils import AttentionGroup

if TYPE_CHECKING:
    from vllm.sampling_params import SamplingParams

logger = init_logger(__name__)
UNO_LORA_ID = 1_000_003


def prepare_uno_inputs_reference(
    buffers: InputBuffers,
    slot_mapping: torch.Tensor,
    sample_idx_mapping: torch.Tensor,
    input_batch: InputBatch,
    num_sampled: torch.Tensor,
    num_rejected: torch.Tensor,
    last_sampled: torch.Tensor,
    next_prefill_tokens: torch.Tensor,
    seeds: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    k: int,
    max_model_len: int,
    noise_seed: int,
    noise_high: int,
    step: int,
) -> None:
    """CPU reference oracle for the fused seed/noise preparation kernel.

    Temporary suffix KV starts after the last verified row. The target overwrites
    it on verification; neither request progress nor prefix ownership advances here.
    """
    n = input_batch.num_reqs
    count = n * k
    state_idx = input_batch.idx_mapping.long()
    valid_end = input_batch.query_start_loc[1 : n + 1] - num_rejected
    first_pos = input_batch.positions[valid_end.long() - 1] + 1
    offsets = torch.arange(k, dtype=torch.int64, device=buffers.device)
    positions = first_pos[:, None] + offsets
    seed_tokens = torch.where(
        num_sampled > 0,
        last_sampled.reshape(-1)[state_idx],
        next_prefill_tokens.reshape(-1, seeds.shape[0])[0, state_idx],
    )
    buffers.input_ids[:count].copy_(seed_tokens.repeat_interleave(k))
    is_noise = (offsets != 0).repeat(n)
    req_seeds = (seeds[state_idx] + noise_seed).repeat_interleave(k)
    fill_uno_noise(buffers.input_ids[:count], is_noise, req_seeds, step, 1, noise_high)
    # Keep unused, out-of-context rows in range for the model. Their KV writes
    # are suppressed below; native verification bounds usable candidates.
    buffers.positions[:count].copy_(positions.clamp(max=max_model_len - 1).flatten())
    buffers.seq_lens[:n].copy_((first_pos + k).clamp(max=max_model_len))
    buffers.seq_lens[n:].zero_()
    torch.arange(n + 1, out=buffers.query_start_loc[: n + 1])
    buffers.query_start_loc[: n + 1].mul_(k)
    buffers.query_start_loc[n + 1 :].fill_(count)
    buffers.input_ids[count:].zero_()
    buffers.positions[count:].zero_()
    sample_idx_mapping[:count].copy_(state_idx.repeat_interleave(k))
    sample_idx_mapping[count:].fill_(-1)

    block_numbers = positions // block_size
    block_ids = (
        block_table[:n]
        .gather(1, block_numbers.clamp(max=block_table.shape[1] - 1))
        .long()
    )
    valid = (
        (positions < max_model_len)
        & (block_numbers < block_table.shape[1])
        & (block_ids > 0)
    )
    slots = block_ids * block_size + positions % block_size
    slot_mapping[:count].copy_(torch.where(valid, slots, PAD_SLOT_ID).flatten())
    slot_mapping[count:].fill_(PAD_SLOT_ID)


def uncovered_draft_request_counts(
    captured_token_counts: Sequence[int],
    k: int,
    max_num_reqs: int,
) -> list[int]:
    """Request counts whose draft batch has no captured graph to pad into.

    A draft batch is ``n * k`` rows and dispatch pads up: a captured graph
    serves any batch at or below its own row count, so ``n`` is covered when
    ``n * k`` is at most the largest captured count. The gap is therefore only
    at the top, and it is silent -- a batch above the largest captured count
    falls back to eager drafting, which is far slower, with nothing said at
    startup. Whether the gap exists depends on ``k``, ``max_num_seqs`` and the
    capture list together: Qwen3-8B at ``max_num_seqs=16`` with the default
    capture sizes covers every request count at ``k=8`` and leaves the top of
    the range uncovered at ``k=3``.
    """
    if k <= 0 or max_num_reqs <= 0:
        return []
    largest = max(captured_token_counts, default=0)
    return [n for n in range(1, max_num_reqs + 1) if n * k > largest]


def draft_warmup_request_counts(max_num_reqs: int) -> list[int]:
    """Request counts whose draft-input kernel must be compiled before serving.

    ``_prepare_uno_inputs_kernel`` takes ``NUM_REQS`` and ``COUNT`` as Triton
    ``constexpr`` arguments, so a batch of three requests and a batch of four
    are different specialisations and each is compiled the first time it is
    seen. Nothing in startup ran the fused path at all -- graph capture uses
    ``prepare_inputs_to_capture`` rather than ``prepare_uno_inputs_fused`` --
    so on an H100 the first served request spent 120 ms inside its draft
    proposal against under 1.2 ms for every step after it, and a cell that
    first reached a new request count paid again mid-run.

    Every count from 1 to ``max_num_seqs`` is reachable in serving, so every
    count is warmed. The list is the shape set, not a sample of it: a warmup
    that covers only the counts one workload happens to hit leaves the next
    workload paying at its own first request.
    """
    if max_num_reqs <= 0:
        return []
    return list(range(1, max_num_reqs + 1))


@dataclass(frozen=True)
class UnoSamplingMode:
    """A served request-state combination that startup must exercise."""

    name: str
    top_k: int | None
    top_p: float | None
    temperature: float = 0.9
    mixed_greedy: bool = False
    logprobs: int | None = None
    penalties: bool = False
    explicit_seed: bool = False

    @property
    def topk_enabled(self) -> bool:
        return self.top_k is not None

    @property
    def topp_enabled(self) -> bool:
        return self.top_p is not None

    @property
    def min_num_reqs(self) -> int:
        return 2 if self.mixed_greedy else 1

    @property
    def needs_logits_processing(self) -> bool:
        return (
            self.temperature not in (0.0, 1.0)
            or self.topk_enabled
            or self.topp_enabled
            or self.penalties
        )

    def sampling_params(self, req_index: int = 0) -> "SamplingParams":
        from vllm import SamplingParams

        temperature = (
            0.0 if self.mixed_greedy and req_index % 2 == 0 else self.temperature
        )
        return SamplingParams(
            max_tokens=2,
            temperature=temperature,
            top_k=-1 if self.top_k is None else self.top_k,
            top_p=1.0 if self.top_p is None else self.top_p,
            logprobs=self.logprobs,
            repetition_penalty=1.1 if self.penalties else 1.0,
            presence_penalty=0.1 if self.penalties else 0.0,
            frequency_penalty=0.1 if self.penalties else 0.0,
            seed=0 if self.explicit_seed else None,
        )


UNO_SAMPLING_MODES = tuple(
    replace(
        mode,
        name=mode.name
        + ("_logprobs" if logprobs is not None else "")
        + ("_penalties" if penalties else ""),
        logprobs=logprobs,
        penalties=penalties,
    )
    for mode in (
        UnoSamplingMode("top_p_only", None, 0.9),
        UnoSamplingMode("top_k_top_p", 50, 0.9),
        UnoSamplingMode("top_k_only", 50, None),
        UnoSamplingMode("neither", None, None),
        UnoSamplingMode("greedy", None, None, temperature=0.0),
        UnoSamplingMode("sampled_unit_temperature", None, None, temperature=1.0),
        UnoSamplingMode("mixed_greedy", 50, 0.9, mixed_greedy=True),
        UnoSamplingMode(
            "mixed_unfiltered", None, None, temperature=1.0, mixed_greedy=True
        ),
    )
    for logprobs in (None, 1)
    for penalties in (False, True)
) + (
    UnoSamplingMode("seeded_top_p_only", None, 0.9, explicit_seed=True),
    UnoSamplingMode("seeded_top_k_top_p", 50, 0.9, explicit_seed=True),
    UnoSamplingMode("seeded_top_k_only", 50, None, explicit_seed=True),
)


def sampler_branch_for_mode(
    mode: UnoSamplingMode,
    *,
    use_flashinfer: bool,
    logprobs_mode: str = "raw_logprobs",
) -> str:
    """Return the ``Sampler._sample_random`` branch serving will take.

    Keep the same request-state conditions as ``Sampler.sample``. Greedy
    sampling still uses the Triton/gumbel path; temperature is runtime data.
    """
    if (
        use_flashinfer
        and (mode.topk_enabled or mode.topp_enabled)
        and mode.temperature != 0.0
        and not mode.mixed_greedy
        and not mode.explicit_seed
        and not (
            mode.logprobs is not None
            and logprobs_mode in ("processed_logits", "processed_logprobs")
        )
    ):
        return "flashinfer"
    return "triton"


@dataclass(frozen=True)
class UnoSamplerWarmup:
    """A real sampler call selected to warm one served sampler branch."""

    num_reqs: int
    num_rows: int
    mode: UnoSamplingMode
    source: str
    sampler_branch: str
    kernel_keys: tuple[tuple[object, ...], ...]


@dataclass(frozen=True)
class UnoServedLaunches:
    """The bounded launch model used by both Uno warmup and CPU contracts."""

    prepare_request_counts: tuple[int, ...]
    sampler_warmups: tuple[UnoSamplerWarmup, ...]
    # This is the raw served domain, separately retained from the minimal
    # representative calls below. It makes a future selector regression fail
    # closed rather than allowing a test to compare a plan to itself.
    served_sampler_keys: frozenset[tuple[object, ...]] = frozenset()

    @property
    def sampler_keys(self) -> frozenset[tuple[object, ...]]:
        return frozenset(
            key for warmup in self.sampler_warmups for key in warmup.kernel_keys
        )


def _triton_integer_bucket(value: int) -> str:
    """Return the plain-integer specialization bucket used by Triton.

    ``BATCH_SIZE`` is a plain integer argument of ``_topk_topp_kernel``.
    It is not in that kernel's ``do_not_specialize`` list, so the binder has
    distinct ``== 1`` and ``% 16 == 0`` cases in addition to the generic
    integer case.
    """
    if value == 1:
        return "one"
    if value % 16 == 0:
        return "multiple_of_16"
    return "generic"


def _served_sampler_shapes(
    max_num_reqs: int,
    k: int,
    max_num_tokens: int,
    max_model_len: int,
) -> tuple[tuple[int, int, str], ...]:
    """Enumerate bounded sampler row layouts the scheduler can create.

    A pure prefill has one logit row per request. Verification has ``K + 1``
    rows per request. Chunked and mixed batches can land at every intermediate
    row count between those endpoints, so they are deliberately represented
    as valid uneven request layouts rather than discarded as non-divisible
    ``num_reqs * K`` shapes.
    """
    if max_num_reqs <= 0:
        return ()
    if k <= 0:
        raise ValueError("Uno sampler launch enumeration requires positive K")
    if max_num_tokens < max_num_reqs:
        raise ValueError(
            "Uno sampler launch enumeration requires one token per request"
        )
    if max_model_len <= 0:
        raise ValueError(
            "Uno sampler launch enumeration requires positive max_model_len"
        )

    # Each verification request has at most K draft rows plus its sampled row.
    # ``max_num_tokens`` bounds scheduler admission and ``max_model_len``
    # bounds each request. Keeping both in the model makes a newly introduced
    # shape axis fail closed until it has a bound here.
    max_rows = min(
        max_num_tokens,
        max_num_reqs * min(k + 1, max_model_len),
    )
    if max_rows < max_num_reqs:
        raise ValueError("Uno sampler row bound excludes reachable prefills")

    shapes: dict[tuple[int, int], str] = {}

    def add(num_reqs: int, num_rows: int, source: str) -> None:
        if num_rows <= min(max_rows, num_reqs * min(k + 1, max_model_len)):
            assert 1 <= num_reqs <= max_num_reqs
            assert num_reqs <= num_rows <= max_num_tokens
            max_request_rows = (num_rows + num_reqs - 1) // num_reqs
            assert max_request_rows <= k + 1
            assert max_request_rows <= max_model_len
            shapes.setdefault((num_reqs, num_rows), source)

    for num_reqs in range(1, max_num_reqs + 1):
        add(num_reqs, num_reqs, "pure_prefill")
        add(num_reqs, num_reqs * (k + 1), "verification")

    # A mixed/chunked batch can distribute any valid total over active request
    # slots. The warmup synthesizes that exact uneven layout in InputBatch.
    for num_rows in range(1, max_rows + 1):
        add(min(num_rows, max_num_reqs), num_rows, "chunked_or_mixed")
        if num_rows > 1:
            # The same total can be a pure prefill or a partial verification.
            # Keep a verification representative even below max_num_reqs.
            add(min(num_rows - 1, max_num_reqs), num_rows, "chunked_or_mixed")

    return tuple(
        (num_reqs, num_rows, source)
        for (num_reqs, num_rows), source in sorted(shapes.items())
    )


def _served_logprobs_counts(max_num_logprobs: int) -> tuple[int, ...]:
    """Cover each bounded logprob gather width and plain-integer bucket."""
    from vllm.v1.worker.gpu.sample.logprob import _MAX_TOPK_BLOCK

    if max_num_logprobs < 0:
        raise ValueError("Uno logprobs warmup requires a finite nonnegative bound")
    representatives: dict[tuple[int, str], int] = {}
    for count in range(max_num_logprobs + 1):
        assert 0 <= count <= max_num_logprobs
        num_columns = count + 1
        block_size = min(1 << (num_columns - 1).bit_length(), _MAX_TOPK_BLOCK)
        key = (block_size, _triton_integer_bucket(num_columns))
        representatives.setdefault(key, count)
    return tuple(representatives.values())


def _sampler_kernel_keys(
    mode: UnoSamplingMode,
    num_rows: int,
    num_sm: int,
    sampler_branch: str,
    logprobs_mode: str = "raw_logprobs",
) -> tuple[tuple[object, ...], ...]:
    """Return varying filter and rejection keys for one configured engine.

    Rejection keeps the engine's draft-logits, block-verification, synthetic,
    FP64, vocab and K settings. Its three target-logits consumers additionally
    specialize on the pointer dtype: processing copies to FP32, while greedy
    and unit-temperature unfiltered batches retain the model's logits dtype.
    """
    if sampler_branch not in ("flashinfer", "triton", "native_verification"):
        raise ValueError(f"unknown Uno sampler branch: {sampler_branch!r}")
    # Use the production split arithmetic and branch threshold rather than
    # recreating either in a CPU-only test.
    from vllm.v1.sample.ops.topk_topp_triton import (
        _SPLIT_MAX_BATCH,
        _topp_split_count,
    )
    from vllm.v1.worker.gpu.sample.logprob import _MAX_TOPK_BLOCK

    keys: list[tuple[object, ...]] = []
    logits_dtype = "fp32" if mode.needs_logits_processing else "model_dtype"
    if mode.penalties:
        keys.append(("_penalties_kernel",))
    if mode.temperature not in (0.0, 1.0):
        keys.append(("_temperature_kernel",))
    if mode.logprobs is not None:
        logprobs_dtype = (
            logits_dtype
            if logprobs_mode in ("processed_logits", "processed_logprobs")
            else "model_dtype"
        )
        if logprobs_mode not in ("raw_logits", "processed_logits"):
            num_columns = mode.logprobs + 1
            block_size = min(1 << (num_columns - 1).bit_length(), _MAX_TOPK_BLOCK)
            keys.append(
                (
                    "_topk_log_softmax_kernel",
                    logprobs_dtype,
                    block_size,
                    _triton_integer_bucket(num_columns),
                )
            )
        keys.append(("_ranks_kernel", logprobs_dtype))
        if sampler_branch == "native_verification":
            keys.append(("_flatten_sampled_kernel",))
    if sampler_branch == "flashinfer":
        return tuple(keys)
    if sampler_branch == "triton":
        keys.append(("_gumbel_sample_kernel", logits_dtype))
    if sampler_branch == "native_verification":
        keys.extend(
            (name, "target_logits_dtype", logits_dtype)
            for name in (
                "_compute_local_logits_stats_kernel",
                "_rejection_kernel",
                "_resample_kernel",
            )
        )
    if not mode.topk_enabled and not mode.topp_enabled:
        return tuple(keys)

    use_split = mode.topp_enabled and num_rows <= _SPLIT_MAX_BATCH
    if not (use_split and not mode.topk_enabled):
        keys.append(
            (
                "_topk_topp_kernel",
                _triton_integer_bucket(num_rows),
                mode.topk_enabled,
                mode.topp_enabled,
                use_split,
            )
        )
    if use_split:
        split_count = _topp_split_count(num_rows, num_sm)
        has_k = mode.topk_enabled
        keys.extend(
            (kernel, has_k, split_count)
            for kernel in (
                "_topp_sb_stats_kernel",
                "_topp_sb_step_kernel",
                "_topp_sb_mask_kernel",
            )
        )
    return tuple(keys)


def enumerate_uno_served_launches(
    *,
    max_num_reqs: int,
    k: int,
    max_num_tokens: int,
    max_model_len: int,
    num_sm: int,
    use_flashinfer: bool,
    logprobs_mode: str = "raw_logprobs",
    max_num_logprobs: int = 20,
) -> UnoServedLaunches:
    """Build the complete bounded Uno warmup plan from serving behavior.

    This is production code: startup drives these calls and the CPU contract
    reads the same plan. The candidate row domain covers pure prefill,
    verification, and all chunked/mixed intermediates. One valid call is kept
    for each actual Triton key, including the plain-integer ``== 1`` and
    ``% 16 == 0`` buckets and every split-count specialization.
    """
    if num_sm <= 0:
        raise ValueError("Uno sampler launch enumeration requires positive num_sm")

    prepare_counts = tuple(draft_warmup_request_counts(max_num_reqs))
    shapes = _served_sampler_shapes(max_num_reqs, k, max_num_tokens, max_model_len)
    logprobs_counts = _served_logprobs_counts(max_num_logprobs)
    # Verification always applies native filters in RejectionSampler._verify.
    # Select it first, then ordinary sampling fills any remaining shared keys.
    candidates = (
        UnoSamplerWarmup(
            num_reqs,
            num_rows,
            mode,
            source,
            branch,
            _sampler_kernel_keys(mode, num_rows, num_sm, branch, logprobs_mode),
        )
        for declared_mode in UNO_SAMPLING_MODES
        for mode in (
            tuple(replace(declared_mode, logprobs=count) for count in logprobs_counts)
            if declared_mode.logprobs is not None
            else (declared_mode,)
        )
        for branch in (
            "native_verification",
            sampler_branch_for_mode(
                mode, use_flashinfer=use_flashinfer, logprobs_mode=logprobs_mode
            ),
        )
        for num_reqs, num_rows, source in shapes
        if num_reqs >= mode.min_num_reqs
        and (num_rows > num_reqs) == (branch == "native_verification")
    )
    served_sampler_keys: set[tuple[object, ...]] = set()
    served_modes: set[tuple[str, str]] = set()
    covered: set[tuple[object, ...]] = set()
    selected: list[UnoSamplerWarmup] = []
    selected_modes: set[tuple[str, str]] = set()
    for candidate in candidates:
        mode_key = (candidate.mode.name, candidate.sampler_branch)
        served_sampler_keys.update(candidate.kernel_keys)
        served_modes.add(mode_key)
        if (
            set(candidate.kernel_keys).difference(covered)
            or mode_key not in selected_modes
        ):
            selected.append(candidate)
            covered.update(candidate.kernel_keys)
            selected_modes.add(mode_key)

    selected_keys = frozenset(key for warmup in selected for key in warmup.kernel_keys)
    missing = served_sampler_keys.difference(selected_keys)
    missing_modes = served_modes.difference(selected_modes)
    if missing or missing_modes:
        raise RuntimeError(
            "Uno sampler warmup selection missed served Triton keys: "
            f"{sorted(missing)!r}; missing modes: {sorted(missing_modes)!r}"
        )
    return UnoServedLaunches(
        prepare_counts, tuple(selected), frozenset(served_sampler_keys)
    )


class UnoSpeculator(DraftModelSpeculator):
    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        if device.type != "cuda" or not current_platform.is_cuda():
            raise ValueError("Uno currently requires an NVIDIA CUDA device")
        super().__init__(vllm_config, device)
        self.k = self.num_speculative_steps
        if self.max_num_reqs * self.k > self.max_num_tokens:
            raise ValueError(
                "Uno requires max_num_batched_tokens >= "
                "max_num_seqs * num_speculative_tokens"
            )
        assert self.speculative_config.uno_lora_path is not None
        self.lora_request = LoRARequest(
            "uno", UNO_LORA_ID, self.speculative_config.uno_lora_path
        )
        self._lora_hook: Callable[[tuple[int, int] | None], None] | None = None
        self.sample_idx_mapping = torch.full(
            (self.max_num_reqs * self.k,), -1, dtype=torch.int32, device=device
        )
        self.sample_col = torch.arange(self.k, dtype=torch.int32, device=device).repeat(
            self.max_num_reqs
        )
        self.cudagraph_manager: CudaGraphManager | None = None
        self._graph_attn_metadata: dict[BatchExecutionDescriptor, dict[str, Any]] = {}
        self._step = 0
        self.num_graph_replays = 0
        self.num_eager_proposals = 0
        # Proposals made while profiling or capturing are counted apart from
        # serving ones. They used to share the counters, so the one-time
        # "using eager execution" line was always emitted by startup warmup and
        # a later fallback in serving could never announce itself.
        self.num_warmup_proposals = 0

    def set_lora_hook(self, hook: Callable[[tuple[int, int] | None], None]) -> None:
        self._lora_hook = hook

    @contextmanager
    def _draft_lora(self, num_reqs: int, num_tokens: int) -> Iterator[None]:
        if self._lora_hook is None:
            raise RuntimeError("Uno adapter routing has not been initialized")
        try:
            self._lora_hook((num_reqs, num_tokens))
            yield
        finally:
            self._lora_hook(None)

    def load_draft_model(
        self, target_model: nn.Module, target_attn_layer_names: set[str]
    ) -> nn.Module:
        return target_model

    def load_model(self, target_model: nn.Module) -> None:
        self.model = target_model
        self._validate_local_argmax_reduction()
        layers = get_layers_from_vllm_config(
            self.vllm_config,
            AttentionLayerBase,  # type: ignore[type-abstract]
        )
        self.draft_attn_layer_names = {
            name
            for name, layer in layers.items()
            if layer.get_kv_cache_spec(self.vllm_config) is not None
        }
        if not self.draft_attn_layer_names:
            raise ValueError("Uno requires shared target attention with a KV cache")
        for name in self.draft_attn_layer_names:
            if layers[name].get_attn_backend().get_name() != "FLASH_ATTN":
                raise ValueError("Uno currently requires the FLASH_ATTN backend")
        self.supports_mm_inputs = False

    def set_attn(
        self,
        model_state: ModelState,
        kv_cache_config: KVCacheConfig,
        block_tables: BlockTables,
        target_input_buffers: InputBuffers,
        target_attn_groups: list[list[AttentionGroup]],
    ) -> None:
        groups = kv_cache_config.kv_cache_groups
        if len(groups) != 1:
            raise ValueError("Uno requires one homogeneous full-attention KV group")
        spec = groups[0].kv_cache_spec
        if (
            type(spec) is not FullAttentionSpec
            or spec.sliding_window is not None
            or spec.attention_chunk_size is not None
        ):
            raise ValueError("Uno requires homogeneous full attention")
        super().set_attn(
            model_state,
            kv_cache_config,
            block_tables,
            target_input_buffers,
            target_attn_groups,
        )
        if any(
            not group.supports_draft_decode_metadata_update
            for groups in self.attn_groups
            for group in groups
        ):
            raise ValueError(
                "Uno requires attention metadata builders supporting "
                "native draft decode updates"
            )

    def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode) -> None:
        self._graph_attn_metadata.clear()
        can_capture = (
            cudagraph_mode.decode_mode() == CUDAGraphMode.FULL
            and self.attn_cg_support.min_cg_support.value
            >= AttentionCGSupport.UNIFORM_BATCH.value
        )
        if not can_capture and cudagraph_mode.decode_mode() == CUDAGraphMode.FULL:
            logger.warning_once(
                "Uno draft CUDA graphs require uniform-batch attention support; "
                "using eager drafting."
            )
        self.cudagraph_manager = CudaGraphManager(
            self.vllm_config,
            self.device,
            CUDAGraphMode.FULL_DECODE_ONLY if can_capture else CUDAGraphMode.NONE,
            decode_query_len=self.k,
            lora_capture_cases=[2 if self.k > 1 else 0],
        )

    def capture(self) -> None:
        assert self.cudagraph_manager is not None
        self.sample_idx_mapping.fill_(-1)
        self.input_buffers.input_ids.zero_()
        self.input_buffers.positions.zero_()

        def create_forward_fn(desc: BatchExecutionDescriptor, warmup: bool):
            assert desc.num_reqs is not None
            attn_metadata, slots = prepare_inputs_to_capture(
                desc.num_reqs,
                desc.num_tokens,
                self.model_state,
                self.input_buffers,
                self.block_tables,
                self.attn_groups,
                self.kv_cache_config,
                full_cudagraph=True,
                max_query_len=self.k,
            )
            assert attn_metadata is not None
            if not warmup:
                self._graph_attn_metadata[desc] = attn_metadata
            assert self._lora_hook is not None
            self._lora_hook((desc.num_reqs, desc.num_tokens))
            return lambda mode: self._generate_draft(
                desc.num_reqs, desc.num_tokens, attn_metadata, slots
            )

        try:
            self.cudagraph_manager.capture(
                create_forward_fn, "Capturing Uno CUDA graphs"
            )
        finally:
            if self._lora_hook is not None:
                self._lora_hook(None)
        self._log_draft_graph_coverage()

    def draft_warmup_token_counts(self) -> list[int]:
        """Dummy-run token counts that compile every draft-input shape.

        ``_dummy_run`` derives ``num_reqs`` as ``min(num_tokens, max_num_seqs)``
        and its draft proposal runs the real fused-input path, so asking for
        ``n`` tokens compiles the specialisation serving will use at ``n``
        concurrent requests. Returning the counts rather than running them
        keeps this arithmetic testable without a GPU.
        """
        return draft_warmup_request_counts(self.max_num_reqs)

    def report_draft_warmup(
        self,
        prepare_shapes: int,
        sampler_calls: int,
        sampler_keys: int,
        sampler_branches: Mapping[str, str],
        sampler_kernels: Mapping[str, tuple[str, ...]],
    ) -> None:
        """State once what warmup covered, so a cold serve stays visible.

        A silent warmup is worse than none: if the shapes it compiles are not
        the shapes serving asks for, nothing says so and the latency reads as
        the model. Naming the count here puts it beside the JIT monitor's own
        "compilation during inference" warnings, so one log answers whether
        warmup covered what the requests asked for.
        """
        for mode, branch in sampler_branches.items():
            kernels = ",".join(sampler_kernels.get(mode, ())) or "none"
            logger.info(
                "Uno sampler warmup backend: mode=%s branch=%s kernels=%s",
                mode,
                branch,
                kernels,
            )
        logger.info(
            "Uno draft kernels warmed: %d prepare request shapes (1..%d "
            "requests at num_speculative_tokens=%d), %d sampler calls, and "
            "%d sampler launch keys; sampling modes=%s. Any later 'JIT "
            "compilation during inference' warning names a shape this missed.",
            prepare_shapes,
            self.max_num_reqs,
            self.k,
            sampler_calls,
            sampler_keys,
            ",".join(
                sorted(
                    {mode.removeprefix("verification/") for mode in sampler_branches}
                )
            ),
        )

    def _log_draft_graph_coverage(self) -> None:
        """Say once, at startup, which request counts have a draft graph.

        Drafting eagerly costs far more per step than replaying a graph, and
        the fallback is chosen per step with no other announcement: the
        one-time eager line can only fire once, and it says nothing about the
        rest of the range. A deployment that will draft eagerly at its own
        concurrency should learn that at startup rather than from its latency.
        """
        assert self.cudagraph_manager is not None
        captured = sorted({desc.num_tokens for desc in self.cudagraph_manager.graphs})
        uncovered = uncovered_draft_request_counts(captured, self.k, self.max_num_reqs)
        if not captured:
            logger.info(
                "Uno draft CUDA graphs: none captured, so every proposal "
                "drafts eagerly. A draft graph needs a cudagraph_capture_sizes "
                "entry of at least %d (num_speculative_tokens), and the "
                "largest usable entry is bounded by max_num_seqs * "
                "num_speculative_tokens = %d.",
                self.k,
                self.max_num_reqs * self.k,
            )
            return
        if uncovered:
            logger.warning(
                "Uno draft CUDA graphs cover %d of %d request counts: "
                "captured draft row counts %s serve up to %d concurrent "
                "requests, and %d..%d will draft eagerly because %d rows "
                "exceed the largest captured count %d. Add a "
                "cudagraph_capture_sizes entry at or above max_num_seqs * "
                "num_speculative_tokens = %d, or lower max_num_seqs.",
                uncovered[0] - 1,
                self.max_num_reqs,
                captured,
                uncovered[0] - 1,
                uncovered[0],
                uncovered[-1],
                self.max_num_reqs * self.k,
                captured[-1],
                self.max_num_reqs * self.k,
            )
            return
        logger.info(
            "Uno draft CUDA graphs cover every request count: captured draft "
            "row counts %s serve all %d concurrent requests at "
            "num_speculative_tokens=%d.",
            captured,
            self.max_num_reqs,
            self.k,
        )

    def _generate_draft(
        self,
        num_reqs: int,
        num_tokens: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor],
    ) -> None:
        with set_forward_context(
            attn_metadata,
            self.vllm_config,
            num_tokens=num_tokens,
            cudagraph_runtime_mode=CUDAGraphMode.NONE,
            slot_mapping=slot_mappings,
        ):
            hidden_states = self.model(
                input_ids=self.input_buffers.input_ids[:num_tokens],
                positions=self.input_buffers.positions[:num_tokens],
                inputs_embeds=None,
            )
        count = num_reqs * self.k
        tokens = self.sample_draft(
            hidden_states[:count],
            self.input_buffers.positions[:count],
            self.sample_idx_mapping[:count],
            self.temperature,
            self.seeds,
            self.sample_col[:count],
            self.draft_logits,
        )
        self.draft_tokens[:num_reqs].copy_(tokens.view(num_reqs, self.k))

    @torch.inference_mode()
    def propose(
        self,
        input_batch: InputBatch,
        attn_metadata: dict[str, Any],
        slot_mappings: dict[str, torch.Tensor],
        last_hidden_states: torch.Tensor,
        aux_hidden_states: list[torch.Tensor] | None,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        last_sampled: torch.Tensor,
        next_prefill_tokens: torch.Tensor,
        temperature: torch.Tensor,
        seeds: torch.Tensor,
        dp_sync: DPSyncState | None = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        is_profile: bool = False,
    ) -> torch.Tensor:
        n = input_batch.num_reqs
        if n == 0:
            return self.draft_tokens[:0]
        count = n * self.k
        self._copy_request_inputs(n, input_batch.idx_mapping, temperature, seeds)
        if dummy_run and skip_attn_for_dummy_run:
            self.input_buffers.input_ids.zero_()
            self.input_buffers.positions.zero_()
            self.sample_idx_mapping.fill_(-1)
            with self._draft_lora(n, count):
                self._generate_draft(n, count, None, {})
            return self.draft_tokens[:n]

        self._step += 1
        assert self.speculative_config.uno_mask_token_id is not None
        prepare_uno_inputs_fused(
            self.input_buffers,
            self.block_tables.slot_mappings[0],
            self.sample_idx_mapping,
            input_batch,
            num_sampled,
            num_rejected,
            last_sampled,
            next_prefill_tokens,
            seeds,
            self.block_tables.input_block_tables[0],
            self.block_tables.kernel_block_sizes[0],
            self.k,
            self.max_model_len,
            self.speculative_config.uno_noise_seed,
            self.speculative_config.uno_mask_token_id,
            self._step,
        )
        if dummy_run:
            self.block_tables.slot_mappings.fill_(PAD_SLOT_ID)
            self.sample_idx_mapping.fill_(-1)
        assert self.cudagraph_manager is not None
        desc = self.cudagraph_manager.dispatch(n, count, self.k, 2 if self.k > 1 else 0)
        if is_profile:
            desc = BatchExecutionDescriptor(CUDAGraphMode.NONE, count, n)
        with self._draft_lora(n, desc.num_tokens):
            if desc.cg_mode == CUDAGraphMode.FULL:
                # The graph already owns the metadata and slot-buffer views.
                # Refresh native backend scheduling state (needed by FA3) using
                # the updated persistent input buffers, without rebuilding the
                # eager metadata or its temporary CPU tensors.
                captured_attn = self._graph_attn_metadata[desc]
                for attn_groups in self.attn_groups:
                    for attn_group in attn_groups:
                        attn_group.update_draft_decode_metadata(captured_attn)
                self.cudagraph_manager.run_fullgraph(desc)
                if dummy_run or is_profile:
                    self.num_warmup_proposals += 1
                else:
                    self.num_graph_replays += 1
                    if self.num_graph_replays == 1:
                        logger.info("Uno draft CUDA graph replay is active.")
            else:
                self.draft_max_seq_len = min(
                    int(input_batch.seq_lens_cpu_upper_bound[:n].max()) + self.k,
                    self.max_model_len,
                )
                draft_attn = self._build_uniform_attn_metadata(
                    desc,
                    n,
                    self.k,
                    input_batch.seq_lens_cpu_upper_bound,
                    step=self.k,
                )
                slots = build_slot_mappings_by_layer(
                    self.block_tables.slot_mappings[:, : desc.num_tokens],
                    self.kv_cache_config,
                )
                self._generate_draft(n, desc.num_tokens, draft_attn, slots)
                if dummy_run or is_profile:
                    self.num_warmup_proposals += 1
                else:
                    self.num_eager_proposals += 1
                    if self.num_eager_proposals == 1:
                        logger.info(
                            "Uno drafting fell back to eager execution for "
                            "%d draft rows (%d requests). Eager drafting is "
                            "far slower than a captured graph; see the draft "
                            "graph coverage line logged at startup.",
                            desc.num_tokens,
                            n,
                        )
        return self.draft_tokens[:n]
