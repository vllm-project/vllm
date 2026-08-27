# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import gc
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from itertools import groupby, product
from typing import TYPE_CHECKING, Any, NamedTuple, Protocol

import torch
import torch.nn as nn
from tqdm import tqdm

from vllm.compilation.breakable_cudagraph import (
    BreakableCUDAGraphWrapper,
    is_breakable_cudagraph_enabled,
)
from vllm.compilation.counter import compilation_counter
from vllm.compilation.cuda_graph import CUDAGraphWrapper
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.config.compilation import CUDAGraphMode
from vllm.distributed.device_communicators.pynccl_allocator import set_graph_pool_id
from vllm.distributed.parallel_state import (
    get_pp_group,
    graph_capture,
    is_global_first_rank,
)
from vllm.forward_context import BatchDescriptor, set_forward_context
from vllm.logger import init_logger
from vllm.model_executor.offloader.base import get_offloader
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.utils.math_utils import round_up
from vllm.utils.torch_utils import current_stream
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.attn_utils import build_slot_mappings_by_layer
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.cp_utils import prepare_dcp_local_seq_lens
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.model_states.interface import ModelState
from vllm.v1.worker.utils import AttentionGroup

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

logger = init_logger(__name__)


class AttentionState(NamedTuple):
    attn_metadata: dict[str, Any] | None
    slot_mappings: dict[str, torch.Tensor]


@dataclass(frozen=True)
class BatchExecutionDescriptor:
    """Describes the shape of the batch and CG mode to run; this is used to make shape
    matches between the capture and runtime."""

    cg_mode: CUDAGraphMode
    num_tokens: int
    num_reqs: int | None  # None means no request padding is needed (PIECEWISE graphs)
    uniform_token_count: int | None = None
    # Upper bound on per-request query length. Varlen decode graphs leave
    # uniform_token_count unset, so this is what keeps a prefill batch out of one.
    max_query_len: int | None = None
    num_active_loras: int = 0


class CreateForwardFn(Protocol):
    """Factory that prepares inputs (OUTSIDE the graph) and returns a
    forward_fn. Called with warmup=True for the warmup pass and warmup=False
    for the captured pass."""

    def __call__(
        self,
        desc: BatchExecutionDescriptor,
        warmup: bool,
    ) -> Callable[[CUDAGraphMode], None]: ...


def _is_compatible(
    desc: BatchExecutionDescriptor,
    num_reqs: int,
    num_tokens: int,
    uniform_token_count: int | None,
    num_active_loras: int,
    max_query_len: int | None,
) -> bool:
    # desc.uniform_token_count=None (PIECEWISE) can handle any uniform_token_count
    # desc.num_reqs=None means no request padding needed (PIECEWISE)
    # desc.max_query_len=None means the graph does not constrain query length; a
    # caller that does not track max_query_len must not match one that does
    return (
        (
            desc.uniform_token_count is None
            or desc.uniform_token_count == uniform_token_count
        )
        and (
            desc.max_query_len is None
            or (max_query_len is not None and desc.max_query_len >= max_query_len)
        )
        and (desc.num_reqs is None or desc.num_reqs >= num_reqs)
        and desc.num_tokens >= num_tokens
        and desc.num_active_loras == num_active_loras
    )


class CudaGraphManager:
    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        cudagraph_mode: CUDAGraphMode,
        decode_query_len: int,
        lora_capture_cases: list[int] | None = None,
        varlen_decode: bool = False,
    ):
        self.vllm_config = vllm_config
        self.device = device
        self.max_num_reqs = vllm_config.scheduler_config.max_num_seqs
        self.compilation_config = vllm_config.compilation_config
        assert self.compilation_config is not None
        self.cudagraph_mode = cudagraph_mode
        self.decode_query_len = decode_query_len
        self.varlen_decode = varlen_decode

        self.dp_size = vllm_config.parallel_config.data_parallel_size
        self.tp_size = vllm_config.parallel_config.tensor_parallel_size
        self.is_first_pp_rank = get_pp_group().is_first_rank
        self.is_last_pp_rank = get_pp_group().is_last_rank
        self.lora_capture_cases = lora_capture_cases or [0]
        # Precompute actual num_active_loras -> captured case mapping so that
        # dispatch() is a plain dict lookup instead of a per-call bisect.
        self._lora_dispatch_map, self._max_lora_case = self._build_lora_dispatch_map()

        self.graphs: dict[BatchExecutionDescriptor, torch.cuda.CUDAGraph] = {}
        self.pool = current_platform.get_global_graph_pool() if cudagraph_mode else None

        self._graphs_captured = False

        # Profiling hooks, set only by profile_cudagraph_memory() below: cap
        # FULL-mode capture at the N largest descriptors and record each
        # captured FULL graph's memory delta for extrapolation.
        self._max_full_descs_to_capture: int | None = None
        self._capture_mem_samples: list[int] | None = None

        self._candidates: dict[tuple[int, int], list[BatchExecutionDescriptor]] = {}
        self._capture_descs: dict[CUDAGraphMode, list[BatchExecutionDescriptor]] = {}

        # Breakable CUDA graph (PW CUDA graph without torch.compile)
        self.use_breakable_cg = (
            is_breakable_cudagraph_enabled()
            and self.cudagraph_mode.has_piecewise_cudagraphs()
        )
        self.breakable_cg_runner: BreakableCUDAGraphWrapper | None = None

        self._init_candidates()

    def _build_lora_dispatch_map(self) -> tuple[dict[int, int], int]:
        """Precompute actual num_active_loras -> effective captured case.

        Mirrors the num_tokens candidate expansion in ``_init_candidates``:
        every possible active-LoRA count is mapped ahead of time to the
        smallest captured case that can serve it, so ``dispatch`` is a plain
        dict lookup instead of a per-call bisect.
        """
        captured_with_lora = sorted(c for c in self.lora_capture_cases if c > 0)
        if not captured_with_lora:
            return {}, 0
        dispatch_map: dict[int, int] = {}
        case_idx = 0
        for n in range(1, captured_with_lora[-1] + 1):
            while captured_with_lora[case_idx] < n:
                case_idx += 1
            dispatch_map[n] = captured_with_lora[case_idx]
        return dispatch_map, captured_with_lora[-1]

    def _resolve_effective_loras(self, num_active_loras: int) -> int:
        """Map an actual active-LoRA count to its captured graph case."""
        if num_active_loras <= 0 or not self._lora_dispatch_map:
            return num_active_loras
        # Counts above the largest captured case clamp to it.
        return self._lora_dispatch_map.get(num_active_loras, self._max_lora_case)

    def _init_candidates(self) -> None:
        """Build priority-ordered candidate lists for each token count."""
        capture_sizes = self.compilation_config.cudagraph_capture_sizes
        if not (self.cudagraph_mode and capture_sizes):
            return

        capture_sizes = sorted(capture_sizes)
        max_decode_tokens = self.max_num_reqs * self.decode_query_len
        decode_mode = self.cudagraph_mode.decode_mode()
        mixed_mode = self.cudagraph_mode.mixed_mode()
        separate_decode_routine = self.cudagraph_mode.separate_routine()
        max_cg_capture_size = self.compilation_config.max_cudagraph_capture_size

        descs_by_mode: defaultdict[CUDAGraphMode, list[BatchExecutionDescriptor]] = (
            defaultdict(list)
        )

        # When using Dynamic SD, num_speculative_tokens is the max number of
        # draft tokens. The scheduler might use a smaller number so we need
        # to capture graphs for all possible values during decode.
        speculative_config = self.vllm_config.speculative_config
        if (
            speculative_config
            and speculative_config.uses_dynamic_speculative_decoding()
        ):
            num_spec_per_batch_size = (
                speculative_config.num_speculative_tokens_per_batch_size
            )
            # uses_dynamic_speculative_decoding() guarantees this is set.
            assert num_spec_per_batch_size is not None
            # decode_query_len = num_speculative_steps + num_new_sampled_tokens
            # _per_step. Recover num_new_sampled_tokens_per_step
            # from the values the manager already has.
            num_new_sampled_tokens_per_step = (
                self.decode_query_len - self.vllm_config.num_speculative_tokens
            )
            # Each entry is (range_start, range_end, num_speculative_tokens).
            decode_query_lens = [
                x[2] + num_new_sampled_tokens_per_step for x in num_spec_per_batch_size
            ]
        else:
            decode_query_lens = [self.decode_query_len]

        capture_varlen_decode = (
            separate_decode_routine and bool(decode_mode) and self.varlen_decode
        )
        for num_tokens, num_active_loras in product(
            capture_sizes, self.lora_capture_cases
        ):
            # Varlen decode graphs take any mix of 1..decode_query_len tokens per
            # request, worst case 1 token per request (or max_num_reqs)
            if capture_varlen_decode and num_tokens <= max_decode_tokens:
                desc = BatchExecutionDescriptor(
                    cg_mode=decode_mode,
                    num_tokens=num_tokens,
                    num_reqs=min(num_tokens, self.max_num_reqs),
                    max_query_len=self.decode_query_len,
                    num_active_loras=num_active_loras,
                )
                descs_by_mode[decode_mode].append(desc)
            # Capture uniform decode specfifc graphs if required
            #  (i.e. separate decode routine)
            elif separate_decode_routine and decode_mode and not self.varlen_decode:
                for decode_query_len in decode_query_lens:
                    rounded_num_tokens = round_up(num_tokens, decode_query_len)
                    rounded_num_reqs = rounded_num_tokens // decode_query_len

                    if (
                        rounded_num_tokens > max_decode_tokens
                        or rounded_num_tokens > max_cg_capture_size
                        or rounded_num_reqs > self.max_num_reqs
                    ):
                        continue

                    desc = BatchExecutionDescriptor(
                        cg_mode=decode_mode,
                        num_tokens=rounded_num_tokens,
                        num_reqs=rounded_num_reqs,
                        uniform_token_count=decode_query_len,
                        num_active_loras=num_active_loras,
                    )

                    # avoid duplicate graphs
                    if desc not in descs_by_mode[decode_mode]:
                        descs_by_mode[decode_mode].append(desc)

            if mixed_mode:
                # for PIECEWISE graphs there is no limit on requests when replaying
                # i.e. no request padding is needed, so we leave it as None.
                # For breakable PW graphs, break-point kernels read the real batch
                # from the forward context; in-graph kernels handle the token padding
                # themselves from the padded slot_mapping (rows with slot == -1).
                num_reqs = None
                if mixed_mode == CUDAGraphMode.FULL:
                    num_reqs = min(num_tokens, self.max_num_reqs)
                desc = BatchExecutionDescriptor(
                    cg_mode=mixed_mode,
                    num_tokens=num_tokens,
                    num_reqs=num_reqs,
                    num_active_loras=num_active_loras,
                )
                descs_by_mode[mixed_mode].append(desc)

        for mode, descs in descs_by_mode.items():
            descs.sort(key=lambda d: d.num_tokens, reverse=True)
            self._capture_descs[mode] = descs

        for mode in (CUDAGraphMode.FULL, CUDAGraphMode.PIECEWISE):
            mode_descs = tuple(reversed(descs_by_mode.get(mode, [])))
            for num_active_loras in self.lora_capture_cases:
                lora_descs = [
                    d for d in mode_descs if d.num_active_loras == num_active_loras
                ]
                current_range_start = 0
                # Dynamic speculative decoding can produce multiple graphs with the same
                # num_tokens. Group them so each graph covers the same candidate range.
                for num_tokens, group in groupby(lora_descs, lambda d: d.num_tokens):
                    matching = list(group)
                    for i in range(current_range_start, num_tokens + 1):
                        key = (i, num_active_loras)
                        self._candidates.setdefault(key, []).extend(matching)
                    current_range_start = num_tokens + 1

    def needs_capture(self) -> bool:
        return len(self._capture_descs) > 0

    @torch.inference_mode()
    def capture(
        self,
        create_forward_fn: CreateForwardFn,
        progress_bar_desc: str = "Capturing CUDA graphs",
    ) -> None:
        """Capture CUDA graphs.

        Args:
            create_forward_fn: Factory that prepares inputs (OUTSIDE graph) and
                returns a forward_fn. For FULL and breakable PIECEWISE modes,
                it is invoked once with warmup=True and again with warmup=False
                because attention backends may mutate or lazily initialize
                metadata during warmup.
        """
        with graph_capture(device=self.device):
            # Capture in order: PIECEWISE first, then FULL. PIECEWISE has larger
            # activations so FULL activations should fit in already allocated
            # buffers in the graph pool.
            for mode in [CUDAGraphMode.PIECEWISE, CUDAGraphMode.FULL]:
                if mode not in self._capture_descs:
                    continue

                descs = self._capture_descs[mode]
                if (
                    mode == CUDAGraphMode.FULL
                    and self._max_full_descs_to_capture is not None
                ):
                    # Profiling only: capture a sample of the largest FULL
                    # graphs; the total cost is extrapolated from their
                    # per-graph memory deltas.
                    descs = descs[: self._max_full_descs_to_capture]
                if is_global_first_rank():
                    descs = tqdm(descs, desc=f"{progress_bar_desc} ({mode.name})")
                for desc in descs:
                    # Prepare inputs and get forward function
                    forward_fn = create_forward_fn(desc, warmup=True)

                    # Warmup
                    forward_fn(CUDAGraphMode.NONE)

                    # Capture
                    logger.debug(
                        "CG Capture: mode=%s, batch_desc=%s", desc.cg_mode.name, desc
                    )
                    if (
                        desc.cg_mode == CUDAGraphMode.PIECEWISE
                        and not self.use_breakable_cg
                    ):
                        forward_fn(CUDAGraphMode.PIECEWISE)
                    else:
                        # Capture with fresh attention state.
                        forward_fn = create_forward_fn(desc, warmup=False)
                        if desc.cg_mode == CUDAGraphMode.PIECEWISE:
                            forward_fn(CUDAGraphMode.PIECEWISE)
                            continue
                        assert desc not in self.graphs, (
                            f"Graph already captured for {desc}"
                        )
                        graph = torch.cuda.CUDAGraph()
                        # Sync offloader's copy stream before capture.
                        # Ensure any pre-capture prefetches from offloader are complete.
                        get_offloader().sync_prev_onload()
                        if self.pool is not None:
                            set_graph_pool_id(self.pool)
                        else:
                            set_graph_pool_id(current_platform.graph_pool_handle())
                        if self._capture_mem_samples is not None:
                            torch.accelerator.synchronize()
                            free_before = torch.accelerator.get_memory_info()[0]
                        with torch.cuda.graph(
                            graph, self.pool, stream=current_stream()
                        ):
                            forward_fn(CUDAGraphMode.NONE)
                            # Join offloader's copy stream after forward to avoid
                            # unjoined stream error. The last layer's start_prefetch
                            # forks copy_stream, but wait_prefetch only happens in
                            # the next forward pass.
                            get_offloader().join_after_forward()
                        if self._capture_mem_samples is not None:
                            torch.accelerator.synchronize()
                            free_after = torch.accelerator.get_memory_info()[0]
                            self._capture_mem_samples.append(free_before - free_after)
                        self.graphs[desc] = graph
                        compilation_counter.num_cudagraph_captured += 1
        self._graphs_captured = True

    def captured_token_counts(self) -> list[int]:
        """Sorted token counts with a captured graph, ignoring LoRA variants."""
        return sorted(
            {desc.num_tokens for desc in self.graphs if desc.num_active_loras == 0}
        )

    def dispatch(
        self,
        num_reqs: int,
        num_tokens: int,
        uniform_token_count: int | None,
        num_active_loras: int,
        max_query_len: int | None = None,
    ) -> BatchExecutionDescriptor:
        """Find matching cudagraph descriptor from priority-ordered candidates."""

        effective_loras = self._resolve_effective_loras(num_active_loras)
        key = (num_tokens, effective_loras)
        if self._graphs_captured and num_tokens > 0 and key in self._candidates:
            for desc in self._candidates[key]:
                if _is_compatible(
                    desc,
                    num_reqs,
                    num_tokens,
                    uniform_token_count,
                    effective_loras,
                    max_query_len,
                ):
                    return desc
        return BatchExecutionDescriptor(
            cg_mode=CUDAGraphMode.NONE,
            num_tokens=num_tokens,
            num_reqs=num_reqs,
            num_active_loras=effective_loras,
        )

    def run_fullgraph(self, desc: BatchExecutionDescriptor):
        """Replay a captured FULL cudagraph."""
        assert desc.cg_mode == CUDAGraphMode.FULL, (
            f"Expected FULL mode, got {desc.cg_mode}"
        )
        assert desc in self.graphs, f"No cudagraph for {desc}"
        # Sync offloader before replay - needed when transitioning from
        # eager/piecewise to full cudagraph (e.g., prefill → decode).
        # The previous eager iteration's start_prefetch may have queued
        # H2D copies on copy_stream that the graph's captured events
        # cannot see. Without this, replay could overwrite static buffers
        # while those copies are still in flight.
        get_offloader().sync_prev_onload()
        self.graphs[desc].replay()

    def init_breakable_cg_runner(self, model: nn.Module) -> None:
        if self.breakable_cg_runner is None:
            self.breakable_cg_runner = BreakableCUDAGraphWrapper(
                model, self.vllm_config
            )

    def run_pw_graph(self, model: nn.Module, model_inputs: dict[str, Any]) -> Any:
        if not self.use_breakable_cg:
            # Default: Use torch-compiled piecewise cudagraph.
            return model(**model_inputs)
        assert self.breakable_cg_runner is not None
        return self.breakable_cg_runner(**model_inputs)


class ModelCudaGraphManager(CudaGraphManager):
    """CudaGraphManager with model-specific capture and hidden state management."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        cudagraph_mode: CUDAGraphMode,
        decode_query_len: int,
        lora_capture_cases: list[int] | None = None,
        varlen_decode: bool = False,
    ):
        super().__init__(
            vllm_config,
            device,
            cudagraph_mode,
            decode_query_len,
            lora_capture_cases=lora_capture_cases,
            varlen_decode=varlen_decode,
        )
        self.hidden_states: torch.Tensor | None = None
        self.aux_hidden_states: list[torch.Tensor] = []
        self.use_aux_hidden_state_outputs = False
        self.intermediate_tensors: IntermediateTensors | None = None

    def capture(
        self,
        model: nn.Module,
        model_state: ModelState,
        input_buffers: InputBuffers,
        intermediate_tensors: IntermediateTensors | None,
        block_tables: BlockTables,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        has_lora: bool = False,
        use_aux_hidden_state_outputs: bool = False,
        lora_capture_hook: Callable[[int, int, int], None] | None = None,
        progress_bar_desc: str = "Capturing CUDA graphs",
    ) -> None:
        """Capture CUDA graphs for model forward pass."""
        self.use_aux_hidden_state_outputs = use_aux_hidden_state_outputs
        if self.use_breakable_cg:
            self.init_breakable_cg_runner(model)

        def create_forward_fn(
            desc: BatchExecutionDescriptor,
            warmup: bool,
        ) -> Callable[[CUDAGraphMode], None]:
            num_tokens = desc.num_tokens
            num_reqs = desc.num_reqs or min(num_tokens, self.max_num_reqs)

            # Set LoRA state before capture so kernels see correct adapters.
            if lora_capture_hook is not None:
                lora_capture_hook(desc.num_active_loras, num_reqs, num_tokens)

            num_tokens_across_dp = (
                torch.full((self.dp_size,), num_tokens, dtype=torch.int32, device="cpu")
                if self.dp_size > 1
                else None
            )

            model_inputs = {
                "input_ids": input_buffers.input_ids[:num_tokens],
                "positions": input_buffers.positions[:num_tokens],
                **model_state.prepare_dummy_inputs(num_reqs, num_tokens),
            }
            if not self.is_first_pp_rank:
                # Update for non-first PP ranks.
                model_inputs["input_ids"] = None
                model_inputs["inputs_embeds"] = None
                assert intermediate_tensors is not None
                model_inputs["intermediate_tensors"] = intermediate_tensors[:num_tokens]

            attn_metadata, slot_mappings = prepare_inputs_to_capture(
                num_reqs,
                num_tokens,
                model_state,
                input_buffers,
                block_tables,
                attn_groups,
                kv_cache_config,
                full_cudagraph=desc.cg_mode == CUDAGraphMode.FULL,
                max_query_len=desc.max_query_len,
            )

            # Capture with dummy rows marked as padding.
            input_buffers.is_padding.fill_(True)

            def forward_fn(cg_mode: CUDAGraphMode) -> None:
                batch_descriptor = None
                if cg_mode == CUDAGraphMode.PIECEWISE:
                    batch_descriptor = BatchDescriptor(
                        num_tokens=num_tokens,
                        has_lora=has_lora,
                        num_active_loras=desc.num_active_loras,
                    )
                with set_forward_context(
                    attn_metadata,
                    self.vllm_config,
                    num_tokens=num_tokens,
                    cudagraph_runtime_mode=cg_mode,
                    num_tokens_across_dp=num_tokens_across_dp,
                    slot_mapping=slot_mappings,
                    batch_descriptor=batch_descriptor,
                    is_padding=input_buffers.is_padding[:num_tokens],
                ):
                    if cg_mode == CUDAGraphMode.PIECEWISE:
                        # PIECEWISE graph (compiled PW or breakable, chosen inside
                        # run_pw_graph).
                        model_output = self.run_pw_graph(model, model_inputs)
                    else:
                        model_output = model(**model_inputs)

                if cg_mode == CUDAGraphMode.PIECEWISE:
                    # PW CUDA graph (compiled or breakable) internally handles the
                    # model outputs. No need to keep track of the hidden states.
                    return None

                if self.is_last_pp_rank:
                    # Last PP rank (common case).
                    if self.use_aux_hidden_state_outputs:
                        hidden_states, aux_hidden_states = model_output
                    else:
                        hidden_states = model_output
                        aux_hidden_states = []
                    if self.hidden_states is None:
                        self.hidden_states = torch.empty_like(hidden_states)
                    self.hidden_states[:num_tokens] = hidden_states
                    if self.use_aux_hidden_state_outputs and not self.aux_hidden_states:
                        self.aux_hidden_states = [
                            torch.empty_like(x) for x in aux_hidden_states
                        ]
                    for i, aux in enumerate(aux_hidden_states):
                        self.aux_hidden_states[i][:num_tokens] = aux
                else:
                    # Non-last PP rank.
                    assert isinstance(model_output, IntermediateTensors)
                    intermediate_tensors = model_output
                    if self.intermediate_tensors is None:
                        self.intermediate_tensors = IntermediateTensors.empty_like(
                            intermediate_tensors
                        )
                    for k, v in intermediate_tensors.tensors.items():
                        self.intermediate_tensors[k][:num_tokens] = v

            return forward_fn

        super().capture(create_forward_fn, progress_bar_desc)

    def run_fullgraph(
        self, desc: BatchExecutionDescriptor
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]] | IntermediateTensors:
        """Replay a captured FULL cudagraph and return hidden states."""
        super().run_fullgraph(desc)
        if not self.is_last_pp_rank:
            assert self.intermediate_tensors is not None
            return self.intermediate_tensors[: desc.num_tokens]

        assert self.hidden_states is not None
        hidden_states = self.hidden_states[: desc.num_tokens]
        if not self.use_aux_hidden_state_outputs:
            return hidden_states
        return hidden_states, [x[: desc.num_tokens] for x in self.aux_hidden_states]


def prepare_inputs_to_capture(
    num_reqs: int,
    num_tokens: int,
    model_state: ModelState,
    input_buffers: InputBuffers,
    block_tables: BlockTables,
    attn_groups: list[list[AttentionGroup]],
    kv_cache_config: KVCacheConfig,
    full_cudagraph: bool,
    max_query_len: int | None = None,
) -> AttentionState:
    input_batch = InputBatch.make_dummy(
        num_reqs, num_tokens, input_buffers, max_query_len=max_query_len
    )
    input_block_tables = block_tables.get_dummy_block_tables(num_reqs)
    slot_mappings = block_tables.get_dummy_slot_mappings(num_tokens)
    slot_mappings_by_layer = build_slot_mappings_by_layer(
        slot_mappings, kv_cache_config
    )

    # HACK(woosuk): Special handling for DCP.
    if block_tables.cp_size > 1:
        prepare_dcp_local_seq_lens(
            input_buffers.dcp_local_seq_lens,
            input_batch.seq_lens,
            num_reqs,
            block_tables.cp_size,
            block_tables.cp_rank,
            block_tables.cp_interleave,
        )
        input_batch.dcp_local_seq_lens = input_buffers.dcp_local_seq_lens[:num_reqs]

    # NOTE(woosuk): Attention metadata is required not just by standard attention
    # kernels, but also by specialized attention-like operations (e.g., Inkling's sconv,
    # DSV4 compressor), which maintain their own states and require special metadata
    # such as block tables.
    # During CUDA graph capture:
    # - For FULL CUDA graphs: We set for_capture=True so that both attention and
    #   attention-like ops produce capturable metadata compatible with CUDA graphs.
    # - For PIECEWISE CUDA graphs: We still build attention metadata, but set
    #   for_capture=False. This is because:
    #     * Attention-like ops (such as sconv or DSV4 compressor) may not be used as
    #       breakpoints in PIECEWISE CUDA graphs, so we must generate their attention
    #       metadata so they can execute and be captured during graph capture.
    #     * Standard attention ops that are treated as breakpoints will be executed
    #       eagerly at capture time (not included in the graph itself), and for these,
    #       setting for_capture=False is essential. Some attention backends
    #       (like linear attention) cannot generate capturable metadata for prefill,
    #       so for_capture=False ensures they execute without issue.
    #     * We assume that attention-like operations intended for capture will still
    #       produce capturable metadata, even when for_capture=False. While this
    #       assumption is brittle, it currently works in practice.
    # In summary: We always generate attention metadata for both FULL and PIECEWISE
    # CUDA graphs, setting for_capture=True for FULL graphs, and for_capture=False
    # for PIECEWISE graphs, to ensure correct execution and capture.
    attn_metadata = model_state.prepare_attn(
        input_batch,
        CUDAGraphMode.NONE,
        input_block_tables,
        slot_mappings,
        attn_groups,
        kv_cache_config,
        for_capture=full_cudagraph,
    )
    return AttentionState(attn_metadata, slot_mappings_by_layer)


# ---------------------------------------------------------------------------
# CUDA graph memory profiling
# ---------------------------------------------------------------------------

# Number of FULL graphs captured during profiling; the total FULL capture
# cost is extrapolated from this sample to avoid a second full capture.
_FULL_GRAPH_PROFILING_SAMPLES = 2
# Floor for the extrapolated per-graph cost (driver overhead per graph).
_MIN_PER_GRAPH_BYTES = 1 << 20


@torch.inference_mode()
def profile_cudagraph_memory(runner: "GPUModelRunner") -> int:
    """Estimate the GPU memory needed for CUDA graph capture.

    Called during memory profiling, *before* the real KV cache is allocated,
    so that ``Worker.determine_available_memory`` can reserve headroom for
    graph capture. Bootstraps a minimal KV cache, runs ``capture_model()``
    once, then releases everything so the real init/capture path starts clean.

    FULL graphs bake in KV cache pointers, so only the largest few are
    captured (into a throwaway pool) and their total cost is extrapolated.
    PIECEWISE, encoder and speculator graphs are measured in full. All
    profiling captures are discarded afterwards: replaying graphs recorded
    against the throwaway profiling state is unsafe (e.g. inductor graph
    partition reclaims the storages of earlier cudagraph recordings once the
    real capture records new ones, leading to use-after-free crashes).
    """
    if runner.compilation_config.cudagraph_mode == CUDAGraphMode.NONE:
        return 0

    gc.collect()
    torch.accelerator.empty_cache()

    # Run the whole profiling phase against a throwaway CUDA graph pool by
    # pointing the global graph pool singleton at it: objects that bind the
    # pool lazily during profiling (speculator cudagraph managers, breakable
    # runners created mid-capture) then land on the throwaway pool too. Pools
    # bound before profiling (piecewise wrappers) are swapped explicitly in
    # the inner block. Profiling graphs captured into the persistent global
    # pool and then discarded would drop its use_count to 0, tripping the c10
    # allocator's create_or_incref_pool assert when the real capture reuses
    # that pool ("use_count > 0 INTERNAL ASSERT FAILED").
    platform_cls = type(current_platform)
    saved_global_pool = platform_cls._global_graph_pool
    throwaway_pool = current_platform.graph_pool_handle()
    platform_cls._global_graph_pool = throwaway_pool

    try:
        with set_current_vllm_config(runner.vllm_config):
            _init_minimal_kv_cache_for_profiling(runner)

        manager = runner.cudagraph_manager
        assert manager is not None

        # Don't count profiling captures; the real capture_model() runs later.
        saved_num_cudagraph_captured = compilation_counter.num_cudagraph_captured
        saved_capture_triggers = compilation_counter.num_gpu_runner_capture_triggers
        all_wrappers: list[Any] = []
        original_pools: dict[int, Any] = {}
        speculator = getattr(runner, "speculator", None)
        spec_manager_names: list[str] = []
        try:
            if not manager.needs_capture():
                return 0
            manager.pool = throwaway_pool
            if manager.use_breakable_cg:
                # Create the breakable runner before the wrapper pool swap so
                # its pool is covered as well.
                manager.init_breakable_cg_runner(runner.model)
            all_wrappers = list(CUDAGraphWrapper._all_instances) + list(
                BreakableCUDAGraphWrapper._all_instances
            )
            for wrapper in all_wrappers:
                original_pools[id(wrapper)] = wrapper.graph_pool
                wrapper.graph_pool = throwaway_pool
            if speculator is not None:
                spec_manager_names = [
                    name
                    for name, value in vars(speculator).items()
                    if isinstance(value, CudaGraphManager)
                ]
            manager._max_full_descs_to_capture = _FULL_GRAPH_PROFILING_SAMPLES
            mem_samples: list[int] = []
            manager._capture_mem_samples = mem_samples

            measured = int(runner.capture_model())

            # The measured delta covers PIECEWISE, encoder and speculator graphs
            # plus the sampled FULL graphs; swap the sampled FULL cost for the
            # extrapolated total. FULL and PIECEWISE share one pool here just as
            # they share the global pool at runtime, so the overlap is not
            # double-counted.
            num_full_graphs = len(manager._capture_descs.get(CUDAGraphMode.FULL, []))
            full_estimate = _extrapolate_full_graph_memory(mem_samples, num_full_graphs)
            return max(measured - sum(mem_samples) + full_estimate, 0)
        finally:
            compilation_counter.num_cudagraph_captured = saved_num_cudagraph_captured
            compilation_counter.num_gpu_runner_capture_triggers = saved_capture_triggers
            CUDAGraphWrapper.clear_all_graphs()
            BreakableCUDAGraphWrapper.clear_all_graphs()
            for wrapper in all_wrappers:
                if id(wrapper) in original_pools:
                    wrapper.graph_pool = original_pools[id(wrapper)]
            # Drop the speculator's cudagraph managers; the real
            # initialize_kv_cache re-creates them. Their profiling graphs
            # release the throwaway pool here rather than after the real init.
            for name in spec_manager_names:
                setattr(speculator, name, None)
            # Drop local references before teardown detaches the runner's
            # manager and flushes the allocator.
            del manager
            _teardown_profiling_state(runner)
    finally:
        platform_cls._global_graph_pool = saved_global_pool


def _extrapolate_full_graph_memory(mem_samples: list[int], total_graphs: int) -> int:
    """Extrapolate the total FULL capture cost from samples of the largest
    graphs. The first capture allocates the pool baseline; later graphs mostly
    reuse it, so the second sample is taken as the per-graph cost."""
    if not mem_samples:
        return 0
    first_capture = mem_samples[0]
    per_graph = max(mem_samples[1], _MIN_PER_GRAPH_BYTES) if len(mem_samples) > 1 else 0
    return first_capture + (total_graphs - 1) * per_graph


def _init_minimal_kv_cache_for_profiling(runner: "GPUModelRunner") -> None:
    """Allocate the smallest KV cache that still lets every graph be captured."""
    from vllm.v1.core.kv_cache_utils import (
        get_kv_cache_config_from_groups,
        get_kv_cache_groups,
    )

    kv_cache_spec = runner.get_kv_cache_spec()
    kv_cache_groups = get_kv_cache_groups(runner.vllm_config, kv_cache_spec)
    # At least one block per sequence is required to capture the graphs.
    min_blocks = (
        min(runner.max_num_reqs, runner.compilation_config.max_cudagraph_capture_size)
        or 1
    )
    saved_override = runner.cache_config.num_gpu_blocks_override
    runner.cache_config.num_gpu_blocks_override = min_blocks
    try:
        minimal_config = get_kv_cache_config_from_groups(
            runner.vllm_config, kv_cache_groups, available_memory=0
        )
    finally:
        runner.cache_config.num_gpu_blocks_override = saved_override

    runner.initialize_kv_cache(minimal_config, is_profiling=True)
    runner.cache_config.num_gpu_blocks = minimal_config.num_blocks


def _teardown_profiling_state(runner: "GPUModelRunner") -> None:
    """Release the profiling KV cache and captured graphs while keeping model
    weights, so the real ``initialize_kv_cache`` starts from a clean slate."""
    torch.accelerator.synchronize()
    if hasattr(runner.model_state, "_mamba_ctx"):
        runner.model_state._mamba_ctx = None
    if hasattr(runner, "kv_caches"):
        runner.kv_caches.clear()
    if hasattr(runner, "attn_groups"):
        runner.attn_groups.clear()
    if hasattr(runner, "kv_cache_config"):
        del runner.kv_cache_config
    # Dropping the manager releases the profiling graphs and throwaway pool.
    runner.cudagraph_manager = None
    # Release encoder graphs captured during profiling; the real
    # capture_model() re-captures them.
    if runner.model_state.supports_mm_inputs:
        runner.model_state.encoder_runner.clear()
    # Detach profiling KV tensors held by attention layers.
    for layer in runner.compilation_config.static_forward_context.values():
        if hasattr(layer, "kv_cache"):
            kv_cache = layer.kv_cache
            layer.kv_cache = (
                torch.tensor([]) if isinstance(kv_cache, torch.Tensor) else []
            )
            del kv_cache
    runner.cache_config.num_gpu_blocks = None
    runner.maybe_remove_all_loras(runner.lora_config)
    gc.collect()
    torch.accelerator.empty_cache()
