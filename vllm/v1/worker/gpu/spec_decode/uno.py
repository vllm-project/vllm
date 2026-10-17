# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Uno shared-model parallel drafting for Model Runner V2."""

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from typing import Any

import torch
import torch.nn as nn

import vllm.envs as envs
from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.config.compilation import CUDAGraphMode
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger
from vllm.lora.layers import LoRAMapping
from vllm.lora.request import LoRARequest
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.platforms import current_platform
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig
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
from vllm.v1.worker.gpu.spec_decode.uno_lora import (
    base_prompt_mapping,
    draft_lora_mapping,
)
from vllm.v1.worker.gpu.spec_decode.uno_prepare import prepare_uno_inputs_fused
from vllm.v1.worker.utils import AttentionGroup

logger = init_logger(__name__)
UNO_LORA_ID = 1_000_003
# Attention backends Uno's tests run it on, per platform. A backend must be
# listed here AND pass the capability checks in set_attn and
# init_cudagraph_manager; one that passes them but was never run with Uno
# (FlashInfer's trtllm-gen decode, for example) is refused at startup.
UNO_CUDA_ATTN_BACKENDS: tuple[str, ...] = ("FLASH_ATTN", "TRITON_ATTN")
UNO_ROCM_ATTN_BACKENDS: tuple[str, ...] = ("TRITON_ATTN",)
UNO_USER_LORA_ERROR = (
    "Uno does not support user LoRA adapters yet: --enable-lora serves only "
    "Uno's own adapter, so runtime LoRA loading and default multimodal LoRAs "
    "are refused (request-specific LoRA requests are refused before queueing)"
)


def validate_uno_config(vllm_config: VllmConfig) -> None:
    """Refuse configurations Uno does not support, before anything is built."""
    speculative_config = vllm_config.speculative_config
    assert speculative_config is not None
    model_config = vllm_config.model_config
    parallel_config = vllm_config.parallel_config
    if speculative_config.num_speculative_tokens_per_batch_size is not None:
        raise ValueError("Uno requires a fixed num_speculative_tokens")
    if speculative_config.rejection_sample_method != "standard":
        raise ValueError("Uno requires rejection_sample_method='standard'")
    if speculative_config.use_local_argmax_reduction:
        raise ValueError("Uno requires full draft logits; local argmax is unsupported")
    if speculative_config.enable_adaptive_verification:
        raise ValueError(
            "Adaptive verification is not supported with method='uno'; "
            "disable enable_adaptive_verification"
        )
    if speculative_config.draft_tensor_parallel_size not in (
        None,
        parallel_config.tensor_parallel_size,
    ) or any(
        getattr(parallel_config, name) != 1
        for name in (
            "pipeline_parallel_size",
            "data_parallel_size",
            "prefill_context_parallel_size",
            "decode_context_parallel_size",
        )
    ):
        raise ValueError(
            "Uno supports tensor parallelism; pipeline/data/context "
            "parallelism are not supported, and the draft must use the "
            "target's tensor_parallel_size"
        )
    if parallel_config.use_ubatching:
        raise ValueError("Uno does not support dual batch overlap")
    if (
        model_config.runner_type != "generate"
        or model_config.is_diffusion
        or model_config.is_multimodal_model
        or model_config.is_encoder_decoder
    ):
        raise ValueError("Uno requires a text-only decoder model")
    if (
        model_config.is_hybrid
        or model_config.is_attention_free
        or model_config.get_sliding_window() is not None
    ):
        raise ValueError("Uno requires a model with full attention in every layer")
    lora_config = vllm_config.lora_config
    if lora_config is None:
        raise ValueError(
            "Uno requires --enable-lora and --max-lora-rank large enough "
            "for the Uno adapter"
        )
    if lora_config.max_loras < 2:
        raise ValueError(
            "Uno requires max_loras >= 2 for shared-adapter warmup and graph capture"
        )
    if envs.VLLM_ALLOW_RUNTIME_LORA_UPDATING or lora_config.default_mm_loras:
        raise ValueError(UNO_USER_LORA_ERROR)
    if (
        vllm_config.kv_transfer_config is not None
        or vllm_config.cache_config.kv_offloading_size is not None
    ):
        raise ValueError("Uno does not support KV cache transfer")
    if vllm_config.cache_config.kv_sharing_fast_prefill:
        raise ValueError(
            "Fast prefill optimization for KV sharing is not compatible "
            "with Uno: Uno's draft layers are the target's layers, so the "
            "KV-sharing eligibility walk would wrap sharing layers "
            "incorrectly, and fast prefill gives incorrect logits for "
            "prompt tokens that Uno's verification requires."
        )
    if not vllm_config.scheduler_config.async_scheduling:
        raise ValueError("Uno requires asynchronous scheduling")
    scheduler_config = vllm_config.scheduler_config
    required_tokens = (
        scheduler_config.max_num_seqs * speculative_config.num_speculative_tokens
    )
    if scheduler_config.max_num_batched_tokens < required_tokens:
        raise ValueError(
            "Uno requires max_num_batched_tokens >= max_num_seqs * "
            f"num_speculative_tokens ({required_tokens}) for draft LoRA routing"
        )
    # An explicitly requested attention backend is checked here, before any
    # weights load; load_model checks the backend every layer actually built.
    attention_config = vllm_config.attention_config
    requested = (attention_config.backend, *attention_config.backend_per_kind.values())
    for backend in requested:
        if backend is not None:
            refusal = uno_attention_backend_refusal(backend.name)
            if refusal is not None:
                raise ValueError(refusal)
    if (
        current_platform.is_rocm()
        and attention_config.backend is None
        and not attention_config.backend_per_kind
    ):
        # ROCm's default is not a backend Uno's tests run.
        raise ValueError(
            "Uno on ROCm requires an explicit tested attention backend: pass "
            '--attention-config \'{"backend":"TRITON_ATTN"}\''
        )


def uno_device_refusal(device: torch.device) -> str | None:
    """Why Uno cannot run on ``device``, or None when it can.

    NVIDIA CUDA or AMD ROCm. HIP devices report the type "cuda", so the
    platform decides between the two. The attention backend, not the card,
    is what Uno restricts (``uno_attention_backend_refusal``).
    """
    if device.type != "cuda":
        return "Uno requires a CUDA or ROCm GPU device"
    if current_platform.is_cuda() or current_platform.is_rocm():
        return None
    return "Uno currently requires an NVIDIA CUDA or AMD ROCm device"


def uno_attention_backend_refusal(backend_name: str) -> str | None:
    """Why Uno cannot run on the attention backend ``backend_name``, or None.

    The tested backends of the current platform (``UNO_CUDA_ATTN_BACKENDS``,
    ``UNO_ROCM_ATTN_BACKENDS``) pass; every other backend is refused, even
    when its metadata builder has the capabilities Uno checks later.
    """
    if current_platform.is_rocm():
        platform, tested = "ROCm", UNO_ROCM_ATTN_BACKENDS
    else:
        platform, tested = "CUDA", UNO_CUDA_ATTN_BACKENDS
    if backend_name in tested:
        return None
    return (
        f"Uno on {platform} is tested only with the {' and '.join(tested)} "
        f"attention backend{'s' if len(tested) > 1 else ''}, not "
        f"{backend_name}. Select a tested backend with --attention-config."
    )


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


class UnoSpeculator(DraftModelSpeculator):
    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        refusal = uno_device_refusal(device)
        if refusal is not None:
            raise ValueError(refusal)
        validate_uno_config(vllm_config)
        super().__init__(vllm_config, device)
        self.k = self.num_speculative_steps
        assert self.speculative_config.uno_lora_path is not None
        self.lora_request = LoRARequest(
            "uno", UNO_LORA_ID, self.speculative_config.uno_lora_path
        )
        self._lora_hook: Callable[[tuple[int, int] | None], None] | None = None
        self.sample_idx_mapping = torch.full(
            (self.max_num_reqs * self.k,), -1, dtype=torch.int64, device=device
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

    def set_lora_manager(self, lora_manager: Any) -> None:
        """Route the draft rows through Uno's adapter with the public LoRA API.

        Each draft pass installs its mapping through the worker manager's
        ``set_active_adapters``, the call the runner makes for every target
        step, so the native Punica metadata is rebuilt in place and a captured
        draft graph reads it on replay. The mapping is not restored after the
        draft: the runner installs the next target step's own mapping, and the
        manager rebuilds it because it differs from the draft mapping.

        No request may select Uno's adapter for its target rows. The frontends
        refuse request-specific LoRA, but a prebuilt ``EngineCoreRequest``
        skips that check and can carry any adapter id, Uno's included. So the
        target-side ``set_active_adapters`` is wrapped: when a target batch
        asks for the id Uno's adapter is registered under, Uno first moves its
        adapter to a fresh id through the manager's public calls. The request's
        own adapter then loads under the id it asked for, and its target rows
        use exactly the weights a plain engine would give them.
        """
        set_active_adapters = lora_manager.set_active_adapters
        set_active_adapters({self.lora_request}, None)

        def set_target_adapters(requests: set[Any], mapping: Any) -> None:
            uno_id = self.lora_request.lora_int_id
            if any(
                request.lora_int_id == uno_id and request is not self.lora_request
                for request in requests
            ):
                self._move_adapter_off(lora_manager, requests)
            set_active_adapters(requests, mapping)

        lora_manager.set_active_adapters = set_target_adapters

        def set_draft_mapping(shape: tuple[int, int] | None) -> None:
            if shape is None:
                return
            num_reqs, num_tokens = shape
            uno_id = self.lora_request.lora_int_id
            set_active_adapters(
                {self.lora_request},
                LoRAMapping(
                    draft_lora_mapping(num_tokens, num_reqs, self.k, uno_id),
                    base_prompt_mapping(num_tokens),
                    is_prefill=True,
                ),
            )

        self.set_lora_hook(set_draft_mapping)

    def _move_adapter_off(self, lora_manager: Any, requests: set[Any]) -> None:
        """Re-register Uno's adapter under an id no current request carries."""
        taken = {request.lora_int_id for request in requests}
        taken |= set(lora_manager.list_adapters())
        old_id = self.lora_request.lora_int_id
        new_id = max(taken | {old_id}) + 1
        moved = LoRARequest(
            self.lora_request.lora_name, new_id, self.lora_request.lora_path
        )
        lora_manager.add_adapter(moved)
        lora_manager.remove_adapter(old_id)
        self.lora_request = moved
        logger.warning(
            "A request asked for LoRA adapter id %d, which Uno's adapter was "
            "registered under; Uno's adapter now uses id %d so the request "
            "gets its own adapter.",
            old_id,
            new_id,
        )

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
        # The backend must be a tested one for this platform; its metadata
        # builder's capabilities are checked as well, in set_attn (native
        # draft decode updates) and init_cudagraph_manager (graph support).
        for name in self.draft_attn_layer_names:
            refusal = uno_attention_backend_refusal(
                layers[name].get_attn_backend().get_name()
            )
            if refusal is not None:
                raise ValueError(refusal)
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
        dp_sync_state: DPSyncState | None = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        is_profile: bool = False,
        num_speculative_tokens: int | None = None,
    ) -> torch.Tensor:
        # Uno drafts all K rows in one launch and requires a fixed K; the
        # runner blanks any rows beyond num_speculative_tokens.
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

        if not (dummy_run or is_profile):
            # Startup and DP dummy proposals must not move the noise step the
            # first served proposal starts from.
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
            self.block_tables.block_sizes[0],
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
                # eager metadata or its temporary CPU tensors. A backend with
                # smaller kernel blocks reads its own persistent kernel block
                # table, which only a metadata build refreshes, so map this
                # step's block tables into it first.
                captured_attn = self._graph_attn_metadata[desc]
                for attn_groups in self.attn_groups:
                    for attn_group in attn_groups:
                        attn_group.map_to_kernel_block_table(
                            self.block_tables.input_block_tables[
                                attn_group.kv_cache_group_id
                            ]
                        )
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
