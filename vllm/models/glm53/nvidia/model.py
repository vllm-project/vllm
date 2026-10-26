# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3 (``GlmMoeDsaForCausalLM``) on NVIDIA GPUs and its runner state."""

import typing
import weakref
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

import torch

from vllm.config import SpeculativeConfig, VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.model_executor.warmup.cutedsl_warmup import (
    CuTeDSLCompileUnit,
    register_cutedsl_warmup_provider,
)
from vllm.models.deepseek_v32.nvidia.model import (
    DeepseekV32ForCausalLM,
    DeepseekV32Model,
)
from vllm.sequence import IntermediateTensors
from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.model_states.default import DefaultModelState
from vllm.v1.worker.gpu.model_states.interface import ModelSpecificAttnMetadata
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.utils import AttentionGroup

from . import switch
from .ops.launch import Glm53Geometry

if TYPE_CHECKING:
    from . import draft as glm53_draft
    from . import weights as glm53_weights


class Glm53Model(DeepseekV32Model):
    # The installed decode segment (decode.SpecializedDecode), or None for stock vLLM.
    glm53_segment: typing.Any = None
    # Latency-specialized path: the checkpoint slices kept during load_weights, then
    # the segment's weight forms built from them.
    glm53_stash: "glm53_weights.CheckpointStash | None" = None
    glm53_forms: "glm53_weights.SegmentWeights | None" = None

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        segment = self.glm53_segment
        if segment is not None and input_ids is not None:
            claimed = segment.forward(input_ids, positions)
            if claimed is not None:
                return self.norm(claimed)
        return super().forward(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        if switch.current().enabled:
            from . import weights as glm53_weights

            tp = get_tensor_model_parallel_world_size()
            world = switch.LowLatencyDecode.WORLD
            if tp != world:
                raise ValueError(
                    f"GLM-5.3 latency-specialized decode needs TP{world}, got TP{tp}"
                )
            record = switch.current()
            narrow, wide = switch.roles(record.row_tiers)
            self.glm53_stash = glm53_weights.CheckpointStash(
                Glm53Geometry.from_config(self.config, tp),
                get_tensor_model_parallel_rank(),
                torch.device("cuda", torch.accelerator.current_device_index()),
                narrow,
                wide,
            )
            weights = self.glm53_stash.tee(weights)
        return super().load_weights(weights)


class _FusedQPrefillWarmup:
    """CuTe DSL warm-up of the fused-q kernel's bf16-query variant
    (``nope_dim = 0``).
    """

    def __init__(self, model: "GlmMoeDsaForCausalLM", dtype: torch.dtype) -> None:
        self._model = weakref.ref(model)
        self._dtype = dtype

    def get_cutedsl_warmup_compile_units(self) -> tuple[CuTeDSLCompileUnit, ...]:
        """Return one compile unit per attention layout of the model (full, shared)."""
        from vllm.cute_utils import _TORCH_TO_CUTE_DTYPE
        from vllm.models.deepseek_v32.nvidia.ops.fused_q_cutedsl import FusedQKernel

        model = self._model()
        if model is None:
            return ()
        keys = set()
        for layer in model.model.layers:
            attention = getattr(layer, "self_attn", None)
            if attention is None:
                continue
            indexer = attention.indexer
            if indexer is not None and not attention.skip_topk:
                index = (
                    indexer.head_dim,
                    indexer.n_head,
                    _TORCH_TO_CUTE_DTYPE[
                        attention.indexer_rope_emb.cos_sin_cache.dtype
                    ],
                    _TORCH_TO_CUTE_DTYPE[self._dtype],
                )
            else:
                index = (0, 0, None, None)
            keys.add(
                (
                    attention.qk_rope_head_dim,
                    0,
                    attention.num_local_heads,
                    _TORCH_TO_CUTE_DTYPE[attention.rotary_emb.cos_sin_cache.dtype],
                    *index,
                    attention._index_rope_interleave,
                )
            )
        return tuple(
            CuTeDSLCompileUnit(
                "glm53_fused_q",
                ("glm53_fused_q", *key),
                lambda key=key: FusedQKernel.compile(*key),  # type: ignore[misc]
            )
            for key in sorted(keys, key=repr)
        )


class GlmMoeDsaForCausalLM(DeepseekV32ForCausalLM):
    """GLM-5.3 (``glm_moe_dsa``) causal LM with the low-latency decode hooks."""

    model_cls = Glm53Model

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        # The low-latency decode switch, read once; refuses an unsupported deployment.
        low_latency = switch.resolve(vllm_config)
        if low_latency.enabled:
            from vllm.models.glm53.nvidia import mtp as glm53_mtp

            glm53_mtp.register()
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.low_latency: switch.LowLatencyDecode = low_latency
        speculative = vllm_config.speculative_config
        self.glm53_drafts: int | None = (
            None if speculative is None else speculative.num_speculative_tokens
        )
        self.glm53_max_num_seqs: int = vllm_config.scheduler_config.max_num_seqs
        self.glm53_max_model_len: int = vllm_config.model_config.max_model_len
        self.glm53_v2_runner: bool = bool(vllm_config.use_v2_model_runner)
        if low_latency.enabled:
            self._fused_q_warmup = _FusedQPrefillWarmup(
                self, vllm_config.model_config.dtype
            )
            register_cutedsl_warmup_provider(self._fused_q_warmup)

    @staticmethod
    def get_model_state_cls() -> type["Glm53ModelState"]:
        """Model Runner V2's state for GLM-5.3 (``Glm53ModelState`` below)."""
        return Glm53ModelState

    def process_weights_after_loading(self) -> None:
        if not switch.current().enabled:
            return
        from . import decode as glm53_decode
        from . import weights as glm53_weights

        stash = self.model.glm53_stash
        if stash is not None:
            attention = self.model.layers[0].self_attn
            if not glm53_weights.is_deepgemm_form(attention.fused_qkv_a_proj):
                raise ValueError(
                    "the GLM-5.3 segment reads vLLM's FP8 linears in their DeepGEMM "
                    "form (vLLM's "
                    "default linear backend on SM10x); serve without --linear-backend"
                )
            self.model.glm53_forms = glm53_weights.build_segment_weights(
                self.model, stash
            )
            self.model.glm53_stash = None
        # Latency-specialized path: the model owns segment 1 (decode.py).
        glm53_decode.install(
            self.model,
            self.model.glm53_forms,
            self.glm53_drafts,
            max_num_seqs=self.glm53_max_num_seqs,
            v2_runner=self.glm53_v2_runner,
            max_model_len=self.glm53_max_model_len,
        )


EXPERT_COUNT_TOKENS = (1, 2, 3, 4, 5, 6, 9, 10, 17, 18, 33, 34, 65, 66)


def warm_up_request_kernels(
    geometry: Glm53Geometry, vocab_size: int, device: torch.device
) -> None:
    """Compile the Triton kernels that the first requests otherwise compile while
    serving.
    """
    from vllm.model_executor.layers.fused_moe.utils import count_expert_num_tokens
    from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p

    rows = geometry.rows
    logits = torch.zeros((rows, vocab_size), dtype=torch.float32, device=device)
    apply_top_k_top_p(logits, None, torch.full((rows,), 0.95, device=device))
    for tokens in EXPERT_COUNT_TOKENS:
        topk_ids = torch.zeros(
            (tokens, geometry.top_k), dtype=torch.int32, device=device
        )
        count_expert_num_tokens(topk_ids, geometry.experts, None)
    torch.accelerator.synchronize(device)


def warm_up_rejection_kernels(
    sampler: Sampler,
    spec: SpeculativeConfig,
    rows: int,
    vocab_size: int,
    device: torch.device,
) -> None:
    """Compile vLLM's rejection-sampling kernels for one request's ``rows``
    logits rows.
    """
    from vllm.v1.worker.gpu.spec_decode.rejection_sampler import RejectionSampler
    from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import (
        rejection_sample,
    )

    verifier = RejectionSampler(sampler, spec, device)
    states = sampler.sampling_states
    i32 = dict(dtype=torch.int32, device=device)
    for dtype in (torch.bfloat16, torch.float32):
        for temperature in (0.0, 1.0):
            rejection_sample(
                torch.zeros((rows, vocab_size), dtype=dtype, device=device),
                None,
                torch.zeros(rows, **i32),
                torch.tensor([0, rows], **i32),
                torch.arange(rows, dtype=torch.int64, device=device),
                torch.zeros(1, **i32),
                torch.zeros(rows, **i32),
                torch.arange(rows, **i32),
                torch.full_like(states.temperature.gpu, temperature),
                states.seeds.gpu,
                verifier.num_speculative_steps,
                verifier.synthetic_conditional_rates,
                use_fp64=sampler.use_fp64_gumbel,
                use_block_verification=verifier.use_block_verification,
            )
    torch.accelerator.synchronize(device)


def _draft() -> "glm53_draft.Glm53Draft | None":
    """Return the installed drafter hook; under ``auto`` none, with nothing imported."""
    if not switch.current().enabled:
        return None
    from . import draft as glm53_draft

    return glm53_draft.current()


class Glm53ModelState(DefaultModelState):
    """``DefaultModelState`` plus the decode segment's runner-side hooks."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._metadata: dict[tuple[int, int, int, int], dict[str, Any]] = {}
        # Model Runner V2's low-latency decode paths, with the segment installed.
        self.low_latency_decode = self._segment() is not None

    def _segment(self) -> Any:
        return self.model.model.glm53_segment

    def claims_draft_decode_capture(self) -> bool:
        """Whether the drafter segment runs the draft decode being captured."""
        draft = _draft()
        return draft is not None and draft.claims_capture()

    def custom_sampler(self, sampler: Sampler) -> None:
        """Keep vLLM's samplers; compile the request-time Triton kernels now."""
        segment = self._segment()
        if segment is None:
            return
        vocab_size = self.vllm_config.model_config.get_vocab_size()
        warm_up_request_kernels(segment.geometry, vocab_size, self.device)
        spec = self.vllm_config.speculative_config
        if spec is not None:
            rows = spec.num_speculative_tokens + 1
            warm_up_rejection_kernels(sampler, spec, rows, vocab_size, self.device)

    def initialize_kv_cache(
        self, kv_cache_config: KVCacheConfig, block_tables: BlockTables
    ) -> None:
        """Drop a segment built over an earlier KV cache (vLLM's graph-memory profiling
        binds a throwaway one).
        """
        super().initialize_kv_cache(kv_cache_config, block_tables)
        segment = self._segment()
        if segment is not None:
            segment.close()
            if block_tables is not None:
                segment.bind_block_tables(block_tables, kv_cache_config)

    def add_request(self, req_index: int, new_req_data: NewRequestData) -> None:
        super().add_request(req_index, new_req_data)
        segment = self._segment()
        if segment is not None:
            segment.request_added(
                new_req_data.req_id, new_req_data.num_computed_tokens, req_index
            )

    def remove_request(self, req_id: str) -> None:
        super().remove_request(req_id)
        segment = self._segment()
        if segment is not None:
            segment.request_removed(req_id)

    def preprocess_state(
        self,
        input_batch: InputBatch,
        block_tables: tuple[torch.Tensor, ...],
        kv_cache_config: KVCacheConfig,
        num_computed_tokens: torch.Tensor,
    ) -> None:
        """Account the step for the segment."""
        super().preprocess_state(
            input_batch, block_tables, kv_cache_config, num_computed_tokens
        )
        segment = self._segment()
        if segment is None:
            return
        num_reqs = input_batch.num_reqs
        segment.step(
            list(input_batch.req_ids),
            input_batch.num_computed_tokens_np[:num_reqs].tolist(),
            input_batch.num_scheduled_tokens[:num_reqs].tolist(),
            input_batch.num_tokens_after_padding,
        )
        draft = _draft()
        if draft is not None:
            # The drafter's row tables of the step, and the rows its catch-up gathers.
            draft.step(list(input_batch.req_ids), input_batch.query_start_loc)

    def postprocess_state(
        self,
        idx_mapping: torch.Tensor,
        num_sampled: torch.Tensor | int,
        num_computed_tokens: torch.Tensor | None = None,
    ) -> None:
        """Hand the drafter the step's sampled-token counts (its sampled rows at
        MTP K = 1).
        """
        super().postprocess_state(idx_mapping, num_sampled, num_computed_tokens)
        draft = _draft()
        if draft is not None:
            draft.observe_sampled(num_sampled)

    def prepare_attn(
        self,
        input_batch: InputBatch,
        cudagraph_mode: CUDAGraphMode,
        block_tables: tuple[torch.Tensor, ...],
        slot_mappings: torch.Tensor,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        for_capture: bool = False,
        ubatch_idx: int = 0,
        model_specific_attn_metadata: ModelSpecificAttnMetadata | None = None,
    ) -> dict[str, Any]:
        """Return the cached metadata of a segment-owned FULL step, or build it."""
        key = None
        segment = self._segment()
        if (
            segment is not None
            and cudagraph_mode == CUDAGraphMode.FULL
            and not for_capture
            and ubatch_idx == 0
            and not input_batch.has_prefill
            and input_batch.max_query_len is None
        ):
            key = (
                input_batch.num_reqs,
                input_batch.num_reqs_after_padding,
                input_batch.num_tokens,
                input_batch.num_tokens_after_padding,
            )
            cached = self._metadata.get(key)
            if cached is not None and segment.owns_attn_metadata(
                input_batch.num_tokens_after_padding
            ):
                return cached
        metadata = super().prepare_attn(
            input_batch,
            cudagraph_mode,
            block_tables,
            slot_mappings,
            attn_groups,
            kv_cache_config,
            for_capture=for_capture,
            ubatch_idx=ubatch_idx,
            model_specific_attn_metadata=model_specific_attn_metadata,
        )
        if key is not None:
            self._metadata.setdefault(key, metadata)
        return metadata

    def capture_inner_cudagraphs(self, *args: Any, **kwargs: Any) -> None:
        """Drop the cached metadata: a capture follows every reallocation of its
        buffers.
        """
        self._metadata.clear()
        return super().capture_inner_cudagraphs(*args, **kwargs)
