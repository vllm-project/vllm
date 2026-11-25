# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable
from copy import copy
from typing import TYPE_CHECKING, Any, ClassVar, NamedTuple

import torch
from torch import nn

from vllm.config import CUDAGraphMode, VllmConfig
from vllm.forward_context import BatchDescriptor, set_forward_context
from vllm.logger import init_logger
from vllm.model_executor.model_loader import get_model
from vllm.model_executor.model_loader.utils import get_draft_load_config
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.attn_utils import build_slot_mappings_by_layer
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.cudagraph_utils import (
    BatchExecutionDescriptor,
    CudaGraphManager,
)
from vllm.v1.worker.gpu.dp_utils import DPSyncState
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.model_states.interface import ModelState
from vllm.v1.worker.gpu.sample.greedy_argmax import greedy_argmax
from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample
from vllm.v1.worker.gpu.spec_decode.eagle.utils import (
    get_target_lm_head,
    maybe_share_target_embed,
    share_target_lm_head,
)
from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator
from vllm.v1.worker.utils import AttentionGroup

from .dfly_inputs import prepare_dfly_inputs, write_dfly_query_slots
from .dfly_target_reuse import HYV4DFlyTargetReuse

if TYPE_CHECKING:
    from vllm.models.hy_v4.nvidia.attention import HYV4MLAAttention
    from vllm.models.hy_v4.nvidia.dfly import HYV4DFlyForCausalLM

logger = init_logger(__name__)


class DFlyProposeInputs(NamedTuple):
    """The `propose` inputs a DFly query block is built from: the target step's
    results and per-request sampling state."""

    query_start_loc: torch.Tensor
    positions: torch.Tensor
    idx_mapping: torch.Tensor
    num_sampled: torch.Tensor
    num_rejected: torch.Tensor
    last_sampled: torch.Tensor
    last_hidden_states: torch.Tensor
    # Copied to the speculator's fixed buffers, which graphs read.
    temperature: torch.Tensor
    seeds: torch.Tensor


class HYV4DFlyQueryModel(nn.Module):
    """Fixed-shape query computation; attention keeps its existing eager breaks."""

    def __init__(
        self,
        model: "HYV4DFlyForCausalLM",
        draft_tokens: torch.Tensor,
        sample_fn: Callable[[int, torch.Tensor], torch.Tensor],
    ):
        super().__init__()
        self.model = model
        self.query_width = model.model.query_width
        self.sample_fn = sample_fn
        self.register_buffer("draft_tokens", draft_tokens, persistent=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        target_hidden: torch.Tensor,
        active: torch.Tensor,
    ) -> torch.Tensor:
        batch = active.shape[0]
        embeddings = self.model.embed_input_ids(input_ids)
        embeddings[:: self.query_width] = self.model.model.mtp_fusion(
            embeddings[:: self.query_width], target_hidden
        )
        hidden = self.model(
            input_ids=input_ids, positions=positions, inputs_embeds=embeddings
        ).view(batch, self.query_width, -1)
        previous = input_ids.view(batch, self.query_width)[:, 0].long()
        for step in range(self.query_width - 1):
            corrected = self.model.apply_hidden_correction(
                hidden[:, step + 1], previous
            )
            previous = self.sample_fn(step, corrected)
            self.draft_tokens[:batch, step] = torch.where(active, previous, -1)
        return self.draft_tokens[:batch]


class HYV4DFlySpeculator(DraftModelSpeculator):
    """Vendor-neutral DFly speculator; vendors bind their draft and attention."""

    can_reuse_target_attn_metadata = False
    draft_model_cls: ClassVar[type["HYV4DFlyForCausalLM"]]
    attention_cls: ClassVar[type["HYV4MLAAttention"]]

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        cache_dtype = vllm_config.cache_config.cache_dtype
        if cache_dtype not in ("auto", "bfloat16", "fp8_ds_mla"):
            raise ValueError("DFly requires BF16 or fp8_ds_mla MLA KV cache")
        spec = vllm_config.speculative_config
        assert spec is not None
        draft_cache_dtype = spec.kv_cache_dtype or cache_dtype
        target_dtype = "bfloat16" if cache_dtype == "auto" else cache_dtype
        draft_dtype = "bfloat16" if draft_cache_dtype == "auto" else draft_cache_dtype
        if draft_dtype != target_dtype:
            raise ValueError(
                "DFly draft and target must use the same MLA KV cache dtype; "
                "omit speculative_config.kv_cache_dtype to inherit the target"
            )
        kv_transfer = vllm_config.kv_transfer_config
        if (
            kv_transfer is not None
            and kv_transfer.kv_connector != "DecodeBenchConnector"
        ):
            raise ValueError("DFly does not support PD disaggregation")
        if (
            vllm_config.lora_config is not None
            or vllm_config.parallel_config.use_ubatching
        ):
            raise ValueError("DFly does not support LoRA or microbatching")
        super().__init__(vllm_config, device)
        config = super().attn_vllm_config
        config.model_config = copy(self.draft_model_config)
        config.parallel_config = copy(vllm_config.parallel_config)
        config.parallel_config.enable_expert_parallel = False
        config.parallel_config.is_moe_model = False
        config.attention_config = copy(config.attention_config)
        config.attention_config.sparse_mla_force_mqa = True
        config.attention_config.use_non_causal = False
        config.compilation_config = copy(config.compilation_config)
        config.cache_config = vllm_config.cache_config
        config.quant_config = None
        self._draft_vllm_config = config
        self.num_query_per_req = self.num_speculative_steps + 1
        self.uses_mrope = self.draft_model_config.uses_mrope
        self.target_hidden_states = torch.zeros(
            self.max_num_reqs, self.hidden_size, dtype=self.dtype, device=device
        )
        self.active_requests = torch.zeros(
            self.max_num_reqs, dtype=torch.bool, device=device
        )
        # The target batch row of each request's last accepted token, and the
        # position of its bonus (anchor) query.
        self.source_rows = torch.zeros(
            self.max_num_reqs, dtype=torch.int64, device=device
        )
        self.anchors = torch.zeros(self.max_num_reqs, dtype=torch.int64, device=device)
        self.mrope_positions = (
            torch.zeros(
                self.draft_model_config.mrope_num_dims,
                self.max_num_tokens,
                dtype=torch.int64,
                device=device,
            )
            if self.uses_mrope
            else None
        )
        # Draft logits cache column of each step, as 0-d views.
        self._step_cols = torch.arange(
            self.num_speculative_steps, dtype=torch.int32, device=device
        )
        self.query_model: HYV4DFlyQueryModel | None = None
        self.cudagraph_manager: CudaGraphManager | None = None
        self.reuse: HYV4DFlyTargetReuse | None = None
        self._storage_validated = False
        self._slot_mappings: torch.Tensor | None = None

    def load_draft_model(
        self, target_model: nn.Module, target_attn_layer_names: set[str]
    ) -> nn.Module:
        config = self._draft_vllm_config
        config.quant_config = VllmConfig.get_quantization_config(
            self.draft_model_config, get_draft_load_config(config)
        )
        draft = get_model(
            vllm_config=config,
            model_config=self.draft_model_config,
            load_config=get_draft_load_config(config),
            prefix="dfly",
        )
        if not isinstance(draft, self.draft_model_cls):
            raise TypeError("DFly speculator requires HYV4DFlyForCausalLM")
        language_model = (
            target_model.get_language_model()
            if hasattr(target_model, "get_language_model")
            else target_model
        )
        maybe_share_target_embed(draft, draft.model, language_model.model)
        head = share_target_lm_head(
            draft,
            get_target_lm_head(target_model, language_model),
            quantization=None,
        )
        if head is None or draft.model.embed_tokens is None:
            raise ValueError("DFly requires compatible target embedding and LM head")
        if (
            head.weight.dtype != self.dtype
            or draft.model.embed_tokens.weight.dtype != self.dtype
        ):
            raise ValueError("DFly requires BF16 shared embedding and LM head weights")
        draft.logits_processor.head_dtype = self.vllm_config.model_config.head_dtype
        self.reuse = HYV4DFlyTargetReuse(
            target_model, draft, self.vllm_config, self.attention_cls
        )
        self.query_model = HYV4DFlyQueryModel(
            draft, self.draft_tokens, self._sample_step
        )
        config.quant_config = draft.model.quant_config
        config.model_config.hf_config = draft.config
        config.model_config.hf_text_config = draft.config
        logger.info(
            "HYV4 DFly: %d layers, N=%d, TP=%d, DP=%d (TP-local draft), "
            "target index/KV=%s/%s, %s writable KV aliases, "
            "max length=%d, draft quantization=%s, draft sampling=%s",
            len(draft.model.layers),
            self.num_speculative_steps,
            config.parallel_config.tensor_parallel_size,
            config.parallel_config.data_parallel_size,
            draft.config.target_index_layer,
            draft.config.target_kv_layers,
            config.cache_config.cache_dtype,
            self.max_model_len,
            self.draft_model_config.quantization or "none",
            self.speculative_config.draft_sample_method,
        )
        return draft

    @property
    def attn_vllm_config(self) -> VllmConfig:
        return self._draft_vllm_config

    def set_attn(
        self,
        model_state: ModelState,
        kv_cache_config: KVCacheConfig,
        block_tables: BlockTables,
        target_input_buffers: InputBuffers,
        target_attn_groups: list[list[AttentionGroup]],
        kernel_block_sizes: list[int],
    ) -> None:
        super().set_attn(
            model_state,
            kv_cache_config,
            block_tables,
            target_input_buffers,
            target_attn_groups,
            kernel_block_sizes,
        )
        assert self.reuse is not None
        self.reuse.validate_groups(kv_cache_config)
        self._storage_validated = False
        self._slot_mappings = torch.full(
            (len(kv_cache_config.kv_cache_groups), self.max_num_tokens),
            PAD_SLOT_ID,
            dtype=torch.int64,
            device=self.device,
        )

    def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode) -> None:
        query_width = self.num_query_per_req
        max_reqs = min(self.max_num_reqs, self.max_num_tokens // query_width)
        if cudagraph_mode.decode_mode() == CUDAGraphMode.FULL:
            cg_support = self.attn_cg_support
            if (
                cg_support.min_cg_support.value
                >= AttentionCGSupport.UNIFORM_BATCH.value
            ):
                # Every query block is a uniform decode batch of query_width
                # tokens per request, so the whole query step can be one graph.
                self.cudagraph_manager = CudaGraphManager(
                    self._draft_vllm_config,
                    self.device,
                    CUDAGraphMode.FULL_DECODE_ONLY,
                    decode_query_lens={query_width: max_reqs},
                )
                logger.info(
                    "DFly CUDA graphs: FULL, %d tokens per request, up to %d requests",
                    query_width,
                    max_reqs,
                )
                return
            logger.warning(
                "DFly draft attention (%s) does not support FULL CUDA graphs; "
                "using breakable PIECEWISE graphs.",
                cg_support.min_cg_attn_backend,
            )

        config = copy(self._draft_vllm_config)
        config.compilation_config = copy(config.compilation_config)
        compilation = config.compilation_config
        request_sizes = set(compilation.decode_cudagraph_capture_request_sizes or ())
        request_sizes.update(
            min(cdiv(size, query_width), max_reqs)
            for size in compilation.prefill_cudagraph_capture_token_sizes or ()
        )
        compilation.prefill_cudagraph_capture_token_sizes = sorted(
            n * query_width for n in request_sizes if 0 < n <= max_reqs
        )
        compilation.use_breakable_cudagraph = True
        mode = (
            CUDAGraphMode.NONE
            if cudagraph_mode == CUDAGraphMode.NONE
            else CUDAGraphMode.PIECEWISE
        )
        self.cudagraph_manager = CudaGraphManager(
            config, self.device, mode, decode_query_lens={query_width: max_reqs}
        )
        if mode != CUDAGraphMode.NONE:
            assert self.query_model is not None
            self.cudagraph_manager.init_breakable_cg_runner(self.query_model)
            logger.info(
                "DFly CUDA graphs: breakable PIECEWISE, query token buckets=%s; "
                "sparse attention and metadata remain eager",
                compilation.prefill_cudagraph_capture_token_sizes,
            )

    @torch.inference_mode()
    def capture(self) -> None:
        manager = self.cudagraph_manager
        if manager is None or not manager.needs_capture():
            return
        assert self.reuse is not None
        self.reuse.validate_storage()

        def create_forward_fn(
            desc: BatchExecutionDescriptor, warmup: bool
        ) -> Callable[[CUDAGraphMode], None]:
            count = desc.num_tokens
            batch = count // self.num_query_per_req
            self.block_tables.get_dummy_block_tables(batch)
            # An all-padding block: no K/V writes, no draft logits writes.
            self._prepare_query(
                self._dummy_propose_inputs(), 0, batch, write_slots=True
            )
            # A FULL graph keeps the metadata it was captured with, so capture
            # it for the longest sequences it may replay.
            self.draft_max_seq_len = (
                self.max_model_len
                if desc.cg_mode == CUDAGraphMode.FULL
                else self.num_query_per_req
            )
            metadata, slots = self._query_attention_state(
                desc, torch.zeros(batch, dtype=torch.int32, device="cpu")
            )

            def forward(cg_mode: CUDAGraphMode) -> None:
                self._run_query_model(count, metadata, slots, cg_mode)

            return forward

        manager.capture(
            create_forward_fn, progress_bar_desc="Capturing DFly CUDA graphs"
        )

    def _dummy_propose_inputs(self) -> DFlyProposeInputs:
        """Placeholder inputs for dummy runs and capture; never read with
        `num_reqs=0`."""
        return DFlyProposeInputs(
            query_start_loc=self.input_buffers.query_start_loc,
            positions=self.input_buffers.positions,
            idx_mapping=self.idx_mapping,
            num_sampled=self.source_rows,
            num_rejected=self.source_rows,
            last_sampled=self.draft_tokens,
            last_hidden_states=self.target_hidden_states,
            temperature=self.temperature,
            seeds=self.seeds,
        )

    def _prepare_query(
        self,
        inputs: DFlyProposeInputs,
        num_reqs: int,
        padded_batch: int,
        write_slots: bool,
    ) -> None:
        """Fill the query block, its slots and the draft layers' sparse indices.
        Rows past `num_reqs` are padding, so `num_reqs=0` gives an all-padding
        block."""
        assert self.reuse is not None
        rope_state = getattr(getattr(self, "model_state", None), "rope_state", None)
        prepare_dfly_inputs(
            input_ids=self.input_buffers.input_ids,
            positions=self.input_buffers.positions,
            is_padding=self.input_buffers.is_padding,
            query_start_loc=self.input_buffers.query_start_loc,
            seq_lens=self.input_buffers.seq_lens,
            active=self.active_requests,
            source_rows=self.source_rows,
            anchors=self.anchors,
            target_hidden_out=self.target_hidden_states,
            mrope_positions=self.mrope_positions,
            draft_idx_mapping=self.idx_mapping,
            out_temperature=self.temperature,
            out_seeds=self.seeds,
            target_query_start_loc=inputs.query_start_loc,
            target_positions=inputs.positions,
            idx_mapping=inputs.idx_mapping,
            num_sampled=inputs.num_sampled,
            num_rejected=inputs.num_rejected,
            last_sampled=inputs.last_sampled,
            target_hidden=inputs.last_hidden_states,
            mrope_delta=rope_state.prefill_delta.gpu if rope_state else None,
            temperature=inputs.temperature,
            seeds=inputs.seeds,
            max_model_len=self.max_model_len_gpu,
            mask_token_id=self.model.config.mask_token_id,
            query_width=self.num_query_per_req,
            num_reqs=num_reqs,
            padded_batch=padded_batch,
        )
        if write_slots:
            assert self._slot_mappings is not None
            for gid, groups in enumerate(self.attn_groups):
                if groups:
                    write_dfly_query_slots(
                        self._slot_mappings[gid],
                        self.input_buffers.positions,
                        self.active_requests,
                        self.block_tables.input_block_tables[gid],
                        self.block_tables.block_sizes[gid],
                        self.max_model_len_gpu,
                        self.num_query_per_req,
                        padded_batch,
                    )
        self.reuse.update_indices(
            self.active_requests, self.source_rows, self.anchors, padded_batch
        )

    def _sample_step(self, step: int, hidden: torch.Tensor) -> torch.Tensor:
        """The draft tokens at query offset `step + 1` of every row."""
        if self.draft_logits is None:
            return greedy_argmax(self.model.compute_logits(hidden))
        batch = hidden.shape[0]
        # Draws are keyed by the position before the predicted token.
        pos = self.anchors[:batch] + step
        return gumbel_sample(
            self.model.compute_logits(hidden),
            self.idx_mapping[:batch],
            self.temperature,
            self.seeds,
            pos,
            apply_temperature=True,
            is_drafting=True,
            logits_cache=self.draft_logits.values,
            logits_cache_col=self._step_cols[step],
            use_fp64=self.use_fp64_gumbel,
            logits_cache_row_mapping=self.draft_logits.row_mapping,
        )

    def _query_attention_state(
        self, desc: BatchExecutionDescriptor, seq_lens_cpu_upper_bound: torch.Tensor
    ) -> tuple[dict[str, Any] | None, dict[str, torch.Tensor]]:
        assert self._slot_mappings is not None
        count = desc.num_tokens
        batch = count // self.num_query_per_req
        metadata = self._build_attn_metadata(
            num_reqs=batch,
            batch_desc=desc,
            query_start_loc_np=self.arange_np[: batch + 1] * self.num_query_per_req,
            seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
            step=self.num_query_per_req,
            causal=True,
            slot_mappings=self._slot_mappings[:, :count],
        )
        slots = build_slot_mappings_by_layer(
            self._slot_mappings[:, :count], self.kv_cache_config
        )
        return metadata, slots

    def _run_query_model(
        self,
        count: int,
        metadata: dict[str, Any] | None,
        slots: dict[str, torch.Tensor] | None,
        cg_mode: CUDAGraphMode,
    ) -> None:
        assert self.query_model is not None
        batch = count // self.num_query_per_req
        positions = (
            self.input_buffers.positions[:count]
            if self.mrope_positions is None
            else self.mrope_positions[:, :count]
        )
        model_inputs = dict(
            input_ids=self.input_buffers.input_ids[:count],
            positions=positions,
            target_hidden=self.target_hidden_states[:batch],
            active=self.active_requests[:batch],
        )
        with set_forward_context(
            metadata,
            self._draft_vllm_config,
            num_tokens=count,
            cudagraph_runtime_mode=cg_mode,
            batch_descriptor=BatchDescriptor(num_tokens=count),
            slot_mapping=slots,
            is_padding=self.input_buffers.is_padding[:count],
        ):
            if cg_mode == CUDAGraphMode.PIECEWISE:
                assert self.cudagraph_manager is not None
                self.cudagraph_manager.run_pw_graph(self.query_model, model_inputs)
            else:
                self.query_model(**model_inputs)

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
        batch = input_batch.num_reqs
        query_width = self.num_query_per_req
        count = batch * query_width
        if count > self.max_num_tokens:
            raise ValueError("DFly query block exceeds max_num_batched_tokens")
        result = self.draft_tokens[:batch]
        if batch == 0 or num_speculative_tokens == 0:
            return result.fill_(-1)
        assert self.reuse is not None
        descriptor = BatchExecutionDescriptor(CUDAGraphMode.NONE, count, batch)
        if not dummy_run and self.cudagraph_manager is not None:
            descriptor = self.cudagraph_manager.dispatch(
                batch, count, uniform_token_count=query_width, num_active_loras=0
            )
        padded_count = descriptor.num_tokens
        assert padded_count % query_width == 0
        padded_batch = padded_count // query_width
        if not dummy_run and not self._storage_validated:
            self.reuse.validate_storage()
            self._storage_validated = True

        with_attn = not (dummy_run and skip_attn_for_dummy_run)
        if dummy_run:
            inputs, num_reqs = self._dummy_propose_inputs(), 0
        else:
            inputs = DFlyProposeInputs(
                query_start_loc=input_batch.query_start_loc,
                positions=input_batch.positions,
                idx_mapping=input_batch.idx_mapping,
                num_sampled=num_sampled,
                num_rejected=num_rejected,
                last_sampled=last_sampled,
                last_hidden_states=last_hidden_states,
                temperature=temperature,
                seeds=seeds,
            )
            num_reqs = batch
        self._prepare_query(inputs, num_reqs, padded_batch, write_slots=with_attn)

        metadata = None
        draft_slots = None
        if with_attn:
            upper_bound = torch.zeros(padded_batch, dtype=torch.int32, device="cpu")
            upper_bound[:batch].copy_(input_batch.seq_lens_cpu_upper_bound[:batch])
            self.draft_max_seq_len = min(
                int(upper_bound.max()) + query_width, self.max_model_len
            )
            metadata, draft_slots = self._query_attention_state(descriptor, upper_bound)

        if descriptor.cg_mode == CUDAGraphMode.FULL:
            # The inputs, slots, indices and metadata above fill the persistent
            # buffers the graph reads.
            assert self.cudagraph_manager is not None
            self.cudagraph_manager.run_fullgraph(descriptor)
        else:
            self._run_query_model(
                padded_count, metadata, draft_slots, descriptor.cg_mode
            )
        return result
