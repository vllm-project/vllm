# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The model state of the specialized decode.

`Qwen4ExpSpecializedDecodeModelState` is `Qwen4ExpModelState` plus:

- the decode executor, built once the weights are loaded: the model runner
  creates the model state after loading the target and draft models, and
  before memory profiling and CUDA graph capture. The state warms up the
  executor's kernels right away, since memory profiling already captures the
  largest decode graphs;
- with `draft_head_fp8_prefilter`, the FP8-prefiltered draft head over the
  target's LM head, which the MTP drafter shares (`logits/draft_head.py`),
  built at the same point;
- for uniform decode steps on FULL graphs, CUDA graphs of the stock attention
  metadata build, recorded with the inner CUDA graphs and replayed in place of
  the eager build, on a side stream beside the forward graph unless
  `decode_metadata_overlap` is off, and the n-gram inputs written by one
  kernel, compiled right after recording (`decode_metadata`);
- the GDN layers' record/commit state protocol (`gdn_records.py`): its
  record header buffers, built with the executor (memory profiling counts them),
  and the commit of every step whose FULL graph replay ran the records core,
  marked in `prepare_attn` and launched in `postprocess_state` before the
  stock postprocess; `prepare_attn` also binds the target model's
  `seq_lens`, which the records core reads in prefix-caching `align`
  mode. It is on
  whenever the fused GDN core is, unless mixed batches also take FULL CUDA
  graphs (`cudagraph_mode=FULL`): then a FULL graph's row count would not
  tell whether it holds the records core, and the GDN layers keep vLLM's
  protocol; or unless the rows per request (and in `align` mode the mamba
  block size) leave no two speculative slots for copies of the records.

Everything else, including `preprocess_state` and the CUDA graph handling of
`MambaHybridModelState`, is the stock state's. `preprocess_state` stays
eager: it moves mamba state across block boundaries, so it cannot run twice
for the runtime check, and its host cost is small: it launches at least two
kernels, the decision kernel (`preprocess_mamba_align_fused_kernel`) and
`precopy_mamba_align_fused_kernel`.
"""

from typing import Any

import torch
from torch import nn

from vllm.config import CUDAGraphMode, VllmConfig
from vllm.logger import init_logger
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
from vllm.v1.worker.gpu.model_states.interface import ModelSpecificAttnMetadata
from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState
from vllm.v1.worker.gpu.states import RequestState
from vllm.v1.worker.utils import AttentionGroup

from ...model import Qwen4ExpForCausalLM, Qwen4ExpForConditionalGeneration
from ...model_state import Qwen4ExpModelState
from .. import DecodeConfig
from ..logits.processor import bind_draft_head
from .decode import SpecializedDecode
from .decode_metadata import (
    DecodeMetadataGraphs,
    Key,
    capture_keys,
    differences,
    fill_ngram_inputs,
    snapshot,
    supported,
    unrecordable_builders,
    warm_up_ngram_inputs,
)
from .gdn_records import GdnRecords, records_fit

logger = init_logger(__name__)


class Qwen4ExpSpecializedDecodeModelState(Qwen4ExpModelState):
    """`Qwen4ExpModelState` that builds the specialized decode executor and
    replays the attention metadata of uniform decode steps."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        model: nn.Module,
        encoder_cache: EncoderCache | None,
        device: torch.device,
    ) -> None:
        """Build the stock state, then the decode executor of the loaded
        target `model` (`Qwen4ExpForCausalLM` or its multimodal
        wrapper)."""
        super().__init__(vllm_config, model, encoder_cache, device)
        language_model = (
            model.language_model
            if isinstance(model, Qwen4ExpForConditionalGeneration)
            else model
        )
        assert isinstance(language_model, Qwen4ExpForCausalLM)
        self.decode: SpecializedDecode = language_model.build_specialized_decode()
        self.gdn_records = self._build_gdn_records()
        # The request rows of the records step to commit (its idx_mapping).
        self._pending_commit: torch.Tensor | None = None
        # Before memory profiling, whose graph captures run the executor.
        self.decode.warm_up()
        if self.gdn_records is not None:
            self.gdn_records.warm_up()
        config = language_model.specialized_decode_config
        assert config is not None
        if config.draft_head_fp8:
            # The drafter shares the target's head; memory profiling counts
            # the FP8 copy.
            bind_draft_head(language_model.logits_processor, language_model.lm_head)
        self.init_decode_metadata(config)
        # The executor's FULL graph captures wait for the overlapped replays.
        if config.metadata_overlap:
            self.decode.metadata = self.metadata

    def _build_gdn_records(self) -> GdnRecords | None:
        """The GDN records protocol of the executor's GDN layers, bound to
        them, or None to keep vLLM's protocol (module docstring)."""
        gdn = self.decode.gdn
        if gdn.core is None:
            return None
        cudagraph_mode = self.vllm_config.compilation_config.cudagraph_mode
        if cudagraph_mode is None or cudagraph_mode.mixed_mode() == CUDAGraphMode.FULL:
            logger.info_once(
                "Qwen4Exp specialized decode: the GDN layers keep vLLM's "
                "speculative state protocol (mixed batches take FULL CUDA graphs)."
            )
            return None
        block_size = 0
        if self._align_mode:
            block_size = self.cache_config.mamba_block_size or 0
        if not records_fit(gdn.tokens_per_request, self._align_mode, block_size):
            logger.info_once(
                "Qwen4Exp specialized decode: the GDN layers keep vLLM's "
                "speculative state protocol (%d rows per request and mamba "
                "blocks of %d tokens leave no two speculative slots for the "
                "records).",
                gdn.tokens_per_request,
                block_size,
            )
            return None
        records = GdnRecords(
            gdn.layers,
            gdn.tokens_per_request,
            min(self.max_num_reqs, gdn.max_requests),
            gdn.key_heads,
            align=self._align_mode,
        )
        gdn.bind_records(records)
        logger.info_once(
            "Qwen4Exp specialized decode: the GDN layers record and commit their "
            "speculative state; the record headers of %d requests take %.2f MiB.",
            records.layout.capacity,
            records.nbytes / (1 << 20),
        )
        return records

    def init_decode_metadata(self, config: DecodeConfig) -> None:
        """Set up the replayed metadata of uniform decode steps, with the
        deployment's specialized decode `config`."""
        self.metadata = DecodeMetadataGraphs(
            self.vllm_config,
            decode_query_len=config.tokens_per_request,
            max_num_reqs=self.max_num_reqs,
            device=self.device,
            check_steps=config.metadata_check_steps,
            overlap=config.metadata_overlap,
        )
        # The batch whose attention metadata replayed, and its key: its n-gram
        # inputs take the kernel.
        self._replayed: tuple[InputBatch, Key] | None = None

    def _unsupported(self, attn_groups: list[list[AttentionGroup]]) -> list[str]:
        """Why the attention metadata of `attn_groups` cannot replay."""
        reasons = []
        if not supported():
            reasons.append("needs CUDA and Triton")
        if self._use_flashinfer_replayssm or self.recoverssm is not None:
            reasons.append("ReplaySSM and RecoverSSM keep per-step host state")
        if self.uses_ngram_embedding and self.disk_offload_modules:
            reasons.append("PLE disk offload stages rows on the host")
        if names := unrecordable_builders(attn_groups):
            reasons.append(f"builders {', '.join(names)} do not record")
        return reasons

    def initialize_kv_cache(
        self, kv_cache_config: KVCacheConfig, block_tables: BlockTables
    ) -> None:
        super().initialize_kv_cache(kv_cache_config, block_tables)
        # Memory profiling's recordings wrote the builders of its KV cache;
        # its forward graphs are dropped and captured again.
        self.metadata.release()
        self.metadata.forget_joins()
        if self.gdn_records is not None:
            block_size = 0
            if self._align_mode:
                block_size = self._get_mamba_group_info(kv_cache_config)[1].block_size
            self.gdn_records.bind_caches(block_size)

    def capture_inner_cudagraphs(
        self,
        input_buffers: InputBuffers,
        block_tables: BlockTables,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        stream: torch.cuda.Stream | None = None,
    ) -> None:
        super().capture_inner_cudagraphs(
            input_buffers, block_tables, attn_groups, kv_cache_config, stream
        )
        self.metadata.release()
        if reasons := self._unsupported(attn_groups):
            logger.warning_once(
                "Qwen4Exp specialized decode: attention metadata builds eagerly: %s.",
                "; ".join(reasons),
            )
            return
        keys = capture_keys(self.vllm_config, self.metadata.decode_query_len)
        stock = super().prepare_attn
        buffers = []
        if self._align_mode and self._mamba_ctx is not None:
            assert self._mamba_ctx.aligned_state_indices is not None
            buffers.append(self._mamba_ctx.aligned_state_indices)

        def stock_build(input_batch: InputBatch) -> dict[str, Any]:
            """The stock build of a batch of a recorded key."""
            num_reqs_padded = input_batch.num_reqs_after_padding
            return stock(
                input_batch,
                CUDAGraphMode.FULL,
                tuple(
                    table[:num_reqs_padded] for table in block_tables.input_block_tables
                ),
                block_tables.slot_mappings[:, : input_batch.num_tokens_after_padding],
                attn_groups,
                kv_cache_config,
            )

        self.metadata.record(
            keys, input_buffers, attn_groups, buffers, stock_build, self.max_model_len
        )
        if self.uses_ngram_embedding:
            warm_up_ngram_inputs(
                input_buffers.query_start_loc,
                self.ngram_context,
                self.ple_query_start_loc,
                self.ngram_eos_token_id,
            )
        logger.info_once(
            "Qwen4Exp specialized decode: recorded the attention metadata of %d "
            "uniform decode shapes; %d replay beside the forward graph.",
            len(keys),
            len(self.metadata.overlapped_keys),
        )

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
        """Replay the recorded build of a uniform decode step; build any other
        step's metadata the stock way. Mark a step whose FULL graph runs the
        GDN records core for the commit after sampling."""
        if self._pending_commit is not None:
            raise RuntimeError(
                "Qwen4Exp specialized decode: a step that ran the GDN records "
                "core was not committed."
            )
        attn_metadata = self._prepare_attn(
            input_batch,
            cudagraph_mode,
            block_tables,
            slot_mappings,
            attn_groups,
            kv_cache_config,
            for_capture,
            ubatch_idx,
            model_specific_attn_metadata,
        )
        records = self.gdn_records
        if records is not None and records.layers[0].prefix in attn_metadata:
            # The target model's batch (the MTP drafter's has its own input
            # buffers and no GDN layers): its seq_lens, which the records core
            # reads in align mode, at an address that must not move.
            records.bind_seq_lens(input_batch.seq_lens)
        if (
            records is not None
            and not for_capture
            and cudagraph_mode == CUDAGraphMode.FULL
            and records.records_step(input_batch.num_tokens_after_padding)
            # The target model's forward: the MTP drafter builds its own
            # metadata here too, for its attention groups, without GDN layers.
            and records.layers[0].prefix in attn_metadata
        ):
            metadata = attn_metadata[records.layers[0].prefix]
            if not (
                isinstance(metadata, GDNAttentionMetadata)
                and metadata.num_prefills == 0
                and metadata.num_decodes == 0
                and metadata.num_spec_decodes == input_batch.num_reqs
            ):
                raise RuntimeError(
                    "Qwen4Exp specialized decode: a FULL CUDA graph with the GDN "
                    "records core replays for a step that is not a pure "
                    f"speculative decode ({metadata})."
                )
            self._pending_commit = input_batch.idx_mapping
        return attn_metadata

    def _prepare_attn(
        self,
        input_batch: InputBatch,
        cudagraph_mode: CUDAGraphMode,
        block_tables: tuple[torch.Tensor, ...],
        slot_mappings: torch.Tensor,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        for_capture: bool,
        ubatch_idx: int,
        model_specific_attn_metadata: ModelSpecificAttnMetadata | None,
    ) -> dict[str, Any]:
        """`prepare_attn`'s metadata: the recorded build of a uniform decode
        step replayed, any other step's built the stock way."""
        self._replayed = None
        key = None
        if not for_capture and model_specific_attn_metadata is None:
            key = self.metadata.key(input_batch, cudagraph_mode, attn_groups)
        if key is None:
            self.metadata.settle()
            return super().prepare_attn(
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
        # decode_backend=latency_specialized refuses micro-batching.
        assert ubatch_idx == 0
        stock = super().prepare_attn
        attn_metadata = self.metadata.replay(
            key,
            input_batch,
            lambda batch: stock(
                batch,
                cudagraph_mode,
                block_tables,
                slot_mappings,
                attn_groups,
                kv_cache_config,
            ),
        )
        self._replayed = (input_batch, key)
        return attn_metadata

    def postprocess_state(
        self,
        idx_mapping: torch.Tensor,
        num_sampled: torch.Tensor | int,
        num_computed_tokens: torch.Tensor | None = None,
    ) -> None:
        """Commit a GDN records step (`gdn_records.py`), then run the stock
        postprocess: the accepted-count scatter and the prefix-caching
        `align` copies, which read the committed slots."""
        pending, self._pending_commit = self._pending_commit, None
        if pending is not None:
            assert self.gdn_records is not None
            if idx_mapping is not pending or not isinstance(num_sampled, torch.Tensor):
                raise RuntimeError(
                    "Qwen4Exp specialized decode: the postprocess of a GDN records "
                    "step is not the step's own (another batch, or no sampled "
                    "counts)."
                )
            self.gdn_records.commit(
                idx_mapping.shape[0], num_sampled, idx_mapping, num_computed_tokens
            )
        super().postprocess_state(idx_mapping, num_sampled, num_computed_tokens)

    def prepare_inputs(
        self,
        input_batch: InputBatch,
        req_states: RequestState,
    ) -> dict[str, Any]:
        """Write the n-gram inputs of a replayed step with one kernel; any
        other step's the stock way."""
        replayed, self._replayed = self._replayed, None
        if (
            replayed is None
            or replayed[0] is not input_batch
            or not self.uses_ngram_embedding
        ):
            return super().prepare_inputs(input_batch, req_states)
        if self.metadata.check_inputs(replayed[1]):
            return self._checked_inputs(input_batch, req_states, replayed[1])
        return self._replayed_inputs(input_batch, req_states)

    def _replayed_inputs(
        self, input_batch: InputBatch, req_states: RequestState
    ) -> dict[str, Any]:
        """The model inputs of a replayed step: the stock ones without the
        n-gram inputs, which one kernel writes."""
        model_inputs = MambaHybridModelState.prepare_inputs(
            self, input_batch, req_states
        )
        fill_ngram_inputs(
            input_batch,
            req_states.num_computed_tokens.gpu,
            req_states.all_token_ids.gpu,
            self.ngram_context,
            self.ple_query_start_loc,
            self.ngram_eos_token_id,
        )
        model_inputs.update(
            query_start_loc=self.ple_query_start_loc,
            ngram_context=self.ngram_context,
        )
        return model_inputs

    def _checked_inputs(
        self, input_batch: InputBatch, req_states: RequestState, key: Key
    ) -> dict[str, Any]:
        """`_replayed_inputs`, checked against the stock inputs written from
        the same buffer state; raises if they differ."""
        buffers = [self.ngram_context, self.ple_query_start_loc]
        before = [buffer.clone() for buffer in buffers]
        model_inputs = self._replayed_inputs(input_batch, req_states)
        fast = snapshot(model_inputs, buffers)
        for buffer, saved in zip(buffers, before):
            buffer.copy_(saved)
        stock = snapshot(super().prepare_inputs(input_batch, req_states), buffers)
        if found := differences(fast, stock):
            raise RuntimeError(
                f"Qwen4Exp decode metadata check: the n-gram inputs of {key} "
                "differ from the stock ones: " + "; ".join(found)
            )
        return model_inputs
