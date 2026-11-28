# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DFly2 proposal over read-only DeepSeek-V4.1 target caches."""

from copy import copy

import torch
from torch import nn

from vllm.config import CUDAGraphMode, VllmConfig
from vllm.config.kernel import MEGA_MOE_BACKENDS
from vllm.logger import init_logger
from vllm.model_executor.model_loader import get_model
from vllm.model_executor.model_loader.utils import get_draft_load_config
from vllm.models.deepseek_v41.attention import DeepseekV4Attention
from vllm.models.hy_v4.common.dfly_speculator import HYV4DFlySpeculator
from vllm.v1.worker.gpu.cudagraph_utils import CudaGraphManager
from vllm.v1.worker.gpu.spec_decode.eagle.utils import (
    get_target_lm_head,
    maybe_share_target_embed,
    share_target_lm_head,
)

from .dfly import DFlyDeepseekV41DraftModel, DFlyQuantConfig
from .ops.dfly import gather_dfly_context

logger = init_logger(__name__)


class DFlyTargetContext:
    def __init__(self, target, draft, config):
        self.draft = draft
        self.target_metadata = None
        self.replay_layers = target.model.decoder_replay_layers
        self.sources = []
        self.indexers = []
        for kv_layer, index_layer in zip(
            draft.config.target_kv_layers, draft.config.target_index_layer
        ):
            source = target.model.layers[kv_layer].attn
            indexer = target.model.layers[index_layer].attn
            if not isinstance(source, DeepseekV4Attention) or not isinstance(
                indexer, DeepseekV4Attention
            ):
                raise ValueError("DFly2 requires native DeepSeek-V4.1 MLA")
            if indexer.indexer is None or indexer.topk_indices_buffer is None:
                raise ValueError("DFly2 index sources must run a target indexer")
            if source.compress_ratio != 1:
                raise ValueError("DFly2 requires ratio-1 target caches")
            if indexer.topk_snapshot is None:
                indexer.topk_snapshot = torch.full_like(indexer.topk_indices_buffer, -1)
            self.sources.append(source)
            self.indexers.append(indexer)
        context = draft.model.layers[0].attn.context
        self.block_indices = torch.arange(
            context.kv.shape[0] * context.kv.shape[1],
            device=context.kv.device,
            dtype=torch.int32,
        ).view(context.kv.shape[:2])[:, context.context_width :]

    def validate_groups(self, config):
        names = {name for group in config.kv_cache_groups for name in group.layer_names}
        for source in self.sources:
            if (
                not {source.compressed_cache_prefix, source.swa_cache_layer.prefix}
                <= names
            ):
                raise ValueError(
                    "DFly2 target KV layers must have allocated cache groups"
                )

    def validate_storage(self):
        for source in self.sources:
            for cache in (
                source._compressed_kv_cache(),
                source.swa_cache_layer.kv_cache,
            ):
                if cache.dtype != torch.uint8 or cache.shape[-1] not in (528, 584):
                    raise ValueError("DFly2 requires FP8 DeepSeek-V4/V4.1 target KV")

    def update_indices(self, active, source_rows, anchors, padded_batch):
        metadata = self.target_metadata
        replay = self.replay_layers
        replay_rows = (
            replay.replay_batch.rows
            if replay is not None and replay.replay_batch is not None
            else None
        )
        for layer, source, indexer in zip(
            self.draft.model.layers, self.sources, self.indexers
        ):
            context = layer.attn.context
            kv = context.kv[:padded_batch]
            indices = context.indices[:padded_batch]
            indices.fill_(-1)
            if metadata is None:
                kv.zero_()
                continue
            compressed = metadata[source.compressed_cache_prefix]
            swa = metadata[source.swa_cache_layer.prefix]
            # Graph padding has no corresponding target request/block-table row.
            batch = min(padded_batch, compressed.block_table.shape[0])
            topk = self.draft.config.index_topk
            for cache, table, block_size, offset, width, window in (
                (
                    source._compressed_kv_cache(),
                    compressed.block_table,
                    compressed.block_size,
                    0,
                    topk,
                    False,
                ),
                (
                    source.swa_cache_layer.kv_cache,
                    swa.block_table,
                    swa.block_size,
                    topk,
                    self.draft.config.sliding_window,
                    True,
                ),
            ):
                gather_dfly_context(
                    cache,
                    table,
                    block_size,
                    indexer.topk_snapshot,
                    source_rows,
                    anchors,
                    active,
                    kv[:batch],
                    indices[:batch],
                    offset=offset,
                    width=width,
                    window=window,
                    replay_rows=replay_rows,
                )
            if batch < padded_batch:
                kv[batch:].zero_()
            start = context.context_width
            indices[:, start : start + context.query_width] = torch.where(
                active[:padded_batch, None], self.block_indices[:padded_batch], -1
            )


class DFlyQueryModel(nn.Module):
    def __init__(self, model, draft_tokens, sample_fn):
        super().__init__()
        self.model = model
        self.sample_fn = sample_fn
        self.register_buffer("draft_tokens", draft_tokens, persistent=False)
        self.query_width = model.model.query_width

    def forward(self, input_ids, positions, target_hidden, active):
        batch = active.shape[0]
        width = self.query_width
        embeddings = self.model.embed_input_ids(input_ids)
        embeddings[::width] = self.model.model.mtp_fusion(
            embeddings[::width], target_hidden
        )
        hidden = self.model(input_ids, positions, embeddings).view(batch, width, -1)
        previous = input_ids.view(batch, width)[:, 0].long()
        position = positions.view(batch, width)
        correction = self.model.model.hidden_correction
        block = hidden.new_empty((batch, width, correction.attention.head_dim))
        # Slot zero is part of the correction history even though it is not sampled.
        combined = correction.combine(
            hidden[:, 0], self.model.embed_input_ids(previous)
        )
        block[:, 0] = correction.attention.block_kv(combined, position[:, 0])
        for slot in range(1, width):
            corrected = correction(
                hidden[:, slot],
                self.model.embed_input_ids(previous),
                position[:, slot],
                block,
                slot,
            )
            previous = self.sample_fn(slot - 1, corrected)
            self.draft_tokens[:batch, slot - 1] = torch.where(active, previous, -1)
        return self.draft_tokens[:batch]


class DeepseekV41DFlySpeculator(HYV4DFlySpeculator):
    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)
        config = self._draft_vllm_config
        config.parallel_config.is_moe_model = True
        config.parallel_config.enable_eplb = False
        config.parallel_config.eplb_config = copy(config.parallel_config.eplb_config)
        config.parallel_config.eplb_config.num_redundant_experts = 0
        config.kernel_config = copy(config.kernel_config)
        if config.kernel_config.moe_backend in MEGA_MOE_BACKENDS:
            config.kernel_config.moe_backend = "auto"

    def load_draft_model(self, target_model, target_attn_layer_names):
        config = self._draft_vllm_config
        hf_quant = getattr(
            self.draft_model_config.hf_config, "quantization_config", None
        )
        config.quant_config = (
            DFlyQuantConfig.from_config(hf_quant) if hf_quant else None
        )
        draft = get_model(
            vllm_config=config,
            model_config=self.draft_model_config,
            load_config=get_draft_load_config(config),
            prefix="dfly",
        )
        if not isinstance(draft, DFlyDeepseekV41DraftModel):
            raise TypeError("DFly2 requires DFlyDeepseekV41DraftModel")
        target = (
            target_model.get_language_model()
            if hasattr(target_model, "get_language_model")
            else target_model
        )
        maybe_share_target_embed(draft, draft.model, target.model)
        head = share_target_lm_head(
            draft, get_target_lm_head(target_model, target), quantization=None
        )
        if head is None or draft.model.embed_tokens is None:
            raise ValueError("DFly2 requires the target embedding and LM head")
        if (
            head.weight.dtype != torch.bfloat16
            or draft.model.embed_tokens.weight.dtype != torch.bfloat16
        ):
            raise ValueError("DFly2 shared embedding and LM head must be BF16")
        draft.logits_processor.head_dtype = self.vllm_config.model_config.head_dtype
        self.reuse = DFlyTargetContext(target, draft, config)
        self.query_model = DFlyQueryModel(draft, self.draft_tokens, self._sample_step)
        logger.info(
            "DeepSeek-V4.1 DFly2: %d layers, %d draft tokens, target KV/index %s/%s",
            len(draft.model.layers),
            self.num_speculative_steps,
            draft.config.target_kv_layers,
            draft.config.target_index_layer,
        )
        return draft

    def init_cudagraph_manager(self, cudagraph_mode):
        width = self.num_query_per_req
        mode = (
            CUDAGraphMode.NONE
            if cudagraph_mode == CUDAGraphMode.NONE
            else CUDAGraphMode.FULL_DECODE_ONLY
        )
        self.cudagraph_manager = CudaGraphManager(
            self._draft_vllm_config,
            self.device,
            mode,
            decode_query_lens={
                width: min(self.max_num_reqs, self.max_num_tokens // width)
            },
        )

    def _query_attention_state(self, desc, seq_lens_cpu_upper_bound):
        return None, {}

    @torch.inference_mode()
    def propose(self, input_batch, attn_metadata, *args, **kwargs):
        assert isinstance(self.reuse, DFlyTargetContext)
        self.reuse.target_metadata = (
            None if kwargs.get("skip_attn_for_dummy_run") else attn_metadata
        )
        return super().propose(input_batch, attn_metadata, *args, **kwargs)
