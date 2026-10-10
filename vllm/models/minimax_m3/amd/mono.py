# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bind native M3 tensors and batch metadata to ATOM's opt-in mono library."""

import weakref
from typing import cast

import torch
import torch.distributed as dist

from vllm.distributed import get_tp_group
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.models.minimax_m3.amd.indexer_aiter import MiniMaxM3IndexerAiterMetadata
from vllm.models.minimax_m3.amd.mono_weights import layer_specs, require
from vllm.models.minimax_m3.amd.sparse_attention_msa import (
    MiniMaxM3SparseAiterPADecodeMetadata,
)
from vllm.models.minimax_m3.common.sparse_attention import MiniMaxM3SparseMetadata

logger = init_logger(__name__)
SUPPORTED_TOKENS = (1, 4, 8, 16)


def _collective_stage(group, name, action, *, allow_fallback=False):
    error, result = None, None
    try:
        result = action()
    except Exception as exc:
        error = (
            isinstance(exc, ValueError),
            f"rank {dist.get_rank(group)} {name}: {type(exc).__name__}: {exc}",
        )
    errors: list[tuple[bool, str] | None] = [None] * dist.get_world_size(group)
    dist.all_gather_object(errors, error, group=group)
    if any(item is not None for item in errors):
        failures = [item for item in errors if item is not None]
        message = "; ".join(item[1] for item in failures)
        if allow_fallback and all(item[0] for item in failures):
            logger.info("ATOM mono unavailable; using native execution: %s", message)
            return None
        raise RuntimeError("MiniMax-M3 ATOM mono initialization failed: " + message)
    return result


def _model_layers(model, config):
    parallel = config.parallel_config
    require(config.use_v2_model_runner, "V2 model runner is required")
    require(config.weight_transfer_config is None, "weight transfer is unsupported")
    require(
        config.cache_config.num_gpu_blocks_override is not None,
        "set num_gpu_blocks_override to reserve space for lazy mono allocation",
    )
    require(
        (
            parallel.tensor_parallel_size,
            parallel.pipeline_parallel_size,
            parallel.data_parallel_size,
        )
        == (4, 1, 1),
        "requires TP4/PP1/DP1",
    )
    require(
        parallel.decode_context_parallel_size
        == parallel.prefill_context_parallel_size
        == 1
        and not parallel.enable_expert_parallel
        and not parallel.use_ubatching,
        "context/expert parallelism and microbatching are unsupported",
    )
    require(
        config.lora_config is None
        and config.kv_transfer_config is None
        and not config.model_config.enable_sleep_mode,
        "LoRA, KV transfer and sleep are unsupported",
    )
    require(
        config.cache_config.cache_dtype == "fp8"
        and config.cache_config.block_size == 128
        and not config.cache_config.enable_prefix_caching,
        "requires FP8, block size 128 and prefix caching disabled",
    )
    spec = config.speculative_config
    require(
        spec is not None
        and spec.method == "eagle3"
        and spec.num_speculative_tokens == 3,
        "requires native EAGLE3 with three speculative tokens",
    )
    require(
        config.model_config.max_model_len == 16384,
        "requires max_model_len=16384",
    )
    cfg = model.config
    require(
        (cfg.hidden_size, cfg.num_attention_heads, cfg.num_key_value_heads)
        == (6144, 64, 4),
        "model dimensions",
    )
    require(cfg.swiglu_alpha == 1.702 and cfg.swiglu_beta == 1.0, "SwiGLU constants")
    sparse = cfg.sparse_attention_config
    require(
        (
            sparse["sparse_topk_blocks"],
            sparse["sparse_block_size"],
            sparse.get("sparse_init_block", 0),
            sparse.get("sparse_local_block", 0),
        )
        == (16, 128, 0, 1),
        "sparse selection configuration",
    )
    layer_ids = [i for i, flag in enumerate(sparse["sparse_attention_freq"]) if flag]
    require(
        layer_ids == list(range(3, len(model.layers))),
        "expected contiguous sparse layers after three dense layers",
    )
    require(len(model.layers) == 60, "expected 60 decoder layers")

    require(
        all(
            getattr(getattr(model.layers[i], "self_attn", None), "indexer", None)
            is not None
            for i in layer_ids
        ),
        "expected sparse attention with indexers",
    )

    metadata = get_forward_context().attn_metadata
    require(isinstance(metadata, dict), "attention metadata is required")
    first = model.layers[layer_ids[0]].self_attn
    require(
        all(
            metadata[model.layers[i].self_attn.layer_name] is metadata[first.layer_name]
            and metadata[model.layers[i].self_attn.indexer.index_cache.prefix]
            is metadata[first.indexer.index_cache.prefix]
            for i in layer_ids
        ),
        "sparse main layers and index layers must each share metadata",
    )
    return layer_ids


class M3Mono:
    layer_ids: list[int]

    def __init__(self, model, layer_ids, specs, caches):
        self.group = get_tp_group().cpu_group
        self.fallback_counts = {}
        self.query_len = 4

        def dependencies():
            from atom.models.minimax_m3.mono.library import (
                AtomM3Mono,
                StepMetadata,
                TPContext,
            )

            return AtomM3Mono, StepMetadata, TPContext

        AtomM3Mono, self.StepMetadata, TPContext = _collective_stage(
            self.group, "ATOM dependencies", dependencies
        )

        self.layer_ids = layer_ids
        self.runtime = AtomM3Mono(
            specs,
            caches,
            TPContext(
                self.group,
                get_tp_group().rank_in_group,
                get_tp_group().world_size,
                next(model.parameters()).device,
            ),
        )
        self.attentions = [model.layers[i].self_attn for i in self.layer_ids]
        logger.info(
            "ATOM mono library ready: sparse layers=%s, buckets=%s",
            self.layer_ids,
            SUPPORTED_TOKENS,
        )

    def close(self):
        self.runtime.close()

    def begin_forward(self, token_count):
        context = get_forward_context()
        reason = None
        if token_count not in SUPPORTED_TOKENS:
            reason = "token_count"
        elif not isinstance(context.attn_metadata, dict):
            reason = "no_metadata"
        else:
            first = self.attentions[0]
            main = cast(
                MiniMaxM3SparseMetadata, context.attn_metadata[first.layer_name]
            )
            index = cast(
                MiniMaxM3IndexerAiterMetadata,
                context.attn_metadata[first.indexer.index_cache.prefix],
            )
            if (
                main.num_prefills
                or index.num_prefills
                or main.decode is None
                or index.decode is None
            ):
                reason = "prefill_or_mixed"
            elif (
                not 0 < main.num_decodes <= 4
                or main.num_decodes != index.num_decodes
                or main.decode.seq_lens.numel() > 4
                or index.decode.seq_lens.numel() > 4
            ):
                reason = "concurrency"
            elif (
                main.decode.decode_query_len not in (1, 4)
                or token_count % main.decode.decode_query_len
                or token_count // main.decode.decode_query_len > 4
                or index.decode.decode_query_len != main.decode.decode_query_len
            ):
                reason = "query_shape"
        if reason is not None:
            self.fallback_counts[reason] = self.fallback_counts.get(reason, 0) + 1
            return False
        assert main.decode is not None and main.page16_slot_mapping is not None
        self.query_len = main.decode.decode_query_len
        decode = cast(MiniMaxM3SparseAiterPADecodeMetadata, main.decode)
        assert decode.page16_block_table is not None
        assert index.decode is not None
        assert context.slot_mapping is not None
        self.runtime.prepare_step(
            self.StepMetadata(
                main_table=decode.page16_block_table,
                index_table=index.decode.block_table,
                seq_lens=main.decode.seq_lens,
                main_slots=main.page16_slot_mapping,
                index_slots=context.slot_mapping[first.indexer.index_cache.prefix],
                token_count=token_count,
                query_len=self.query_len,
            )
        )
        return True

    def forward_layer(self, index, hidden, residual, positions):
        return self.runtime.forward_layer(
            self.layer_ids[index], hidden, residual, positions
        )


def prepare_model(model, config):
    """Initialize on eager warmup, including temporary profiling caches."""
    if not isinstance(get_forward_context().attn_metadata, dict):
        return
    require(
        not torch.cuda.is_current_stream_capturing(),
        "mono must be initialized by an eager forward before graph capture",
    )
    group = get_tp_group().cpu_group
    layer_ids = _collective_stage(
        group,
        "model support",
        lambda: _model_layers(model, config),
        allow_fallback=True,
    )
    if layer_ids is None:
        model._mono_config = None
        return
    attentions = [model.layers[i].self_attn for i in layer_ids]
    ready = all(
        attn.kv_cache.numel() > 0 and attn.indexer.index_cache.kv_cache.numel() > 0
        for attn in attentions
    )
    readiness = [False] * dist.get_world_size(group)
    dist.all_gather_object(readiness, ready, group=group)
    if not all(readiness):
        return
    bindings = _collective_stage(
        group,
        "weight support",
        lambda: layer_specs(model, layer_ids),
        allow_fallback=True,
    )
    if bindings is None:
        model._mono_config = None
        return
    model._mono = M3Mono(model, layer_ids, *bindings)
    for attn in attentions:
        attn._mono_model_ref = weakref.ref(model)


def detach_model_cache(model, replacement):
    """All TP ranks detach together after graph teardown, before rebinding."""
    if model._mono is not None:
        require(
            replacement.numel() == 0,
            "detach caches after destroying graphs before binding new storage",
        )
        model._mono.close()
        model._mono = None
        logger.info("ATOM mono library closed for detached model cache")
