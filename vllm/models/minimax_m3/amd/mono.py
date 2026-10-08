# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bind native M3 tensors and batch metadata to ATOM's opt-in mono library."""

from typing import cast

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


def _collective_stage(group, name, action):
    error, result = None, None
    try:
        result = action()
    except Exception as exc:
        error = f"rank {dist.get_rank(group)} {name}: {type(exc).__name__}: {exc}"
    errors: list[str | None] = [None] * dist.get_world_size(group)
    dist.all_gather_object(errors, error, group=group)
    if any(item is not None for item in errors):
        raise RuntimeError(
            "MiniMax-M3 ATOM mono initialization failed: "
            + "; ".join(item for item in errors if item is not None)
        )
    return result


class M3Mono:
    def __init__(self, model, config, kv_cache_config):
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

        def validate():
            parallel = config.parallel_config
            require(config.use_v2_model_runner, "V2 model runner is required")
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
                config.scheduler_config.max_num_seqs <= 4
                and config.model_config.max_model_len == 16384,
                "requires at most four sequences and max_model_len=16384",
            )
            cfg = model.config
            require(
                (cfg.hidden_size, cfg.num_attention_heads, cfg.num_key_value_heads)
                == (6144, 64, 4),
                "model dimensions",
            )
            require(
                cfg.swiglu_alpha == 1.702 and cfg.swiglu_beta == 1.0, "SwiGLU constants"
            )
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
            layer_ids = [
                i for i, flag in enumerate(sparse["sparse_attention_freq"]) if flag
            ]
            require(
                layer_ids == list(range(3, len(model.layers))),
                "expected contiguous sparse layers after three dense layers",
            )
            require(len(model.layers) == 60, "expected 60 decoder layers")
            self.layer_ids = list(range(3, 60))
            names_to_group = {
                name: i
                for i, group in enumerate(kv_cache_config.kv_cache_groups)
                for name in group.layer_names
            }
            main_groups = {
                names_to_group[model.layers[i].self_attn.layer_name] for i in layer_ids
            }
            index_groups = {
                names_to_group[model.layers[i].self_attn.indexer.index_cache.prefix]
                for i in layer_ids
            }
            require(
                len(main_groups) == len(index_groups) == 1,
                "sparse main layers and index layers must each share metadata",
            )

        _collective_stage(self.group, "configuration", validate)
        specs, caches = _collective_stage(
            self.group,
            "native model binding",
            lambda: layer_specs(model, self.layer_ids),
        )
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
                main.decode.decode_query_len not in (1, 4)
                or token_count % main.decode.decode_query_len
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


def prepare_model(model, config, kv_cache_config):
    from vllm.models.minimax_m3.amd.model import MiniMaxM3Model

    group = get_tp_group().cpu_group

    def target():
        language_model = (
            model.get_language_model()
            if hasattr(model, "get_language_model")
            else model
        )
        require(
            isinstance(language_model.model, MiniMaxM3Model), "unsupported model class"
        )
        require(
            language_model.model._mono is None,
            "cache reinitialization is unsupported while graphs reference "
            "mono resources",
        )
        return language_model.model

    target_model = _collective_stage(group, "target model", target)
    target_model._mono = M3Mono(target_model, config, kv_cache_config)


def release_model(model):
    """Release mono-owned IPC only after the runner has drained its graphs."""
    language_model = (
        model.get_language_model() if hasattr(model, "get_language_model") else model
    )
    target = getattr(language_model, "model", None)
    adapter = getattr(target, "_mono", None)
    if target is not None and adapter is not None:
        adapter.close()
        target._mono = None
