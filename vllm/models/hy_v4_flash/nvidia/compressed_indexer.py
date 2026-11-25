# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Flash coarse-only gated IndexPool with bounded paged FP8 scoring."""

import torch
from torch import nn

from vllm import envs
from vllm.config import CUDAGraphMode
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.indexer_topk import get_indexer_topk
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    ReplicatedLinear,
)
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import (
    fp8_fp4_paged_mqa_logits,
    get_paged_mqa_logits_metadata,
    has_deep_gemm,
    mqa_logits_mx_inputs,
    mqa_logits_use_mx_scales,
)
from vllm.utils.math_utils import cdiv
from vllm.utils.platform_utils import num_compute_units

from ..common.indexpool_cache import POOL_PAGE_SIZE, IndexPoolCache, IndexPoolMetadata
from .indexpool_ops import pool_quantize_write, quantize_query, select_expand_groups
from .indexpool_prepare import prepare_groups, write_tails


class FlashCompressedIndexer(nn.Module):
    def __init__(self, vllm_config, config, topk_indices_buffer, prefix):
        super().__init__()
        if not has_deep_gemm():
            raise RuntimeError("Flash IndexPool requires DeepGEMM paged MQA")
        self.ratio = config.compressed_indexer_compress_ratio
        self.group_budget = config.compressed_indexer_group_topk - 1
        self.topk = get_indexer_topk(
            vllm_config.kernel_config.sparse_indexer_topk_backend
        )
        self.num_heads = config.compressed_indexer_num_attention_heads
        if not 1 <= self.num_heads <= 64:
            raise ValueError("IndexPool requires 1..64 scoring heads")
        if config.qk_rope_head_dim != 64:
            raise ValueError("IndexPool kernels require a 64-dim rotary tail")
        self.rope_parameters = config.rope_parameters
        self.eps = config.rms_norm_eps
        self.topk_indices_buffer = topk_indices_buffer
        self.max_model_len = vllm_config.model_config.max_model_len
        self.logits_budget = envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB * 1024 * 1024 // 4
        for name, input_size, output_size in (
            ("query_projection", config.q_lora_rank, self.num_heads * 128),
            ("head_weight_projection", config.hidden_size, self.num_heads),
            ("token_key_projection", config.hidden_size, 128),
        ):
            setattr(
                self,
                name,
                ReplicatedLinear(
                    input_size,
                    output_size,
                    bias=False,
                    quant_config=vllm_config.quant_config,
                    prefix=f"{prefix}.{name}",
                ),
            )
        self.token_key_norm = RMSNorm(128, eps=self.eps)
        # Value then gate projection, replicated, so one GEMM produces both.
        self.key_projection = MergedColumnParallelLinear(
            128,
            [128, 128],
            bias=False,
            params_dtype=torch.float32,
            prefix=f"{prefix}.key_projection",
            return_bias=False,
            disable_tp=True,
        )
        self.key_norm = nn.RMSNorm(128, eps=self.eps, dtype=torch.float32)
        self.intra_group_ape = nn.Parameter(
            torch.empty(self.ratio, 128, dtype=torch.float32)
        )
        self.k_cache = IndexPoolCache(
            f"{prefix}.k_cache", vllm_config.cache_config, self.ratio
        )
        self.tail_cache = IndexPoolCache(
            f"{prefix}.tail_cache",
            vllm_config.cache_config,
            self.ratio,
            tail=True,
        )
        self.num_sms = num_compute_units()
        self.varlen = current_platform.is_device_capability_family(100)

    def forward(self, hidden_states, q_c, positions, rope=None):
        output = self.topk_indices_buffer[: hidden_states.shape[0]]
        metadata = get_forward_context().attn_metadata
        if metadata is None:
            return output.fill_(-1)
        if not isinstance(metadata, dict):
            raise TypeError("IndexPool requires per-layer metadata")
        km = metadata[self.k_cache.prefix]
        tm = metadata[self.tail_cache.prefix]
        if not isinstance(km, IndexPoolMetadata) or not isinstance(
            tm, IndexPoolMetadata
        ):
            raise TypeError("IndexPool cache metadata has an incompatible type")
        tokens = km.num_actual_tokens
        if tokens == 0 or km.seq_lens.numel() == 0:
            return output.fill_(-1)
        # Scoring rows below `tokens` are fully written, padding included.
        if tokens < output.shape[0]:
            output[tokens:].fill_(-1)
        keys = self.token_key_norm(
            self.token_key_projection(hidden_states)[0]
        ).bfloat16()
        cache, tail = self.k_cache.kv_cache, self.tail_cache.kv_cache
        (
            raw,
            group_positions,
            group_owner,
            groups,
            group_live,
            owner,
            logical,
            upper,
            score_lengths,
            token_table,
            live,
        ) = prepare_groups(
            keys,
            positions,
            tail,
            tm.block_table,
            km.block_table,
            km.query_start_loc,
            km.seq_lens,
            tokens,
            self.ratio,
        )
        if raw.shape[0]:
            pool_quantize_write(
                self.key_projection(raw),
                self.intra_group_ape,
                self.key_norm.weight,
                self.eps,
                group_positions,
                self.rope_parameters,
                cache,
                km.block_table,
                group_owner,
                groups,
                group_live,
                mqa_logits_use_mx_scales(),
            )
        write_tails(
            keys, positions, tail, tm.block_table, km.query_start_loc, km.seq_lens
        )
        quant_q, weights = quantize_query(
            self.query_projection(q_c)[0][:tokens].view(tokens, self.num_heads, 128),
            self.head_weight_projection(hidden_states)[0][:tokens],
            positions[..., :tokens],
            self.rope_parameters,
            32 if self.num_heads <= 32 else 64,
        )
        quant_q = quant_q[:, None]
        mx_scales, weights = mqa_logits_mx_inputs(quant_q, None, weights, torch.float32)
        max_seq_len = (
            self.max_model_len
            if get_forward_context().cudagraph_runtime_mode == CUDAGraphMode.FULL
            else km.max_seq_len
        )
        width = max(
            POOL_PAGE_SIZE,
            cdiv(max_seq_len, self.ratio * POOL_PAGE_SIZE) * POOL_PAGE_SIZE,
        )
        if self.logits_budget < width:
            raise ValueError("IndexPool logits budget cannot hold a single scoring row")
        chunk = max(1, self.logits_budget // width)
        for begin in range(0, tokens, chunk):
            end = min(tokens, begin + chunk)
            bounds = upper[begin:end]
            lengths = score_lengths[begin:end]
            indices = owner[begin:end] if self.varlen else None
            schedule = get_paged_mqa_logits_metadata(
                lengths, POOL_PAGE_SIZE, self.num_sms, indices=indices
            )
            table = token_table[begin:end]
            logits = fp8_fp4_paged_mqa_logits(
                (
                    quant_q[begin:end],
                    mx_scales[begin:end] if mx_scales is not None else None,
                ),
                cache.unsqueeze(2),
                weights[begin:end],
                lengths,
                table,
                schedule,
                max_model_len=width,
                indices=indices,
            )
            select_expand_groups(
                logits,
                bounds,
                self.group_budget,
                logical[begin:end],
                live[begin:end],
                self.ratio,
                output[begin:end],
                self.topk,
            )
        return output
