# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Flash coarse-only gated IndexPool with bounded paged FP8 scoring."""

import torch
import torch.nn.functional as F
from torch import nn

from vllm import envs
from vllm.config import CUDAGraphMode
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.linear import ReplicatedLinear
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

from .indexpool_cache import POOL_PAGE_SIZE, IndexPoolCache, IndexPoolMetadata
from .indexpool_ops import (
    expand_groups,
    fwht_quantize,
    gated_pool,
    rotate_indexer,
    select_groups,
    write_pool,
)
from .indexpool_prepare import prepare_groups, write_tails


class FlashCompressedIndexer(nn.Module):
    def __init__(self, vllm_config, config, topk_indices_buffer, prefix):
        super().__init__()
        if not has_deep_gemm():
            raise RuntimeError("Flash IndexPool requires DeepGEMM paged MQA")
        self.ratio = config.compressed_indexer_compress_ratio
        self.group_budget = config.compressed_indexer_group_topk - 1
        self.num_heads = config.compressed_indexer_num_attention_heads
        if not 1 <= self.num_heads <= 64:
            raise ValueError("IndexPool requires 1..64 scoring heads")
        self.rope_dim = config.qk_rope_head_dim
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
        self.token_key_norm = nn.RMSNorm(128, eps=self.eps)
        self.key_value_projection = nn.Linear(128, 128, bias=False, dtype=torch.float32)
        self.key_gate_projection = nn.Linear(128, 128, bias=False, dtype=torch.float32)
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
        output.fill_(-1)
        metadata = get_forward_context().attn_metadata
        if metadata is None:
            return output
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
            return output
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
            live,
        ) = prepare_groups(
            keys,
            positions,
            tail,
            tm.block_table,
            km.query_start_loc,
            km.seq_lens,
            tokens,
            self.ratio,
        )
        if raw.shape[0]:
            values = self.key_value_projection(raw.float())
            gates = self.key_gate_projection(raw.float())
            pooled = gated_pool(
                values, gates, self.intra_group_ape, self.key_norm.weight, self.eps
            ).bfloat16()
            quant, scales = fwht_quantize(
                rotate_indexer(
                    pooled, group_positions, self.rope_dim, self.rope_parameters
                )
            )
            if mqa_logits_use_mx_scales():
                scales = (scales.view(torch.int32) >> 23) * 0x01010101
            write_pool(
                cache, km.block_table, group_owner, groups, group_live, quant, scales
            )
        write_tails(
            keys, positions, tail, tm.block_table, km.query_start_loc, km.seq_lens
        )
        query = self.query_projection(q_c)[0][:tokens].reshape(
            tokens, self.num_heads, 128
        )
        query = rotate_indexer(
            query, positions[..., :tokens], self.rope_dim, self.rope_parameters
        )
        quant_q, scales_q = fwht_quantize(query.reshape(-1, 128).bfloat16())
        heads = 32 if self.num_heads <= 32 else 64
        quant_q = F.pad(
            quant_q.view(torch.uint8).reshape(tokens, self.num_heads, 128),
            (0, 0, 0, heads - self.num_heads),
        ).view(torch.float8_e4m3fn)
        weights = self.head_weight_projection(hidden_states)[0][:tokens].float()
        weights = weights * (self.num_heads**-0.5 * 128**-0.5)
        weights = F.pad(
            weights * scales_q.reshape(tokens, self.num_heads),
            (0, heads - self.num_heads),
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
            lengths = bounds.clamp_min(1).int()[:, None].contiguous()
            indices = owner[begin:end].int().contiguous() if self.varlen else None
            schedule = get_paged_mqa_logits_metadata(
                lengths, POOL_PAGE_SIZE, self.num_sms, indices=indices
            )
            table = km.block_table[owner[begin:end]]
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
            recalled = select_groups(logits, bounds, self.group_budget)
            output[begin:end].copy_(
                expand_groups(
                    recalled,
                    logical[begin:end],
                    self.ratio,
                    active=live[begin:end],
                )
            )
        return output
