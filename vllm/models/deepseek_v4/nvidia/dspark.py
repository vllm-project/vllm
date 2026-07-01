# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4 DSpark draft model.

DSpark is not a regular next-n MTP head.  The checkpoint stores three draft
decoder blocks under ``mtp.0``-``mtp.2``.  Each block attends over a small
draft block plus a circular window of target-model hidden-state KVs.
"""

import typing
from collections.abc import Callable, Iterable

import regex as re
import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.logger import init_logger
from vllm.model_executor.kernels.mhc.tilelang import (
    hc_head_fused_kernel_tilelang,
    mhc_fused_post_pre_tilelang,
    mhc_post_tilelang,
    mhc_pre_tilelang,
)
from vllm.model_executor.layers.fused_moe import (
    fused_moe_make_expert_params_mapping,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.utils import get_draft_quant_config
from vllm.models.deepseek_v4.common.ops import fused_q_kv_rmsnorm
from vllm.models.deepseek_v4.common.ops.fused_inv_rope_fp8_quant import (
    fused_inv_rope_fp8_quant,
)
from vllm.models.deepseek_v4.nvidia.ops.fp8_einsum import (
    deepseek_v4_fp8_einsum,
    deepseek_v4_sm12x_fp8_einsum_quant,
)
from vllm.v1.attention.backends.mla.sparse_mla_env import (
    triton_sparse_mla_head_block_size,
)
from vllm.v1.attention.backends.mla.sparse_mla_kernels import (
    matmul_sparse_mla_attention_with_sink,
)

from .model import (
    DeepseekV4MoE,
    _select_dsv4_attn_cls,
    make_deepseek_v4_expert_params_mapping,
)
from .dspark_triton import (
    dspark_context_kv_store,
    dspark_qkv_postprocess,
    dspark_triton_attention,
)

logger = init_logger(__name__)

_EXPERT_SCALE_RE = re.compile(r"\.experts\.\d+\.w[123]\.scale$")


def _spec_bool(vllm_config: VllmConfig, attr: str, default: bool = True) -> bool:
    spec_config = vllm_config.speculative_config
    return bool(getattr(spec_config, attr, default))


def _linear_output(output: torch.Tensor | tuple[torch.Tensor, object]) -> torch.Tensor:
    if isinstance(output, tuple):
        return output[0]
    return output


def _rmsnorm_no_weight(x: torch.Tensor, eps: float) -> torch.Tensor:
    x_float = x.float()
    return x_float.mul(torch.rsqrt(x_float.square().mean(-1, keepdim=True) + eps))


def _apply_rope_gptj_last(
    x: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
) -> torch.Tensor:
    """Apply DS4 RoPE to the last rope_dim features.

    This mirrors the DS4 fused qnorm/kv-insert kernels' PyTorch reference:
    GPT-J/interleaved pairs, cos||sin cache layout, and RoPE on the tail of the
    head dimension.
    """
    rope_dim = cos_sin_cache.shape[-1]
    half = rope_dim // 2
    head_dim = x.shape[-1]
    nope_dim = head_dim - rope_dim

    cs = cos_sin_cache.index_select(0, positions.long()).to(torch.float32)
    cos = cs[..., :half]
    sin = cs[..., half:]

    rope = x[..., nope_dim:].float()
    shape = rope.shape
    rope = rope.reshape(*shape[:-1], half, 2)
    even = rope[..., 0]
    odd = rope[..., 1]

    while cos.dim() < even.dim():
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

    new_even = torch.addcmul(-odd * sin, even, cos)
    new_odd = torch.addcmul(odd * cos, even, sin)
    rope_rotated = torch.stack((new_even, new_odd), dim=-1).reshape(shape)

    out = x.clone()
    out[..., nope_dim:] = rope_rotated.to(out.dtype)
    return out


class DeepSeekV4DSparkLayer(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        *,
        dspark_layer_idx: int,
        prefix: str,
    ) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config
        self.hidden_size = config.hidden_size
        self.rms_norm_eps = config.rms_norm_eps
        self.hc_mult = config.hc_mult
        self.hc_sinkhorn_iters = config.hc_sinkhorn_iters
        self.hc_eps = config.hc_eps
        self.hc_post_alpha = 2.0
        self.block_size = int(getattr(config, "dspark_block_size", 0))
        runtime_layer_idx = config.num_hidden_layers + dspark_layer_idx
        runtime_prefix = (
            f"{prefix}.layers.{runtime_layer_idx}"
            if prefix
            else f"layers.{runtime_layer_idx}"
        )

        self.attn = _select_dsv4_attn_cls(vllm_config)(
            vllm_config,
            prefix=f"{runtime_prefix}.attn",
            topk_indices_buffer=None,
            aux_stream_list=None,
        )
        self.ffn = DeepseekV4MoE(vllm_config, prefix=f"{runtime_prefix}.ffn")
        self.attn_norm = RMSNorm(self.hidden_size, self.rms_norm_eps)
        self.ffn_norm = RMSNorm(self.hidden_size, self.rms_norm_eps)

        mix_hc = (2 + self.hc_mult) * self.hc_mult
        hc_dim = self.hc_mult * self.hidden_size
        self.hc_attn_fn = nn.Parameter(
            torch.empty((mix_hc, hc_dim), dtype=torch.float32),
            requires_grad=False,
        )
        self.hc_ffn_fn = nn.Parameter(
            torch.empty((mix_hc, hc_dim), dtype=torch.float32),
            requires_grad=False,
        )
        self.hc_attn_base = nn.Parameter(
            torch.empty(mix_hc, dtype=torch.float32),
            requires_grad=False,
        )
        self.hc_ffn_base = nn.Parameter(
            torch.empty(mix_hc, dtype=torch.float32),
            requires_grad=False,
        )
        self.hc_attn_scale = nn.Parameter(
            torch.empty(3, dtype=torch.float32),
            requires_grad=False,
        )
        self.hc_ffn_scale = nn.Parameter(
            torch.empty(3, dtype=torch.float32),
            requires_grad=False,
        )

        max_batch = max(1, int(vllm_config.scheduler_config.max_num_seqs))
        window = int(config.sliding_window)
        self.register_buffer(
            "_main_kv_cache",
            torch.zeros(
                max_batch,
                window,
                config.head_dim,
                dtype=vllm_config.model_config.dtype,
            ),
            persistent=False,
        )
        self.use_materialized_attention = _spec_bool(
            vllm_config,
            "dspark_materialized_attention",
        )
        self.use_triton_attention = _spec_bool(
            vllm_config,
            "dspark_triton_attention",
        )
        self.use_triton_qkv_postprocess = _spec_bool(
            vllm_config,
            "dspark_triton_qkv_postprocess",
        )
        self.use_triton_context_kv_store = _spec_bool(
            vllm_config,
            "dspark_triton_context_kv_store",
        )
        self.use_fused_o_proj_quant = _spec_bool(
            vllm_config,
            "dspark_fused_o_proj_quant",
        )
        self.materialized_head_block_size = triton_sparse_mla_head_block_size() or 1
        self.register_buffer(
            "_score_buffer",
            torch.empty(
                max_batch * self.block_size,
                self.attn.n_local_heads,
                window + self.block_size,
                dtype=torch.float32,
            ),
            persistent=False,
        )

    @property
    def window_size(self) -> int:
        return int(self.attn.window_size)

    def _wo_b_output(self, wo_b_input: torch.Tensor) -> torch.Tensor:
        return _linear_output(self.attn.wo_b(wo_b_input))

    def _o_proj_fused_quant(
        self, attn, o: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor | None:
        """Fused inv-RoPE+quant -> FP8-epilogue einsum -> wo_b GEMM.

        Has the einsum kernel emit FP8 z plus its matching per-token/128-block
        scale directly, instead of BF16 z that wo_b's normal forward would
        then re-quantize in a separate kernel launch. Returns None (caller
        falls back to the standard path) if wo_b's underlying block-scaled MM
        kernel can't be found in the expected shape, or if the wo_a weight
        needs the TP weight-group narrowing this path hasn't validated.
        """
        quant_method = getattr(attn.wo_b, "quant_method", None)
        kernel = getattr(quant_method, "w8a8_block_fp8_linear", None) or getattr(
            quant_method, "fp8_linear", None
        )
        if kernel is None or not hasattr(kernel, "apply_block_scaled_mm"):
            return None
        wo_b_weight = getattr(attn.wo_b, "weight", None)
        wo_b_scale = getattr(attn.wo_b, "weight_scale_inv", None)
        if wo_b_scale is None:
            wo_b_scale = getattr(attn.wo_b, "weight_scale", None)
        if wo_b_weight is None or wo_b_scale is None:
            return None

        n_groups = attn.n_local_groups
        o_lora_rank = attn.o_lora_rank
        hidden_size = attn.wo_a.weight.shape[-1]
        if attn.wo_a.weight.shape[0] // o_lora_rank != n_groups:
            return None
        wo_a_weight_3d = attn.wo_a.weight.view(n_groups, o_lora_rank, hidden_size)
        wo_a_scale = getattr(attn.wo_a, "weight_scale_inv", None)
        if wo_a_scale is None:
            wo_a_scale = attn.wo_a.weight_scale
        if wo_a_scale.dim() == 2:
            scale_out_blocks = (o_lora_rank + 127) // 128
            scale_hidden_blocks = (hidden_size + 127) // 128
            if wo_a_scale.shape[0] // scale_out_blocks != n_groups:
                return None
            wo_a_scale = wo_a_scale.view(n_groups, scale_out_blocks, scale_hidden_blocks)
        elif wo_a_scale.dim() != 3 or wo_a_scale.shape[0] != n_groups:
            return None

        o_fp8, o_scale = fused_inv_rope_fp8_quant(
            o,
            positions,
            attn.rotary_emb.cos_sin_cache,
            n_groups=n_groups,
            heads_per_group=attn.n_local_heads // n_groups,
            nope_dim=attn.nope_head_dim,
            rope_dim=attn.rope_head_dim,
            tma_aligned_scales=attn._tma_aligned_scales,
        )
        num_tokens = o.shape[0]
        z_fp8 = torch.empty(
            (num_tokens, n_groups * o_lora_rank),
            device=o.device,
            dtype=torch.float8_e4m3fn,
        )
        z_scale = torch.empty(
            (num_tokens, n_groups * o_lora_rank // 128),
            device=o.device,
            dtype=torch.float32,
        )
        deepseek_v4_sm12x_fp8_einsum_quant(
            o_fp8,
            o_scale,
            wo_a_weight_3d,
            wo_a_scale,
            z_fp8,
            z_scale,
        )
        output = kernel.apply_block_scaled_mm(
            A=z_fp8,
            B=wo_b_weight,
            As=z_scale,
            Bs=wo_b_scale,
        )
        if get_tensor_model_parallel_world_size() > 1:
            output = tensor_model_parallel_all_reduce(output)
        logger.info_once(
            "DSpark fused o_proj quant engaged: z_fp8 shape=%s scale shape=%s",
            tuple(z_fp8.shape),
            tuple(z_scale.shape),
        )
        return output

    def _project_main_kv(
        self,
        main_x: torch.Tensor,
        main_positions: torch.Tensor,
    ) -> torch.Tensor:
        kv = self._project_main_kv_raw(main_x)
        kv = self.attn.kv_norm(kv)
        return _apply_rope_gptj_last(
            kv, main_positions, self.attn.rotary_emb.cos_sin_cache
        )

    def _project_main_kv_raw(self, main_x: torch.Tensor) -> torch.Tensor:
        qr_kv = _linear_output(self.attn.fused_wqa_wkv(main_x))
        _, kv = qr_kv.split([self.attn.q_lora_rank, self.attn.head_dim], dim=-1)
        return kv

    def store_main_kv(
        self,
        main_x: torch.Tensor,
        main_positions: torch.Tensor,
        query_start_loc: torch.Tensor,
        batch_size: int,
        num_rejected_tokens: torch.Tensor | None = None,
    ) -> None:
        if main_x.numel() == 0:
            return
        if self.use_triton_context_kv_store:
            main_kv_raw = self._project_main_kv_raw(main_x)
            dspark_context_kv_store(
                main_kv_raw,
                self._main_kv_cache[:batch_size],
                main_positions,
                query_start_loc,
                batch_size,
                num_rejected_tokens,
                self.attn.kv_norm.weight.data,
                self.attn.rotary_emb.cos_sin_cache,
                self.attn.eps,
            )
            return

        main_kv = self._project_main_kv(main_x, main_positions)
        starts = query_start_loc[:-1].long()
        ends = query_start_loc[1:].long()
        if num_rejected_tokens is not None:
            ends = ends - num_rejected_tokens[:batch_size].long()
        window = self.window_size
        for req_idx in range(batch_size):
            start = int(starts[req_idx].item())
            end = int(ends[req_idx].item())
            if end <= start:
                continue
            req_positions = main_positions[start:end].long()
            slots = torch.remainder(req_positions, window)
            self._main_kv_cache[req_idx, slots] = main_kv[start:end].to(
                self._main_kv_cache.dtype
            )

    def _project_draft_q_kv(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        qr_kv = _linear_output(self.attn.fused_wqa_wkv(x))
        qr, kv = qr_kv.split([self.attn.q_lora_rank, self.attn.head_dim], dim=-1)
        qr, kv = fused_q_kv_rmsnorm(
            qr,
            kv,
            self.attn.q_norm.weight.data,
            self.attn.kv_norm.weight.data,
            self.attn.eps,
        )
        q = _linear_output(self.attn.wq_b(qr)).view(
            -1, self.attn.n_local_heads, self.attn.head_dim
        )
        if self.use_triton_qkv_postprocess:
            q, kv = dspark_qkv_postprocess(
                q.contiguous(),
                kv.contiguous(),
                positions,
                self.attn.rotary_emb.cos_sin_cache,
                self.attn.eps,
            )
        else:
            q = _rmsnorm_no_weight(q, self.attn.eps).to(x.dtype)
            q = _apply_rope_gptj_last(
                q, positions, self.attn.rotary_emb.cos_sin_cache
            )
            kv = _apply_rope_gptj_last(
                kv, positions, self.attn.rotary_emb.cos_sin_cache
            )
        return q, kv

    def _dspark_attention(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        main_x: torch.Tensor,
        main_positions: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        del main_x
        block_size = self.block_size
        q, draft_kv = self._project_draft_q_kv(x, positions)
        q = q.view(batch_size, block_size, self.attn.n_local_heads, self.attn.head_dim)
        draft_kv = draft_kv.view(batch_size, block_size, self.attn.head_dim)

        cache_kv = self._main_kv_cache[:batch_size].to(draft_kv.dtype)
        if self.use_triton_attention:
            o = dspark_triton_attention(
                q,
                cache_kv,
                draft_kv,
                main_positions.long(),
                self.attn.attn_sink[: self.attn.n_local_heads],
                float(self.attn.scale),
            ).reshape(
                batch_size * block_size,
                self.attn.n_local_heads,
                self.attn.head_dim,
            )
        else:
            kv = torch.cat([cache_kv, draft_kv], dim=1)

            cache_arange = torch.arange(
                self.window_size,
                device=positions.device,
                dtype=main_positions.dtype,
            )
            valid_cache = cache_arange.unsqueeze(0) <= torch.minimum(
                main_positions.long().unsqueeze(1),
                torch.full_like(
                    main_positions.long().unsqueeze(1), self.window_size - 1
                ),
            )
            valid = torch.cat(
                [
                    valid_cache,
                    torch.ones(
                        batch_size,
                        block_size,
                        dtype=torch.bool,
                        device=positions.device,
                    ),
                ],
                dim=1,
            )
            if self.use_materialized_attention:
                q_tokens = q.reshape(
                    batch_size * block_size,
                    self.attn.n_local_heads,
                    self.attn.head_dim,
                )
                kv_tokens = (
                    kv.unsqueeze(1)
                    .expand(batch_size, block_size, kv.shape[1], kv.shape[2])
                    .reshape(batch_size * block_size, kv.shape[1], kv.shape[2])
                    .contiguous()
                )
                valid_tokens = (
                    valid.unsqueeze(1)
                    .expand(batch_size, block_size, valid.shape[1])
                    .reshape(batch_size * block_size, valid.shape[1])
                    .contiguous()
                )
                o = torch.empty_like(q_tokens)
                score_buffer = self._score_buffer[
                    : q_tokens.shape[0],
                    : self.attn.n_local_heads,
                    : kv.shape[1],
                ]
                matmul_sparse_mla_attention_with_sink(
                    q_tokens,
                    kv_tokens,
                    valid_tokens,
                    float(self.attn.scale),
                    self.attn.attn_sink,
                    o,
                    num_heads=self.attn.n_local_heads,
                    score_buffer=score_buffer,
                    head_block_size=self.materialized_head_block_size,
                )
            else:
                scores = torch.einsum("bqhd,bnd->bqhn", q.float(), kv.float())
                scores *= float(self.attn.scale)
                scores = scores.masked_fill(
                    ~valid[:, None, None, :], float("-inf")
                )

                sink = self.attn.attn_sink[: self.attn.n_local_heads].view(
                    1, 1, -1, 1
                )
                max_score = torch.maximum(
                    scores.max(dim=-1, keepdim=True).values, sink
                )
                exp_scores = torch.exp(scores - max_score).masked_fill(
                    ~valid[:, None, None, :], 0.0
                )
                denom = exp_scores.sum(dim=-1, keepdim=True) + torch.exp(
                    sink - max_score
                )
                o = torch.einsum(
                    "bqhn,bnd->bqhd", exp_scores / denom, kv.float()
                )
                o = o.to(x.dtype).reshape(
                    batch_size * block_size,
                    self.attn.n_local_heads,
                    self.attn.head_dim,
                )
        return self._o_proj(o, positions)

    def _o_proj(self, o: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        attn = self.attn
        if not self.use_fused_o_proj_quant:
            return _linear_output(attn._o_proj(o, positions))

        required_attrs = (
            "_einsum_recipe",
            "_tma_aligned_scales",
            "wo_a",
            "wo_b",
            "rotary_emb",
            "n_local_groups",
            "n_local_heads",
            "nope_head_dim",
            "rope_head_dim",
            "o_lora_rank",
        )
        if not all(hasattr(attn, attr) for attr in required_attrs):
            return _linear_output(attn._o_proj(o, positions))

        if (
            self.use_fused_o_proj_quant
            and tuple(attn._einsum_recipe) == (1, 128, 128)
            and not attn._tma_aligned_scales
        ):
            fused_output = self._o_proj_fused_quant(attn, o, positions)
            if fused_output is not None:
                return fused_output

        o_fp8, o_scale = fused_inv_rope_fp8_quant(
            o,
            positions,
            attn.rotary_emb.cos_sin_cache,
            n_groups=attn.n_local_groups,
            heads_per_group=attn.n_local_heads // attn.n_local_groups,
            nope_dim=attn.nope_head_dim,
            rope_dim=attn.rope_head_dim,
            tma_aligned_scales=attn._tma_aligned_scales,
        )
        z = torch.empty(
            (o.shape[0], attn.n_local_groups, attn.o_lora_rank),
            device=o.device,
            dtype=torch.bfloat16,
        )
        wo_a_scale = getattr(attn.wo_a, "weight_scale_inv", None)
        if wo_a_scale is None:
            wo_a_scale = attn.wo_a.weight_scale
        deepseek_v4_fp8_einsum(
            o_fp8,
            o_scale,
            attn.wo_a.weight,
            wo_a_scale,
            z,
            "bhr,hdr->bhd",
            list(attn._einsum_recipe),
        )
        wo_b_input = z.flatten(1)
        output = self._wo_b_output(wo_b_input)
        return output

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        input_ids: torch.Tensor,
        main_x: torch.Tensor,
        main_positions: torch.Tensor,
        batch_size: int,
        post_mix: torch.Tensor | None = None,
        res_mix: torch.Tensor | None = None,
        residual: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        attn_norm_weight = self.attn_norm.weight.data
        attn_norm_eps = self.attn_norm.variance_epsilon
        if residual is None:
            residual = x
            post_mix, res_mix, x = mhc_pre_tilelang(
                x,
                self.hc_attn_fn,
                self.hc_attn_scale,
                self.hc_attn_base,
                self.rms_norm_eps,
                self.hc_eps,
                self.hc_eps,
                self.hc_post_alpha,
                self.hc_sinkhorn_iters,
                norm_weight=attn_norm_weight,
                norm_eps=attn_norm_eps,
            )
        else:
            residual, post_mix, res_mix, x = mhc_fused_post_pre_tilelang(
                x,
                residual,
                post_mix,
                res_mix,
                self.hc_attn_fn,
                self.hc_attn_scale,
                self.hc_attn_base,
                self.rms_norm_eps,
                self.hc_eps,
                self.hc_eps,
                self.hc_post_alpha,
                self.hc_sinkhorn_iters,
                n_splits=1,
                tile_n=1,
                norm_weight=attn_norm_weight,
                norm_eps=attn_norm_eps,
            )

        x = self._dspark_attention(x, positions, main_x, main_positions, batch_size)

        ffn_norm_weight = self.ffn_norm.weight.data
        ffn_norm_eps = self.ffn_norm.variance_epsilon
        residual, post_mix, res_mix, x = mhc_fused_post_pre_tilelang(
            x,
            residual,
            post_mix,
            res_mix,
            self.hc_ffn_fn,
            self.hc_ffn_scale,
            self.hc_ffn_base,
            self.rms_norm_eps,
            self.hc_eps,
            self.hc_eps,
            self.hc_post_alpha,
            self.hc_sinkhorn_iters,
            n_splits=1,
            tile_n=1,
            norm_weight=ffn_norm_weight,
            norm_eps=ffn_norm_eps,
        )
        x = self.ffn(x, input_ids)
        return x, residual, post_mix, res_mix

    def finalize_mega_moe_weights(self) -> None:
        self.ffn.finalize_mega_moe_weights()


class DeepSeekV4DSpark(nn.Module):
    """DSpark draft model for fixed-block speculative decoding."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        assert vllm_config.speculative_config is not None
        config = vllm_config.speculative_config.draft_model_config.hf_config
        self.config = config
        self.block_size = int(getattr(config, "dspark_block_size", 0))
        self.target_layer_ids = tuple(getattr(config, "dspark_target_layer_ids", ()))
        if self.block_size <= 0:
            raise ValueError("DSpark requires dspark_block_size > 0")
        if not self.target_layer_ids:
            raise ValueError("DSpark requires dspark_target_layer_ids")

        self.hc_mult = config.hc_mult
        self.hc_eps = config.hc_eps
        self.rms_norm_eps = config.rms_norm_eps
        self.num_dspark_layers = len(self.target_layer_ids)
        self.runtime_num_dspark_layers = self.num_dspark_layers
        self.max_batch = max(1, int(vllm_config.scheduler_config.max_num_seqs))
        hidden_size = int(config.hidden_size)
        dtype = vllm_config.model_config.dtype
        quant_config = get_draft_quant_config(vllm_config)
        self.dspark_aux_hidden_size = hidden_size * len(self.target_layer_ids)

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            hidden_size,
            quant_config=quant_config,
            prefix=f"{prefix}embed_tokens",
        )
        self.main_proj = ReplicatedLinear(
            self.dspark_aux_hidden_size,
            hidden_size,
            bias=False,
            params_dtype=dtype,
            quant_config=quant_config,
            prefix=f"{prefix}mtp.0.main_proj",
            return_bias=False,
        )
        self.main_norm = RMSNorm(hidden_size, eps=config.rms_norm_eps)
        self.layers = nn.ModuleDict(
            {
                str(idx): DeepSeekV4DSparkLayer(
                    vllm_config,
                    dspark_layer_idx=idx,
                    prefix=prefix.rstrip("."),
                )
                for idx in range(self.num_dspark_layers)
            }
        )

        hc_dim = self.hc_mult * hidden_size
        self.hc_head_fn = nn.Parameter(
            torch.empty(self.hc_mult, hc_dim, dtype=torch.float32),
            requires_grad=False,
        )
        self.hc_head_base = nn.Parameter(
            torch.empty(self.hc_mult, dtype=torch.float32),
            requires_grad=False,
        )
        self.hc_head_scale = nn.Parameter(
            torch.empty(1, dtype=torch.float32),
            requires_grad=False,
        )
        self.norm = RMSNorm(hidden_size, eps=config.rms_norm_eps)
        self.head = ParallelLMHead(
            config.vocab_size,
            hidden_size,
            prefix=f"{prefix}head",
        )
        self.logits_processor = LogitsProcessor(config.vocab_size)

        markov_rank = int(getattr(config, "dspark_markov_rank", 256))
        self.markov_w1 = VocabParallelEmbedding(
            config.vocab_size,
            markov_rank,
            prefix=f"{prefix}mtp.2.markov_head.markov_w1",
        )
        self.markov_w2 = ReplicatedLinear(
            markov_rank,
            config.vocab_size,
            bias=False,
            params_dtype=dtype,
            prefix=f"{prefix}mtp.2.markov_head.markov_w2",
            return_bias=False,
        )
        self.use_markov_inplace_add = _spec_bool(
            vllm_config,
            "dspark_markov_inplace_add",
        )
        self.confidence_head = ReplicatedLinear(
            hidden_size + markov_rank,
            1,
            bias=False,
            params_dtype=torch.float32,
            prefix=f"{prefix}mtp.2.confidence_head.proj",
            return_bias=False,
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def precompute_and_store_context_kv(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mapping: torch.Tensor | None = None,
        *,
        query_start_loc: torch.Tensor | None = None,
        batch_size: int | None = None,
        num_rejected_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        del context_slot_mapping
        if query_start_loc is None or context_states.numel() == 0:
            return None
        if batch_size is None:
            batch_size = int(query_start_loc.shape[0] - 1)
        main_x = _linear_output(self.main_proj(context_states))
        main_x = self.main_norm(main_x)
        for idx, layer in enumerate(self.layers.values()):
            if idx >= self.runtime_num_dspark_layers:
                break
            layer.store_main_kv(
                main_x,
                context_positions,
                query_start_loc,
                batch_size,
                num_rejected_tokens,
            )
        return main_x

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        hidden_states: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        main_positions: torch.Tensor | None = None,
        main_x: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self._forward_impl(
            input_ids,
            positions,
            hidden_states=hidden_states,
            inputs_embeds=inputs_embeds,
            main_positions=main_positions,
            main_x=main_x,
        )

    def _forward_impl(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        hidden_states: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        main_positions: torch.Tensor | None = None,
        main_x: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if input_ids is None:
            raise ValueError("DSpark requires input_ids")

        num_input_rows = input_ids.shape[0]
        inferred_batch = (num_input_rows + self.block_size - 1) // self.block_size
        batch_size = min(inferred_batch, self.max_batch)
        process_rows = batch_size * self.block_size
        if process_rows < num_input_rows:
            input_ids = input_ids[:process_rows]
            positions = positions[:process_rows]
        elif process_rows != num_input_rows:
            pad_rows = process_rows - num_input_rows
            pad_token = int(getattr(self.config, "dspark_noise_token_id", 0))
            input_ids = torch.cat(
                [
                    input_ids,
                    input_ids.new_full((pad_rows,), pad_token),
                ],
                dim=0,
            )
            positions = torch.cat(
                [positions, positions.new_zeros((pad_rows,))],
                dim=0,
            )
        if hidden_states is None or hidden_states.shape[0] != batch_size:
            new_hidden_states = torch.zeros(
                batch_size,
                self.dspark_aux_hidden_size,
                device=input_ids.device,
                dtype=self.main_norm.weight.dtype,
            )
            if hidden_states is not None and hidden_states.numel() > 0:
                rows = min(batch_size, hidden_states.shape[0])
                new_hidden_states[:rows].copy_(hidden_states[:rows])
            hidden_states = new_hidden_states
        if main_positions is None:
            main_positions = torch.clamp(
                positions.view(batch_size, self.block_size)[:, 0] - 1,
                min=0,
            )
        else:
            main_positions = torch.clamp(main_positions, min=0)
            if main_positions.shape[0] != batch_size:
                new_main_positions = torch.zeros(
                    batch_size,
                    device=positions.device,
                    dtype=positions.dtype,
                )
                rows = min(batch_size, main_positions.shape[0])
                new_main_positions[:rows].copy_(main_positions[:rows])
                main_positions = new_main_positions

        if main_x is None or main_x.shape[0] != batch_size:
            main_x = _linear_output(self.main_proj(hidden_states))
            main_x = self.main_norm(main_x)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        x = inputs_embeds.unsqueeze(-2).repeat(1, self.hc_mult, 1)

        residual, post_mix, res_mix = None, None, None
        for idx, layer in enumerate(self.layers.values()):
            if idx >= self.runtime_num_dspark_layers:
                break
            x, residual, post_mix, res_mix = layer(
                x,
                positions,
                input_ids,
                main_x,
                main_positions,
                batch_size,
                post_mix,
                res_mix,
                residual,
            )
        x = mhc_post_tilelang(x, residual, post_mix, res_mix)
        x = hc_head_fused_kernel_tilelang(
            x,
            self.hc_head_fn,
            self.hc_head_scale,
            self.hc_head_base,
            self.rms_norm_eps,
            self.hc_eps,
        )
        x = self.norm(x)
        if x.shape[0] < num_input_rows:
            x = torch.cat(
                [
                    x,
                    x.new_zeros(num_input_rows - x.shape[0], x.shape[1]),
                ],
                dim=0,
            )
        return x[:num_input_rows]

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.head, hidden_states)

    def _markov_w1_embedding(self, prev_token_ids: torch.Tensor) -> torch.Tensor:
        return self.markov_w1(prev_token_ids.long())

    def apply_dspark_markov_bias(
        self,
        base_logits: torch.Tensor,
        prev_token_ids: torch.Tensor,
        step_idx: int,
    ) -> torch.Tensor:
        del step_idx
        markov_embed = self._markov_w1_embedding(prev_token_ids)
        markov_bias = _linear_output(self.markov_w2(markov_embed))
        if self.use_markov_inplace_add and markov_bias.dtype == base_logits.dtype:
            markov_bias.add_(base_logits)
            return markov_bias
        return base_logits + markov_bias

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            ("gate_up_proj", "w1", 0),
            ("gate_up_proj", "w3", 1),
            ("attn.fused_wqa_wkv", "attn.wq_a", 0),
            ("attn.fused_wqa_wkv", "attn.wkv", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        n_head = self.config.num_attention_heads
        n_local_head = n_head // tp_size
        head_rank_start = n_local_head * tp_rank
        head_rank_end = n_local_head * (tp_rank + 1)

        first_layer = next(iter(self.layers.values()))
        if first_layer.ffn.use_mega_moe:
            expert_mapping = make_deepseek_v4_expert_params_mapping(
                self.config.n_routed_experts
            )
        else:
            expert_mapping = fused_moe_make_expert_params_mapping(
                self,
                ckpt_gate_proj_name="w1",
                ckpt_down_proj_name="w2",
                ckpt_up_proj_name="w3",
                num_experts=self.config.n_routed_experts,
            )

        expert_scale_suffix = (
            ".weight_scale"
            if getattr(self.config, "expert_dtype", "fp4") == "fp4"
            else ".weight_scale_inv"
        )

        def map_name(name: str) -> str | None:
            if name == "embed.weight":
                return "embed_tokens.weight"
            if name == "head.weight":
                return "head.weight"
            if not name.startswith("mtp."):
                return None
            parts = name.split(".")
            if len(parts) < 3:
                return None
            layer_idx = int(parts[1])
            rest = ".".join(parts[2:])
            if layer_idx >= self.num_dspark_layers:
                return None
            if layer_idx == 0 and rest.startswith("main_proj."):
                return "main_proj." + rest.removeprefix("main_proj.")
            if layer_idx == 0 and rest == "main_norm.weight":
                return "main_norm.weight"
            if layer_idx == self.num_dspark_layers - 1:
                final_map = {
                    "norm.weight": "norm.weight",
                    "hc_head_fn": "hc_head_fn",
                    "hc_head_base": "hc_head_base",
                    "hc_head_scale": "hc_head_scale",
                    "markov_head.markov_w1.weight": "markov_w1.weight",
                    "markov_head.markov_w2.weight": "markov_w2.weight",
                    "confidence_head.proj.weight": "confidence_head.weight",
                }
                if rest in final_map:
                    return final_map[rest]
            return f"layers.{layer_idx}.{rest}"

        for original_name, loaded_weight in weights:
            name = map_name(original_name)
            if name is None:
                continue
            if name.endswith(".scale"):
                suffix = (
                    expert_scale_suffix
                    if _EXPERT_SCALE_RE.search(name)
                    else ".weight_scale_inv"
                )
                name = name.removesuffix(".scale") + suffix
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if not name.startswith("layers."):
                    continue
                if ".experts." in name:
                    continue
                if weight_name not in name:
                    continue
                mapped_name = name.replace(weight_name, param_name)
                param = params_dict[mapped_name]
                param.weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(mapped_name)
                break
            else:
                if ".experts." in name:
                    if (
                        "weight_scale" in name
                        and loaded_weight.dtype == torch.float8_e8m0fnu
                    ):
                        loaded_weight = loaded_weight.view(torch.uint8)
                    for mapping in expert_mapping:
                        param_name, weight_name, expert_id, expert_shard_id = mapping
                        if weight_name not in name:
                            continue
                        name_mapped = name.replace(weight_name, param_name)
                        param = params_dict[name_mapped]
                        weight_loader = typing.cast(
                            Callable[..., bool], param.weight_loader
                        )
                        success = weight_loader(
                            param,
                            loaded_weight,
                            name_mapped,
                            shard_id=expert_shard_id,
                            expert_id=expert_id,
                            return_success=True,
                        )
                        if success:
                            loaded_params.add(name_mapped)
                            break
                    continue
                if "attn_sink" in name:
                    narrow_weight = loaded_weight[head_rank_start:head_rank_end]
                    n = narrow_weight.shape[0]
                    params_dict[name][:n].copy_(narrow_weight)
                    loaded_params.add(name)
                    continue
                if ".shared_experts.w2" in name:
                    name = name.replace(".shared_experts.w2", ".shared_experts.down_proj")
                if name.endswith(".ffn.gate.bias"):
                    name = name.replace(
                        ".ffn.gate.bias",
                        ".ffn.gate.e_score_correction_bias",
                    )
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(name)

        for layer_idx in range(self.num_dspark_layers):
            layer_prefix = f"layers.{layer_idx}."
            if not any(name.startswith(layer_prefix) for name in loaded_params):
                raise ValueError(
                    f"DSpark layer mtp.{layer_idx} weights missing from checkpoint."
                )
        self.finalize_mega_moe_weights()
        logger.info_once("DSpark draft model loaded: %d params", len(loaded_params))
        return loaded_params

    def finalize_mega_moe_weights(self) -> None:
        for layer in self.layers.values():
            layer.finalize_mega_moe_weights()
