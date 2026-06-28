# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4 DSpark draft module scaffolding.

DSpark checkpoints store their draft module under ``mtp.*`` but the module is
not the serial DeepSeek V4 MTP architecture.  This file provides a draft-only
module with parameter names and weight loading matching the DSpark checkpoint so
the runtime can be implemented incrementally without accidentally falling back
to serial MTP.
"""

import contextlib
import json
import os
import re
import typing
from collections.abc import Callable, Iterable

import torch
import torch.nn as nn
from safetensors.torch import safe_open

from vllm.config import VllmConfig
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger
from vllm.model_executor.kernels.mhc.tilelang import (
    hc_head_fused_kernel_tilelang,
    mhc_post_tilelang,
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
from vllm.model_executor.model_loader.utils import process_weights_after_loading
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import set_default_torch_dtype
from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
    _decode_e8m0_scales,
    rocm_inv_rope_einsum,
)
from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample

from .model import (
    DeepseekV4DecoderLayer,
    make_deepseek_v4_expert_params_mapping,
)
from .ops.dspark_sparse_attn_tilelang import (
    dspark_sparse_attn,
)

logger = init_logger(__name__)

_EXPERT_SCALE_RE = re.compile(r"\.experts\.\d+\.w[123]\.scale$")


def _build_dspark_topk_idxs(
    *,
    window_size: int,
    batch_size: int,
    block_size: int,
    positions: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Build DSpark's rolling-window+draft attention index tensor.

    The public DSpark reference uses exactly this dense gather contract: all
    available target-window positions followed by the local draft block rows.
    """
    target = torch.arange(window_size, device=device, dtype=torch.int32)
    valid_target = target.view(1, 1, window_size) <= positions.to(torch.int32).view(
        batch_size, 1, 1
    )
    target_idxs = torch.where(
        valid_target,
        target.view(1, 1, window_size),
        target.new_full((1, 1, window_size), -1),
    )
    target_idxs = target_idxs.expand(batch_size, block_size, window_size)
    draft_idxs = (
        window_size + torch.arange(block_size, device=device, dtype=torch.int32)
    ).view(1, 1, block_size)
    draft_idxs = draft_idxs.expand(batch_size, block_size, block_size)
    return torch.cat([target_idxs, draft_idxs], dim=-1).contiguous()


@triton.jit
def _store_dspark_context_kv_kernel(
    cache_ptr,
    main_kv_ptr,
    positions_ptr,
    query_start_loc_ptr,
    num_rejected_ptr,
    cache_indices_ptr,
    cache_stride_batch: tl.constexpr,
    cache_stride_window: tl.constexpr,
    cache_stride_dim: tl.constexpr,
    main_kv_stride_tokens: tl.constexpr,
    main_kv_stride_dim: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    WINDOW_SIZE: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    req_idx = tl.program_id(0)
    tail_idx = tl.program_id(1)
    dim_block = tl.program_id(2)
    cache_idx = tl.load(cache_indices_ptr + req_idx).to(tl.int64)

    start = tl.load(query_start_loc_ptr + req_idx)
    end = tl.load(query_start_loc_ptr + req_idx + 1)
    rejected = tl.load(num_rejected_ptr + req_idx)
    valid_end = end - rejected
    valid_len = tl.maximum(valid_end - start, 0)
    tail_len = tl.minimum(valid_len, WINDOW_SIZE)
    token_idx = valid_end - tail_len + tail_idx
    safe_token_idx = tl.maximum(token_idx, 0)

    dims = dim_block * BLOCK_D + tl.arange(0, BLOCK_D)
    dim_mask = dims < HEAD_DIM
    valid = tail_idx < tail_len

    pos = tl.load(positions_ptr + safe_token_idx, mask=valid, other=0)
    slot = pos % WINDOW_SIZE
    values = tl.load(
        main_kv_ptr
        + safe_token_idx * main_kv_stride_tokens
        + dims * main_kv_stride_dim,
        mask=valid & dim_mask,
        other=0.0,
    )
    tl.store(
        cache_ptr
        + cache_idx * cache_stride_batch
        + slot * cache_stride_window
        + dims * cache_stride_dim,
        values,
        mask=valid & dim_mask,
    )


def _fake_fp8_e4m3_mxfp_inplace(x: torch.Tensor, block_size: int = 64) -> None:
    """Mirror the public DSpark QAT KV fake-quant path.

    The HF DSpark inference code applies an in-place FP8 E4M3 quant+dequant to
    the non-RoPE KV dimensions with power-of-two scales.  This keeps the draft
    numerics aligned with the QAT-trained module without changing the storage
    format of the rolling KV cache.
    """
    if x.numel() == 0:
        return
    if x.shape[-1] % block_size != 0:
        raise ValueError(
            "DSpark fake-FP8 block size must divide the last dimension: "
            f"{x.shape[-1]} % {block_size} != 0."
        )
    view = x.reshape(-1, x.shape[-1] // block_size, block_size)
    amax = view.abs().amax(dim=-1, keepdim=True).clamp_min(1.0e-4)
    scale = torch.exp2(torch.ceil(torch.log2(amax / 448.0)))
    quant = torch.clamp(view / scale, -448.0, 448.0).to(torch.float8_e4m3fn)
    view.copy_(quant.to(view.dtype) * scale.to(view.dtype))


def _apply_dspark_kv_qat_(kv: torch.Tensor, rope_dim: int) -> None:
    non_rope = kv[..., :-rope_dim] if rope_dim > 0 else kv
    _fake_fp8_e4m3_mxfp_inplace(non_rope, block_size=64)


def _apply_dspark_rope_hf(
    x: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    rope_dim: int,
) -> torch.Tensor:
    """Apply DSpark HF-style RoPE to the trailing RoPE lanes."""
    if rope_dim <= 0:
        return x
    out = x.clone()
    rope = x[..., -rope_dim:].float()
    cos_sin = cos_sin_cache.index_select(0, positions.reshape(-1)).float()
    cos, sin = cos_sin.chunk(2, dim=-1)
    rope_pairs = rope.view(*rope.shape[:-1], -1, 2)
    even = rope_pairs[..., 0]
    odd = rope_pairs[..., 1]
    rotated = torch.stack(
        (
            even * cos[:, None, :] - odd * sin[:, None, :],
            odd * cos[:, None, :] + even * sin[:, None, :],
        ),
        dim=-1,
    ).flatten(-2)
    out[..., -rope_dim:] = rotated.to(out.dtype)
    return out


def _linear_output(output: torch.Tensor | tuple[torch.Tensor, ...]) -> torch.Tensor:
    return output[0] if isinstance(output, tuple) else output


def _dequantize_e4m3_e8m0_block_weight(
    weight: torch.Tensor,
    scale: torch.Tensor,
    *,
    block_size: int = 128,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize DSpark's standalone FP8 block-scaled linears.

    The public DSpark inference path dispatches these weights through its own
    FP8 GEMM.  vLLM's DeepGEMM block-FP8 path currently illegal-accesses on the
    DSpark main projection shape, so keep this draft-only projection as a plain
    BF16 linear until a native safe kernel is wired in.
    """
    if weight.ndim != 2 or scale.ndim != 2:
        raise ValueError(
            "DSpark FP8 block dequant expects 2D weight and scale tensors, "
            f"got weight={tuple(weight.shape)} scale={tuple(scale.shape)}."
        )
    out_features, in_features = weight.shape
    expected_scale_shape = (
        (out_features + block_size - 1) // block_size,
        (in_features + block_size - 1) // block_size,
    )
    if tuple(scale.shape) != expected_scale_shape:
        raise ValueError(
            "DSpark FP8 block scale shape mismatch: "
            f"expected {expected_scale_shape}, got {tuple(scale.shape)}."
        )
    expanded_scale = (
        scale.to(torch.float32)
        .repeat_interleave(block_size, dim=0)
        .repeat_interleave(block_size, dim=1)
    )
    expanded_scale = expanded_scale[:out_features, :in_features]
    return (weight.to(torch.float32) * expanded_scale).to(out_dtype)


def _read_dspark_num_layers(model_path: str, default: int) -> int:
    """Infer DSpark stage count from local checkpoint metadata.

    The public DSpark top-level config currently keeps
    ``num_nextn_predict_layers=1`` even though the attached DSpark module has
    three ``mtp.N`` stages.  The HF inference directory carries
    ``n_mtp_layers``; if unavailable, fall back to the safetensors index.
    """
    inference_config = os.path.join(model_path, "inference", "config.json")
    if os.path.exists(inference_config):
        with open(inference_config, encoding="utf-8") as f:
            n_layers = json.load(f).get("n_mtp_layers")
        if n_layers:
            return int(n_layers)

    index_path = os.path.join(model_path, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path, encoding="utf-8") as f:
            weight_map = json.load(f).get("weight_map", {})
        stage_ids: set[int] = set()
        for name in weight_map:
            parts = name.split(".")
            if len(parts) > 2 and parts[0] == "mtp":
                with contextlib.suppress(ValueError):
                    stage_ids.add(int(parts[1]))
        if stage_ids:
            return max(stage_ids) + 1
    return int(default)


def _iter_mtp_safetensors(model_path: str) -> Iterable[tuple[str, torch.Tensor]]:
    """Yield only ``mtp.*`` tensors from a local safetensors checkpoint."""
    index_path = os.path.join(model_path, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        raise FileNotFoundError(
            "DSpark loader currently requires a local safetensors index at "
            f"{index_path!r}."
        )
    with open(index_path, encoding="utf-8") as f:
        weight_map = json.load(f)["weight_map"]

    files = sorted(
        {filename for name, filename in weight_map.items() if name.startswith("mtp.")}
    )
    if not files:
        raise ValueError(f"No mtp.* DSpark weights found in {index_path!r}.")

    for filename in files:
        path = os.path.join(model_path, filename)
        with safe_open(path, framework="pt", device="cpu") as f:
            for name in f.keys():  # noqa: SIM118
                if name.startswith("mtp."):
                    yield name, f.get_tensor(name)


class DSparkMarkovHead(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix: str) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        rank = int(getattr(config, "dspark_markov_rank", 256))
        self.markov_w1 = VocabParallelEmbedding(
            config.vocab_size,
            rank,
            prefix=f"{prefix}.markov_w1",
        )
        self.markov_w2 = ParallelLMHead(
            config.vocab_size,
            rank,
            params_dtype=torch.float32,
            org_num_embeddings=config.vocab_size,
            prefix=f"{prefix}.markov_w2",
        )
        self.logits_processor = LogitsProcessor(config.vocab_size)

    def forward(self, token_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embeds = self.markov_w1(token_ids)
        logits = self.logits_processor(
            self.markov_w2,
            embeds.view(-1, embeds.shape[-1]).float(),
        )
        return logits.view(*embeds.shape[:-1], -1), embeds


class DSparkConfidenceHead(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix: str) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        rank = int(getattr(config, "dspark_markov_rank", 256))
        self.proj = ReplicatedLinear(
            config.hidden_size + rank,
            1,
            bias=False,
            params_dtype=torch.float32,
            quant_config=None,
            prefix=f"{prefix}.proj",
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        markov_embeds: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([hidden_states, markov_embeds], dim=-1)
        confidence = _linear_output(self.proj(x.float()))
        return confidence.squeeze(-1)


class DeepSeekV4DSparkLayer(DeepseekV4DecoderLayer):
    """One DSpark stage.

    It intentionally keeps the normal DeepSeek V4 decoder-layer parameter names
    for attention/MoE/HC weights, then adds the DSpark-only stage-0 and final
    stage heads with names matching the checkpoint.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        *,
        stage_id: int,
        num_dspark_layers: int,
        prefix: str,
        topk_indices_buffer: torch.Tensor,
        aux_stream_list: list[torch.cuda.Stream] | None,
    ) -> None:
        super().__init__(
            vllm_config,
            prefix=prefix,
            topk_indices_buffer=topk_indices_buffer,
            aux_stream_list=aux_stream_list,
        )
        self.prefix = prefix
        config = vllm_config.model_config.hf_config
        if stage_id == 0:
            target_layer_ids = tuple(
                int(i) for i in getattr(config, "dspark_target_layer_ids", ())
            )
            if not target_layer_ids:
                raise ValueError("DSpark requires dspark_target_layer_ids.")
            self.main_proj = ReplicatedLinear(
                config.hidden_size * len(target_layer_ids),
                config.hidden_size,
                bias=False,
                quant_config=None,
                prefix=f"{prefix}.main_proj",
            )
            self.main_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

        if stage_id == num_dspark_layers - 1:
            self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
            self.markov_head = DSparkMarkovHead(
                vllm_config,
                prefix=f"{prefix}.markov_head",
            )
            self.confidence_head = DSparkConfidenceHead(
                vllm_config,
                prefix=f"{prefix}.confidence_head",
            )
            self.hc_head_fn = nn.Parameter(
                torch.empty(
                    config.hc_mult,
                    config.hc_mult * config.hidden_size,
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )
            self.hc_head_base = nn.Parameter(
                torch.empty(config.hc_mult, dtype=torch.float32),
                requires_grad=False,
            )
            self.hc_head_scale = nn.Parameter(
                torch.empty(1, dtype=torch.float32),
                requires_grad=False,
            )

        self.dspark_block_size = int(getattr(config, "dspark_block_size", 0) or 0)
        self.dspark_noise_token_id = int(getattr(config, "dspark_noise_token_id", -1))
        if self.dspark_block_size <= 0:
            raise ValueError("DSpark requires dspark_block_size in the config.")
        if self.dspark_noise_token_id < 0:
            raise ValueError("DSpark requires dspark_noise_token_id in the config.")
        self.register_buffer(
            "dspark_kv_cache",
            torch.zeros(
                vllm_config.scheduler_config.max_num_seqs,
                config.sliding_window,
                config.head_dim,
                dtype=vllm_config.model_config.dtype,
            ),
            persistent=False,
        )
        # DSpark reuses the DeepSeek V4 attention linears/output projection, but
        # not the normal vLLM paged MLA/SWA cache path.  Remove those attention
        # layers from static_forward_context so DraftModelSpeculator.set_attn()
        # does not allocate/build unused draft attention metadata for them.
        static_forward_context = vllm_config.compilation_config.static_forward_context
        static_forward_context.pop(self.attn.prefix, None)
        static_forward_context.pop(self.attn.swa_cache_layer.prefix, None)

    def _compute_main_kv(
        self,
        main_x: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = main_x.shape[0]
        qr_kv = _linear_output(self.attn.fused_wqa_wkv(main_x.view(batch_size, -1)))
        _, kv = qr_kv.split([self.attn.q_lora_rank, self.attn.head_dim], dim=-1)
        kv_normed = self.attn.kv_norm(kv)
        kv = _apply_dspark_rope_hf(
            kv_normed.view(batch_size, 1, self.attn.head_dim),
            positions.view(-1),
            self.attn.rotary_emb.cos_sin_cache,
            self.attn.rope_head_dim,
        )
        _apply_dspark_kv_qat_(kv, self.attn.rope_head_dim)
        return kv.view(batch_size, self.attn.head_dim)

    def precompute_dspark_context_kv(
        self,
        main_x: torch.Tensor,
        positions: torch.Tensor,
        cache_start: int = 0,
    ) -> None:
        """Populate this stage's rolling target-KV window from target hidden states."""
        batch_size, seq_len, _ = main_x.shape
        window_size = self.attn.window_size
        cache = self.dspark_kv_cache[cache_start : cache_start + batch_size]
        main_kv = self._compute_main_kv(
            main_x.reshape(-1, main_x.shape[-1]), positions.reshape(-1)
        )
        main_kv = main_kv.view(batch_size, seq_len, self.attn.head_dim)
        tail_len = min(seq_len, window_size)
        tail_kv = main_kv[:, -tail_len:]
        tail_positions = positions[:, -tail_len:]
        slots = tail_positions.remainder(window_size).long()
        batch_indices = torch.arange(
            batch_size,
            dtype=torch.long,
            device=main_x.device,
        ).view(batch_size, 1)
        cache[batch_indices, slots] = tail_kv

    def precompute_dspark_context_kv_flat(
        self,
        main_x: torch.Tensor,
        positions: torch.Tensor,
        query_start_loc: torch.Tensor,
        num_rejected: torch.Tensor,
        cache_indices: torch.Tensor,
        num_reqs: int,
    ) -> None:
        """Populate this stage's rolling target-KV window from flat target rows."""
        if num_reqs <= 0:
            return
        main_kv = self._compute_main_kv(main_x, positions).contiguous()
        block_d = 64
        grid = (
            num_reqs,
            self.attn.window_size,
            triton.cdiv(self.attn.head_dim, block_d),
        )
        _store_dspark_context_kv_kernel[grid](
            self.dspark_kv_cache,
            main_kv,
            positions,
            query_start_loc,
            num_rejected,
            cache_indices,
            self.dspark_kv_cache.stride(0),
            self.dspark_kv_cache.stride(1),
            self.dspark_kv_cache.stride(2),
            main_kv.stride(0),
            main_kv.stride(1),
            HEAD_DIM=self.attn.head_dim,
            WINDOW_SIZE=self.attn.window_size,
            BLOCK_D=block_d,
        )

    def cache_dspark_wo_a_bf16(self) -> None:
        """Preserve DSpark's HF-style BF16 WO-A before FP8 kernel repacking.

        DSpark attention uses the public reference path:
        inverse-RoPE -> BF16 WO-A einsum -> WO-B.  The target DeepSeek V4
        linears are repacked by process_weights_after_loading for DeepGEMM, so
        after that point wo_a.weight_scale_inv no longer has checkpoint block
        layout. Cache the canonical BF16 WO-A while the loaded params still
        match the checkpoint layout.
        """
        if hasattr(self.attn.wo_a, "weight_scale_inv"):
            flat_weight = self.attn.wo_a.weight.reshape(
                self.attn.n_local_groups * self.attn.o_lora_rank,
                -1,
            )
            scale = _decode_e8m0_scales(self.attn.wo_a.weight_scale_inv).reshape(
                -1, self.attn.wo_a.weight_scale_inv.shape[-1]
            )
            cached = flat_weight.unflatten(0, (-1, 128)).unflatten(-1, (-1, 128)).to(
                torch.float32
            ) * scale[:, None, :, None].to(torch.float32)
            cached = (
                cached.flatten(2, 3)
                .flatten(0, 1)
                .to(torch.bfloat16)
                .view(
                    self.attn.n_local_groups,
                    self.attn.o_lora_rank,
                    flat_weight.shape[-1],
                )
            )
        else:
            cached = self.attn.wo_a.weight.view(
                self.attn.n_local_groups,
                self.attn.o_lora_rank,
                -1,
            ).to(torch.bfloat16)

        if "_dsv4_wo_a_bf16" in self.attn.wo_a._buffers:
            self.attn.wo_a._buffers["_dsv4_wo_a_bf16"] = cached
        else:
            self.attn.wo_a.register_buffer(
                "_dsv4_wo_a_bf16",
                cached,
                persistent=False,
            )

    def _dspark_output_projection(
        self,
        out: torch.Tensor,
        draft_positions: torch.Tensor,
        batch_size: int,
        block_size: int,
        hidden_size: int,
    ) -> torch.Tensor:
        flat_positions = draft_positions.reshape(-1)
        # DSpark's public reference uses a BF16 inverse-RoPE + WO-A/WO-B path
        # here.  The target DeepSeek V4 CUDA path uses a fused FP8 _o_proj that
        # assumes target-layer RoPE/cache invariants and corrupts DSpark draft
        # numerics for compress_ratio=0.
        z = rocm_inv_rope_einsum(
            self.attn.rotary_emb,
            out,
            flat_positions,
            self.attn.rope_head_dim,
            self.attn.n_local_groups,
            self.attn.o_lora_rank,
            self.attn.wo_a,
        )
        out_final = self.attn.wo_b(z.flatten(1)).view(
            batch_size,
            block_size,
            hidden_size,
        )
        return out_final

    def _dspark_attention(
        self,
        positions: torch.Tensor,
        x: torch.Tensor,
        main_x: torch.Tensor,
        cache_indices: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, block_size, hidden_size = x.shape
        if block_size != self.dspark_block_size:
            raise ValueError(
                "DSpark draft block size mismatch: "
                f"expected {self.dspark_block_size}, got {block_size}."
            )
        if main_x.shape[1] != 1:
            raise ValueError(
                "DSpark decode currently expects one target hidden state per request."
            )

        main_positions = positions.view(batch_size)
        main_kv = self._compute_main_kv(main_x[:, 0], main_positions)

        flat_x = x.reshape(batch_size * block_size, hidden_size)
        qr_kv = _linear_output(self.attn.fused_wqa_wkv(flat_x))
        qr, kv = qr_kv.split([self.attn.q_lora_rank, self.attn.head_dim], dim=-1)
        q_normed = self.attn.q_norm(qr)
        q = _linear_output(self.attn.wq_b(q_normed))
        kv_normed = self.attn.kv_norm(kv)
        q = q.view(batch_size * block_size, self.attn.n_local_heads, self.attn.head_dim)
        q *= torch.rsqrt(q.square().mean(-1, keepdim=True) + self.attn.eps)
        kv = kv_normed.view(batch_size * block_size, 1, self.attn.head_dim)

        draft_positions = positions.view(batch_size, 1) + torch.arange(
            1,
            block_size + 1,
            dtype=positions.dtype,
            device=positions.device,
        ).view(1, block_size)
        q = _apply_dspark_rope_hf(
            q,
            draft_positions.reshape(-1),
            self.attn.rotary_emb.cos_sin_cache,
            self.attn.rope_head_dim,
        )
        kv = _apply_dspark_rope_hf(
            kv,
            draft_positions.reshape(-1),
            self.attn.rotary_emb.cos_sin_cache,
            self.attn.rope_head_dim,
        )
        _apply_dspark_kv_qat_(kv, self.attn.rope_head_dim)
        q = q.view(batch_size, block_size, self.attn.n_local_heads, self.attn.head_dim)
        kv = kv.view(batch_size, block_size, self.attn.head_dim)

        window_size = self.attn.window_size
        cache_rows = cache_indices.view(batch_size).long()
        cache_slots = positions.view(batch_size).remainder(window_size).long()
        self.dspark_kv_cache[cache_rows, cache_slots].copy_(main_kv)
        cache_window = self.dspark_kv_cache[cache_rows]
        all_kv = torch.cat([cache_window, kv], dim=1)
        topk_idxs = _build_dspark_topk_idxs(
            window_size=window_size,
            batch_size=batch_size,
            block_size=block_size,
            positions=positions.view(batch_size),
            device=x.device,
        )
        out = dspark_sparse_attn(
            q,
            all_kv,
            self.attn.attn_sink[: self.attn.n_local_heads],
            topk_idxs,
            self.attn.scale,
        )
        out = out.reshape(
            batch_size * block_size,
            self.attn.n_local_heads,
            self.attn.head_dim,
        )
        return self._dspark_output_projection(
            out,
            draft_positions,
            batch_size,
            block_size,
            hidden_size,
        )

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        input_ids: torch.Tensor,
        main_x: torch.Tensor,
        cache_indices: torch.Tensor,
        post_mix: torch.Tensor | None = None,
        res_mix: torch.Tensor | None = None,
        residual: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, block_size = input_ids.shape
        hidden_size = self.hidden_size
        if residual is None:
            if x.ndim != 4:
                raise ValueError(
                    "First DSpark stage expects [batch, block, hc, hidden] input."
                )
            hc_mult = x.shape[2]
            x_current = x.reshape(batch_size * block_size, hc_mult, hidden_size)
        else:
            if x.ndim != 3:
                raise ValueError(
                    "Subsequent DSpark stages expect [batch, block, hidden] input."
                )
            x_current = x.reshape(batch_size * block_size, hidden_size)
        input_ids_flat = input_ids.reshape(-1)

        attn_norm_weight = self.attn_norm.weight.data
        attn_norm_eps = self.attn_norm.variance_epsilon
        if residual is None:
            residual = x_current
            x_attn, post_mix, res_mix = self.hc_pre(
                residual,
                self.hc_attn_fn,
                self.hc_attn_scale,
                self.hc_attn_base,
                norm_weight=attn_norm_weight,
                norm_eps=attn_norm_eps,
            )
        else:
            assert post_mix is not None
            assert res_mix is not None
            residual, post_mix, res_mix, x_attn = self.hc_post_pre(
                x_current,
                residual,
                post_mix,
                res_mix,
                self.hc_attn_fn,
                self.hc_attn_scale,
                self.hc_attn_base,
                norm_weight=attn_norm_weight,
                norm_eps=attn_norm_eps,
            )

        x_attn = self._dspark_attention(
            positions,
            x_attn.view(batch_size, block_size, hidden_size),
            main_x,
            cache_indices,
        ).reshape(batch_size * block_size, hidden_size)

        ffn_norm_weight = self.ffn_norm.weight.data
        ffn_norm_eps = self.ffn_norm.variance_epsilon
        residual, post_mix, res_mix, x_ffn = self.hc_post_pre(
            x_attn,
            residual,
            post_mix,
            res_mix,
            self.hc_ffn_fn,
            self.hc_ffn_scale,
            self.hc_ffn_base,
            norm_weight=ffn_norm_weight,
            norm_eps=ffn_norm_eps,
            hc_fn_bf16=self.hc_ffn_fn_bf16,
        )
        x_ffn = self.ffn(x_ffn, input_ids_flat)
        return (
            x_ffn.view(batch_size, block_size, hidden_size),
            residual,
            post_mix,
            res_mix,
        )

class DeepSeekV4DSparkDraft(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        self.config = vllm_config.model_config.hf_config
        self.model_path = vllm_config.model_config.model
        self.num_dspark_layers = _read_dspark_num_layers(
            self.model_path,
            getattr(self.config, "num_nextn_predict_layers", 1),
        )
        self.dspark_start_layer_idx = self.config.num_hidden_layers
        self.checkpoint_weight_name_prefixes = tuple(
            f"mtp.{idx}." for idx in range(self.num_dspark_layers)
        )

        self.topk_indices_buffer = torch.empty(
            vllm_config.scheduler_config.max_num_batched_tokens,
            self.config.index_topk,
            dtype=torch.int32,
        )
        aux_stream_list = [torch.cuda.Stream() for _ in range(3)]

        self.layers = nn.ModuleDict()
        for idx in range(self.num_dspark_layers):
            layer_idx = self.dspark_start_layer_idx + idx
            layer_prefix = (
                f"{prefix}.layers.{layer_idx}" if prefix else f"layers.{layer_idx}"
            )
            self.layers[str(layer_idx)] = DeepSeekV4DSparkLayer(
                vllm_config,
                stage_id=idx,
                num_dspark_layers=self.num_dspark_layers,
                prefix=layer_prefix,
                topk_indices_buffer=self.topk_indices_buffer,
                aux_stream_list=aux_stream_list,
            )
        self.embed_tokens: VocabParallelEmbedding | None = None
        self.lm_head: ParallelLMHead | None = None
        self.logits_processor = LogitsProcessor(self.config.vocab_size)

    def attach_target_modules(
        self,
        embed_tokens: VocabParallelEmbedding,
        lm_head: ParallelLMHead,
    ) -> None:
        self.embed_tokens = embed_tokens
        self.lm_head = lm_head

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self.embed_tokens is None:
            raise RuntimeError("DSpark draft has no shared target embedding.")
        return self.embed_tokens(input_ids)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.lm_head is None:
            raise RuntimeError("DSpark draft has no shared target lm_head.")
        return self.logits_processor(
            self.lm_head,
            hidden_states.view(-1, hidden_states.shape[-1]),
        ).view(*hidden_states.shape[:-1], -1)

    def forward_embed(
        self,
        main_hidden: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        first_layer = self.layers[str(self.dspark_start_layer_idx)]
        assert isinstance(first_layer, DeepSeekV4DSparkLayer)
        main_x = _linear_output(first_layer.main_proj(main_hidden))
        main_x = first_layer.main_norm(main_x)
        batch_size = input_ids.shape[0]
        draft_input_ids = input_ids.new_full(
            (batch_size, first_layer.dspark_block_size),
            first_layer.dspark_noise_token_id,
        )
        draft_input_ids[:, 0] = input_ids
        x = self.embed_input_ids(draft_input_ids.reshape(-1))
        x = x.view(batch_size, first_layer.dspark_block_size, self.config.hidden_size)
        x = x.unsqueeze(2).repeat(1, 1, self.config.hc_mult, 1)
        return x, main_x.unsqueeze(1), draft_input_ids

    def precompute_context_kv(
        self,
        main_hidden: torch.Tensor,
        positions: torch.Tensor,
        context_ranges: list[tuple[int, int]],
    ) -> None:
        """Populate DSpark rolling target-KV caches for scheduled target tokens."""
        if not context_ranges:
            return
        first_layer = self.layers[str(self.dspark_start_layer_idx)]
        assert isinstance(first_layer, DeepSeekV4DSparkLayer)
        main_x = _linear_output(first_layer.main_proj(main_hidden))
        main_x = first_layer.main_norm(main_x)

        for req_idx, (start, end) in enumerate(context_ranges):
            if end <= start:
                continue
            req_main_x = main_x[start:end].unsqueeze(0)
            req_positions = positions[start:end].unsqueeze(0)
            for layer in self.layers.values():
                assert isinstance(layer, DeepSeekV4DSparkLayer)
                layer.precompute_dspark_context_kv(
                    req_main_x,
                    req_positions,
                    cache_start=req_idx,
                )

    def precompute_context_kv_flat(
        self,
        main_hidden: torch.Tensor,
        positions: torch.Tensor,
        query_start_loc: torch.Tensor,
        num_rejected: torch.Tensor,
        cache_indices: torch.Tensor,
        num_reqs: int,
    ) -> None:
        """Populate DSpark rolling target-KV caches from flat target rows."""
        if num_reqs <= 0 or main_hidden.numel() == 0:
            return
        first_layer = self.layers[str(self.dspark_start_layer_idx)]
        assert isinstance(first_layer, DeepSeekV4DSparkLayer)
        main_x = _linear_output(first_layer.main_proj(main_hidden))
        main_x = first_layer.main_norm(main_x)

        for layer in self.layers.values():
            assert isinstance(layer, DeepSeekV4DSparkLayer)
            layer.precompute_dspark_context_kv_flat(
                main_x,
                positions,
                query_start_loc,
                num_rejected,
                cache_indices,
                num_reqs,
            )

    def reset_dspark_kv_cache(self) -> None:
        for layer in self.layers.values():
            assert isinstance(layer, DeepSeekV4DSparkLayer)
            layer.dspark_kv_cache.zero_()

    def forward_head(
        self,
        x: torch.Tensor,
        input_ids: torch.Tensor,
        *,
        idx_mapping: torch.Tensor | None = None,
        temperature: torch.Tensor | None = None,
        seeds: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        draft_logits: torch.Tensor | None = None,
        draft_step_cols: torch.Tensor | None = None,
        active_num_reqs: torch.Tensor | None = None,
        use_fp64_gumbel: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        final_layer = self.layers[
            str(self.dspark_start_layer_idx + self.num_dspark_layers - 1)
        ]
        assert isinstance(final_layer, DeepSeekV4DSparkLayer)
        batch_size, block_size, hc_mult, hidden_size = x.shape
        x_flat = x.reshape(batch_size * block_size, hc_mult, hidden_size)
        hidden = hc_head_fused_kernel_tilelang(
            x_flat,
            final_layer.hc_head_fn,
            final_layer.hc_head_scale,
            final_layer.hc_head_base,
            final_layer.rms_norm_eps,
            final_layer.hc_eps,
        ).view(batch_size, block_size, hidden_size)
        logits_hidden = final_layer.norm(hidden)
        logits = self.compute_logits(logits_hidden)

        output_ids = input_ids.new_empty(batch_size, block_size + 1)
        output_ids[:, 0] = input_ids
        markov_embeds = []
        for idx in range(block_size):
            logits_bias, markov_embed = final_layer.markov_head(output_ids[:, idx])
            logits[:, idx].add_(logits_bias)
            markov_embeds.append(markov_embed)
            if draft_logits is not None:
                assert idx_mapping is not None
                assert temperature is not None
                assert seeds is not None
                assert positions is not None
                assert draft_step_cols is not None
                output_ids[:, idx + 1] = gumbel_sample(
                    logits[:, idx],
                    idx_mapping,
                    temperature,
                    seeds,
                    positions + idx + 1,
                    apply_temperature=True,
                    output_processed_logits=draft_logits,
                    output_processed_logits_col=draft_step_cols[idx],
                    output_processed_logits_active_rows=active_num_reqs,
                    use_fp64=use_fp64_gumbel,
                )
            else:
                output_ids[:, idx + 1] = logits[:, idx].argmax(dim=-1)
        confidence = final_layer.confidence_head(
            hidden,
            torch.stack(markov_embeds, dim=1),
        )
        return output_ids, logits, confidence

    def forward_spec(
        self,
        input_ids: torch.Tensor,
        main_hidden: torch.Tensor,
        positions: torch.Tensor,
        *,
        idx_mapping: torch.Tensor | None = None,
        temperature: torch.Tensor | None = None,
        seeds: torch.Tensor | None = None,
        draft_logits: torch.Tensor | None = None,
        draft_step_cols: torch.Tensor | None = None,
        active_num_reqs: torch.Tensor | None = None,
        use_fp64_gumbel: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if positions.numel() == 0:
            raise ValueError("DSpark forward_spec requires at least one position.")
        if idx_mapping is None:
            raise ValueError("DSpark forward_spec requires idx_mapping.")
        x, main_x, draft_input_ids = self.forward_embed(main_hidden, input_ids)
        residual, post_mix, res_mix = None, None, None
        for layer in self.layers.values():
            assert isinstance(layer, DeepSeekV4DSparkLayer)
            x, residual, post_mix, res_mix = layer(
                x,
                positions,
                draft_input_ids,
                main_x,
                idx_mapping,
                post_mix,
                res_mix,
                residual,
            )
        assert residual is not None
        assert post_mix is not None
        assert res_mix is not None
        x_flat = mhc_post_tilelang(
            x.reshape(-1, self.config.hidden_size),
            residual,
            post_mix,
            res_mix,
        )
        x = x_flat.view(
            input_ids.shape[0],
            draft_input_ids.shape[1],
            self.config.hc_mult,
            self.config.hidden_size,
        )
        output_ids, logits, confidence = self.forward_head(
            x,
            input_ids,
            idx_mapping=idx_mapping,
            temperature=temperature,
            seeds=seeds,
            positions=positions,
            draft_logits=draft_logits,
            draft_step_cols=draft_step_cols,
            active_num_reqs=active_num_reqs,
            use_fp64_gumbel=use_fp64_gumbel,
        )
        return output_ids, logits, confidence

    def _get_dspark_layer_idx_from_weight_name(self, name: str) -> int | None:
        for idx in range(self.num_dspark_layers):
            layer_idx = self.dspark_start_layer_idx + idx
            if name.startswith(f"layers.{layer_idx}.") or name.startswith(
                f"model.layers.{layer_idx}."
            ):
                return layer_idx
        return None

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
        n_local_head = self.config.num_attention_heads // tp_size
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

        pending_main_proj: dict[str, dict[str, torch.Tensor]] = {}

        def maybe_load_main_proj(
            name: str,
            loaded_weight: torch.Tensor,
        ) -> bool:
            if ".main_proj." not in name:
                return False
            base_name, suffix = name.rsplit(".", 1)
            if suffix not in ("weight", "scale"):
                return False
            entry = pending_main_proj.setdefault(base_name, {})
            entry[suffix] = loaded_weight
            if "weight" not in entry or "scale" not in entry:
                return True
            param_name = f"{base_name}.weight"
            param = params_dict[param_name]
            dequant_weight = _dequantize_e4m3_e8m0_block_weight(
                entry["weight"],
                entry["scale"],
                out_dtype=param.dtype,
            )
            dequant_weight = dequant_weight.to(device=param.device, dtype=param.dtype)
            with torch.no_grad():
                param.data.copy_(dequant_weight)
            loaded_params.add(param_name)
            del pending_main_proj[base_name]
            return True

        for name, loaded_weight in weights:
            if not name.startswith("mtp."):
                continue
            mtp_layer_idx = int(name.split(".", 2)[1])
            name = name.replace(
                f"mtp.{mtp_layer_idx}.",
                f"layers.{self.dspark_start_layer_idx + mtp_layer_idx}.",
            )
            spec_layer = self._get_dspark_layer_idx_from_weight_name(name)
            if spec_layer is None:
                continue

            if maybe_load_main_proj(name, loaded_weight):
                continue

            if name.endswith(".scale"):
                suffix = (
                    expert_scale_suffix
                    if _EXPERT_SCALE_RE.search(name)
                    else ".weight_scale_inv"
                )
                name = name.removesuffix(".scale") + suffix

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if ".experts." in name:
                    continue
                if weight_name in ("w1", "w3") and ".shared_experts." not in name:
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
                    # E8M0 expert scales are stored as float8_e8m0fnu in the
                    # checkpoint, but MXFP4 MoE scale params keep raw exponent
                    # bytes. Non-expert FP8 linears use float8 scale params, so
                    # keep this conversion scoped to experts only.
                    if (
                        "weight_scale" in name
                        and loaded_weight.dtype == torch.float8_e8m0fnu
                    ):
                        loaded_weight = loaded_weight.view(torch.uint8)
                    for mapping in expert_mapping:
                        param_name, weight_name, expert_id, expert_shard_id = mapping
                        if weight_name not in name:
                            continue
                        mapped_name = name.replace(weight_name, param_name)
                        param = params_dict[mapped_name]
                        weight_loader = typing.cast(
                            Callable[..., bool], param.weight_loader
                        )
                        if weight_loader(
                            param,
                            loaded_weight,
                            mapped_name,
                            shard_id=expert_shard_id,
                            expert_id=expert_id,
                            return_success=True,
                        ):
                            loaded_params.add(mapped_name)
                            break
                    continue
                if "attn_sink" in name:
                    narrow_weight = loaded_weight[head_rank_start:head_rank_end]
                    n = narrow_weight.shape[0]
                    params_dict[name][:n].copy_(narrow_weight)
                    loaded_params.add(name)
                    continue
                if ".shared_experts.w2" in name:
                    name = name.replace(
                        ".shared_experts.w2", ".shared_experts.down_proj"
                    )
                if name.endswith(".ffn.gate.bias"):
                    name = name.replace(
                        ".ffn.gate.bias",
                        ".ffn.gate.e_score_correction_bias",
                    )
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(name)

        if pending_main_proj:
            raise ValueError(
                "Incomplete DSpark main_proj FP8 block weights: "
                f"{sorted(pending_main_proj)}"
            )

        expected_layers = set(
            range(
                self.dspark_start_layer_idx,
                self.dspark_start_layer_idx + self.num_dspark_layers,
            )
        )
        loaded_layers = {
            layer
            for param_name in loaded_params
            if (layer := self._get_dspark_layer_idx_from_weight_name(param_name))
            is not None
        }
        missing_layers = expected_layers - loaded_layers
        if missing_layers:
            raise ValueError(
                "DSpark draft layer weights missing from checkpoint: "
                f"{sorted(missing_layers)}"
            )

        for layer in self.layers.values():
            layer.cache_dspark_wo_a_bf16()
            layer.ffn.finalize_mega_moe_weights()

        logger.info_once(
            "DSpark draft weights loaded: %d params across %d stages",
            len(loaded_params),
            self.num_dspark_layers,
        )
        return loaded_params


def load_dspark_model(vllm_config: VllmConfig) -> DeepSeekV4DSparkDraft:
    target_device = vllm_config.device_config.device
    with set_default_torch_dtype(vllm_config.model_config.dtype):
        model = DeepSeekV4DSparkDraft(vllm_config=vllm_config)
    model.load_weights(_iter_mtp_safetensors(vllm_config.model_config.model))
    process_weights_after_loading(
        model,
        vllm_config.model_config,
        target_device,
    )
    model.to(target_device)
    return model
