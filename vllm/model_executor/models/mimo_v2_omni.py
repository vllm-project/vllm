# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable, Iterable
from functools import partial

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig

from vllm.config import VllmConfig
from vllm.distributed import parallel_state
from vllm.distributed import utils as dist_utils
from vllm.model_executor.layers.activation import get_act_and_mul_fn
from vllm.model_executor.layers.attention import MMEncoderAttention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.rotary_embedding.common import ApplyRotaryEmb
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.vision import is_vit_use_data_parallel

from .interfaces import (
    SupportsMultiModal,
    SupportsPP,
)
from .mimo_v2_flash import MiMoV2FlashForCausalLM
from .qwen2_5_vl import (
    Qwen2_5_VisionMLP,
    Qwen2_5_VisionPatchEmbed,
    Qwen2_5_VisionPatchMerger,
)
from .utils import maybe_prefix


class MiMoVisionMLP(Qwen2_5_VisionMLP):
    pass


class MiMoVisionPatchEmbed(Qwen2_5_VisionPatchEmbed):
    pass


class MiMoVisionPatchMerger(Qwen2_5_VisionPatchMerger):
    pass


class MiMoVisionAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_kv_heads: int,
        qk_channels: int,
        kv_channels: int,
        use_sink: bool = False,
        visual_token_window_size: int = 64,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        use_data_parallel = is_vit_use_data_parallel()
        self.tp_size = (
            1
            if use_data_parallel
            else parallel_state.get_tensor_model_parallel_world_size()
        )
        self.tp_rank = parallel_state.get_tensor_model_parallel_rank()

        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.qk_channels = qk_channels
        self.kv_channels = kv_channels
        self.embed_dim = embed_dim

        self.num_heads_per_partition = dist_utils.divide(num_heads, self.tp_size)
        self.num_kv_heads_per_partition = dist_utils.divide(num_kv_heads, self.tp_size)

        # Attention scale uses the Q/K head dimension (qk_channels)
        self.scale = qk_channels**-0.5

        # QKV: Q is (num_heads * qk_channels), KV are (num_kv_heads * kv_channels)
        self.qkv = QKVParallelLinear(
            hidden_size=embed_dim,
            head_size=qk_channels,
            total_num_heads=num_heads,
            total_num_kv_heads=num_kv_heads,
            v_head_size=kv_channels,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv",
            disable_tp=use_data_parallel,
        )

        # Output projection: input is (num_heads * kv_channels) after attention
        self.proj = RowParallelLinear(
            input_size=num_heads * kv_channels,
            output_size=embed_dim,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.proj",
            disable_tp=use_data_parallel,
        )

        # For full attention (non-window blocks)
        self.attn = MMEncoderAttention(
            num_heads=self.num_heads_per_partition,
            head_size=kv_channels,
            scale=self.scale,
            num_kv_heads=self.num_kv_heads_per_partition,
            prefix=f"{prefix}.attn",
        )

        # Rotary embeddings applied separately to Q and K
        self.apply_rotary_emb = ApplyRotaryEmb(enforce_enable=True)

        # Sink attention weights (loaded but not used in vLLM flash_attn)
        # The checkpoint stores these only for non-full-attention blocks
        self.use_sink = use_sink
        if use_sink:
            self.sinks = nn.Parameter(
                torch.zeros(num_heads, dtype=torch.bfloat16),
                requires_grad=False,
            )
        else:
            self.sinks = None

        self.visual_token_window_size = visual_token_window_size

    def _forward_window_attn(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ) -> torch.Tensor:
        """Window attention via flash_attn_varlen_func with window_size."""
        from vllm.vllm_flash_attn import flash_attn_varlen_func

        w = self.visual_token_window_size
        output = flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            dropout_p=0.0,
            softmax_scale=self.scale,
            causal=False,
            window_size=[w, w],
        )
        return output

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb_cos: torch.Tensor,
        rotary_pos_emb_sin: torch.Tensor,
        max_seqlen: int,
        full_attn: bool = True,
    ) -> torch.Tensor:
        """
        Args:
            x: [seq_len, batch=1, embed_dim]  (seq-first convention)
            cu_seqlens: cumulative sequence lengths [num_seqs+1], int32
            rotary_pos_emb_cos: [seq_len, qk_channels // 2]
            rotary_pos_emb_sin: [seq_len, qk_channels // 2]
            max_seqlen: maximum sequence length
            full_attn: if True, full attention; if False, window attention
        """
        # [seq_len, 1, embed_dim] -> QKV projection
        qkv, _ = self.qkv(x)  # [seq_len, 1, q_size + kv_size + kv_size]
        seq_len, batch_size, _ = qkv.shape

        q_size = self.num_heads_per_partition * self.qk_channels
        kv_size = self.num_kv_heads_per_partition * self.kv_channels
        q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)

        # Rearrange to [batch, seq, head, head_dim] for rotary application
        q = einops.rearrange(q, "s b (h d) -> b s h d", h=self.num_heads_per_partition)
        k = einops.rearrange(
            k, "s b (h d) -> b s h d", h=self.num_kv_heads_per_partition
        )
        v = einops.rearrange(
            v, "s b (h d) -> b s h d", h=self.num_kv_heads_per_partition
        )

        # Apply rotary embeddings to Q and K independently (handles GQA)
        if rotary_pos_emb_cos is not None and rotary_pos_emb_sin is not None:
            q = self.apply_rotary_emb(q, rotary_pos_emb_cos, rotary_pos_emb_sin)
            k = self.apply_rotary_emb(k, rotary_pos_emb_cos, rotary_pos_emb_sin)

        if full_attn:
            # Full attention via MMEncoderAttention
            # Flatten to [batch, seq, heads * head_dim]
            q_flat = q.reshape(batch_size, seq_len, -1)
            k_flat = k.reshape(batch_size, seq_len, -1)
            v_flat = v.reshape(batch_size, seq_len, -1)
            context_layer = self.attn(
                query=q_flat,
                key=k_flat,
                value=v_flat,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )
            # context_layer: [batch, seq, num_heads, head_dim] or [batch, seq, hidden]
            # Ensure shape is [seq, batch, num_heads * kv_channels]
            if context_layer.dim() == 4:
                context_layer = einops.rearrange(
                    context_layer, "b s h d -> s b (h d)"
                ).contiguous()
            else:
                context_layer = einops.rearrange(
                    context_layer, "b s d -> s b d"
                ).contiguous()
        else:
            # Window attention via flash_attn_varlen_func with window_size
            # Flatten batch dimension: [seq, head, head_dim]
            q_varlen = einops.rearrange(q, "b s h d -> (b s) h d")
            k_varlen = einops.rearrange(k, "b s h d -> (b s) h d")
            v_varlen = einops.rearrange(v, "b s h d -> (b s) h d")
            output = self._forward_window_attn(
                q_varlen, k_varlen, v_varlen, cu_seqlens, max_seqlen
            )
            # output: [total_tokens, num_heads, kv_channels]
            context_layer = einops.rearrange(
                output, "(b s) h d -> s b (h d)", b=batch_size
            ).contiguous()

        output, _ = self.proj(context_layer)
        return output


class MiMoVisionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        qk_channels: int,
        kv_channels: int,
        mlp_hidden_dim: int,
        act_fn: Callable[[torch.Tensor], torch.Tensor] = F.silu,
        norm_eps: float = 1e-6,
        use_sink: bool = False,
        visual_token_window_size: int = 64,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.norm1 = RMSNorm(dim, eps=norm_eps)
        self.norm2 = RMSNorm(dim, eps=norm_eps)
        self.attn = MiMoVisionAttention(
            embed_dim=dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            qk_channels=qk_channels,
            kv_channels=kv_channels,
            use_sink=use_sink,
            visual_token_window_size=visual_token_window_size,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )
        self.mlp = MiMoVisionMLP(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_fn=act_fn,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb_cos: torch.Tensor,
        rotary_pos_emb_sin: torch.Tensor,
        max_seqlen: int,
        full_attn: bool = True,
    ) -> torch.Tensor:
        # x: [seq_len, batch=1, dim]
        x_attn = self.attn(
            self.norm1(x),
            cu_seqlens=cu_seqlens,
            rotary_pos_emb_cos=rotary_pos_emb_cos,
            rotary_pos_emb_sin=rotary_pos_emb_sin,
            max_seqlen=max_seqlen,
            full_attn=full_attn,
        )
        # Fused residual add + norm2
        x_norm, residual = self.norm2(x, residual=x_attn)
        x = residual + self.mlp(x_norm)
        return x


class MiMoVisionTransformer(nn.Module):
    def __init__(
        self,
        vision_cfg: PretrainedConfig,
        *,
        norm_eps: float = 1e-6,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.spatial_merge_size = vision_cfg.spatial_merge_size
        self.spatial_merge_unit = self.spatial_merge_size**2
        self.fullatt_block_indexes = vision_cfg.fullatt_block_indexes
        self.vit_window_attn_types = vision_cfg.vit_window_attn_types
        self.visual_token_window_size = vision_cfg.visual_token_window_size
        self.hidden_size = vision_cfg.hidden_size
        self.num_heads = vision_cfg.num_heads
        self.num_kv_heads = vision_cfg.num_key_value_heads
        self.qk_channels = vision_cfg.qk_channels
        self.kv_channels = vision_cfg.kv_channels

        self.patch_embed = MiMoVisionPatchEmbed(
            patch_size=vision_cfg.patch_size,
            temporal_patch_size=vision_cfg.temporal_patch_size,
            in_channels=vision_cfg.in_channels,
            embed_dim=vision_cfg.hidden_size,
        )

        norm_layer = partial(RMSNorm, eps=norm_eps)

        # Rotary embedding for 2D positions.
        # With partial_rotary_factor=0.5 and head_size=qk_channels:
        #   rotary_dim = qk_channels // 2
        #   get_cos_sin returns cos, sin each of shape [pos, rotary_dim // 2]
        # After indexing with 2D pos_ids and flattening:
        #   result shape = [tokens, rotary_dim] = [tokens, qk_channels // 2]
        # which is what ApplyRotaryEmb expects as cos/sin input.
        self.rotary_pos_emb = get_rope(
            head_size=vision_cfg.qk_channels,
            max_position=8192,
            is_neox_style=True,
            rope_parameters={"partial_rotary_factor": 0.5},
        )

        self.blocks = nn.ModuleList(
            [
                MiMoVisionBlock(
                    dim=vision_cfg.hidden_size,
                    num_heads=vision_cfg.num_heads,
                    num_kv_heads=vision_cfg.num_key_value_heads,
                    qk_channels=vision_cfg.qk_channels,
                    kv_channels=vision_cfg.kv_channels,
                    mlp_hidden_dim=vision_cfg.intermediate_size,
                    act_fn=get_act_and_mul_fn(vision_cfg.hidden_act),
                    norm_eps=norm_eps,
                    use_sink=(
                        vision_cfg.use_sink
                        and i not in vision_cfg.fullatt_block_indexes
                    ),
                    visual_token_window_size=vision_cfg.visual_token_window_size,
                    quant_config=quant_config,
                    prefix=f"{prefix}.blocks.{i}",
                )
                for i in range(vision_cfg.depth)
            ]
        )

        self.merger = MiMoVisionPatchMerger(
            d_model=vision_cfg.out_hidden_size,
            context_dim=vision_cfg.hidden_size,
            norm_layer=norm_layer,
            spatial_merge_size=vision_cfg.spatial_merge_size,
            quant_config=quant_config,
            prefix=f"{prefix}.merger",
        )

    @property
    def dtype(self) -> torch.dtype:
        return self.patch_embed.proj.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.patch_embed.proj.weight.device

    def apply_index(self, tensor: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        """Reindex tensor at the spatial_merge_unit granularity."""
        tensor = tensor.unflatten(0, (-1, self.spatial_merge_unit))
        tensor = tensor[index]
        tensor = tensor.flatten(0, 1)
        return tensor

    def get_window_index_1d(
        self, grid_thw: torch.Tensor, col: bool = True
    ) -> torch.Tensor:
        """Compute 1D window indices for col-based or row-based SWA reordering."""
        window_index: list[torch.Tensor] = []
        window_index_id = 0
        for grid_t, grid_h, grid_w in grid_thw:
            llm_grid_h = grid_h // self.spatial_merge_size
            llm_grid_w = grid_w // self.spatial_merge_size
            index = torch.arange(grid_t * llm_grid_h * llm_grid_w).reshape(
                grid_t, llm_grid_h, llm_grid_w
            )
            index_new = index.transpose(1, 2).reshape(-1) if col else index.reshape(-1)
            window_index.append(index_new + window_index_id)
            window_index_id += int((grid_t * llm_grid_h * llm_grid_w).item())
        return torch.cat(window_index, dim=0)

    def rot_pos_emb(self, grid_thw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute 2D rotary position embedding cos/sin for given grid sizes.

        Returns:
            cos: [total_tokens, qk_channels // 2]
            sin: [total_tokens, qk_channels // 2]
        """
        cos_list, sin_list = [], []
        for i in range(grid_thw.size(0)):
            t, h, w = int(grid_thw[i, 0]), int(grid_thw[i, 1]), int(grid_thw[i, 2])

            # Build 2D position IDs with spatial_merge_size interleaving
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            hpos_ids = (
                hpos_ids.reshape(
                    h // self.spatial_merge_size,
                    self.spatial_merge_size,
                    w // self.spatial_merge_size,
                    self.spatial_merge_size,
                )
                .permute(0, 2, 1, 3)
                .flatten()
            )
            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            wpos_ids = (
                wpos_ids.reshape(
                    h // self.spatial_merge_size,
                    self.spatial_merge_size,
                    w // self.spatial_merge_size,
                    self.spatial_merge_size,
                )
                .permute(0, 2, 1, 3)
                .flatten()
            )
            pos_ids = torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1)
            # pos_ids: [t*h*w, 2]

            max_grid_size = max(h, w)
            # get_cos_sin returns cos, sin each of shape [max_grid_size, rotary_dim//2]
            # where rotary_dim = qk_channels // 2 (from partial_rotary_factor=0.5)
            cos, sin = self.rotary_pos_emb.get_cos_sin(max_grid_size)

            # [t*h*w, 2, rotary_dim//2] -> [t*h*w, rotary_dim] (= qk_channels // 2)
            cos_img = cos[pos_ids].flatten(1)
            sin_img = sin[pos_ids].flatten(1)
            cos_list.append(cos_img)
            sin_list.append(sin_img)

        return torch.cat(cos_list, dim=0), torch.cat(sin_list, dim=0)

    def forward(self, x: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [total_tokens, C] pre-flattened patches
            grid_thw: [num_images, 3] tensor of (t, h, w) for each image/video
        Returns:
            [merged_tokens, out_hidden_size]
        """
        # Ensure grid_thw is a tensor
        if not isinstance(grid_thw, torch.Tensor):
            grid_thw = torch.tensor(grid_thw, dtype=torch.long)

        # Move to visual model device/dtype
        x = x.to(device=self.device, dtype=self.dtype)

        # Patch embedding: [total_tokens, hidden_size]
        x = self.patch_embed(x)

        # Compute 2D rotary positional embeddings
        # cos, sin: [total_tokens, qk_channels // 2]
        rotary_cos, rotary_sin = self.rot_pos_emb(grid_thw)
        rotary_cos = rotary_cos.to(device=x.device)
        rotary_sin = rotary_sin.to(device=x.device)

        # Compute cu_seqlens for flash_attn (per-image/video sequence lengths)
        seqlens = torch.repeat_interleave(
            grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
        )
        cu_seqlens = torch.cat(
            [
                torch.tensor([0], device=x.device, dtype=torch.int32),
                seqlens.cumsum(dim=0).to(device=x.device, dtype=torch.int32),
            ]
        )
        max_seqlen = int(seqlens.max().item())

        # Precompute col-based window index for type=1 (col SWA) layers
        window_index_1d_col = self.get_window_index_1d(grid_thw, col=True).to(
            device=x.device
        )
        reverse_window_index_1d_col = torch.argsort(window_index_1d_col)

        # Col-based rotary embeddings (reordered at spatial_merge_unit granularity).
        # apply_index reorders groups of spatial_merge_unit tokens, just like x.
        col_cos = self.apply_index(rotary_cos, window_index_1d_col)
        col_sin = self.apply_index(rotary_sin, window_index_1d_col)

        # Add batch dimension: [total_tokens, 1, hidden_size]
        x = x.unsqueeze(1)

        for i, blk in enumerate(self.blocks):
            window_attn_type = self.vit_window_attn_types[i]

            # Reorder tokens to col-based layout when entering col-SWA region
            if window_attn_type == 1 and (
                i == 0 or self.vit_window_attn_types[i - 1] != 1
            ):
                x = self.apply_index(x, window_index_1d_col)

            # Restore row-based order when leaving col-SWA region
            if (
                i > 0
                and window_attn_type != 1
                and self.vit_window_attn_types[i - 1] == 1
            ):
                x = self.apply_index(x, reverse_window_index_1d_col)

            # Use col-based embeddings for col-SWA layers
            cos_now = col_cos if window_attn_type == 1 else rotary_cos
            sin_now = col_sin if window_attn_type == 1 else rotary_sin

            full_attn = i in self.fullatt_block_indexes
            x = blk(
                x,
                cu_seqlens=cu_seqlens,
                rotary_pos_emb_cos=cos_now,
                rotary_pos_emb_sin=sin_now,
                max_seqlen=max_seqlen,
                full_attn=full_attn,
            )

        # Restore row-based order if last block was col-SWA
        if self.vit_window_attn_types[-1] == 1:
            x = self.apply_index(x, reverse_window_index_1d_col)

        # Remove batch dim and merge spatial tokens
        # x: [total_tokens, 1, hidden_size] -> [total_tokens, hidden_size]
        x = x.squeeze(1)
        x = self.merger(x)
        return x

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            ("mlp.gate_up_proj", "mlp.gate_proj", 0),
            ("mlp.gate_up_proj", "mlp.up_proj", 1),
        ]
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_params: set[str] = set()

        for name, loaded_weight in weights:
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class MiMoV2OmniForCausalLM(nn.Module, SupportsMultiModal, SupportsPP):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        # Omni ViT/Audio Encoder BF16
        self.visual = MiMoVisionTransformer(
            config.vision_config,
            norm_eps=getattr(vllm_config, "rms_norm_eps", 1e-6),
            quant_config=None,
            prefix=maybe_prefix("visual", prefix),
        )
        # self.audio_config = config.audio_config
        # self.audio_encoder = MimoAudioEncoder(self.audio_config)
        self.language_model = MiMoV2FlashForCausalLM(
            vllm_config=vllm_config.with_hf_config(config.text_config),
            prefix=maybe_prefix("language_model", prefix),
        )
