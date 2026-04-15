# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable, Iterable, Mapping, Sequence
from functools import partial
from typing import Any

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BatchFeature, PretrainedConfig

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
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargsItems
from vllm.multimodal.parse import MultiModalDataItems
from vllm.multimodal.processing import (
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
)
from vllm.transformers_utils.processors.mimo_omni import (
    MiMoOmniProcessor,
    _format_timestamp,
)

from .interfaces import (
    MultiModalEmbeddings,
    SupportsMultiModal,
    SupportsPP,
)
from .mimo_v2_flash import MiMoV2FlashForCausalLM
from .qwen2_5_vl import (
    Qwen2_5_VisionMLP,
    Qwen2_5_VisionPatchEmbed,
    Qwen2_5_VisionPatchMerger,
    Qwen2_5_VLDummyInputsBuilder,
    Qwen2_5_VLImageEmbeddingInputs,
    Qwen2_5_VLImageInputs,
    Qwen2_5_VLImagePixelInputs,
    Qwen2_5_VLMultiModalProcessor,
    Qwen2_5_VLProcessingInfo,
    Qwen2_5_VLVideoEmbeddingInputs,
    Qwen2_5_VLVideoInputs,
    Qwen2_5_VLVideoPixelInputs,
)
from .utils import AutoWeightsLoader, IntermediateTensors, WeightsMapper, maybe_prefix


class Mimo_VLVisionConfig(PretrainedConfig):
    model_type = "mimovl"
    base_config_key = "vision_config"

    def __init__(
        self,
        depth=28,
        hidden_size=1280,
        hidden_act="silu",
        intermediate_size=4608,
        num_heads=32,
        in_channels=3,
        patch_size=16,
        spatial_merge_size=2,
        temporal_patch_size=2,
        tokens_per_second=2,
        window_size=128,
        out_hidden_size=2048,
        fullatt_block_indexes=None,
        initializer_range=0.02,
        kv_channels=64,  # HACK
        qk_channels=64,
        num_query_groups=4,
        num_key_value_heads=8,
        vit_window_attn_types=None,
        visual_token_window_size=64,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.depth = depth
        self.hidden_size = hidden_size
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.num_heads = num_heads
        # Support GQA: if num_key_value_heads is not provided,
        # default to num_heads (MHA)
        if num_key_value_heads is None:
            num_key_value_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.temporal_patch_size = temporal_patch_size
        self.tokens_per_second = tokens_per_second
        self.window_size = window_size
        self.fullatt_block_indexes = (
            fullatt_block_indexes
            if fullatt_block_indexes is not None
            else [7, 15, 23, 31]
        )
        self.out_hidden_size = out_hidden_size
        self.initializer_range = initializer_range
        self.kv_channels = kv_channels
        self.qk_channels = qk_channels
        self.num_query_groups = num_query_groups
        self.vit_window_attn_types = vit_window_attn_types or [-1] * depth
        self.visual_token_window_size = visual_token_window_size


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
        max_seqlen: torch.Tensor,
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
        max_seqlen: torch.Tensor,
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
        max_seqlen: torch.Tensor,
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
            hidden_size=vision_cfg.hidden_size,
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
        max_seqlen = seqlens.max()

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


class MiMoV2OmniProcessingInfo(Qwen2_5_VLProcessingInfo):
    def get_hf_config(self):
        config = self.ctx.get_hf_config()
        if isinstance(config.vision_config, dict):
            config.vision_config = Mimo_VLVisionConfig.from_dict(config.vision_config)
        return config

    def get_hf_processor(self, **kwargs: object) -> MiMoOmniProcessor:
        hf_config = self.get_hf_config()
        tokenizer = self.get_tokenizer()
        return MiMoOmniProcessor.from_hf_config(tokenizer, hf_config)

    def get_image_processor(self, **kwargs: object):
        return self.get_hf_processor(**kwargs).image_processor


class MiMoV2OmniMultiModalProcessor(Qwen2_5_VLMultiModalProcessor):
    """vLLM multimodal processor for MiMo-Omni (image + video).

    Key differences from Qwen2.5-VL:
    - Videos use timestamp tokens between temporal grid positions.
    - The HF processor expects ``(TCHW_tensor, timestamps_T_tensor)`` video
      tuples rather than plain numpy arrays.
    - ``video_start_times`` is tracked so prompt-update reconstruction can
      regenerate the exact same timestamp token IDs.
    """

    # fps assumed for vllm-decoded video (numpy T,H,W,C arrays).
    # The video loader samples ~32 frames; treat each frame as 1 s apart so
    # MiMoVLProcessor sees 1 fps input and resamples internally.
    _INPUT_FPS: float = 1.0

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return dict(
            **super()._get_mm_fields_config(hf_inputs, hf_processor_mm_kwargs),
            video_start_times=MultiModalFieldConfig.batched("video"),
        )

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        """Convert numpy video arrays to (TCHW, timestamps) tuples for MiMo."""
        if "videos" in mm_data:
            converted: list[tuple[torch.Tensor, torch.Tensor]] = []
            for video in mm_data["videos"]:
                if (
                    isinstance(video, tuple)
                    and len(video) == 2
                    and isinstance(video[0], torch.Tensor)
                    and isinstance(video[1], torch.Tensor)
                ):
                    # already in MiMo format
                    converted.append(video)
                else:
                    # numpy (T, H, W, C) or torch (T, H, W, C) / (T, C, H, W)
                    if isinstance(video, np.ndarray):
                        frames = torch.from_numpy(video)
                    elif isinstance(video, torch.Tensor):
                        frames = video
                    else:
                        frames = torch.tensor(np.array(video))

                    if frames.ndim == 4 and frames.shape[-1] in (1, 3, 4):
                        # THWC → TCHW
                        frames = frames.permute(0, 3, 1, 2).float()
                    else:
                        frames = frames.float()

                    T = frames.shape[0]
                    timestamps = torch.arange(
                        T, dtype=torch.float32
                    ) / self._INPUT_FPS
                    converted.append((frames, timestamps))

            mm_data = {**mm_data, "videos": converted}

        return super()._call_hf_processor(prompt, mm_data, mm_kwargs, tok_kwargs)

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        hf_processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        hf_config = self.info.get_hf_config()
        tokenizer = self.info.get_tokenizer()
        vocab = tokenizer.get_vocab()

        merge_size = hf_config.vision_config.spatial_merge_size
        p = hf_processor.mimo_processor

        image_pad_id = vocab[hf_processor.image_token]
        video_pad_id = vocab[hf_processor.video_token]
        vision_start_id = p.vision_start_token_id
        vision_end_id = p.vision_end_token_id
        video_start_id = p.video_start_token_id
        video_end_id = p.video_end_token_id

        def get_image_replacement(item_idx: int) -> PromptUpdateDetails:
            out_item = out_mm_kwargs["image"][item_idx]
            grid_thw = out_item["image_grid_thw"].data
            T = int(grid_thw[0])
            H = int(grid_thw[1])
            W = int(grid_thw[2])
            n_tokens = T * H * W // (merge_size * merge_size)
            full = (
                [vision_start_id]
                + [image_pad_id] * n_tokens
                + [vision_end_id]
            )
            embed_mask = (
                [False] + [True] * n_tokens + [False]
            )
            embed_t = torch.tensor(embed_mask)
            return PromptUpdateDetails(
                full=full,
                is_embed=lambda _tok, _seq: embed_t,
            )

        def get_video_replacement(item_idx: int) -> PromptUpdateDetails:
            out_item = out_mm_kwargs["video"][item_idx]
            grid_thw = out_item["video_grid_thw"].data
            spt = float(out_item["second_per_grid_ts"].data)
            start = float(out_item["video_start_times"].data)

            T = int(grid_thw[0])
            H = int(grid_thw[1])
            W = int(grid_thw[2])
            n_per_grid = H * W // (merge_size * merge_size)

            full: list[int] = [video_start_id]
            is_embed_mask: list[bool] = [False]

            for j in range(T):
                ts_text = _format_timestamp(start + j * spt)
                ts_ids = tokenizer.encode(
                    ts_text, add_special_tokens=False
                )
                full.extend(ts_ids)
                is_embed_mask.extend([False] * len(ts_ids))
                full.append(vision_start_id)
                is_embed_mask.append(False)
                full.extend([video_pad_id] * n_per_grid)
                is_embed_mask.extend([True] * n_per_grid)
                full.append(vision_end_id)
                is_embed_mask.append(False)

            full.append(video_end_id)
            is_embed_mask.append(False)

            embed_t = torch.tensor(is_embed_mask)
            return PromptUpdateDetails(
                full=full,
                is_embed=lambda _tok, _seq: embed_t,
            )

        return [
            PromptReplacement(
                modality="image",
                target=[image_pad_id],
                replacement=get_image_replacement,
            ),
            PromptReplacement(
                modality="video",
                target=[video_pad_id],
                replacement=get_video_replacement,
            ),
        ]


class MiMoV2OmniDummyInputsBuilder(Qwen2_5_VLDummyInputsBuilder):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_images = mm_counts.get("image", 0)
        num_videos = mm_counts.get("video", 0)
        image_ph = "<|vision_start|><|image_pad|><|vision_end|>"
        video_ph = "<|vision_start|><|video_pad|><|vision_end|>"
        return image_ph * num_images + video_ph * num_videos


@MULTIMODAL_REGISTRY.register_processor(
    MiMoV2OmniMultiModalProcessor,
    info=MiMoV2OmniProcessingInfo,
    dummy_inputs=MiMoV2OmniDummyInputsBuilder,
)
class MiMoV2OmniForCausalLM(nn.Module, SupportsMultiModal, SupportsPP):
    # To ensure correct weight loading and mapping.
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            # mapping for new names in checkpoint saved after transformers v4.52
            "model.language_model.": "language_model.model.",
            "model.visual.": "visual.",
            # mapping for original checkpoint
            "lm_head.": "language_model.lm_head.",
            "model.": "language_model.model.",
        }
    )

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return "<|vision_start|><|image_pad|><|vision_end|>"
        if modality.startswith("video"):
            return "<|vision_start|><|video_pad|><|vision_end|>"

        raise ValueError("Only image or video modality is supported")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        # Omni ViT/Audio Encoder BF16
        vision_config = (
            Mimo_VLVisionConfig.from_dict(config.vision_config)
            if isinstance(config.vision_config, dict)
            else config.vision_config
        )
        self.visual = MiMoVisionTransformer(
            vision_config,
            norm_eps=getattr(vllm_config, "rms_norm_eps", 1e-6),
            quant_config=None,
            prefix=maybe_prefix("visual", prefix),
        )
        # self.audio_config = config.audio_config
        # self.audio_encoder = MimoAudioEncoder(self.audio_config)
        self.language_model = MiMoV2FlashForCausalLM(
            vllm_config=vllm_config,
            prefix=maybe_prefix("language_model", prefix),
        )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

    def _parse_and_validate_image_input(
        self, **kwargs: object
    ) -> Qwen2_5_VLImageInputs | None:
        pixel_values = kwargs.pop("pixel_values", None)
        image_embeds = kwargs.pop("image_embeds", None)
        image_grid_thw = kwargs.pop("image_grid_thw", None)

        if pixel_values is None and image_embeds is None:
            return None

        if pixel_values is not None:
            return Qwen2_5_VLImagePixelInputs(
                type="pixel_values",
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
            )

        if image_embeds is not None:
            return Qwen2_5_VLImageEmbeddingInputs(
                type="image_embeds",
                image_embeds=image_embeds,
                image_grid_thw=image_grid_thw,
            )

    def _parse_and_validate_video_input(
        self, **kwargs: object
    ) -> Qwen2_5_VLVideoInputs | None:
        pixel_values_videos = kwargs.pop("pixel_values_videos", None)
        video_embeds = kwargs.pop("video_embeds", None)
        video_grid_thw = kwargs.pop("video_grid_thw", None)
        second_per_grid_ts = kwargs.pop("second_per_grid_ts", None)

        if pixel_values_videos is None and video_embeds is None:
            return None

        if pixel_values_videos is not None:
            return Qwen2_5_VLVideoPixelInputs(
                type="pixel_values_videos",
                pixel_values_videos=pixel_values_videos,
                video_grid_thw=video_grid_thw,
                second_per_grid_ts=second_per_grid_ts,
            )

        if video_embeds is not None:
            return Qwen2_5_VLVideoEmbeddingInputs(
                type="video_embeds",
                video_embeds=video_embeds,
                video_grid_thw=video_grid_thw,
                second_per_grid_ts=second_per_grid_ts,
            )

    def _process_image_input(
        self, image_input: Qwen2_5_VLImageInputs
    ) -> tuple[torch.Tensor, ...]:
        grid_thw = image_input["image_grid_thw"]
        assert grid_thw.ndim == 2
        grid_thw_list = grid_thw.tolist()

        if image_input["type"] == "image_embeds":
            image_embeds = image_input["image_embeds"].type(self.visual.dtype)
        else:
            pixel_values = image_input["pixel_values"]
            # if self.use_data_parallel:
            #     return run_dp_sharded_vision_model(
            #         self.visual, pixel_values, grid_thw_list, rope_type="rope_3d"
            #     )
            # else:
            #     image_embeds = self.visual(pixel_values, grid_thw=grid_thw_list)
            image_embeds = self.visual(pixel_values, grid_thw=grid_thw_list)

        # Split concatenated embeddings for each image item.
        merge_size = self.visual.spatial_merge_size
        sizes = (grid_thw.prod(-1) // merge_size // merge_size).tolist()
        return image_embeds.split(sizes)

    def _process_video_input(
        self, video_input: Qwen2_5_VLVideoInputs
    ) -> tuple[torch.Tensor, ...]:
        grid_thw = video_input["video_grid_thw"]
        assert grid_thw.ndim == 2
        grid_thw_list = grid_thw.tolist()

        if video_input["type"] == "video_embeds":
            video_embeds = video_input["video_embeds"].type(self.visual.dtype)
        else:
            pixel_values_videos = video_input["pixel_values_videos"]
            # if self.use_data_parallel:
            #     return run_dp_sharded_vision_model(
            #         self.visual,
            #         pixel_values_videos,
            #         grid_thw_list,
            #         rope_type="rope_3d",
            #     )
            # else:
            #     video_embeds = self.visual(
            #         pixel_values_videos, grid_thw=grid_thw_list
            #     )
            video_embeds = self.visual(pixel_values_videos, grid_thw=grid_thw_list)

        # Split concatenated embeddings for each video item.
        merge_size = self.visual.spatial_merge_size
        sizes = (grid_thw.prod(-1) // merge_size // merge_size).tolist()
        return video_embeds.split(sizes)

    def _parse_and_validate_multimodal_inputs(self, **kwargs: object) -> dict:
        mm_input_by_modality = {}

        # Preserve the order of modalities if there are multiple of them
        # from the order of kwargs.
        for input_key in kwargs:
            if (
                input_key in ("pixel_values", "image_embeds")
                and "image" not in mm_input_by_modality
            ):
                mm_input_by_modality["image"] = self._parse_and_validate_image_input(
                    **kwargs
                )
            if (
                input_key in ("pixel_values_videos", "video_embeds")
                and "video" not in mm_input_by_modality
            ):
                mm_input_by_modality["video"] = self._parse_and_validate_video_input(
                    **kwargs
                )
        return mm_input_by_modality

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        mm_input_by_modality = self._parse_and_validate_multimodal_inputs(**kwargs)
        if not mm_input_by_modality:
            return []

        # The result multimodal_embeddings is tuple of tensors, with each
        # tensor correspoending to a multimodal data item (image or video).
        multimodal_embeddings: tuple[torch.Tensor, ...] = ()

        # NOTE: It is important to iterate over the keys in this dictionary
        # to preserve the order of the modalities.
        for modality in mm_input_by_modality:
            multimodal_input = mm_input_by_modality[modality]
            if modality == "image":
                image_embeddings = self._process_image_input(multimodal_input)
                multimodal_embeddings += tuple(image_embeddings)
            if modality == "video":
                video_embeddings = self._process_video_input(multimodal_input)
                multimodal_embeddings += tuple(video_embeddings)
        return multimodal_embeddings

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        """Run forward pass for Qwen2.5-VL.

        Args:
            input_ids: Flattened (concatenated) input_ids corresponding to a
                batch.
            positions: Flattened (concatenated) position ids corresponding to a
                batch. **NOTE**: If mrope is enabled (default setting for
                Qwen2.5-VL opensource models), the shape will be `(3, seq_len)`,
                otherwise it will be `(seq_len,).
        """

        if intermediate_tensors is not None:
            inputs_embeds = None

        hidden_states = self.language_model.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        return self.language_model.compute_logits(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=["audio_encoder.", "speech_embeddings."],
        )
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
