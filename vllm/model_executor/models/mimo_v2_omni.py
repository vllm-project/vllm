# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable
from functools import partial

import einops
import torch
import torch.nn.functional as F
from torch import nn

from vllm.config import (
    VllmConfig,
)
from vllm.distributed import parallel_state
from vllm.distributed import utils as dist_utils
from vllm.model_executor.layers.linear import (
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding.common import (
    ApplyRotaryEmb,
)

from .interfaces import SupportsMultiModal, SupportsPP
from .mimo_v2_flash import MiMoV2FlashForCausalLM
from .qwen2_5_vl import (
    Qwen2_5_VisionMLP as MiMoVisionMLP,
)
from .utils import (
    maybe_prefix,
)
from .vision import is_vit_use_data_parallel


class MiMoVisionAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        projection_size: int,
        use_sink: bool = False,
        window_size: tuple[int, int] = (-1, -1),
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        # Per attention head and per partition values.
        use_data_parallel = is_vit_use_data_parallel()
        self.tp_size = (
            1
            if use_data_parallel
            else parallel_state.get_tensor_model_parallel_world_size()
        )
        self.tp_rank = parallel_state.get_tensor_model_parallel_rank()
        self.hidden_size_per_attention_head = dist_utils.divide(
            projection_size, num_heads
        )
        self.num_attention_heads_per_partition = dist_utils.divide(
            num_heads, self.tp_size
        )

        self.qkv = QKVParallelLinear(
            hidden_size=embed_dim,
            head_size=self.hidden_size_per_attention_head,
            total_num_heads=num_heads,
            total_num_kv_heads=num_heads,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv",
            disable_tp=use_data_parallel,
        )

        self.proj = RowParallelLinear(
            input_size=projection_size,
            output_size=embed_dim,
            quant_config=quant_config,
            prefix=f"{prefix}.proj",
            disable_tp=use_data_parallel,
        )

        # self.attn = MMEncoderAttention(
        #     num_heads=self.num_attention_heads_per_partition,
        #     head_size=self.hidden_size_per_attention_head,
        #     scale=self.hidden_size_per_attention_head**-0.5,
        #     prefix=f"{prefix}.attn",
        # )

        self.apply_rotary_emb = ApplyRotaryEmb(enforce_enable=True)

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb_cos: torch.Tensor,
        rotary_pos_emb_sin: torch.Tensor,
        max_seqlen: torch.Tensor,  # Only used for Flash Attention
        sequence_lengths: torch.Tensor,  # Only used for FlashInfer CuDNN backend
    ) -> torch.Tensor:
        """
        x: [b, s, embed_dim]
        cu_seqlens: [b]
        """
        # [s, b, c] --> [s, b, head * 3 * head_dim]
        x, _ = self.qkv(x)
        seq_len, batch_size, _ = x.shape

        qkv = einops.rearrange(
            x,
            "b s (three head head_dim) -> b s three head head_dim",
            three=3,
            head=self.num_attention_heads_per_partition,
        )

        if rotary_pos_emb_cos is not None and rotary_pos_emb_sin is not None:
            qk, v = qkv[:, :, :2], qkv[:, :, 2]

            qk_reshaped = einops.rearrange(
                qk, "b s two head head_dim -> (two b) s head head_dim", two=2
            )
            qk_reshaped = qk_reshaped.contiguous()
            qk_rotated = self.apply_rotary_emb(
                qk_reshaped,
                rotary_pos_emb_cos,
                rotary_pos_emb_sin,
            )
            qk_rotated = qk_rotated.view(
                2,
                batch_size,
                seq_len,
                self.num_attention_heads_per_partition,
                self.hidden_size_per_attention_head,
            )
            q, k = qk_rotated.unbind(dim=0)
        else:
            q, k, v = qkv.unbind(dim=2)

        # TODO(Isotr0py): bidirectional SWA
        context_layer = self.attn(
            query=q,
            key=k,
            value=v,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            sequence_lengths=sequence_lengths,
        )

        output, _ = self.proj(context_layer)
        return output


class MiMoVisionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_hidden_dim: int,
        act_fn: Callable[[torch.Tensor], torch.Tensor] = F.silu,
        use_sink: bool = False,
        window_size: tuple[int, int] = (-1, -1),
        norm_layer: Callable[[int], nn.Module] | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.attn = MiMoVisionAttention(
            embed_dim=dim,
            num_heads=num_heads,
            projection_size=dim,
            use_sink=use_sink,
            window_size=window_size,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )
        self.mlp = MiMoVisionMLP(
            dim,
            mlp_hidden_dim,
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
        max_seqlen: torch.Tensor,  # Only used for Flash Attention
    ) -> torch.Tensor:
        x_attn = self.attn(
            self.norm1(x),
            cu_seqlens=cu_seqlens,
            rotary_pos_emb_cos=rotary_pos_emb_cos,
            rotary_pos_emb_sin=rotary_pos_emb_sin,
            max_seqlen=max_seqlen,
            sequence_lengths=None,
        )
        x_fused_norm, residual = self.norm2(x, residual=x_attn)
        x = residual + self.mlp(x_fused_norm)
        return x


class MiMoVisionTransformer(nn.Module):
    def __init__(
        self,
        config,
        *,
        norm_eps=1e-6,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.norm_eps = norm_eps
        self.quant_config = quant_config
        self.prefix = prefix

    def apply_index(self, tensor: torch.Tensor, index: torch.Tensor):
        tensor = tensor.unflatten(0, (-1, self.spatial_merge_unit))
        tensor = tensor[index]
        tensor = tensor.flatten(0, 1)
        return tensor

    def get_window_index_1d(self, grid_thw, col=True):
        window_index: list = []
        window_index_id = 0
        for grid_t, grid_h, grid_w in grid_thw:
            llm_grid_h, llm_grid_w = (
                grid_h // self.spatial_merge_size,
                grid_w // self.spatial_merge_size,
            )
            index = torch.arange(grid_t * llm_grid_h * llm_grid_w).reshape(
                grid_t, llm_grid_h, llm_grid_w
            )
            index_new = index.transpose(1, 2).reshape(-1) if col else index.reshape(-1)
            window_index.append(index_new + window_index_id)
            window_index_id += (grid_t * llm_grid_h * llm_grid_w).item()
        window_index = torch.cat(
            window_index,
            dim=0,
        )
        return window_index

    def forward(self, x: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
        # TODO(Isotr0py): ViT SWA
        pass


class MiMoV2OmniForCausalLM(nn.Module, SupportsMultiModal, SupportsPP):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
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
