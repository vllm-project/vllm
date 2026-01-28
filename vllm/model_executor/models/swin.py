# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable

import torch
import torch.nn as nn
from transformers import SwinConfig
from transformers.models.swin.modeling_swin import SwinEmbeddings, SwinPatchMerging
from transformers.models.swin.modeling_swin import SwinLayer as HFSwinLayer
from transformers.pytorch_utils import meshgrid

from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.model_loader.weight_utils import default_weight_loader


class SwinSelfAttention(nn.Module):
    def __init__(
        self,
        config: SwinConfig,
        dim: int,
        num_heads: int,
        window_size: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(
                f"The hidden size ({dim}) is not a multiple of the number of "
                f"attention heads ({num_heads})"
            )

        self.num_attention_heads = num_heads
        self.attention_head_size = int(dim / num_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.window_size = (
            window_size
            if isinstance(window_size, Iterable)
            else (window_size, window_size)
        )
        self.scale = self.attention_head_size**-0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(
                (2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1), num_heads
            )
        )

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(meshgrid([coords_h, coords_w], indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)

        self.relative_position_index = nn.Parameter(
            relative_position_index, requires_grad=False
        )

        self.qkv = QKVParallelLinear(
            hidden_size=dim,
            head_size=self.attention_head_size,
            total_num_heads=self.num_attention_heads,
            bias=config.qkv_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv",
        )

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def _get_rel_pos_bias(self) -> torch.Tensor:
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ]
        relative_position_bias = relative_position_bias.view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1,
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        return relative_position_bias.unsqueeze(0)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.FloatTensor | None = None,
        output_attentions: bool | None = False,
    ) -> tuple[torch.Tensor, ...]:
        batch_size, dim, num_channels = hidden_states.shape

        qkv_output, _ = self.qkv(hidden_states)
        query_layer, key_layer, value_layer = qkv_output.chunk(3, dim=-1)

        key_layer = self.transpose_for_scores(key_layer)
        value_layer = self.transpose_for_scores(value_layer)
        query_layer = self.transpose_for_scores(query_layer)

        attention_scores = self._get_rel_pos_bias()
        if attention_mask is not None:
            mask_shape = attention_mask.shape[0]
            attention_mask_expanded = attention_mask.view(
                1, mask_shape, 1, dim, dim
            ).expand(
                batch_size // mask_shape, mask_shape, self.num_attention_heads, dim, dim
            )
            attention_scores = attention_scores + attention_mask_expanded.unsqueeze(
                1
            ).unsqueeze(0)
            attention_scores = attention_scores.view(
                -1, self.num_attention_heads, dim, dim
            )

        context_layer = torch.nn.functional.scaled_dot_product_attention(
            query_layer,
            key_layer,
            value_layer,
            attn_mask=attention_scores,
            dropout_p=0.0,
        )
        attention_probs = None

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (
            (context_layer, attention_probs) if output_attentions else (context_layer,)
        )

        return outputs


class SwinSelfOutput(nn.Module):
    def __init__(
        self,
        config: SwinConfig,
        dim: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.dense = RowParallelLinear(
            input_size=dim,
            output_size=dim,
            quant_config=quant_config,
            prefix=f"{prefix}.dense",
        )

    def forward(
        self, hidden_states: torch.Tensor, input_tensor: torch.Tensor
    ) -> torch.Tensor:
        hidden_states, _ = self.dense(hidden_states)

        return hidden_states


class SwinAttention(nn.Module):
    def __init__(
        self,
        config: SwinConfig,
        dim: int,
        num_heads: int,
        window_size: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.self = SwinSelfAttention(
            config,
            dim,
            num_heads,
            window_size,
            quant_config=quant_config,
            prefix=f"{prefix}.self",
        )
        self.output = SwinSelfOutput(
            config, dim, quant_config=quant_config, prefix=f"{prefix}.output"
        )
        self.pruned_heads = set()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.FloatTensor | None = None,
        output_attentions: bool | None = False,
    ) -> tuple[torch.Tensor]:
        self_outputs = self.self(hidden_states, attention_mask, output_attentions)
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]
        return outputs


class SwinIntermediate(nn.Module):
    def __init__(
        self,
        config: SwinConfig,
        dim: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.dense = ColumnParallelLinear(
            dim,
            int(config.mlp_ratio * dim),
            quant_config=quant_config,
            prefix=f"{prefix}.dense",
        )
        self.intermediate_act_fn = get_act_fn(config.hidden_act)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class SwinOutput(nn.Module):
    def __init__(
        self,
        config: SwinConfig,
        dim: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.dense = RowParallelLinear(
            int(config.mlp_ratio * dim),
            dim,
            quant_config=quant_config,
            prefix=f"{prefix}.dense",
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.dense(hidden_states)
        return hidden_states


class SwinLayer(HFSwinLayer):
    def __init__(
        self,
        config: SwinConfig,
        dim: int,
        input_resolution: int,
        num_heads: int,
        drop_path_rate: float = 0.0,
        shift_size: int = 0,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__(
            config=config,
            dim=dim,
            input_resolution=input_resolution,
            num_heads=num_heads,
            drop_path_rate=drop_path_rate,
            shift_size=shift_size,
        )

        self.attention = SwinAttention(
            config,
            dim,
            num_heads,
            window_size=self.window_size,
            quant_config=quant_config,
            prefix=f"{prefix}.attention",
        )
        self.intermediate = SwinIntermediate(
            config, dim, quant_config=quant_config, prefix=f"{prefix}.intermediate"
        )
        self.output = SwinOutput(
            config, dim, quant_config=quant_config, prefix=f"{prefix}.output"
        )


class SwinStage(nn.Module):
    def __init__(
        self,
        config: SwinConfig,
        dim: int,
        input_resolution: int,
        depth: int,
        num_heads: int,
        drop_path: list[float],
        downsample: SwinPatchMerging | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.dim = dim
        self.blocks = nn.ModuleList(
            [
                SwinLayer(
                    config=config,
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    drop_path_rate=drop_path[layer_idx],
                    shift_size=0 if (layer_idx % 2 == 0) else config.window_size // 2,
                    quant_config=quant_config,
                    prefix=f"{prefix}.blocks.{layer_idx}",
                )
                for layer_idx in range(depth)
            ]
        )

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(
                input_resolution, dim=dim, norm_layer=nn.LayerNorm
            )
        else:
            self.downsample = None

        self.pointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_dimensions: tuple[int, int],
        output_attentions: bool | None = False,
        always_partition: bool | None = False,
    ) -> tuple[torch.Tensor]:
        height, width = input_dimensions
        for i, layer_module in enumerate(self.blocks):
            layer_outputs = layer_module(
                hidden_states,
                input_dimensions,
                output_attentions,
                always_partition,
            )

            hidden_states = layer_outputs[0]

        hidden_states_before_downsampling = hidden_states
        if self.downsample is not None:
            height_downsampled, width_downsampled = (height + 1) // 2, (width + 1) // 2
            output_dimensions = (height, width, height_downsampled, width_downsampled)
            hidden_states = self.downsample(
                hidden_states_before_downsampling, input_dimensions
            )
        else:
            output_dimensions = (height, width, height, width)

        stage_outputs = (
            hidden_states,
            hidden_states_before_downsampling,
            output_dimensions,
        )

        if output_attentions:
            stage_outputs += layer_outputs[1:]
        return stage_outputs


class SwinEncoder(nn.Module):
    def __init__(
        self,
        config: SwinConfig,
        grid_size: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.num_layers = len(config.depths)
        self.config = config
        dpr = [
            x.item()
            for x in torch.linspace(
                0, config.drop_path_rate, sum(config.depths), device="cpu"
            )
        ]
        self.layers = nn.ModuleList(
            [
                SwinStage(
                    config=config,
                    dim=int(config.embed_dim * 2**layer_idx),
                    input_resolution=(
                        grid_size[0] // (2**layer_idx),
                        grid_size[1] // (2**layer_idx),
                    ),
                    depth=config.depths[layer_idx],
                    num_heads=config.num_heads[layer_idx],
                    drop_path=dpr[
                        sum(config.depths[:layer_idx]) : sum(
                            config.depths[: layer_idx + 1]
                        )
                    ],
                    downsample=SwinPatchMerging
                    if (layer_idx < self.num_layers - 1)
                    else None,
                    quant_config=quant_config,
                    prefix=f"{prefix}.layers.{layer_idx}",
                )
                for layer_idx in range(self.num_layers)
            ]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_dimensions: tuple[int, int],
        output_attentions: bool | None = False,
        always_partition: bool | None = False,
    ) -> tuple[torch.Tensor]:
        for i, layer_module in enumerate(self.layers):
            layer_outputs = layer_module(
                hidden_states,
                input_dimensions,
                output_attentions,
                always_partition,
            )

            hidden_states = layer_outputs[0]
            output_dimensions = layer_outputs[2]

            input_dimensions = (output_dimensions[-2], output_dimensions[-1])

        return hidden_states


class SwinModel(nn.Module):
    config_class: SwinConfig

    def __init__(
        self,
        config: SwinConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.num_layers = len(config.depths)
        self.num_features = int(config.embed_dim * 2 ** (self.num_layers - 1))

        self.embeddings = SwinEmbeddings(config)
        self.encoder = SwinEncoder(
            config,
            self.embeddings.patch_grid,
            quant_config=quant_config,
            prefix=f"{prefix}.encoder",
        )

    def forward(
        self,
        pixel_values: torch.FloatTensor | None = None,
        output_attentions: bool | None = None,
    ) -> tuple[torch.Tensor]:
        embedding_output, input_dimensions = self.embeddings(pixel_values)

        encoder_outputs = self.encoder(
            embedding_output,
            input_dimensions,
            output_attentions=output_attentions,
        )

        return encoder_outputs

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            ("qkv", "query", "q"),
            ("qkv", "key", "k"),
            ("qkv", "value", "v"),
        ]
        params_dict = dict(self.named_parameters())
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
