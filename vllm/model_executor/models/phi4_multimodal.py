# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Annotated, Any, Literal, TypeAlias

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    BatchFeature,
    Phi4MultimodalAudioConfig,
    Phi4MultimodalConfig,
    Phi4MultimodalFeatureExtractor,
    Phi4MultimodalImageProcessorFast,
)
from transformers import Phi4MultimodalProcessor as Phi4MMProcessor
from transformers.models.phi4_multimodal.modeling_phi4_multimodal import (
    Phi4MultimodalAudioConvModule,
    Phi4MultimodalAudioNemoConvSubsampling,
    Phi4MultimodalAudioRelativeAttentionBias,
    adaptive_enc_mask,
    unfold_tensor,
)

from vllm.config import VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.distributed import (
    divide,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.model_executor.layers.activation import MulAndSilu, get_act_fn
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalDataDict,
    MultiModalFieldConfig,
    MultiModalKwargsItems,
    NestedTensors,
)
from vllm.multimodal.parse import (
    AudioProcessorItems,
    ImageEmbeddingItems,
    ImageProcessorItems,
    ImageSize,
    MultiModalDataItems,
    MultiModalDataParser,
)
from vllm.multimodal.processing import (
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
)
from vllm.multimodal.profiling import BaseDummyInputsBuilder
from vllm.sequence import IntermediateTensors
from vllm.utils.tensor_schema import TensorSchema, TensorShape

from .idefics2_vision_model import Idefics2VisionTransformer
from .interfaces import MultiModalEmbeddings, SupportsLoRA, SupportsMultiModal
from .utils import (
    AutoWeightsLoader,
    WeightsMapper,
    init_vllm_registered_model,
    maybe_prefix,
)

_AUDIO_MAX_SOUNDFILE_SIZE = 241_000


def _get_padding_size(
    orig_width: int, orig_height: int, target_height: int, target_width: int
):
    ratio_width = target_width / orig_width
    ratio_height = target_height / orig_height

    if ratio_width < ratio_height:
        padding_width = 0
        padding_height = target_height - int(orig_height * ratio_width)
    else:
        padding_width = target_width - int(orig_width * ratio_height)
        padding_height = 0
    return padding_height, padding_width


class Phi4MMProjector(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.up = ColumnParallelLinear(input_size, hidden_size)
        self.down = RowParallelLinear(hidden_size, hidden_size)
        self.act = get_act_fn("gelu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.up(x)
        x = self.act(x)
        x, _ = self.down(x)
        return x


class Phi4MMImageEmbedding(nn.Module):
    """Image embedding."""

    def __init__(self, config: Phi4MultimodalConfig):
        super().__init__()
        self.config = config
        self.layer_idx = config.vision_config.feature_layer
        self.crop_size = config.vision_config.crop_size
        self.image_dim_out = config.vision_config.hidden_size

        n_patches = config.vision_config.image_size // config.vision_config.patch_size
        if n_patches % 2 != 0:
            self.img_processor_padding = nn.ReflectionPad2d((0, 1, 0, 1))
            n_patches += 1
        self.num_img_tokens = (n_patches // 2) ** 2

        num_hidden_layers = (
            config.vision_config.num_hidden_layers + self.layer_idx + 1
            if self.layer_idx < 0
            else self.layer_idx + 1
        )
        self.img_processor = Idefics2VisionTransformer(
            config.vision_config,
            require_post_norm=False,
            num_hidden_layers_override=num_hidden_layers,
        )
        self.image_token_compression = nn.AvgPool2d(kernel_size=2, stride=2)
        self.img_projection = Phi4MMProjector(self.image_dim_out, config.hidden_size)
        self.global_img_feature_extensor = nn.Parameter(
            torch.zeros([1, 1, self.image_dim_out])
        )
        self.sub_img_feature_extensor = nn.Parameter(
            torch.zeros([1, 1, 1, self.image_dim_out])
        )

    def get_img_features(
        self,
        img_embeds: torch.FloatTensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.FloatTensor:
        img_feature = self.img_processor(
            img_embeds, patch_attention_mask=attention_mask
        )

        patch_feature = img_feature
        # reshape to 2D tensor
        width = int(math.sqrt(patch_feature.size(1)))
        patch_feature = patch_feature.view(-1, width, width, patch_feature.size(-1))
        # convert to NCHW
        patch_feature = patch_feature.permute(0, 3, 1, 2)
        if getattr(self, "img_processor_padding", None) is not None:
            patch_feature = self.img_processor_padding(patch_feature)
        patch_feature = self.image_token_compression(patch_feature)
        # convert to NHWC
        patch_feature = patch_feature.permute(0, 2, 3, 1)
        patch_feature = patch_feature.view(
            -1, patch_feature.size(1) * patch_feature.size(2), patch_feature.size(-1)
        )
        return patch_feature

    def forward(
        self,
        image_pixel_values: torch.FloatTensor,
        image_sizes: torch.Tensor | None = None,
        image_attention_mask: torch.Tensor | None = None,
    ) -> torch.FloatTensor:
        image_pixel_values = image_pixel_values.to(
            self.img_processor.embeddings.patch_embedding.weight.dtype
        )

        target_device = self.img_projection.up.bias.device
        target_dtype = self.img_projection.up.bias.dtype

        batch_size = image_pixel_values.shape[0]

        img_features = self.get_img_features(
            image_pixel_values.flatten(0, 1),
            attention_mask=image_attention_mask.flatten(0, 1).to(
                dtype=bool, device=target_device
            ),
        )
        base_feat_size = int(np.sqrt(img_features.shape[1]))
        img_features = img_features.view(
            batch_size, -1, base_feat_size**2, self.image_dim_out
        )
        image_sizes = image_sizes.view(-1, 2)

        output_imgs = []
        for idx in range(batch_size):
            height, width = image_sizes[idx]
            height_ratio = height // self.crop_size
            width_ratio = width // self.crop_size
            area_ratio = height_ratio * width_ratio

            global_img = img_features[idx, :1]
            global_img = global_img.reshape(
                1, base_feat_size, base_feat_size, self.image_dim_out
            ).contiguous()
            temporary_extensor = self.sub_img_feature_extensor.repeat(
                1, base_feat_size, 1, 1
            )
            global_img = torch.cat([global_img, temporary_extensor], dim=2).reshape(
                1, -1, self.image_dim_out
            )

            sub_img = img_features[idx, 1:]
            sub_img = sub_img[:area_ratio]
            sub_img = (
                sub_img.reshape(
                    height_ratio,
                    width_ratio,
                    base_feat_size,
                    base_feat_size,
                    self.image_dim_out,
                )
                .transpose(1, 2)
                .reshape(
                    1,
                    height_ratio * base_feat_size,
                    width_ratio * base_feat_size,
                    self.image_dim_out,
                )
                .contiguous()
            )

            if image_attention_mask is not None:
                reshaped_image_attention_mask = (
                    image_attention_mask[idx, 1 : area_ratio + 1, 0::2, 0::2]
                    .reshape(height_ratio, width_ratio, base_feat_size, base_feat_size)
                    .transpose(1, 2)
                    .reshape(
                        1, height_ratio * base_feat_size, width_ratio * base_feat_size
                    )
                )
                useful_height = int(reshaped_image_attention_mask[0, :, 0].sum().item())
                useful_width = int(reshaped_image_attention_mask[0, 0, :].sum().item())
                sub_img = sub_img[:, :useful_height, :useful_width]
                temporary_extensor = self.sub_img_feature_extensor.repeat(
                    1, useful_height, 1, 1
                )
            else:
                temporary_extensor = self.sub_img_feature_extensor.repeat(
                    1, height_ratio * base_feat_size, 1, 1
                )

            sub_img = torch.cat([sub_img, temporary_extensor], dim=2).reshape(
                1, -1, self.image_dim_out
            )

            # Merge global and sub
            output_imgs.append(
                torch.cat(
                    [sub_img, self.global_img_feature_extensor, global_img], dim=1
                )
            )

        img_set_tensor = []
        for output_img in output_imgs:
            output_img = output_img.to(device=target_device, dtype=target_dtype)
            img_feature_proj = self.img_projection(output_img)
            img_set_tensor.append(img_feature_proj.flatten(0, 1))

        return img_set_tensor


class Phi4MultimodalAudioMLP(nn.Module):
    def __init__(
        self,
        config: Phi4MultimodalAudioConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.act_fn = MulAndSilu()
        self.gate_up_proj = MergedColumnParallelLinear(
            config.hidden_size,
            [config.intermediate_size] * 2,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.layer_norm(hidden_states)
        hidden_states, _ = self.gate_up_proj(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        hidden_states, _ = self.down_proj(hidden_states)
        return hidden_states


class Phi4MultimodalAudioAttention(nn.Module):
    def __init__(
        self,
        config: Phi4MultimodalAudioConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.total_num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.total_num_heads
        if self.head_dim * self.total_num_heads != self.embed_dim:
            raise ValueError(
                "embed_dim must be divisible by num_heads "
                f"(got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim**-0.5

        self.qkv_proj = QKVParallelLinear(
            hidden_size=self.embed_dim,
            head_size=self.head_dim,
            total_num_heads=self.total_num_heads,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )

        self.o_proj = RowParallelLinear(
            input_size=self.embed_dim,
            output_size=self.embed_dim,
            quant_config=quant_config,
            prefix=f"{prefix}.out_proj",
        )

        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        self.num_heads = divide(self.total_num_heads, self.tp_size)

    def split_attn_mask(self, attention_mask: torch.Tensor) -> torch.Tensor:
        start_idx = self.num_heads * self.tp_rank
        end_idx = self.num_heads * (self.tp_rank + 1)
        return attention_mask[:, start_idx:end_idx]

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        qkv_states, _ = self.qkv_proj(hidden_states)
        query, key, value = qkv_states.chunk(3, dim=-1)

        bsz, seq_len, _ = query.size()
        query = query.view(bsz, seq_len, self.num_heads, self.head_dim)
        key = key.view(bsz, seq_len, self.num_heads, self.head_dim)
        value = value.view(bsz, seq_len, self.num_heads, self.head_dim)
        query, key, value = (x.transpose(1, 2) for x in (query, key, value))

        attention_mask = self.split_attn_mask(attention_mask)
        out = F.scaled_dot_product_attention(
            query,
            key,
            value,
            scale=self.scale,
            attn_mask=attention_mask,
        )
        out = out.transpose(1, 2).reshape(bsz, seq_len, -1)

        attn_output, _ = self.o_proj(out)

        return attn_output


class Phi4MultimodalAudioConformerEncoderLayer(nn.Module):
    def __init__(self, config: Phi4MultimodalAudioConfig):
        super().__init__()

        self.feed_forward_in = Phi4MultimodalAudioMLP(config)
        self.self_attn = Phi4MultimodalAudioAttention(config)
        self.conv = Phi4MultimodalAudioConvModule(config)
        self.feed_forward_out = Phi4MultimodalAudioMLP(config)
        self.layer_norm_att = nn.LayerNorm(config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states + 0.5 * self.feed_forward_in(hidden_states)
        hidden_states = self.layer_norm_att(residual)

        hidden_states = residual + self.self_attn(hidden_states, attention_mask)
        hidden_states = hidden_states + self.conv(hidden_states)
        hidden_states = hidden_states + 0.5 * self.feed_forward_out(hidden_states)

        out = self.layer_norm(hidden_states)

        return out


class Phi4MMAudioMeanVarianceNormLayer(nn.Module):
    """Mean/variance normalization layer.

    Will subtract mean and multiply input by inverted standard deviation.
    Typically used as a very first layer in a model.

    Args:
        config: [Phi4MultimodalAudioConfig](https://huggingface.co/docs/transformers/model_doc/phi4_multimodal#transformers.Phi4MultimodalAudioConfig)
            object containing model parameters.
    """

    def __init__(self, config: Phi4MultimodalAudioConfig):
        super().__init__()
        self.global_mean = nn.Parameter(torch.zeros(config.input_size))
        self.global_invstd = nn.Parameter(torch.ones(config.input_size))

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        """MeanVarianceNormLayer Forward

        Args:
            input_: torch.Tensor
                input tensor.
        """
        return (input_ - self.global_mean) * self.global_invstd


class Phi4MultimodalAudioModel(nn.Module):
    def __init__(self, config: Phi4MultimodalAudioConfig):
        super().__init__()
        self.config = config

        self.encoder_embedding = Phi4MMAudioMeanVarianceNormLayer(config)
        self.embed = Phi4MultimodalAudioNemoConvSubsampling(config)
        self.relative_attention_bias_layer = Phi4MultimodalAudioRelativeAttentionBias(
            config
        )
        self.encoders = nn.ModuleList(
            [
                Phi4MultimodalAudioConformerEncoderLayer(config)
                for _ in range(config.num_blocks)
            ]
        )

    def _streaming_mask(
        self,
        seq_len: int,
        batch_size: int,
        chunk_size: int,
        left_chunk: int,
    ):
        # Create mask matrix for streaming
        # S stores start index. if chunksize is 18, s is [0,18,36,....]
        chunk_start_idx = np.arange(0, seq_len, chunk_size)

        enc_streaming_mask = (
            adaptive_enc_mask(seq_len, chunk_start_idx, left_window=left_chunk)
            .unsqueeze(0)
            .expand([batch_size, -1, -1])
        )
        return enc_streaming_mask

    def forward_embeddings(
        self,
        hidden_states: torch.Tensor,
        masks: torch.Tensor,
    ):
        """Forwarding the inputs through the top embedding layers"""
        seq_len = math.ceil(hidden_states.shape[1] / self.config.time_reduction)
        if seq_len <= 0:
            raise ValueError(
                f"Sequence length after time reduction is invalid: {seq_len}."
                "Your input feature is too short."
            )

        batch_size = hidden_states.shape[0]

        enc_streaming_mask = self._streaming_mask(
            seq_len, batch_size, self.config.chunk_size, self.config.left_chunk
        )
        enc_streaming_mask = enc_streaming_mask.to(hidden_states.device)

        hidden_states, masks = self.embed(hidden_states, masks)

        streaming_mask = enc_streaming_mask
        if streaming_mask is not None and masks is not None:
            hs_mask = masks & streaming_mask
        elif masks is not None:
            hs_mask = masks
        else:
            hs_mask = streaming_mask

        return hidden_states, hs_mask, masks

    def calculate_hs_mask(
        self, hidden_states: torch.Tensor, device: torch.device, mask: torch.Tensor
    ):
        max_audio_length = hidden_states.shape[1]
        batch_size = hidden_states.shape[0]
        enc_streaming_mask = self._streaming_mask(
            max_audio_length, batch_size, self.config.chunk_size, self.config.left_chunk
        )
        enc_streaming_mask = enc_streaming_mask.to(device)
        if mask is None:
            return enc_streaming_mask

        feature_lens = mask.sum(1)
        padding_length = feature_lens
        pad_mask = torch.arange(0, max_audio_length, device=device).expand(
            padding_length.size(0), -1
        ) < padding_length.unsqueeze(1)
        pad_mask = pad_mask.unsqueeze(1)
        pad_mask = pad_mask & enc_streaming_mask
        return pad_mask

    def forward(self, hidden_states: torch.Tensor, mask: torch.Tensor | None = None):
        hidden_states = self.encoder_embedding(hidden_states)
        hidden_states, hs_mask, mask = self.forward_embeddings(hidden_states, mask)

        unfolded = False
        bs, seq_len, _ = hidden_states.shape
        max_seq_len = 500  # maximum position for absolute positional encoding
        if seq_len > max_seq_len:
            # audio sequence is longer than max_seq_len,
            # unfold it into chunks of max_seq_len
            unfolded = True
            # the unfold op will drop residual frames,
            # pad it to the multiple of max_seq_len
            if seq_len % max_seq_len > 0:
                chunk_pad_size = max_seq_len - (seq_len % max_seq_len)
            else:
                chunk_pad_size = 0
            if chunk_pad_size > 0:
                hidden_states_pad = F.pad(
                    hidden_states, (0, 0, 0, chunk_pad_size), "constant", 0
                )
                hidden_states = hidden_states_pad.to(hidden_states.device)

            hidden_states = unfold_tensor(hidden_states, max_seq_len)
            masks_unfold = None
            if mask is not None:
                # revise hs_mask here because the previous calculated hs_mask
                # did not consider extra pad
                subsampled_pad_mask = mask.squeeze(1)  # [bz, subsampled_unmask_seq_len]
                extra_padded_subsamlped_pad_mask = F.pad(
                    subsampled_pad_mask, (0, chunk_pad_size), "constant", False
                )  # extra padding to the pad mask
                extra_padded_subsamlped_pad_mask = (
                    extra_padded_subsamlped_pad_mask.unsqueeze(-1).float()
                )
                masks_unfold = unfold_tensor(
                    extra_padded_subsamlped_pad_mask, max_seq_len
                )  # unfold the pad mask like we did to the input tensor
                masks_unfold = masks_unfold.squeeze(
                    -1
                ).bool()  # unfold op does not support bool tensor
            hs_mask = self.calculate_hs_mask(
                hidden_states, hidden_states.device, masks_unfold
            )  # calculate hs_mask based on the unfolded pad mask

        relative_attention_bias = self.relative_attention_bias_layer(hidden_states)
        attention_mask = hs_mask.unsqueeze(1) + relative_attention_bias

        for layer in self.encoders:
            hidden_states = layer(hidden_states, attention_mask)

        if unfolded:
            embed_dim = hidden_states.shape[-1]
            hidden_states = hidden_states.reshape(bs, -1, embed_dim)
            # if we ever padded before unfolding, we need to remove the padding
            if chunk_pad_size > 0:
                hidden_states = hidden_states[:, :-chunk_pad_size, :]

        return hidden_states


class Phi4MMAudioEmbedding(nn.Module):
    def __init__(self, config: Phi4MultimodalConfig):
        super().__init__()
        self.config = config
        self.layer_idx = config.audio_config.feature_layer

        self.encoder = Phi4MultimodalAudioModel(config.audio_config)

        audio_config = config.audio_config
        proj_input_size = audio_config.hidden_size * audio_config.downsample_rate
        self.vision_speech_projection = Phi4MMProjector(
            proj_input_size, config.hidden_size
        )
        self.speech_projection = Phi4MMProjector(proj_input_size, config.hidden_size)

    def get_projection(
        self,
        audio_projection_mode: Literal["speech", "vision"],
    ) -> Phi4MMProjector:
        if audio_projection_mode == "speech":
            return self.speech_projection
        elif audio_projection_mode == "vision":
            return self.vision_speech_projection

    def forward(
        self,
        audio_input_features: torch.FloatTensor,
        audio_embed_sizes=None,
        audio_attention_mask=None,
        audio_projection_mode="speech",
    ) -> torch.FloatTensor:
        audio_projection = self.get_projection(audio_projection_mode)

        target_device = audio_projection.up.bias.device
        target_dtype = audio_projection.up.bias.dtype

        audio_input_features = audio_input_features.to(
            device=target_device, dtype=target_dtype
        )

        audio_encoder_hidden_states = self.encoder(
            audio_input_features, audio_attention_mask
        )
        audio_embeds = audio_projection(audio_encoder_hidden_states)

        return audio_embeds.flatten(0, 1)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
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


class Phi4MMImagePixelInputs(TensorSchema):
    """
    Dimensions:
        - bn: Batch size * number of images
        - p: Number of patches (1 + num_patches)
        - c: Number of channels (3)
        - h: Height of each image patch
        - w: Width of each image patch
        - nc: Number of crops
        - H_mask: Height of attention mask
        - W_mask: Width of attention mask
    """

    type: Literal["pixel_values"]

    pixel_values: Annotated[
        torch.Tensor | list[torch.Tensor],
        TensorShape(
            "bn", "p", 3, "h", "w", dynamic_dims={"p"}
        ),  # may be different per batch and image
    ]

    image_sizes: Annotated[
        torch.Tensor,
        TensorShape("bn", 2),  # (height, width)
    ]

    num_img_tokens: Annotated[
        list[int],
        TensorShape("bn"),
    ]

    image_attention_mask: Annotated[
        torch.Tensor,
        TensorShape("bn", "nc", 32, 32),  # H_mask, W_mask
    ]


class Phi4MMImageEmbeddingInputs(TensorSchema):
    """
    Dimensions:
        - bn: Batch size * number of images
        - f: Image feature size
        - h: Hidden size (must match language model backbone)
    """

    type: Literal["image_embeds"]

    data: Annotated[
        torch.Tensor | list[torch.Tensor],
        TensorShape("bn", "f", "h"),
    ]


class Phi4MMAudioFeatureInputs(TensorSchema):
    """
    Dimensions:
        - bn: Batch size * number of audios
        - f: Number of Mel filterbank bins (80)
        - t: Time frames (M)
    """

    type: Literal["audio_features"]

    audio_features: Annotated[
        torch.Tensor | list[torch.Tensor],
        TensorShape("bn", "t", 80, dynamic_dims={"t"}),
    ]


class Phi4MMAudioEmbeddingInputs(TensorSchema):
    """
    Dimensions:
        - b: Batch size
        - n: Number of audios
        - f: Audio feature size
        - h: Hidden size (must match language model backbone)
    """

    type: Literal["audio_embeds"]

    data: Annotated[
        NestedTensors,
        TensorShape("b", "n", "f", "h"),
    ]


Phi4MMImageInput: TypeAlias = Phi4MMImagePixelInputs | Phi4MMImageEmbeddingInputs
Phi4MMAudioInputs: TypeAlias = Phi4MMAudioFeatureInputs | Phi4MMAudioEmbeddingInputs


def cat_with_pad(tensors, dim, padding_value=0):
    """
    cat along dim, while pad to max for all other dims
    """
    ndim = tensors[0].dim()
    assert all(t.dim() == ndim for t in tensors[1:]), (
        "All tensors must have the same number of dimensions"
    )

    out_size = [max(t.shape[i] for t in tensors) for i in range(ndim)]
    out_size[dim] = sum(t.shape[dim] for t in tensors)
    output = tensors[0].new_full(out_size, padding_value)

    index = 0
    for t in tensors:
        # Create a slice list where every dimension except dim is full slice
        slices = [slice(0, t.shape[d]) for d in range(ndim)]
        # Update only the concat dimension slice
        slices[dim] = slice(index, index + t.shape[dim])

        output[slices] = t
        index += t.shape[dim]

    return output


class Phi4MMProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self) -> Phi4MultimodalConfig:
        return self.ctx.get_hf_config(Phi4MultimodalConfig)

    def get_hf_processor(self, **kwargs: object) -> Phi4MMProcessor:
        return self.ctx.get_hf_processor(Phi4MMProcessor, **kwargs)

    def get_feature_extractor(self, **kwargs: object) -> Phi4MultimodalFeatureExtractor:
        return self.get_hf_processor(**kwargs).audio_processor

    def get_image_processor(
        self,
        processor: Phi4MMProcessor | None = None,
    ) -> Phi4MultimodalImageProcessorFast:
        if processor is None:
            processor = self.get_hf_processor()
        return processor.image_processor

    def get_dynamic_hd(
        self,
        processor: Phi4MMProcessor | None = None,
    ) -> int:
        return self.get_image_processor(processor).dynamic_hd

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"audio": None, "image": None}

    def _find_target_aspect_ratio(
        self,
        orig_width: int,
        orig_height: int,
        image_size: int,
        max_num: int,
        min_num: int,
    ):
        w_crop_num = math.ceil(orig_width / float(image_size))
        h_crop_num = math.ceil(orig_height / float(image_size))
        if w_crop_num * h_crop_num > max_num:
            aspect_ratio = orig_width / orig_height

            # calculate the existing image aspect ratio
            target_ratios = set(
                (i, j)
                for i in range(1, max_num + 1)
                for j in range(1, max_num + 1)
                if i * j <= max_num and i * j >= min_num
            )
            target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

            # find the closest aspect ratio to the target
            image_processor = self.get_image_processor()
            target_aspect_ratio = image_processor.find_closest_aspect_ratio(
                aspect_ratio,
                target_ratios,
                orig_width,
                orig_height,
                image_size,
            )

            # calculate the target width and height
            target_width = image_size * target_aspect_ratio[0]
            target_height = image_size * target_aspect_ratio[1]
        else:
            target_width = image_size * w_crop_num
            target_height = image_size * h_crop_num
            target_aspect_ratio = (w_crop_num, h_crop_num)
        return target_aspect_ratio, target_height, target_width

    def _compute_num_image_tokens(
        self,
        orig_width: int,
        orig_height: int,
        dynamic_hd_size: int,
        vit_image_size: int,
        vit_patch_size: int,
        token_compression_factor: int = 2,
    ):
        """
        compute the number of tokens an image is expected to take up considering
        the image encoder architecture and exclude output features containing
        only padding pixels

        for siglip, vit_image_size=448, vit_patch_size=14, so output will be
        32x32 feature map
        NOTE right now, Phi4MM uses hard-coded token_compression_factor=2
        """
        assert vit_image_size % vit_patch_size == 0, (
            "vit_image_size must be divisible by vit_patch_size"
        )
        assert vit_image_size // vit_patch_size % token_compression_factor == 0, (
            "vit_image_size // vit_patch_size must be divisible by "
            "token_compression_factor"
        )

        target_aspect_ratio, target_height, target_width = (
            self._find_target_aspect_ratio(
                orig_width, orig_height, vit_image_size, dynamic_hd_size, min_num=1
            )
        )
        assert target_aspect_ratio[0] * vit_image_size == target_width, (
            f"{target_aspect_ratio[0]} * {vit_image_size} != {target_width}"
        )
        assert target_aspect_ratio[1] * vit_image_size == target_height, (
            f"{target_aspect_ratio[1]} * {vit_image_size} != {target_height}"
        )
        assert (
            target_height % vit_image_size == 0 and target_width % vit_image_size == 0
        )

        padding_height, padding_width = _get_padding_size(
            orig_width, orig_height, target_height, target_width
        )
        assert padding_width == 0 or padding_height == 0, (
            "padding_width or padding_height must be 0"
        )

        target_feat_width = target_width // vit_patch_size
        target_feat_height = target_height // vit_patch_size
        if padding_width >= vit_patch_size:
            assert padding_height == 0, "padding_height not 0"
            non_pad_feat_width = target_feat_width - math.floor(
                padding_width / vit_patch_size
            )
            non_pad_feat_height = target_feat_height
        elif padding_height >= vit_patch_size:
            assert padding_width == 0, "padding_width not 0"
            non_pad_feat_height = target_feat_height - math.floor(
                padding_height / vit_patch_size
            )
            non_pad_feat_width = target_feat_width
        else:
            # small padding shorter than a vit patch
            non_pad_feat_width = target_feat_width
            non_pad_feat_height = target_feat_height

        feat_width = non_pad_feat_width // token_compression_factor
        feat_height = non_pad_feat_height // token_compression_factor
        # NOTE it's possible that the non-padding feature is not divisible
        if non_pad_feat_width % token_compression_factor != 0:
            feat_width += 1
        if non_pad_feat_height % token_compression_factor != 0:
            feat_height += 1
        num_hd_patch_tokens = feat_width * feat_height
        num_hd_newline_tokens = feat_height
        vit_feature_size = vit_image_size // vit_patch_size
        num_global_image_tokens = (vit_feature_size // token_compression_factor) ** 2
        num_sep_tokens = 1
        num_global_image_newline_tokens = vit_feature_size // token_compression_factor

        return (
            num_global_image_tokens
            + num_sep_tokens
            + num_hd_patch_tokens
            + num_hd_newline_tokens
            + num_global_image_newline_tokens
        )

    def get_num_image_tokens(
        self,
        *,
        image_width: int,
        image_height: int,
        processor: Phi4MMProcessor | None = None,
    ) -> int:
        hf_config = self.get_hf_config()
        vision_config = hf_config.vision_config
        vit_image_size = vision_config.image_size
        vit_patch_size = vision_config.patch_size

        dynamic_hd_size = self.get_dynamic_hd(processor=processor)

        # we use default `token_compression_factor=2`,
        # since it's not in HF vision config.
        image_num_tokens = self._compute_num_image_tokens(
            image_width,
            image_height,
            dynamic_hd_size=dynamic_hd_size,
            vit_image_size=vit_image_size,
            vit_patch_size=vit_patch_size,
        )

        return image_num_tokens

    def get_image_size_with_most_features(
        self,
        processor: Phi4MMProcessor | None = None,
    ) -> ImageSize:
        vit_image_size = self.get_hf_config().vision_config.image_size

        max_side = vit_image_size * self.get_dynamic_hd(processor=processor)
        return ImageSize(height=max_side, width=vit_image_size)

    def get_audio_num_frames(self, audio_len: int, sr: float) -> int:
        """
        Compute the output size of the `extract_features` method.

        Args:
            audio_len (int): Length of the input waveform in samples.
            sr (float): Sampling rate of the waveform, either 16000 or 8000.

        Returns:
            tuple (int, int): Output size as (T, D), where:
                T: Number of time frames.
                D: Number of Mel filterbank bins (80).
        """

        # Resample to 16000 or 8000 if needed
        if sr > 16000:
            audio_len //= sr // 16000
        elif 8000 <= sr < 16000:
            # We'll resample to 16K from 8K
            audio_len *= 2
        elif sr < 8000:
            raise RuntimeError(f"Unsupported sample rate {sr}")

        # Spectrogram parameters for 16 kHz
        win_length = 400  # Frame length in samples
        hop_length = 160  # Frame shift in samples

        # Calculate number of frames (T)
        num_frames = (audio_len - win_length) // hop_length + 1
        if num_frames < 1:
            raise ValueError("Waveform too short for given parameters.")

        # Return time frames (T)
        return num_frames

    def _compute_audio_embed_size(self, audio_frames: int) -> int:
        """
        Compute the size of audio embeddings from the number of audio frames.
        """
        # `_compute_audio_embed_size` in audio_processor use torch for
        # computation, therefore we re-implement it to use pythonic
        # numeric computation to avoid extra tensor conversion.
        audio_processor = self.get_feature_extractor()
        audio_compression_rate = audio_processor.audio_compression_rate
        audio_downsample_rate = audio_processor.audio_downsample_rate

        integer = audio_frames // audio_compression_rate
        remainder = audio_frames % audio_compression_rate
        result = integer + int(remainder > 0)

        integer = result // audio_downsample_rate
        remainder = result % audio_downsample_rate
        result = integer + int(remainder > 0)  # qformer compression

        return result


class Phi4MMDummyInputsBuilder(BaseDummyInputsBuilder[Phi4MMProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_audios = mm_counts.get("audio", 0)
        num_images = mm_counts.get("image", 0)

        tokenizer = self.info.get_tokenizer()
        image_tokens: str = tokenizer.image_token * num_images
        audio_tokens: str = tokenizer.audio_token * num_audios

        return image_tokens + audio_tokens

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions] | None = None,
    ) -> MultiModalDataDict:
        num_audios = mm_counts.get("audio", 0)
        num_images = mm_counts.get("image", 0)

        target_width, target_height = self.info.get_image_size_with_most_features()

        image_overrides = mm_options.get("image") if mm_options else None
        audio_overrides = mm_options.get("audio") if mm_options else None

        mm_data = {
            "image": self._get_dummy_images(
                width=target_width,
                height=target_height,
                num_images=num_images,
                overrides=image_overrides,
            ),
            "audio": self._get_dummy_audios(
                length=_AUDIO_MAX_SOUNDFILE_SIZE,
                num_audios=num_audios,
                overrides=audio_overrides,
            ),
        }

        return mm_data


class Phi4MMMultiModalProcessor(BaseMultiModalProcessor[Phi4MMProcessingInfo]):
    def _get_data_parser(self) -> MultiModalDataParser:
        feature_extractor = self.info.get_feature_extractor()
        return MultiModalDataParser(target_sr=feature_extractor.sampling_rate)

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        if not mm_data:
            prompt_ids = self.info.get_tokenizer().encode(prompt)
            prompt_ids = self._apply_hf_processor_tokens_only(prompt_ids)
            return BatchFeature(dict(input_ids=[prompt_ids]), tensor_type="pt")

        audio_data = mm_data.pop("audios", [])
        if audio_data:
            mm_data["audio"] = audio_data

        processed_outputs = super()._call_hf_processor(
            prompt, mm_data, mm_kwargs, tok_kwargs
        )

        if "image_pixel_values" in processed_outputs:
            num_img_tokens = [
                self.info.get_num_image_tokens(
                    image_width=img_size[0], image_height=img_size[1]
                )
                for img_size in processed_outputs["image_sizes"]
            ]
            processed_outputs["num_img_tokens"] = num_img_tokens

        if audio_data:
            audio_features = processed_outputs["audio_input_features"]
            sr = self.info.get_feature_extractor(**mm_kwargs).sampling_rate
            feature_sizes = [
                self.info.get_audio_num_frames(len(audio), sr) for audio in audio_data
            ]
            processed_outputs["audio_input_features"] = [
                audio_features[idx, :size] for idx, size in enumerate(feature_sizes)
            ]

        return processed_outputs

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return dict(
            image_pixel_values=MultiModalFieldConfig.batched("image"),
            image_attention_mask=MultiModalFieldConfig.batched("image"),
            image_sizes=MultiModalFieldConfig.batched("image"),
            num_img_tokens=MultiModalFieldConfig.batched("image"),
            audio_input_features=MultiModalFieldConfig.batched("audio"),
        )

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        tokenizer = self.info.get_tokenizer()
        image_token_id: int = tokenizer.vocab[tokenizer.image_token]
        audio_token_id: int = tokenizer.vocab[tokenizer.audio_token]

        hf_processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        audio_processor = self.info.get_feature_extractor(**hf_processor_mm_kwargs)

        def get_image_replacement_phi4mm(item_idx: int):
            images = mm_items.get_items(
                "image", (ImageEmbeddingItems, ImageProcessorItems)
            )

            if isinstance(images, ImageEmbeddingItems):
                num_image_tokens = images.get_feature_size(item_idx)
            else:
                image_size = images.get_image_size(item_idx)
                num_image_tokens = self.info.get_num_image_tokens(
                    image_width=image_size.width,
                    image_height=image_size.height,
                    processor=hf_processor,
                )

            return [image_token_id] * num_image_tokens

        def get_audio_replacement_phi4mm(item_idx: int):
            audios = mm_items.get_items("audio", AudioProcessorItems)
            # TODO(Isotr0py): support embedding inputs
            audio_len = audios.get_audio_length(item_idx)
            audio_frames = self.info.get_audio_num_frames(
                audio_len, audio_processor.sampling_rate
            )
            audio_embed_size = self.info._compute_audio_embed_size(audio_frames)

            return [audio_token_id] * audio_embed_size

        return [
            PromptReplacement(
                modality="audio",
                target=[audio_token_id],
                replacement=get_audio_replacement_phi4mm,
            ),
            PromptReplacement(
                modality="image",
                target=[image_token_id],
                replacement=get_image_replacement_phi4mm,
            ),
        ]


@MULTIMODAL_REGISTRY.register_processor(
    Phi4MMMultiModalProcessor,
    info=Phi4MMProcessingInfo,
    dummy_inputs=Phi4MMDummyInputsBuilder,
)
class Phi4MultimodalForCausalLM(nn.Module, SupportsLoRA, SupportsMultiModal):
    """
    Implements the Phi-4-multimodal-instruct model in vLLM.
    """

    merge_by_field_config = True

    packed_modules_mapping = {
        "qkv_proj": [
            "qkv_proj",
        ],
        "gate_up_proj": [
            "gate_up_proj",
        ],
    }

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            # Multimodal embedding
            "model.embed_tokens_extend.": "",
            # LLM backbone
            "model.": "language_model.model.",
        },
        orig_to_new_substr={
            # projection
            ".img_projection_": ".img_projection.",
            ".up_proj_for_speech.": ".speech_projection.up.",
            ".up_proj_for_vision_speech.": ".vision_speech_projection.up.",
            ".down_proj_for_speech.": ".speech_projection.down.",
            ".down_proj_for_vision_speech.": ".vision_speech_projection.down.",
        },
    )

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return "<|image|>"
        if modality.startswith("audio"):
            return "<|audio|>"

        raise ValueError("Only image or audio modality is supported")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        multimodal_config = vllm_config.model_config.multimodal_config
        self.config = config
        self.multimodal_config = multimodal_config

        # TODO: Optionally initializes these for supporting input embeddings.
        self.image_embed = Phi4MMImageEmbedding(
            config,
            # prefix=maybe_prefix(prefix, "image_embed"),
        )
        self.audio_embed = Phi4MMAudioEmbedding(
            config,
            # prefix=maybe_prefix(prefix, "audio_embed"),
        )

        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "language_model"),
            architectures=["Phi3ForCausalLM"],
        )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

    def _parse_and_validate_audio_input(
        self, **kwargs: object
    ) -> Phi4MMAudioInputs | None:
        """
        Parse and validate the audio input to the model.  This handles both
        audio features and audio embeddings, but only the former is used for
        now.

        Args:
            kwargs (object): Keyword arguments.

        Returns:
            Optional[Phi4MMAudioInputs]: Parsed and validated audio inputs.
        """
        audio_features = kwargs.pop("audio_input_features", None)
        audio_embeds = kwargs.pop("audio_embeds", None)

        if audio_features is None and audio_embeds is None:
            return None

        if audio_features is not None:
            return Phi4MMAudioFeatureInputs(
                type="audio_features",
                audio_features=audio_features,
            )

        if audio_embeds is not None:
            return Phi4MMAudioEmbeddingInputs(type="audio_embeds", data=audio_embeds)

        raise AssertionError("This line should be unreachable.")

    def _process_audio_input(
        self, audio_input: Phi4MMAudioInputs, audio_projection_mode: str
    ) -> NestedTensors:
        """
        Create the audio embeddings from the audio input, where the audio input
        is pairs of audio features and audio embed lengths.  The audio input is
        created by `input_mapper_for_phi4mm_audio`.

        Args:
            audio_input (Phi4MMAudioInputs): Audio input.

        Returns:
            NestedTensors: Audio embeddings
        """
        if audio_input["type"] == "audio_embeds":
            return audio_input["data"]

        audio_features = audio_input["audio_features"]
        # (e.g. multiple examples) and the second dim is the multi-audio dim
        # (e.g. multiple audios in the same example)

        dtype = next(self.audio_embed.parameters()).dtype
        audio_embeds = [
            self.audio_embed(
                features.unsqueeze(0).to(dtype),
                audio_projection_mode=audio_projection_mode,
            )
            for features in audio_features
        ]
        return audio_embeds

    def _parse_and_validate_image_input(
        self, **kwargs: object
    ) -> Phi4MMImagePixelInputs | None:
        pixel_values = kwargs.get("image_pixel_values")
        if pixel_values is None:
            return None

        image_sizes = kwargs.get("image_sizes")
        image_attention_mask = kwargs.get("image_attention_mask")
        num_img_tokens = kwargs.get("num_img_tokens")
        assert (
            image_sizes is not None
            and image_attention_mask is not None
            and num_img_tokens is not None
        ), "Missing image inputs"

        return Phi4MMImagePixelInputs(
            type="pixel_values",
            pixel_values=pixel_values,
            image_sizes=image_sizes,
            image_attention_mask=image_attention_mask,
            num_img_tokens=num_img_tokens,
        )

    def _parse_and_validate_multimodal_inputs(self, **kwargs: object) -> dict:
        modalities = {}

        # Preserve the order of modalities if there are multiple of them
        # from the order of kwargs.
        for input_key in kwargs:
            if (
                input_key in ("image_pixel_values", "image_embeds")
                and "images" not in modalities
            ):
                modalities["images"] = self._parse_and_validate_image_input(**kwargs)
            if (
                input_key in ("audio_input_features", "audio_embeds")
                and "audios" not in modalities
            ):
                modalities["audios"] = self._parse_and_validate_audio_input(**kwargs)

        return modalities

    def _process_image_input(
        self, image_input: Phi4MMImagePixelInputs
    ) -> list[torch.Tensor]:
        if image_input["type"] == "image_embeds":
            image_embeds = image_input["image_embeds"].type(self.visual.dtype)
        else:
            dtype = next(self.image_embed.parameters()).dtype
            pixel_values = image_input["pixel_values"].to(dtype)
            image_sizes = image_input["image_sizes"]
            image_attention_mask = image_input["image_attention_mask"]
            image_embeds = self.image_embed(
                pixel_values, image_sizes, image_attention_mask
            )
        return image_embeds

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        modalities = self._parse_and_validate_multimodal_inputs(**kwargs)
        if not modalities:
            return []

        # The result multimodal_embeddings is tuple of tensors, with each
        # tensor corresponding to a multimodal data item (image or video).
        multimodal_embeddings: tuple[torch.Tensor, ...] = ()

        # NOTE: It is important to iterate over the keys in this dictionary
        # to preserve the order of the modalities.
        audio_projection_mode = "speech"
        for modality in modalities:
            # make sure process images first
            if modality == "images":
                audio_projection_mode = "vision"
                image_input = modalities["images"]
                image_embeddings = self._process_image_input(image_input)
                multimodal_embeddings += tuple(image_embeddings)
            if modality == "audios":
                audio_input = modalities["audios"]
                audio_embeddings = self._process_audio_input(
                    audio_input, audio_projection_mode=audio_projection_mode
                )
                multimodal_embeddings += tuple(audio_embeddings)

        return multimodal_embeddings

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        if intermediate_tensors is not None:
            inputs_embeds = None

        hidden_states = self.language_model(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        return self.language_model.compute_logits(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

    def get_mm_mapping(self) -> MultiModelKeys:
        """
        Get the module prefix in multimodal models
        """
        return MultiModelKeys.from_string_field(
            language_model="language_model.",
            connector=[
                "img_projection",
                "vision_speech_projection",
                "speech_projection",
            ],
            tower_model=["image_embed", "audio_embed"],
        )

    def get_language_model(self) -> torch.nn.Module:
        return self.language_model
