# SPDX-License-Identifier: Apache-2.0
import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Dict, List, Literal, Optional, Tuple, TypedDict, Union

import numpy as np
import torch
import torch.nn as nn
from transformers import (BatchFeature, Phi4MultimodalVisionConfig, Phi4MultimodalAudioConfig, Phi4MultimodalConfig,
                          SequenceFeatureExtractor, SiglipVisionConfig)
from transformers import Phi4MultimodalProcessor as Phi4MMProcessor

from vllm.config import VllmConfig
from vllm.distributed import get_pp_group
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE, ParallelLMHead)
from vllm.model_executor.models.llama import LlamaModel
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (MultiModalDataDict, MultiModalFieldConfig,
                                    MultiModalKwargs, NestedTensors)
from vllm.multimodal.parse import (AudioProcessorItems, ImageEmbeddingItems,
                                   ImageProcessorItems, ImageSize,
                                   MultiModalDataItems, MultiModalDataParser)
from vllm.multimodal.processing import (BaseMultiModalProcessor,
                                        BaseProcessingInfo, PromptReplacement,
                                        PromptUpdate)
from vllm.multimodal.profiling import BaseDummyInputsBuilder
from vllm.sequence import IntermediateTensors
from vllm.utils import is_list_of

from .idefics2_vision_model import Idefics2VisionTransformer
from .interfaces import MultiModalEmbeddings, SupportsLoRA, SupportsMultiModal
from .utils import (AutoWeightsLoader, WeightsMapper, flatten_bn, maybe_prefix,
                    merge_multimodal_embeddings)

# <|endoftext10|> (see vocab.json in hf model)
_IMAGE_PLACEHOLDER_TOKEN_ID = 200010
# <|endoftext11|>
_AUDIO_PLACEHOLDER_TOKEN_ID = 200011

_AUDIO_MAX_SOUNDFILE_SIZE = 241_000

SIGLIP_NAME = "siglip-so400m-patch14-448"
VISION_ENCODER_TO_PROCESSING_CONFIG = {
    'siglip-so400m-patch14-448': {
        'vit_image_size': 448,
        'vit_patch_size': 14,
        'token_compression_factor': 2,
    },
}


def _get_padding_size(orig_width: int, orig_height: int, target_height: int,
                      target_width: int):
    ratio_width = target_width / orig_width
    ratio_height = target_height / orig_height

    if ratio_width < ratio_height:
        padding_width = 0
        padding_height = target_height - int(orig_height * ratio_width)
    else:
        padding_width = target_width - int(orig_width * ratio_height)
        padding_height = 0
    return padding_height, padding_width


def get_navit_vision_model(layer_idx: int = -1, **kwargs):
    vision_config = {
        "hidden_size": 1152,
        "image_size": 448,
        "intermediate_size": 4304,
        "model_type": "siglip_vision_model",
        "num_attention_heads": 16,
        "num_hidden_layers": 27,
        "patch_size": 14,
    }

    model_config = SiglipVisionConfig(**vision_config, **kwargs)
    if layer_idx < 0:
        num_hidden_layers = model_config.num_hidden_layers \
            + layer_idx + 1
    else:
        num_hidden_layers = layer_idx + 1

    vision_model = Idefics2VisionTransformer(
        config=model_config,
        require_post_norm=False,
        num_hidden_layers_override=num_hidden_layers,
    )

    return vision_model


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

        self.img_processor = get_navit_vision_model(layer_idx=self.layer_idx)
        self.image_token_compression = nn.AvgPool2d(kernel_size=2, stride=2)
        self.img_projection_up = nn.Linear(self.image_dim_out, config.hidden_size)
        self.img_projection_down = nn.Linear(config.hidden_size, config.hidden_size)
        self.global_img_feature_extensor = nn.Parameter(torch.zeros([1, 1, self.image_dim_out]))
        self.sub_img_feature_extensor = nn.Parameter(torch.zeros([1, 1, 1, self.image_dim_out]))

    def get_img_features(self, img_embeds: torch.FloatTensor, attention_mask=None) -> torch.FloatTensor:
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
        patch_feature = patch_feature.view(-1, patch_feature.size(1) * patch_feature.size(2), patch_feature.size(-1))
        return patch_feature

    def forward(
        self,
        image_pixel_values: torch.FloatTensor,
        image_sizes: Optional[torch.Tensor] = None,
        image_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        image_pixel_values = image_pixel_values.to(self.img_processor.embeddings.patch_embedding.weight.dtype)

        target_device = self.img_projection_up.bias.device
        target_dtype = self.img_projection_up.bias.dtype

        batch_size = image_pixel_values.shape[0]

        img_features = self.get_img_features(
            image_pixel_values.flatten(0, 1),
            attention_mask=image_attention_mask.flatten(0, 1).to(dtype=bool, device=target_device),
        )
        base_feat_size = int(np.sqrt(img_features.shape[1]))
        img_features = img_features.view(batch_size, -1, base_feat_size**2, self.image_dim_out)
        image_sizes = image_sizes.view(-1, 2)

        output_imgs = []
        for idx in range(batch_size):
            height, width = image_sizes[idx]
            height_ratio = height // self.crop_size
            width_ratio = width // self.crop_size
            area_ratio = height_ratio * width_ratio

            global_img = img_features[idx, :1]
            global_img = global_img.reshape(1, base_feat_size, base_feat_size, self.image_dim_out).contiguous()
            temporary_extensor = self.sub_img_feature_extensor.repeat(1, base_feat_size, 1, 1)
            global_img = torch.cat([global_img, temporary_extensor], dim=2).reshape(1, -1, self.image_dim_out)

            sub_img = img_features[idx, 1:]
            sub_img = sub_img[:area_ratio]
            sub_img = (
                sub_img.reshape(height_ratio, width_ratio, base_feat_size, base_feat_size, self.image_dim_out)
                .transpose(1, 2)
                .reshape(1, height_ratio * base_feat_size, width_ratio * base_feat_size, self.image_dim_out)
                .contiguous()
            )

            if image_attention_mask is not None:
                reshaped_image_attention_mask = (
                    image_attention_mask[idx, 1 : area_ratio + 1, 0::2, 0::2]
                    .reshape(height_ratio, width_ratio, base_feat_size, base_feat_size)
                    .transpose(1, 2)
                    .reshape(1, height_ratio * base_feat_size, width_ratio * base_feat_size)
                )
                useful_height = int(reshaped_image_attention_mask[0, :, 0].sum().item())
                useful_width = int(reshaped_image_attention_mask[0, 0, :].sum().item())
                sub_img = sub_img[:, :useful_height, :useful_width]
                temporary_extensor = self.sub_img_feature_extensor.repeat(1, useful_height, 1, 1)
            else:
                temporary_extensor = self.sub_img_feature_extensor.repeat(1, height_ratio * base_feat_size, 1, 1)

            sub_img = torch.cat([sub_img, temporary_extensor], dim=2).reshape(1, -1, self.image_dim_out)

            # Merge global and sub
            output_imgs.append(torch.cat([sub_img, self.global_img_feature_extensor, global_img], dim=1))

        img_set_tensor = []
        for output_img in output_imgs:
            output_img = output_img.to(device=target_device, dtype=target_dtype)
            img_feature_proj = self.img_projection_up(output_img)
            img_feature_proj = nn.functional.gelu(img_feature_proj)
            img_feature_proj = self.img_projection_down(img_feature_proj)
            img_set_tensor.append(img_feature_proj)

        return img_set_tensor


########################################################## AUDIO #############################################


class Phi4MultimodalAudioMLP(nn.Module):
    def __init__(self, config: Phi4MultimodalAudioConfig):
        super().__init__()
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.act_fn = ACT2FN[config.activation]
        self.gate_up_proj = nn.Linear(config.hidden_size, config.intermediate_size * 2)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states):
        hidden_states = self.layer_norm(hidden_states)
        up_states = self.gate_up_proj(hidden_states)
        up_states, gate = up_states.chunk(2, dim=-1)
        up_states = up_states * self.act_fn(gate)
        up_states = self.dropout(up_states)
        hidden_states = self.down_proj(up_states)
        out = self.dropout(hidden_states)

        return out


class Phi4MultimodalAudioAttention(nn.Module):
    def __init__(self, config: Phi4MultimodalAudioConfig):
        super().__init__()
        self.config = config
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.dropout_rate
        self.is_causal = True

        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs,
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        attention_interface: Callable = simple_eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, _ = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output


class Phi4MultimodalAudioDepthWiseSeperableConv1d(nn.Module):
    def __init__(self, config: Phi4MultimodalAudioConfig, padding: int = 0):
        super().__init__()
        self.dw_conv = nn.Conv1d(
            config.hidden_size,
            config.hidden_size * config.depthwise_multiplier,
            config.kernel_size,
            1,
            padding=padding,
            groups=config.hidden_size,
        )
        self.pw_conv = nn.Conv1d(
            config.hidden_size * config.depthwise_multiplier, config.depthwise_seperable_out_channel, 1, 1, 0
        )

    def forward(self, hidden_states):
        return self.pw_conv(self.dw_conv(hidden_states))


class Phi4MultimodalAudioGluPointWiseConv(nn.Module):
    def __init__(self, config: Phi4MultimodalAudioConfig):
        super().__init__()
        self.config = config
        self.output_dim = config.ext_pw_out_channel

        self.ext_pw_conv_1d = nn.Conv1d(config.hidden_size, config.ext_pw_out_channel * 2, kernel_size=1, stride=1)
        self.glu_act = ACT2FN[config.conv_glu_type]
        self.b1 = nn.Parameter(torch.zeros(1, config.ext_pw_out_channel, 1))
        self.b2 = nn.Parameter(torch.zeros(1, config.ext_pw_out_channel, 1))

    def forward(self, hidden_states):
        # we assume the input always has the #channel (#dim) in the last dimension of the
        # tensor, so need to switch the dimension first for 1D-Conv case
        hidden_states = hidden_states.permute([0, 2, 1])
        hidden_states = self.ext_pw_conv_1d(hidden_states)
        out = hidden_states[:, 0 : self.output_dim, :] + self.b1
        out = out * self.glu_act(hidden_states[:, self.output_dim : self.output_dim * 2, :] + self.b2)
        return out.permute([0, 2, 1])


class Phi4MultimodalAudioConvModule(nn.Module):
    def __init__(self, config: Phi4MultimodalAudioConfig):
        super().__init__()
        self.config = config
        self.kernel_size = config.kernel_size

        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.glu = Phi4MultimodalAudioGluPointWiseConv(config)
        self.dw_sep_conv_1d = Phi4MultimodalAudioDepthWiseSeperableConv1d(config, padding=config.kernel_size - 1)
        self.act = ACT2FN[config.conv_activation]
        self.ext_pw_conv_1d = nn.Conv1d(config.hidden_size, config.ext_pw_out_channel, kernel_size=1, stride=1)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = self.glu(self.layer_norm(hidden_states))
        hidden_states = self.dw_sep_conv_1d(hidden_states.permute([0, 2, 1]))

        if self.kernel_size > 1:
            hidden_states = hidden_states[:, :, : -(self.kernel_size - 1)]

        hidden_states = self.act(hidden_states)
        hidden_states = self.ext_pw_conv_1d(hidden_states)
        out = self.dropout(hidden_states.permute([0, 2, 1]))
        return out


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
    ):
        residual = hidden_states + 0.5 * self.feed_forward_in(hidden_states)
        hidden_states = self.layer_norm_att(residual)

        hidden_states = residual + self.self_attn(hidden_states, attention_mask)
        hidden_states = hidden_states + self.conv(hidden_states)
        hidden_states = hidden_states + 0.5 * self.feed_forward_out(hidden_states)

        out = self.layer_norm(hidden_states)

        return out


class Phi4MultimodalAudioNemoConvSubsampling(torch.nn.Module):
    def __init__(self, config: Phi4MultimodalAudioConfig):
        super().__init__()
        self.subsampling_factor = config.time_reduction
        self.sampling_num = int(math.log(self.subsampling_factor, 2))
        self.act_fn = ACT2FN[config.nemo_activation]
        conv_channels = config.nemo_conv_channels

        layers = [
            nn.Conv2d(1, conv_channels, kernel_size=3, stride=2, padding=1),
            self.act_fn,
        ]
        for _ in range(self.sampling_num - 1):
            layers.extend(
                [
                    nn.Conv2d(conv_channels, conv_channels, kernel_size=3, stride=2, padding=1, groups=conv_channels),
                    nn.Conv2d(conv_channels, conv_channels, kernel_size=1, stride=1, padding=0, groups=1),
                    self.act_fn,
                ]
            )

        # Aggregate the layers
        self.conv = torch.nn.Sequential(*layers)
        self.out = torch.nn.Linear(conv_channels * config.nemo_final_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor, mask: Optional[torch.Tensor]):
        # Unsqueeze Channel Axis
        hidden_states = hidden_states.unsqueeze(1)
        hidden_states = self.conv(hidden_states)

        # Flatten Channel and Frequency Axes
        b, _, t, _ = hidden_states.size()
        hidden_states = self.out(hidden_states.transpose(1, 2).reshape(b, t, -1))

        if mask is None:
            return hidden_states, None

        max_audio_length = hidden_states.shape[1]
        feature_lens = mask.sum(1)
        padding_length = torch.ceil(feature_lens / self.subsampling_factor)
        arange_ = torch.arange(0, max_audio_length, device=hidden_states.device)
        pad_mask = arange_.expand(padding_length.size(0), -1) < padding_length.unsqueeze(1)
        return hidden_states, pad_mask.unsqueeze(1)


class Phi4MultimodalAudioRelativeAttentionBias(nn.Module):
    def __init__(self, config: Phi4MultimodalAudioConfig):
        super().__init__()

        self.max_distance = config.bias_max_distance
        self.symmetric = config.bias_symmetric
        self.num_buckets = self.max_distance
        if not config.bias_symmetric:
            self.num_buckets *= 2
        self.bias_values = nn.Embedding(self.num_buckets, config.num_attention_heads)

    def forward(self, x):
        # instantiate bias compatible with shape of x
        max_pos = x.size(1)
        context_position = torch.arange(max_pos, device=x.device, dtype=torch.long)[:, None]
        memory_position = torch.arange(max_pos, device=x.device, dtype=torch.long)[None, :]
        relative_position = memory_position - context_position
        # clipping to a maximum distance using ops that play well with ONNX export
        relative_position = relative_position.masked_fill(relative_position < -self.max_distance, -self.max_distance)
        relative_position = relative_position.masked_fill(
            relative_position > self.max_distance - 1, self.max_distance - 1
        )

        # mapping from relative position to index in the bias parameter
        bias_idx = relative_position
        bias_idx = bias_idx.abs() if self.symmetric else bias_idx + self.num_buckets // 2

        att_bias = self.bias_values(bias_idx)
        att_bias = att_bias.permute(2, 0, 1).unsqueeze(0)

        return att_bias


class Phi4MultimodalAudioMeanVarianceNormLayer(nn.Module):
    def __init__(self, config: Phi4MultimodalAudioConfig):
        super().__init__()
        self.register_buffer("global_mean", torch.zeros(config.input_size))
        self.register_buffer("global_invstd", torch.ones(config.input_size))

    def forward(self, x):
        return (x - self.global_mean) * self.global_invstd


class Phi4MultimodalAudioModel(Phi4MultimodalAudioPreTrainedModel):
    def __init__(self, config: Phi4MultimodalAudioConfig):
        super().__init__(config)
        self.config = config

        self.encoder_embedding = Phi4MultimodalAudioMeanVarianceNormLayer(config)
        self.embed = Phi4MultimodalAudioNemoConvSubsampling(config)
        self.relative_attention_bias_layer = Phi4MultimodalAudioRelativeAttentionBias(config)
        self.encoders = nn.ModuleList(
            [Phi4MultimodalAudioConformerEncoderLayer(config) for _ in range(config.num_blocks)]
        )
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def _streaming_mask(self, seq_len, batch_size, chunk_size, left_chunk):
        # Create mask matrix for streaming
        # S stores start index. if chunksize is 18, s is [0,18,36,....]
        chunk_start_idx = np.arange(0, seq_len, chunk_size)
        # avoid randomness when run evaluation or decoding
        if self.training and np.random.rand() > 0.5:
            # Either first or last chunk is not complete.
            # If only the last one is not complete, EOS is not effective
            chunk_start_idx = seq_len - chunk_start_idx
            chunk_start_idx = chunk_start_idx[::-1]
            chunk_start_idx = chunk_start_idx[:-1]
            chunk_start_idx = np.insert(chunk_start_idx, 0, 0)

        enc_streaming_mask = (
            adaptive_enc_mask(seq_len, chunk_start_idx, left_window=left_chunk)
            .unsqueeze(0)
            .expand([batch_size, -1, -1])
        )
        return enc_streaming_mask

    def forward_embeddings(self, hidden_states, masks):
        """Forwarding the inputs through the top embedding layers"""
        seq_len = math.ceil(hidden_states.shape[1] / self.config.time_reduction)
        if seq_len <= 0:
            raise ValueError(
                f"The squence length after time reduction is invalid: {seq_len}. Your input feature is too short."
            )

        batch_size = hidden_states.shape[0]

        enc_streaming_mask = self._streaming_mask(seq_len, batch_size, self.config.chunk_size, self.config.left_chunk)
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

    def calculate_hs_mask(self, hidden_states, device, mask):
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

    def forward(self, hidden_states: torch.Tensor, mask: Optional[torch.Tensor]):
        hidden_states = self.encoder_embedding(hidden_states)
        hidden_states, hs_mask, mask = self.forward_embeddings(hidden_states, mask)

        unfolded = False
        bs, seq_len, _ = hidden_states.shape
        max_seq_len = 500  # maxium position for absolute positional encoding
        if seq_len > max_seq_len:
            # audio sequence is longer than max_seq_len, unfold it into chunks of max_seq_len
            unfolded = True
            # the unfold op will drop residual frames, pad it to the multiple of max_seq_len
            if seq_len % max_seq_len > 0:
                chunk_pad_size = max_seq_len - (seq_len % max_seq_len)
            else:
                chunk_pad_size = 0
            if chunk_pad_size > 0:
                hidden_states_pad = F.pad(hidden_states, (0, 0, 0, chunk_pad_size), "constant", 0)
                hidden_states = hidden_states_pad.to(hidden_states.device)

            hidden_states = unfold_tensor(hidden_states, max_seq_len)
            masks_unfold = None
            if mask is not None:
                # revise hs_mask here because the previous calculated hs_mask did not consider extra pad
                subsampled_pad_mask = mask.squeeze(1)  # [bz, subsampled_unmask_seq_len]
                extra_padded_subsamlped_pad_mask = F.pad(
                    subsampled_pad_mask, (0, chunk_pad_size), "constant", False
                )  # extra padding to the pad mask
                extra_padded_subsamlped_pad_mask = extra_padded_subsamlped_pad_mask.unsqueeze(-1).float()
                masks_unfold = unfold_tensor(
                    extra_padded_subsamlped_pad_mask, max_seq_len
                )  # unfold the pad mask like we did to the input tensor
                masks_unfold = masks_unfold.squeeze(-1).bool()  # unfold op does not support bool tensor
            hs_mask = self.calculate_hs_mask(
                hidden_states, hidden_states.device, masks_unfold
            )  # calculate hs_mask based on the unfolded pad mask

        relative_attention_bias = self.relative_attention_bias_layer(hidden_states)
        attention_mask = hs_mask.unsqueeze(1) + relative_attention_bias

        for layer in self.encoders:
            if self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(
                    layer.__call__,
                    hidden_states,
                    attention_mask,
                )
            else:
                hidden_states = layer(hidden_states, attention_mask)

        if unfolded:
            embed_dim = hidden_states.shape[-1]
            hidden_states = hidden_states.reshape(bs, -1, embed_dim)
            # if we ever padded before unfolding, we need to remove the padding
            if chunk_pad_size > 0:
                hidden_states = hidden_states[:, :-chunk_pad_size, :]

        return hidden_states


def unfold_tensor(tensor, max_seq_len):
    """
    For a given tensor with shape of (N, T, D), if sequence length T is longer than max_seq_len,
    this function unfold it to a (NT', max_seq_len, D) where T' is T // max_seq_len.
    Args:
        tensor: N, T, D
    """
    _, _, D = tensor.shape
    tensor = tensor.transpose(-1, -2)
    # N x D x 1 x T => N x (D x max_seq_len) x T'
    tensor = F.unfold(tensor[..., None, :], kernel_size=(1, max_seq_len), stride=(1, max_seq_len))

    new_bsz, _, slen = tensor.shape
    tensor = tensor.view(new_bsz, -1, max_seq_len, slen)
    tensor = tensor.permute(0, 3, 2, 1)
    tensor = tensor.view(-1, max_seq_len, D).contiguous()
    return tensor


def adaptive_enc_mask(x_len, chunk_start_idx, left_window=0, right_window=0):
    """
    The function is very important for Transformer Transducer Streaming mode
    Args:
        xs_len (int): sequence length
        chunk_start_idx (list): first idx of each chunk, such as [0,18,36,48]. It also supports adaptive chunk size [0,10,15,45]
        left_window (int): how many left chunks can be seen
        right_window (int): how many right chunks can be seen. It is used for chunk overlap model.
        Returns:
            mask (torch.Tensor): a mask tensor for streaming model
    """
    chunk_start_idx = torch.Tensor(chunk_start_idx).long()
    start_pad = torch.nn.functional.pad(
        chunk_start_idx, (1, 0)
    )  # append 0 to the beginning, so it becomes [0, 0, 18, 36, 48]
    end_pad = torch.nn.functional.pad(
        chunk_start_idx, (0, 1), value=x_len
    )  # append x_len to the end, so it becomes [0,18,36,48, x_len]
    seq_range = torch.arange(0, x_len).unsqueeze(-1)
    idx = ((seq_range < end_pad) & (seq_range >= start_pad)).nonzero()[:, 1]
    seq_range_expand = torch.arange(0, x_len).unsqueeze(0).expand(x_len, -1)
    idx_left = idx - left_window
    idx_left[idx_left < 0] = 0
    boundary_left = start_pad[idx_left]
    mask_left = seq_range_expand >= boundary_left.unsqueeze(-1)
    idx_right = idx + right_window
    idx_right[idx_right > len(chunk_start_idx)] = len(chunk_start_idx)
    boundary_right = end_pad[idx_right]
    mask_right = seq_range_expand < boundary_right.unsqueeze(-1)
    return mask_left & mask_right


class Phi4MultimodalAudioEmbedding(nn.Module):
    def __init__(self, config: Phi4MultimodalConfig):
        super().__init__()
        self.config = config
        self.layer_idx = config.audio_config.feature_layer

        self.drop = nn.Dropout(config.embd_pdrop)
        self.encoder = Phi4MultimodalAudioModel._from_config(config.audio_config)
        self.up_proj_for_speech = nn.Linear(
            config.audio_config.hidden_size * config.audio_config.downsample_rate, config.hidden_size
        )
        self.down_proj_for_speech = nn.Linear(config.hidden_size, config.hidden_size)
        self.up_proj_for_vision_speech = nn.Linear(
            config.audio_config.hidden_size * config.audio_config.downsample_rate, config.hidden_size
        )
        self.down_proj_for_vision_speech = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.Tensor,
        audio_input_features: torch.FloatTensor,
        audio_embed_sizes=None,
        audio_attention_mask=None,
        audio_projection_mode="speech",
    ) -> torch.FloatTensor:
        with torch.no_grad():
            positions_tuple = torch.nonzero(input_ids == self.config.audio_config.audio_token_id, as_tuple=True)

        up_proj = self.up_proj_for_speech if audio_projection_mode == "speech" else self.up_proj_for_vision_speech
        down_proj = (
            self.down_proj_for_speech if audio_projection_mode == "speech" else self.down_proj_for_vision_speech
        )

        target_device = up_proj.bias.device
        target_dtype = up_proj.bias.dtype

        audio_input_features = audio_input_features.to(device=target_device, dtype=target_dtype)

        audio_encoder_hidden_states = self.encoder(audio_input_features, audio_attention_mask)
        audio_encoder_hidden_states = up_proj(audio_encoder_hidden_states)
        audio_encoder_hidden_states = nn.functional.gelu(audio_encoder_hidden_states)
        audio_embeds = down_proj(audio_encoder_hidden_states)

        merged_audio_embeds = torch.cat(
            [audio_embeds[i, : audio_embed_sizes[i], :] for i in range(len(audio_embed_sizes))], dim=0
        )
        merged_audio_embeds = merged_audio_embeds.to(dtype=inputs_embeds.dtype, device=inputs_embeds.device)
        # Temporarily disable autocast to avoid issue on bf16 tensors
        # Ref: https://github.com/pytorch/pytorch/issues/132715
        with torch.autocast(device_type=inputs_embeds.device.type, enabled=False):
            audio_embeds = inputs_embeds.index_put(
                indices=positions_tuple, values=merged_audio_embeds, accumulate=False
            )

        audio_embeds = self.drop(audio_embeds)

        return audio_embeds


class Phi4MMImagePixelInputs(TypedDict):
    type: Literal["pixel_values"]
    data: Union[torch.Tensor, List[torch.Tensor]]
    """
    Shape:
    `(batch_size * num_images, 1 + num_patches, num_channels, height, width)`

    Note that `num_patches` may be different per batch and image,
    in which case the data is passed as a list instead of a batched tensor.
    """

    image_sizes: torch.Tensor
    """
    Shape: `(batch_size * num_images, 2)`

    This should be in `(height, width)` format.
    """

    num_img_tokens: list[int]
    """Shape: `(batch_size * num_images)`"""

    image_attention_mask: torch.Tensor
    """Shape: `(batch_size * num_images, H_mask, W_mask)`"""


class Phi4MMImageEmbeddingInputs(TypedDict):
    type: Literal["image_embeds"]
    data: Union[torch.Tensor, List[torch.Tensor]]
    """Shape: `(batch_size * num_images, image_feature_size, hidden_size)`

    `hidden_size` must match the hidden size of language model backbone.
    """


class Phi4MMAudioFeatureInputs(TypedDict):
    type: Literal["audio_features"]
    data: Union[torch.Tensor, List[torch.Tensor]]
    """Shape: `(batch_size * num_audios, 80, M)"""


class Phi4MMAudioEmbeddingInputs(TypedDict):
    type: Literal["audio_embeds"]
    data: NestedTensors
    """Shape: `(batch_size, num_audios, audio_feature_size, hidden_size)"""


Phi4MMImageInput = Union[Phi4MMImagePixelInputs, Phi4MMImageEmbeddingInputs]
Phi4MMAudioInputs = Union[Phi4MMAudioFeatureInputs, Phi4MMAudioEmbeddingInputs]


def cat_with_pad(tensors, dim, padding_value=0):
    """
    cat along dim, while pad to max for all other dims
    """
    ndim = tensors[0].dim()
    assert all(
        t.dim() == ndim for t in
        tensors[1:]), "All tensors must have the same number of dimensions"

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

    def get_hf_processor(
        self,
        *,
        dynamic_hd: Optional[int] = None,
        **kwargs: object,
    ) -> Phi4MMProcessor:
        if dynamic_hd is not None:
            kwargs["dynamic_hd"] = dynamic_hd

        return self.ctx.get_hf_processor(**kwargs)

    @property
    def image_tokens(self) -> list[str]:
        return [f"<|image_{i+1}|>" for i in range(100)]

    @property
    def audio_tokens(self) -> list[str]:
        return [f"<|audio_{i+1}|>" for i in range(100)]

    def get_dynamic_hd(
        self,
        processor: Optional[Phi4MMProcessor] = None,
    ) -> int:
        if processor is None:
            processor = self.get_hf_processor()
        image_processor = processor.image_processor
        return image_processor.dynamic_hd

    def get_feature_extractor(self) -> SequenceFeatureExtractor:
        return self.get_hf_processor().audio_processor

    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
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
            target_ratios = set((i, j) for i in range(1, max_num + 1)
                                for j in range(1, max_num + 1)
                                if i * j <= max_num and i * j >= min_num)
            target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

            # find the closest aspect ratio to the target
            image_processor = self.get_hf_processor().image_processor
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
            "vit_image_size must be divisible by vit_patch_size")
        assert (vit_image_size // vit_patch_size %
                token_compression_factor == 0), (
                    "vit_image_size // vit_patch_size must be divisible by "
                    "token_compression_factor")

        target_aspect_ratio, target_height, target_width = (
            self._find_target_aspect_ratio(orig_width,
                                           orig_height,
                                           vit_image_size,
                                           dynamic_hd_size,
                                           min_num=1))
        assert target_aspect_ratio[0] * vit_image_size == target_width, (
            f"{target_aspect_ratio[0]} * {vit_image_size} != {target_width}")
        assert target_aspect_ratio[1] * vit_image_size == target_height, (
            f"{target_aspect_ratio[1]} * {vit_image_size} != {target_height}")
        assert (target_height % vit_image_size == 0
                and target_width % vit_image_size == 0)

        padding_height, padding_width = _get_padding_size(
            orig_width, orig_height, target_height, target_width)
        assert padding_width == 0 or padding_height == 0, \
            "padding_width or padding_height must be 0"

        target_feat_width = target_width // vit_patch_size
        target_feat_height = target_height // vit_patch_size
        if padding_width >= vit_patch_size:
            assert padding_height == 0, "padding_height not 0"
            non_pad_feat_width = target_feat_width - math.floor(
                padding_width / vit_patch_size)
            non_pad_feat_height = target_feat_height
        elif padding_height >= vit_patch_size:
            assert padding_width == 0, "padding_width not 0"
            non_pad_feat_height = target_feat_height - math.floor(
                padding_height / vit_patch_size)
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
        num_global_image_tokens = (vit_feature_size //
                                   token_compression_factor)**2
        num_sep_tokens = 1
        num_global_image_newline_tokens = \
            vit_feature_size // token_compression_factor

        return (num_global_image_tokens + num_sep_tokens +
                num_hd_patch_tokens + num_hd_newline_tokens +
                num_global_image_newline_tokens)

    def get_num_image_tokens(
        self,
        *,
        image_width: int,
        image_height: int,
        processor: Optional[Phi4MMProcessor] = None,
    ) -> int:
        hf_config = self.get_hf_config()
        vision_encoder_name = hf_config.img_processor
        if vision_encoder_name is None:
            vision_encoder_name = SIGLIP_NAME
        prepro_config = VISION_ENCODER_TO_PROCESSING_CONFIG[
            vision_encoder_name]
        vit_image_size = prepro_config['vit_image_size']
        vit_patch_size = prepro_config['vit_patch_size']
        token_compression_factor = prepro_config['token_compression_factor']

        dynamic_hd_size = self.get_dynamic_hd(processor=processor)

        image_num_tokens = self._compute_num_image_tokens(
            image_width,
            image_height,
            dynamic_hd_size=dynamic_hd_size,
            vit_image_size=vit_image_size,
            vit_patch_size=vit_patch_size,
            token_compression_factor=token_compression_factor,
        )

        return image_num_tokens

    def get_image_size_with_most_features(
        self,
        processor: Optional[Phi4MMProcessor] = None,
    ) -> ImageSize:
        hf_config = self.get_hf_config()
        vision_encoder_name = hf_config.img_processor
        if vision_encoder_name is None:
            vision_encoder_name = SIGLIP_NAME
        prepro_config = VISION_ENCODER_TO_PROCESSING_CONFIG[
            vision_encoder_name]
        vit_image_size = prepro_config['vit_image_size']

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
        Compute the audio embedding size based on the audio frames and
        compression rate.
        """
        hf_config = self.get_hf_config()
        compression_rate = hf_config.embd_layer['audio_embd_layer'][
            'compression_rate']
        # NOTE: this is a hard-coded value but might be configurable
        # in the future
        qformer_compression_rate = 1
        integer = audio_frames // compression_rate
        remainder = audio_frames % compression_rate

        result = integer if remainder == 0 else integer + 1

        integer = result // qformer_compression_rate
        remainder = result % qformer_compression_rate
        # qformer compression
        result = integer if remainder == 0 else integer + 1

        return result


class Phi4MMDummyInputsBuilder(BaseDummyInputsBuilder[Phi4MMProcessingInfo]):

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_audios = mm_counts.get("audio", 0)
        num_images = mm_counts.get("image", 0)

        image_tokens: list[str] = self.info.image_tokens[:num_images]
        audio_tokens: list[str] = self.info.audio_tokens[:num_audios]

        return "".join(image_tokens + audio_tokens)

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> MultiModalDataDict:
        num_audios = mm_counts.get("audio", 0)
        num_images = mm_counts.get("image", 0)

        target_width, target_height = \
            self.info.get_image_size_with_most_features()

        mm_data = {
            "image":
            self._get_dummy_images(width=target_width,
                                   height=target_height,
                                   num_images=num_images),
            "audio":
            self._get_dummy_audios(length=_AUDIO_MAX_SOUNDFILE_SIZE,
                                   num_audios=num_audios),
        }

        return mm_data


class Phi4MMMultiModalProcessor(BaseMultiModalProcessor[Phi4MMProcessingInfo]):

    def _get_data_parser(self) -> MultiModalDataParser:
        feature_extractor = self.info.get_feature_extractor()
        return MultiModalDataParser(target_sr=feature_extractor.sampling_rate,
                                    audio_resample_method="scipy")

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        if not mm_data:
            prompt_ids = self.info.get_tokenizer().encode(prompt)
            prompt_ids = self._apply_hf_processor_tokens_only(prompt_ids)
            return BatchFeature(dict(input_ids=[prompt_ids]), tensor_type="pt")

        sr = self.info.get_feature_extractor().sampling_rate
        if (audio_data := mm_data.get("audios", [])):
            mm_data['audios'] = [(data, sr) for data in audio_data]

        processed_outputs = super()._call_hf_processor(prompt, mm_data,
                                                       mm_kwargs)

        num_img_tokens = [
            self.info.get_num_image_tokens(image_width=img_size[0],
                                           image_height=img_size[1])
            for img_size in processed_outputs["image_sizes"]
        ]
        processed_outputs["num_img_tokens"] = num_img_tokens

        audio_features = processed_outputs['input_audio_embeds']
        feature_sizes = [
            self.info.get_audio_num_frames(len(audio), sr)
            for audio in audio_data
        ]
        processed_outputs['input_audio_embeds'] = [
            audio_features[idx, :size]
            for idx, size in enumerate(feature_sizes)
        ]

        return processed_outputs

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return dict(
            input_image_embeds=MultiModalFieldConfig.batched("image"),
            image_attention_mask=MultiModalFieldConfig.batched("image"),
            image_sizes=MultiModalFieldConfig.batched("image"),
            num_img_tokens=MultiModalFieldConfig.batched("image"),
            input_audio_embeds=MultiModalFieldConfig.batched("audio"),
        )

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs: MultiModalKwargs,
    ) -> Sequence[PromptUpdate]:
        image_tokens: list[str] = self.info.image_tokens  # type: ignore
        audio_tokens: list[str] = self.info.audio_tokens  # type: ignore
        feature_extractor = self.info.get_feature_extractor()
        hf_processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)

        def get_image_replacement_phi4mm(item_idx: int):
            images = mm_items.get_items(
                "image", (ImageEmbeddingItems, ImageProcessorItems))

            if isinstance(images, ImageEmbeddingItems):
                num_image_tokens = images.get_feature_size(item_idx)
            else:
                image_size = images.get_image_size(item_idx)
                num_image_tokens = self.info.get_num_image_tokens(
                    image_width=image_size.width,
                    image_height=image_size.height,
                    processor=hf_processor,
                )

            image_tokens = [_IMAGE_PLACEHOLDER_TOKEN_ID] * num_image_tokens

            return image_tokens

        def get_audio_replacement_phi4mm(item_idx: int):
            audios = mm_items.get_items("audio", AudioProcessorItems)
            # TODO(Isotr0py): support embedding inputs
            audio_len = audios.get_audio_length(item_idx)
            audio_frames = self.info.get_audio_num_frames(
                audio_len, feature_extractor.sampling_rate)
            audio_embed_size = self.info._compute_audio_embed_size(
                audio_frames)

            audio_tokens = [_AUDIO_PLACEHOLDER_TOKEN_ID] * audio_embed_size

            return audio_tokens

        num_images = mm_items.get_count("image", strict=False)
        num_audios = mm_items.get_count("audio", strict=False)

        image_repl = [
            PromptReplacement(
                modality="image",
                target=image_token,
                replacement=get_image_replacement_phi4mm,
            ) for image_token in image_tokens[:num_images]
        ]
        audio_repl = [
            PromptReplacement(
                modality="audio",
                target=audio_token,
                replacement=get_audio_replacement_phi4mm,
            ) for audio_token in audio_tokens[:num_audios]
        ]
        return image_repl + audio_repl


@MULTIMODAL_REGISTRY.register_processor(
    Phi4MMMultiModalProcessor,
    info=Phi4MMProcessingInfo,
    dummy_inputs=Phi4MMDummyInputsBuilder,
)
class Phi4MMForCausalLM(nn.Module, SupportsLoRA, SupportsMultiModal):
    """
    Implements the Phi-4-multimodal-instruct model in vLLM.
    """
    packed_modules_mapping = {
        "qkv_proj": [
            "qkv_proj",
        ],
        "gate_up_proj": [
            "gate_up_proj",
        ],
    }

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_substr={
            "base_layer.": "",
        },
        orig_to_new_prefix={
            "model.embed_tokens_extend.audio_embed.audio_projection.vision.":
            "embed_tokens_extend.audio_projection_for_vision.",
            "model.embed_tokens_extend.audio_embed.audio_projection.speech.":
            "embed_tokens_extend.audio_projection.",
            "model.embed_tokens_extend.audio_embed.": "embed_tokens_extend.",
            "model.embed_tokens_extend.image_embed.": "vision_encoder.",
        },
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        multimodal_config = vllm_config.model_config.multimodal_config
        assert multimodal_config, "multimodal_config is required"
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config

        self.config = config
        self.multimodal_config = multimodal_config
        self.quant_config = quant_config
        self.lora_config = lora_config

        # Tensor/Pipeline parallel not supported for now.
        assert get_pp_group(
        ).world_size == 1, "pipeline parallel is not supported"

        self.vision_encoder = Phi4MMImageEncoder(
            config,
            quant_config,
            prefix="model.vision_embed_tokens",
            model_dir=config._name_or_path)

        if isinstance(config.embd_layer["audio_embd_layer"], dict):
            embedding_config = {
                "embedding_cls":
                config.embd_layer["audio_embd_layer"]["embedding_cls"],
                **config.embd_layer["audio_embd_layer"],
            }
        else:
            embedding_config = {
                "embedding_cls": self.config.embd_layer["embedding_cls"]
            }

        self.embed_tokens_extend = AudioEmbedding(config, **embedding_config)
        self.model = LlamaModel(vllm_config=vllm_config,
                                prefix=maybe_prefix(prefix, "model"))

        self.unpadded_vocab_size = config.vocab_size
        if lora_config:
            self.unpadded_vocab_size += lora_config.lora_extra_vocab_size
        self.lm_head = ParallelLMHead(
            self.unpadded_vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
            padding_size=(
                DEFAULT_VOCAB_PADDING_SIZE
                # We need bigger padding if using lora for kernel
                # compatibility
                if not lora_config else lora_config.lora_vocab_padding_size),
            quant_config=quant_config,
        )
        if config.tie_word_embeddings:
            self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)
        logit_scale = getattr(config, "logit_scale", 1.0)
        self.logits_processor = LogitsProcessor(self.unpadded_vocab_size,
                                                config.vocab_size, logit_scale)

    def _parse_and_validate_audio_input(
            self, **kwargs: object) -> Optional[Phi4MMAudioInputs]:
        """
        Parse and validate the audio input to the model.  This handles both 
        audio features and audio embeddings, but only the former is used for
        now.

        Args:
            kwargs (object): Keyword arguments.

        Returns:
            Optional[Phi4MMAudioInputs]: Parsed and validated audio inputs.
        """
        audio_features = kwargs.pop("input_audio_embeds", None)
        audio_embeds = kwargs.pop("audio_embeds", None)

        if audio_features is None and audio_embeds is None:
            return None

        if audio_features is not None:
            if not isinstance(audio_features, (torch.Tensor, list)):
                raise ValueError("Incorrect type of audio features. "
                                 f"Got type: {type(audio_features)}")

            return Phi4MMAudioFeatureInputs(type="audio_features",
                                            data=flatten_bn(audio_features))

        if audio_embeds is not None:
            if not isinstance(audio_embeds, (torch.Tensor, list)):
                raise ValueError("Incorrect type of audio embeds. "
                                 f"Got type: {type(audio_embeds)}")

            return Phi4MMAudioEmbeddingInputs(type="audio_embeds",
                                              data=audio_embeds)

        raise AssertionError("This line should be unreachable.")

    def _process_audio_input(self, audio_input: Phi4MMAudioInputs,
                             audio_projection_mode: str) -> NestedTensors:
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

        audio_features = audio_input["data"]
        # (e.g. multiple examples) and the second dim is the multi-audio dim
        # (e.g. multiple audios in the same example)

        dtype = next(self.embed_tokens_extend.parameters()).dtype
        audio_embeds = [
            self.embed_tokens_extend(
                features.to(dtype),
                audio_projection_mode=audio_projection_mode,
            ) for features in audio_features
        ]
        return audio_embeds

    def _parse_and_validate_image_input(self,
                                        **kwargs: object) -> Optional[Dict]:
        input_image_embeds: NestedTensors = kwargs.get("input_image_embeds")
        if input_image_embeds is None:
            return None

        image_sizes = kwargs.get("image_sizes")
        image_attention_mask = kwargs.get("image_attention_mask")
        num_img_tokens = kwargs.get("num_img_tokens")
        assert image_sizes is not None and image_attention_mask is not None\
              and num_img_tokens is not None, "Missing image inputs"

        if is_list_of(input_image_embeds, torch.Tensor):
            assert all(p.dim() == 5
                       for p in input_image_embeds), "Incorrect image inputs"
            # list len is batch_size.
            # each tensor has dimension: num_img_per_example, num_hd_patches,
            # channels, height, width.
            # need to pad along num_hd_patches.
            # mask size num_img_per_prompt, num_hd_patches, feat_h, heat_w.
            input_image_embeds = cat_with_pad(input_image_embeds, dim=0)
        elif isinstance(input_image_embeds, torch.Tensor):
            # dimension: batch_size, num_img_per_example, num_hd_patches,
            # channels, height, width.
            # we flatten first 2 dims to make it a single large batch for
            # SigLIP Encoder.
            assert input_image_embeds.dim() == 6, "Incorrect image inputs"
            input_image_embeds = input_image_embeds.flatten(0, 1)
        else:
            raise ValueError("Incorrect input_image_embeds inputs")

        if isinstance(image_attention_mask, list):
            image_attention_mask = cat_with_pad(image_attention_mask, dim=0)
        elif isinstance(image_attention_mask, torch.Tensor):
            image_attention_mask = image_attention_mask.flatten(0, 1)
        else:
            raise ValueError("Incorrect image_attention_mask inputs")

        if isinstance(image_sizes, list):
            image_sizes = torch.cat(image_sizes, dim=0)
        elif isinstance(image_sizes, torch.Tensor):
            image_sizes = image_sizes.flatten(0, 1)
        else:
            raise ValueError("Incorrect image_attention_mask inputs")

        if isinstance(num_img_tokens, list):
            num_img_tokens = [
                n for num_tensor in num_img_tokens
                for n in num_tensor.tolist()
            ]
        elif isinstance(num_img_tokens, torch.Tensor):
            num_img_tokens = num_img_tokens.flatten(0, 1).tolist()
        else:
            raise ValueError("Incorrect image_attention_mask inputs")

        return Phi4MMImagePixelInputs(
            type="pixel_values",
            data=input_image_embeds,
            image_sizes=image_sizes,
            image_attention_mask=image_attention_mask,
            num_img_tokens=num_img_tokens,
        )

    def _parse_and_validate_multimodal_inputs(self, **kwargs: object) -> dict:
        modalities = {}

        # Preserve the order of modalities if there are multiple of them
        # from the order of kwargs.
        for input_key in kwargs:
            if input_key in ("input_image_embeds",
                             "image_embeds") and "images" not in modalities:
                modalities["images"] = self._parse_and_validate_image_input(
                    **kwargs)
            if input_key in ("input_audio_embeds",
                             "audio_embeds") and "audios" not in modalities:
                modalities["audios"] = self._parse_and_validate_audio_input(
                    **kwargs)

        return modalities

    def _process_image_input(
            self, image_input: Phi4MMImagePixelInputs) -> list[torch.Tensor]:
        if image_input["type"] == "image_embeds":
            image_embeds = image_input["image_embeds"].type(self.visual.dtype)
        else:
            dtype = next(self.vision_encoder.parameters()).dtype
            pixel_values = image_input['data'].to(dtype)
            image_sizes = image_input['image_sizes']
            image_attention_mask = image_input['image_attention_mask']
            image_embeds = self.vision_encoder(pixel_values, image_sizes,
                                               image_attention_mask)
        return image_embeds

    def get_multimodal_embeddings(
            self, **kwargs: object) -> Optional[MultiModalEmbeddings]:

        modalities = self._parse_and_validate_multimodal_inputs(**kwargs)
        if not modalities:
            return None

        # The result multimodal_embeddings is tuple of tensors, with each
        # tensor correspoending to a multimodal data item (image or video).
        multimodal_embeddings: tuple[torch.Tensor, ...] = ()

        # NOTE: It is important to iterate over the keys in this dictionary
        # to preserve the order of the modalities.
        audio_projection_mode = 'speech'
        for modality in modalities:
            # make sure process images first
            if modality == "images":
                audio_projection_mode = "vision"
                image_input = modalities["images"]
                vision_embeddings = self._process_image_input(image_input)
                multimodal_embeddings += tuple(vision_embeddings)
            if modality == "audios":
                audio_input = modalities["audios"]
                audio_embeddings = self._process_audio_input(
                    audio_input, audio_projection_mode=audio_projection_mode)
                multimodal_embeddings += tuple(audio_embeddings)

        return multimodal_embeddings

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[MultiModalEmbeddings] = None,
    ) -> torch.Tensor:
        inputs_embeds = self.model.embed_tokens(input_ids)
        if multimodal_embeddings is not None:
            inputs_embeds = merge_multimodal_embeddings(
                input_ids, inputs_embeds, multimodal_embeddings,
                [_IMAGE_PLACEHOLDER_TOKEN_ID, _AUDIO_PLACEHOLDER_TOKEN_ID])
        return inputs_embeds

    def get_input_embeddings_v0(
        self,
        input_ids: torch.Tensor,
        image_input: Optional[Phi4MMImagePixelInputs] = None,
        audio_input: Optional[Phi4MMAudioFeatureInputs] = None,
    ) -> torch.Tensor:
        audio_projection_mode = 'speech'
        inputs_embeds = self.get_input_embeddings(input_ids)
        if image_input is not None:
            image_embeds = self._process_image_input(image_input)
            inputs_embeds = merge_multimodal_embeddings(
                input_ids,
                inputs_embeds,
                image_embeds,
                placeholder_token_id=_IMAGE_PLACEHOLDER_TOKEN_ID,
            )
            audio_projection_mode = 'vision'

        if audio_input is not None:
            audio_embeds = self._process_audio_input(
                audio_input, audio_projection_mode=audio_projection_mode)
            inputs_embeds = merge_multimodal_embeddings(
                input_ids,
                inputs_embeds,
                audio_embeds,
                placeholder_token_id=_AUDIO_PLACEHOLDER_TOKEN_ID,
            )
        return inputs_embeds

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> torch.Tensor:
        if intermediate_tensors is not None:
            inputs_embeds = None

        # NOTE: In v1, inputs_embeds is always generated at model runner from
        # `get_multimodal_embeddings` and `get_input_embeddings`, this
        # condition is only for v0 compatibility.
        elif inputs_embeds is None:
            image_input = self._parse_and_validate_image_input(**kwargs)
            audio_input = self._parse_and_validate_audio_input(**kwargs)

            if image_input is None and audio_input is None:
                inputs_embeds = None
            else:
                inputs_embeds = self.get_input_embeddings_v0(
                    input_ids,
                    image_input=image_input,
                    audio_input=audio_input)
                input_ids = None

        hidden_states = self.model(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        logits = self.logits_processor(self.lm_head, hidden_states,
                                       sampling_metadata)
        return logits

    def load_weights(self, weights: Iterable[Tuple[str,
                                                   torch.Tensor]]) -> None:
        weights = ((name, data) for name, data in weights
                   if "lora" not in name)
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

    def get_mm_mapping(self) -> MultiModelKeys:
        """
        Get the module prefix in multimodal models
        """
        return MultiModelKeys.from_string_field(
            language_model="model.",
            connector=["audio_projection_for_vision", "audio_projection"],
            tower_model=["vision_encoder", "embed_tokens_extend"],
        )

    def get_language_model(self) -> torch.nn.Module:
        return self.model
