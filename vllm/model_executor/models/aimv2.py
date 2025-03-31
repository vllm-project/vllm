# A modified implementation of the AIMv2 Transformer
# inserted here also the image tokenizer used by Ovis
from typing import Optional, Tuple, Union

import torch
from vllm.attention import Attention, AttentionType
from vllm.config import VllmConfig, CacheConfig
from vllm.model_executor.layers.linear import QKVParallelLinear, RowParallelLinear, ColumnParallelLinear
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.quantization.quark.quark import QuarkConfig

from .configuration_aimv2 import AIMv2Config
from torch import nn
from torch.nn import functional as F
from transformers.modeling_outputs import BaseModelOutputWithNoAttention
from transformers.modeling_utils import PreTrainedModel

import PIL
import torch
from torch import softmax
from torch.nn.functional import gumbel_softmax, pad
from transformers import AutoImageProcessor
from vllm.config import QuantizationConfig
from vllm.model_executor.layers.linear import ColumnParallelLinear

IMAGE_INDICATOR_IDS = [-301, -302, -303, -304, -305] # kept for vocab prefixed tokens


class BaseVisualTokenizerConfig(PretrainedConfig):
    def __init__(
        self,
        vocab_size=16384,
        tokenize_function="softmax",
        tau=1.0,
        depths=None,
        drop_cls_token=False,
        backbone_config: Optional[Union[PretrainedConfig, dict]] = None,
        hidden_stride: int = 1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.tokenize_function = tokenize_function
        self.tau = tau
        if isinstance(depths, str):
            depths = [int(x) for x in depths.split('|')]
        self.depths = depths
        self.backbone_kwargs = {}
        self.drop_cls_token = drop_cls_token
        if backbone_config is not None:
            assert isinstance(backbone_config, (PretrainedConfig, dict)), \
                f"expect `backbone_config` to be instance of PretrainedConfig or dict, but got {type(backbone_config)} type"
            if not isinstance(backbone_config, PretrainedConfig):
                model_type = backbone_config['model_type']
                backbone_config.pop('model_type')
                backbone_config = AutoConfig.for_model(model_type, **backbone_config)
        self.backbone_config = backbone_config
        self.hidden_stride = hidden_stride


class Aimv2VisualTokenizerConfig(BaseVisualTokenizerConfig):
    model_type = "aimv2_visual_tokenizer"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.drop_cls_token:
            self.drop_cls_token = False
        if self.depths:
            assert len(self.depths) == 1
            self.backbone_kwargs['num_hidden_layers'] = self.depths[0]


class Aimv2VisualTokenizer(torch.nn.Module):

    def __init__(self, config: BaseVisualTokenizerConfig, quant_config: QuantizationConfig, prefix: str = "", **kwargs):
        super().__init__()
        self.image_processor = AutoImageProcessor.from_pretrained(kwargs['image_processor_name_or_path'])
        self.config = config
        self.image_processor.do_center_crop = False
        self.backbone = AIMv2Model(
            config = config.backbone_config, # noqa
            quant_config=quant_config,
            prefix=f"{prefix}.visual_tokenizer"
        )
        head_dim = config.vocab_size - len(IMAGE_INDICATOR_IDS)  # reserved tokens for IMAGE_INDICATORS
        self.head = torch.nn.Sequential(
            ColumnParallelLinear(
                config.backbone_config.hidden_size * config.hidden_stride * config.hidden_stride, head_dim,
                bias=False,
            ),
            torch.nn.LayerNorm(head_dim)
        )

        assert all((self.image_processor.do_resize,
                    not getattr(self.image_processor, 'do_center_crop', False),
                    self.image_processor.do_rescale,
                    self.image_processor.do_normalize
                    )), f"image_processor `{self.image_processor}` is not supported currently"

    @property
    def dtype(self):
        return self.backbone.dtype

    @property
    def device(self):
        return self.backbone.device

    def get_backbone(self):
        return self.backbone

    def get_image_processor(self):
        return self.image_processor

    def get_head(self):
        return self.head

    def get_image_size(self):
        raise NotImplementedError


    def tokenize(self, logits):
        def st_argmax(y_soft, dim):  # straight-through softmax
            index = y_soft.max(dim, keepdim=True)[1]
            y_hard = torch.zeros_like(y_soft, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
            ret = y_hard - y_soft.detach() + y_soft
            return ret

        if self.config.tokenize_function == 'softmax':
            tokens = softmax(logits, dim=-1)
        elif self.config.tokenize_function == 'gumbel_argmax':
            tokens = gumbel_softmax(logits, tau=self.config.tau, hard=True)
        elif self.config.tokenize_function == 'st_argmax':
            tokens = st_argmax(logits, dim=-1)
        else:
            raise ValueError(
                f'Invalid `max_type`, expected softmax or gumbel_argmax or st_argmax, but got {self.config.tokenize_function}')
        return tokens

    def encode(self, pixel_values):
        features = self.backbone(pixel_values)
        if self.config.drop_cls_token:
            features = features[:, 1:, :]

        # merge number of `hidden_stride * hidden_stride` hidden states together to reduce token sequence length
        # e.g., for hidden_stride=2, this leads to a token length reduction: 1024 -> 256 for aimv2
        if self.config.hidden_stride > 1:
            n, l, d = features.shape  # this `d` maybe different from the above `d
            sqrt_l = int(l ** 0.5)
            assert sqrt_l ** 2 == l, "The token sequence length should be a perfect square."
            features = features.reshape(n, sqrt_l, sqrt_l, d)
            pl = (self.config.hidden_stride - (sqrt_l % self.config.hidden_stride)) % self.config.hidden_stride
            features = pad(features, (0, 0, 0, pl, 0, pl), "constant", 0)
            sqrt_l += pl
            features = features.reshape(n, sqrt_l // self.config.hidden_stride, self.config.hidden_stride,
                                        sqrt_l // self.config.hidden_stride, self.config.hidden_stride, d)
            features = features.permute(0, 1, 3, 2, 4, 5)  # [n, sqrt_l/hs, sqrt_l/hs, hs, hs, d]
            features = features.flatten(3)  # [n, sqrt_l/hs, sqrt_l/hs, hs*hs*d]
            features = features.reshape(
                n, -1, self.config.hidden_stride * self.config.hidden_stride * d)

        return features

    def forward(self, pixel_values) -> torch.Tensor:  # [BatchSize, ImageShape] -> [BatchSize, #Token, VocabSize]
        features = self.encode(pixel_values)
        logits, _ = self.head[0](features) # we spllit the sequncial here for not throwing an error
        logits = self.head[1](logits)
        tokens = self.tokenize(logits)
        # tokens' shape is [BatchSize, #Token, VocabSize-5], so padding with [BatchSize, #Token, 5], after
        # which, tokens' shape should become [BatchSize, #Token, VocabSize]
        batch_size, token_len, _ = tokens.shape
        padding_tensor = torch.zeros(size=(batch_size, token_len, len(IMAGE_INDICATOR_IDS)),
                                     dtype=tokens.dtype,
                                     device=tokens.device,
                                     layout=tokens.layout,
                                     requires_grad=False)
        tokens = torch.cat((tokens, padding_tensor), dim=2)
        return tokens


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

    def extra_repr(self) -> str:
        return f"{tuple(self.weight.shape)}, eps={self.eps}"

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


class AIMv2SwiGLUFFN(nn.Module):
    def __init__(self, config: AIMv2Config, quant_config: QuantizationConfig, prefix: str):
        super().__init__()
        hidden_features = config.intermediate_size
        in_features = config.hidden_size
        bias = config.use_bias

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)#ColumnParallelLinear(in_features,
                   #              hidden_features,
                   #              bias=bias,
                   #              quant_config=quant_config,
                   #              prefix=f"{prefix}.fc1")
        self.fc2 = nn.Linear(hidden_features, in_features, bias=bias)#ColumnParallelLinear(hidden_features,
                   #              in_features,
                   #              bias=bias,
                   #              quant_config=quant_config,
                   #              prefix=f"{prefix}.fc2")
        self.fc3 = nn.Linear(in_features, hidden_features, bias=bias)#RowParallelLinear(in_features,
                   #           hidden_features,
                   #           bias=bias,
                   #           quant_config=quant_config,
                   #           prefix=f"{prefix}.fc3")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_parallel= self.fc1(x)#, _ = self.fc1(x)
        gate= self.fc3(x)#, _ = self.fc3(x)
        x_parallel = F.silu(x_parallel) * gate
        out =self.fc2(x_parallel)#, _ = self.fc2(x_parallel)
        return out



class AIMv2PatchEmbed(nn.Module):
    def __init__(self, config: AIMv2Config):
        super().__init__()
        self.proj = nn.Conv2d(
            config.num_channels,
            config.hidden_size,
            kernel_size=(config.patch_size, config.patch_size),
            stride=(config.patch_size, config.patch_size),
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


class AIMv2ViTPreprocessor(nn.Module):
    def __init__(self, config: AIMv2Config):
        super().__init__()
        num_patches = (config.image_size // config.patch_size) ** 2

        self.patchifier = AIMv2PatchEmbed(config)
        self.pos_embed = nn.Parameter(torch.zeros((1, num_patches, config.hidden_size)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.patchifier(x)
        _, N, _ = tokens.shape
        pos_embed = self.pos_embed.to(tokens.device)
        tokens = tokens + pos_embed[:, :N]
        return tokens


class AIMv2Attention(nn.Module):
    def __init__(self, config: AIMv2Config, quant_config: QuantizationConfig, prefix: str):
        super().__init__()
        dim = config.hidden_size

        self.num_heads = config.num_attention_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=config.qkv_bias)#QKVParallelLinear(
                   # hidden_size=dim,
                   # head_size=dim // config.num_attention_heads,
                   # total_num_heads=config.num_attention_heads,
                   # bias=config.qkv_bias,
                   # quant_config=quant_config,
                   # prefix=f"{prefix}.qkv")
        self.attn_drop = nn.Dropout(config.attention_dropout)
        self.proj = nn.Linear(dim, dim, bias=config.use_bias)#RowParallelLinear(input_size=dim,
                    #                  output_size=dim,
                    #                  bias = config.use_bias,
                    #                  quant_config=quant_config,
                    #                  prefix=f"{prefix}.proj")

        self.proj_drop = nn.Dropout(config.projection_dropout)

    def forward( # todo might implement multiple attn implementations
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x) #, _ = self.qkv(x)

        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        q, k, v = qkv.unbind(0)

        x = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        x = x.transpose(1, 2).contiguous().reshape(B, N, C)
        x= self.proj(x)#, _ = self.proj(x)
        x = self.proj_drop(x)
        return x


class AIMv2Block(nn.Module):
    def __init__(self, config: AIMv2Config, quant_config: QuantizationConfig, prefix: str):
        super().__init__()
        self.attn = AIMv2Attention(config, quant_config=quant_config, prefix=f"{prefix}.attn")
        self.norm_1 = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = AIMv2SwiGLUFFN(config, quant_config=quant_config, prefix=f"{prefix}.mlp")
        self.norm_2 = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = x + self.attn(self.norm_1(x), mask)
        x = x + self.mlp(self.norm_2(x))
        return x


class AIMv2Transformer(nn.Module):
    def __init__(self, config: AIMv2Config, quant_config: QuantizationConfig, prefix: str):
        super().__init__()

        self.blocks = nn.ModuleList(
            [AIMv2Block(config, quant_config, prefix=f"{prefix}.blocks.{i}") for i in range(config.num_hidden_layers)]
        )
        self.post_trunk_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        tokens: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, ...]]]:
        #outputs = []
        for block in self.blocks: # they take the -1 as the ref embeddings, like a clip skip
            tokens = block(tokens, mask)
            #outputs.append(tokens)
        #tokens = self.post_trunk_norm(tokens) NO NORM IN THE OG IMPLEMENTATION
        return tokens


class AIMv2Model(torch.nn.Module):
    def __init__(self, config: AIMv2Config, quant_config: QuantizationConfig, prefix: str = ""):
        super().__init__()
        self.preprocessor = AIMv2ViTPreprocessor(config)
        self.trunk = AIMv2Transformer(config, quant_config=quant_config, prefix=f"{prefix}.trunk")

    @property
    def dtype(self):
        return self.trunk.blocks[0].attn.qkv.weight.dtype

    @property
    def device(self):
        return self.trunk.blocks[0].attn.qkv.device

    def forward(
        self,
        pixel_values: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Union[
        Tuple[torch.Tensor],
        Tuple[torch.Tensor, Tuple[torch.Tensor, ...]],
        BaseModelOutputWithNoAttention,
    ]:

        x = self.preprocessor(pixel_values)
        x = self.trunk(
            x, mask
        )

        return x
