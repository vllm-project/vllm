# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import namedtuple
# adapted from https://huggingface.co/OpenGVLab/InternVL2-4B/blob/main/modeling_intern_vit.py
# --------------------------------------------------------
# InternVL
# Copyright (c) 2023 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
from collections.abc import Iterable
from typing import Callable, Dict, List, NamedTuple, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models import VisionTransformer, checkpoint_seq, create_model
from transformers import PretrainedConfig

from vllm.model_executor.layers.quantization import QuantizationConfig

#from .enable_cpe_support import enable_cpe
#from .input_conditioner import InputConditioner
#from .adaptor_base import AdaptorBase, RadioOutput, AdaptorInput
from . import internvl_cl_eradio_model


#from .feature_normalizer import FeatureNormalizer, IntermediateFeatureNormalizer
#### adatptor_base.py
class AdaptorInput(NamedTuple):
    images: torch.Tensor
    summary: torch.Tensor
    features: torch.Tensor
    feature_fmt: str
    patch_size: int


class RadioOutput(NamedTuple):
    summary: torch.Tensor
    features: torch.Tensor

    def to(self, *args, **kwargs):
        return RadioOutput(
            self.summary.to(*args, **kwargs)
            if self.summary is not None else None,
            self.features.to(*args, **kwargs)
            if self.features is not None else None,
        )


class AdaptorBase(nn.Module):

    def forward(self, input: AdaptorInput) -> RadioOutput:
        raise NotImplementedError("Subclasses must implement this!")


#### input_conditioner.py
norm_t = Union[Tuple[float, float, float], torch.Tensor]


class InputConditioner(nn.Module):

    def __init__(
        self,
        input_scale: float,
        norm_mean: norm_t,
        norm_std: norm_t,
        dtype: torch.dtype = None,
    ):
        super().__init__()

        self.dtype = dtype

        self.register_buffer("norm_mean", _to_tensor(norm_mean) / input_scale)
        self.register_buffer("norm_std", _to_tensor(norm_std) / input_scale)

    def forward(self, x: torch.Tensor):
        y = (x - self.norm_mean) / self.norm_std
        if self.dtype is not None:
            y = y.to(self.dtype)
        return y


def get_default_conditioner():
    from timm.data.constants import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD

    return InputConditioner(
        input_scale=1.0,
        norm_mean=OPENAI_CLIP_MEAN,
        norm_std=OPENAI_CLIP_STD,
    )


def _to_tensor(v: norm_t):
    return torch.as_tensor(v, dtype=torch.float32).view(-1, 1, 1)


####feature_normalizer
def _run_kernel(x: torch.Tensor, mean: torch.Tensor, tx: torch.Tensor):
    if x.ndim <= 3:
        x = x - mean
        x = x @ tx.T
    elif x.ndim == 4:
        x = x - mean.reshape(1, -1, 1, 1)
        kernel = tx.reshape(*tx.shape, 1, 1)
        x = torch.nn.functional.conv2d(x,
                                       weight=kernel,
                                       bias=None,
                                       stride=1,
                                       padding=0)
    else:
        raise ValueError(
            f'Unsupported input dimension: {x.ndim}, shape: {x.shape}')
    return x


class FeatureNormalizer(nn.Module):

    def __init__(self, embed_dim: int, dtype: torch.dtype = torch.float32):
        super().__init__()

        self.register_buffer('mean', torch.zeros(embed_dim, dtype=dtype))
        self.register_buffer('tx', torch.eye(embed_dim, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _run_kernel(x, self.mean, self.tx)
        return x


class InterFeatState(NamedTuple):
    y: torch.Tensor
    alpha: torch.Tensor


class IntermediateFeatureNormalizerBase(nn.Module):

    def forward(self,
                x: torch.Tensor,
                index: int,
                rot_index: int = None,
                skip: Optional[int] = None) -> InterFeatState:
        raise NotImplementedError()


class IntermediateFeatureNormalizer(IntermediateFeatureNormalizerBase):

    def __init__(self,
                 num_intermediates: int,
                 embed_dim: int,
                 rot_per_layer: bool = False,
                 dtype: torch.dtype = torch.float32):
        super().__init__()
        self.register_buffer('alphas',
                             torch.ones(num_intermediates, dtype=dtype))

        rot = torch.eye(embed_dim, dtype=dtype)
        if rot_per_layer:
            rot = rot.unsqueeze(0).repeat(num_intermediates, 1, 1)

        self.register_buffer('rotation', rot.contiguous())
        self.register_buffer(
            'means', torch.zeros(num_intermediates, embed_dim, dtype=dtype))

    def forward(self,
                x: torch.Tensor,
                index: int,
                rot_index: int = None,
                skip: Optional[int] = None) -> InterFeatState:
        if rot_index is None:
            rot_index = index

        if skip:
            assert x.ndim == 3, 'Cannot use the `skip` parameter when the `x` tensor isn\'t 3-dimensional.'
            prefix, x = x[:, :skip], x[:, skip:]

        rotation = self._get_rotation(rot_index)
        y = _run_kernel(x, self.means[index], rotation)

        alpha = self.alphas[index]
        if skip:
            alpha = torch.cat([
                torch.ones(skip, dtype=alpha.dtype, device=alpha.device),
                alpha[None].expand(y.shape[1]),
            ]).reshape(1, -1, 1)
            y = torch.cat([prefix, y], dim=1)
        else:
            if x.ndim == 3:
                alpha = alpha.reshape(1, 1, 1).expand(1, y.shape[1], 1)
            elif x.ndim == 4:
                alpha = alpha.reshape(1, 1, 1, 1).expand(1, 1, *y.shape[2:])
            else:
                raise ValueError(f'Unsupported input dimension: {x.ndim}')

        return InterFeatState(y, alpha)

    def _get_rotation(self, rot_index: int) -> torch.Tensor:
        if self.rotation.ndim == 2:
            return self.rotation
        return self.rotation[rot_index]


class NullIntermediateFeatureNormalizer(IntermediateFeatureNormalizerBase):
    instances = dict()

    def __init__(self, dtype: torch.dtype, device: torch.device):
        super().__init__()
        self.register_buffer('alpha',
                             torch.tensor(1, dtype=dtype, device=device))

    @staticmethod
    def get_instance(dtype: torch.dtype, device: torch.device):
        instance = NullIntermediateFeatureNormalizer.instances.get(
            (dtype, device), None)
        if instance is None:
            instance = NullIntermediateFeatureNormalizer(dtype, device)
            NullIntermediateFeatureNormalizer.instances[(dtype,
                                                         device)] = instance
        return instance

    def forward(self,
                x: torch.Tensor,
                index: int,
                rot_index: int = None,
                skip: Optional[int] = None) -> InterFeatState:
        return InterFeatState(x, self.alpha)


####forward_intermediates.py
def _take_indices(
    num_blocks: int,
    n: Optional[Union[int, List[int], Tuple[int]]],
) -> Tuple[Set[int], int]:
    if isinstance(n, int):
        assert n >= 0
        take_indices = {x for x in range(num_blocks - n, num_blocks)}
    else:
        take_indices = {num_blocks + idx if idx < 0 else idx for idx in n}
    return take_indices, max(take_indices)


def forward_intermediates(
    model: nn.Module,
    patch_extractor: Callable[[torch.Tensor], torch.Tensor],
    norm: nn.Module,
    num_summary_tokens: int,
    num_cls_tokens: int,
    x: torch.Tensor,
    indices: Optional[Union[int, List[int], Tuple[int]]] = None,
    return_prefix_tokens: bool = False,
    stop_early: bool = False,
    output_fmt: str = 'NCHW',
    intermediates_only: bool = False,
    aggregation: Optional[str] = "sparse",
    inter_feature_normalizer: Optional[
        IntermediateFeatureNormalizerBase] = None,
    norm_alpha_scheme="post-alpha",
    block_kwargs: Dict = None,
) -> Union[List[torch.Tensor], Tuple[torch.Tensor, List[torch.Tensor]]]:
    """ Forward features that returns intermediates.
    The Dense layer aggregation method is inspired from the paper: "Dense Connector for MLLMs"
    by Yao, Huanjin et al. (2024). arXiv preprint arXiv:2405.13800}
    Args:
        x: Input image tensor
        indices: Take last n blocks if int, select matching indices if sequence
        return_prefix_tokens: Return both prefix and spatial intermediate tokens
        norm: Apply norm layer to all intermediates
        stop_early: Stop iterating over blocks when last desired intermediate hit
        output_fmt: Shape of intermediate feature outputs
        intermediates_only: Only return intermediate features
        aggregation: intermediate layer aggregation method (sparse or dense)
        norm_alpha_scheme: apply alpha before ("pre-alpha") or after accumulation ("post-alpha")
    Returns:
    """
    assert output_fmt in ('NCHW',
                          'NLC'), 'Output format must be one of NCHW or NLC.'
    assert aggregation in (
        'sparse', 'dense'), 'Aggregation must be one of sparse or dense.'
    reshape = output_fmt == 'NCHW'
    intermediates = []

    block_kwargs = block_kwargs or dict()

    blocks = model.blocks

    take_indices, max_index = _take_indices(len(blocks), indices)
    take_indices = sorted(take_indices)
    # forward pass
    B, _, height, width = x.shape

    x = patch_extractor(x)

    if stop_early:
        blocks = blocks[:max_index + 1]

    if inter_feature_normalizer is None or norm_alpha_scheme == 'none':
        inter_feature_normalizer = NullIntermediateFeatureNormalizer.get_instance(
            x.dtype, x.device)

    assert norm_alpha_scheme in (
        'none', 'pre-alpha',
        'post-alpha'), f'Unsupported alpha scheme: {norm_alpha_scheme}'
    post_alpha_scheme = norm_alpha_scheme == 'post-alpha'

    accumulator = 0
    alpha_sum = 0
    num_accumulated = 0

    take_off = 0

    for i, blk in enumerate(blocks):
        x = blk(x, **block_kwargs)
        if aggregation == "dense":
            # Arbitrarily use the rotation matrix from the final layer in the dense group
            y, alpha = inter_feature_normalizer(
                x,
                i,
                rot_index=take_indices[take_off],
                skip=num_summary_tokens)
            if post_alpha_scheme:
                accumulator = accumulator + y
                alpha_sum = alpha_sum + alpha
            else:
                accumulator = accumulator + (alpha * y)
                alpha_sum += 1
            num_accumulated += 1
        if i == take_indices[take_off]:
            if aggregation == "dense":
                alpha = alpha_sum / num_accumulated
                x_ = alpha * accumulator / num_accumulated
                num_accumulated = 0
                accumulator = 0
                alpha_sum = 0
            else:
                y, alpha = inter_feature_normalizer(x,
                                                    i,
                                                    skip=num_summary_tokens)
                x_ = alpha * y
            # normalize intermediates with final norm layer if enabled
            intermediates.append(norm(x_))
            take_off = min(take_off + 1, len(take_indices) - 1)

    # process intermediates

    # split prefix (e.g. class, distill) and spatial feature tokens
    prefix_tokens = [y[:, :num_cls_tokens] for y in intermediates]
    intermediates = [y[:, num_summary_tokens:] for y in intermediates]

    if reshape:
        # reshape to BCHW output format
        H = height // model.patch_size
        W = width // model.patch_size
        intermediates = [
            y.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            for y in intermediates
        ]
    if not torch.jit.is_scripting() and return_prefix_tokens:
        # return_prefix not support in torchscript due to poor type handling
        intermediates = list(zip(prefix_tokens, intermediates))
    if intermediates_only:
        return intermediates
    x = norm(x)
    return x, intermediates


#### extra_models.py
_has_torch_sdpa = hasattr(F, 'scaled_dot_product_attention')


class PaliGemmaWrapper(nn.Module):

    def __init__(self, vis_model: nn.Module, embed_dim: int):
        super().__init__()

        self.vis_model = vis_model
        self.embed_dim = embed_dim

    @property
    def patch_size(self):
        return self.vis_model.embeddings.patch_size

    @property
    def blocks(self):
        return self.vis_model.encoder.layers

    @property
    def embed_dim(self):
        return self.vis_model.embeddings.embed_dim

    def forward(self, x: torch.Tensor):
        outputs = self.vis_model(
            x,
            return_dict=False,
            interpolate_pos_encoding=True,
        )

        features = outputs[0].to(torch.float32)

        summary = features.mean(dim=1)

        return summary, features

    def forward_features(self, x: torch.Tensor):
        return self(x)


def _get_paligemma_model(repo: str,
                         embed_dim: int = None,
                         dtype: torch.dtype = torch.bfloat16):
    from transformers import PaliGemmaForConditionalGeneration
    from transformers import __version__ as tx_version

    if LooseVersion(tx_version) > LooseVersion('4.44.2'):
        warnings.warn(
            f'Your transformers version "{tx_version}" is higher than 4.44.2, and for whatever reason, PaliGemma might be broken.'
        )

    extra_args = dict()

    if dtype is not None:
        extra_args['torch_dtype'] = dtype
        rev = str(dtype).split('.')[-1]
        extra_args['revision'] = rev

    model = PaliGemmaForConditionalGeneration.from_pretrained(
        repo, **extra_args)

    vis_model = model.vision_tower.vision_model

    vis_model = PaliGemmaWrapper(vis_model, embed_dim)

    return vis_model


@register_model
def paligemma_896_student(**kwargs):
    model = _get_paligemma_model('google/paligemma-3b-pt-896',
                                 embed_dim=1152,
                                 dtype=None)

    return model


def dv2_sdpa(self, x: torch.Tensor) -> torch.Tensor:
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                              C // self.num_heads).permute(2, 0, 3, 1, 4)

    q, k, v = qkv[0], qkv[1], qkv[2]
    x = F.scaled_dot_product_attention(
        q,
        k,
        v,
        is_causal=False,
        dropout_p=self.attn_drop.p if self.training else 0.,
        scale=self.scale,
    )
    x = x.transpose(1, 2).reshape(B, N, C)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x


def _load_dino_v2(dino_v2_model,
                  cache_dir: Optional[str] = None,
                  pretrained=True,
                  **kwargs):
    if cache_dir:
        torch.hub.set_dir(cache_dir)
    model: nn.Module = torch.hub.load(
        'facebookresearch/dinov2',
        dino_v2_model,
        pretrained=pretrained,
        # **kwargs,
    )

    if _has_torch_sdpa:
        for n, m in model.named_modules():
            if n.endswith('.attn'):
                m.forward = MethodType(dv2_sdpa, m)

    return model


class DinoWrapper(nn.Module):

    def __init__(self, dino_model: nn.Module):
        super().__init__()

        self.inner = dino_model
        dino_model.blocks = nn.Sequential(*dino_model.blocks)

    @property
    def embed_dim(self):
        return self.inner.embed_dim

    @property
    def patch_size(self):
        return self.inner.patch_size

    @property
    def num_cls_tokens(self):
        return getattr(self.inner, 'num_tokens', 1)

    @property
    def num_registers(self):
        return getattr(self.inner, 'num_register_tokens', 0)

    @property
    def num_summary_tokens(self):
        return self.num_cls_tokens + self.num_registers

    @property
    def blocks(self):
        return self.inner.blocks

    def forward(self, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        parts = self.inner.forward_features(*args, **kwargs)

        cls_token = parts['x_norm_clstoken']
        features = parts['x_norm_patchtokens']

        return cls_token, features

    def forward_features(self, x: torch.Tensor):
        x = self.inner.prepare_tokens_with_masks(x)
        x = self.inner.blocks(x)
        x_norm = self.inner.norm(x)

        return x_norm[:, 0], x_norm[:, self.num_summary_tokens:]

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        return self.inner.prepare_tokens_with_masks(x)

    def forward_intermediates(
        self,
        x: torch.Tensor,
        norm: bool = False,
        **kwargs,
    ) -> Union[List[torch.Tensor], Tuple[torch.Tensor, List[torch.Tensor]]]:
        return forward_intermediates(
            self,
            patch_extractor=self.inner.prepare_tokens_with_masks,
            num_summary_tokens=self.num_summary_tokens,
            num_cls_tokens=self.num_cls_tokens,
            norm=self.inner.norm if norm else lambda y: y,
            x=x,
            **kwargs,
        )


def _dino_student(arch: str, **kwargs):
    from . import dinov2_arch

    factory = getattr(dinov2_arch, arch)
    model = factory()

    model = DinoWrapper(model)

    conditioner = InputConditioner(
        input_scale=1.0,
        norm_mean=IMAGENET_DEFAULT_MEAN,
        norm_std=IMAGENET_DEFAULT_STD,
    )

    model.input_conditioner = conditioner

    return model


@register_model
def dino_v2_l_student(**kwargs):
    return _dino_student('dinov2_vitl14_reg', **kwargs)


@register_model
def dino_v2_g_student(**kwargs):
    return _dino_student('dinov2_vitg14_reg', **kwargs)


from .dual_hybrid_vit import HybridModel
####enable_cpe_support
from .extra_models import DinoWrapper
from .forward_intermediates import forward_intermediates
from .vit_patch_generator import ViTPatchGenerator


def _forward_cpe(self: VisionTransformer, x: torch.Tensor) -> torch.Tensor:
    x = self.patch_generator(x)
    if getattr(self, 'grad_checkpointing',
               False) and not torch.jit.is_scripting():
        x = checkpoint_seq(self.blocks, x)
    else:
        x = self.blocks(x)
    x = self.norm(x)
    return x


def _take_indices(
    num_blocks: int,
    n: Optional[Union[int, List[int], Tuple[int]]],
) -> Tuple[Set[int], int]:
    if isinstance(n, int):
        assert n >= 0
        take_indices = {x for x in range(num_blocks - n, num_blocks)}
    else:
        take_indices = {num_blocks + idx if idx < 0 else idx for idx in n}
    return take_indices, max(take_indices)


def _forward_intermediates_cpe(
    self,
    x: torch.Tensor,
    norm: bool = False,
    **kwargs,
) -> Union[List[torch.Tensor], Tuple[torch.Tensor, List[torch.Tensor]]]:
    return forward_intermediates(
        self,
        patch_extractor=self.patch_generator,
        num_summary_tokens=self.patch_generator.num_skip,
        num_cls_tokens=self.patch_generator.num_cls_tokens,
        norm=self.norm if norm else lambda y: y,
        x=x,
        **kwargs,
    )


def _forward_cpe_dinov2(self: DinoWrapper, x: torch.Tensor) -> torch.Tensor:
    y = _forward_cpe(self.inner, x)

    return y[:, 0], y[:, self.num_summary_tokens:]


def _forward_intermediates_cpe_dinov2(self: DinoWrapper, *args, **kwargs):
    return _forward_intermediates_cpe(self.inner, *args, **kwargs)


def _enable_cpe_for_timm_vit(
    model: VisionTransformer,
    max_img_size: Union[int, Tuple[int, int]] = 1024,
    num_cls_tokens: int = 1,
    pos_dropout: float = 0.1,
    register_multiple: int = Optional[None],
    num_registers: int = Optional[None],
):
    if not isinstance(model, VisionTransformer):
        raise ValueError("CPE only support for VisionTransformer models!")

    patch_size = model.patch_embed.patch_size[0]
    embed_dim = model.embed_dim
    input_dims = model.patch_embed.img_size
    normalize_patches = not isinstance(model.patch_embed.norm, nn.Identity)
    cls_token = model.cls_token is not None

    max_img_size = int(round(max_img_size / patch_size) * patch_size)

    patch_generator = ViTPatchGenerator(
        patch_size=patch_size,
        embed_dim=embed_dim,
        input_dims=input_dims,
        normalize_patches=normalize_patches,
        cls_token=cls_token,
        max_input_dims=max_img_size,
        pos_dropout=pos_dropout,
        num_cls_tokens=num_cls_tokens,
        register_multiple=register_multiple,
        num_registers=num_registers,
    )

    model.patch_generator = patch_generator
    model.patch_embed = None
    model.cls_token = None
    model.pos_embed = None
    model.pos_drop = None
    model.patch_size = patch_size
    model.num_cls_tokens = num_cls_tokens
    model.num_registers = patch_generator.num_registers

    model.forward_features = MethodType(_forward_cpe, model)
    model.forward_intermediates = MethodType(_forward_intermediates_cpe, model)


def _enable_cpe_for_dv2_reg_vit(
    model: DinoWrapper,
    max_img_size: Union[int, Tuple[int, int]] = 1024,
    num_cls_tokens: int = 1,
    pos_dropout: float = 0.1,
    register_multiple: int = Optional[None],
    num_registers: int = Optional[None],
):
    patch_size = model.patch_size
    embed_dim = model.embed_dim
    input_dims = model.inner.patch_embed.patches_resolution
    normalize_patches = not isinstance(model.inner.patch_embed.norm,
                                       nn.Identity)
    cls_token = True

    max_img_size = int(round(max_img_size / patch_size) * patch_size)

    patch_generator = ViTPatchGenerator(
        patch_size=patch_size,
        embed_dim=embed_dim,
        input_dims=input_dims,
        normalize_patches=normalize_patches,
        cls_token=cls_token,
        max_input_dims=max_img_size,
        pos_dropout=pos_dropout,
        num_cls_tokens=num_cls_tokens,
        register_multiple=register_multiple,
        num_registers=num_registers,
        patch_bias=True,
    )

    inner = model.inner
    inner.patch_generator = patch_generator
    inner.patch_embed = None
    inner.cls_token = None
    inner.pos_embed = None
    inner.register_tokens = None
    inner.patch_size = patch_size

    model.forward_features = MethodType(_forward_cpe_dinov2, model)
    model.forward_intermediates = MethodType(_forward_intermediates_cpe_dinov2,
                                             model)


def enable_cpe(
    model: nn.Module,
    *args,
    **kwargs,
):
    if isinstance(model, VisionTransformer):
        _enable_cpe_for_timm_vit(model, *args, **kwargs)
    elif isinstance(model, DinoWrapper):
        _enable_cpe_for_dv2_reg_vit(model, *args, **kwargs)
    elif isinstance(model, HybridModel):
        _enable_cpe_for_timm_vit(model.vit, *args, **kwargs)
    else:
        raise ValueError(
            f'CPE not supported for this model type: {type(model)}')


################################################################################
class RADIOModelBase(nn.Module):

    def __init__(
        self,
        model: nn.Module,
        input_conditioner: InputConditioner,
        patch_size: int,
        max_resolution: int,
        preferred_resolution: Resolution,
        summary_idxs: Optional[torch.Tensor] = None,
        window_size: int = None,
        adaptors: Dict[str, AdaptorBase] = None,
        feature_normalizer: Optional[FeatureNormalizer] = None,
        inter_feature_normalizer: Optional[
            IntermediateFeatureNormalizer] = None,
    ):
        super().__init__()

        self.model = model
        self.input_conditioner = input_conditioner
        if summary_idxs is not None:
            self.register_buffer('summary_idxs', summary_idxs)
        else:
            self.summary_idxs = None

        self._preferred_resolution = preferred_resolution
        self._patch_size = patch_size
        self._max_resolution = max_resolution
        self._window_size = window_size

        adaptors = adaptors or dict()
        self.adaptors = nn.ModuleDict(adaptors)

        if feature_normalizer is None:
            feature_normalizer = nn.Identity()
        self.feature_normalizer = feature_normalizer
        self.inter_feature_normalizer = inter_feature_normalizer

    @property
    def num_summary_tokens(self) -> int:
        if hasattr(self.model, 'num_summary_tokens'):
            return self.model.num_summary_tokens

        patch_gen = getattr(self.model, "patch_generator", None)
        if patch_gen is not None:
            return patch_gen.num_skip
        elif getattr(self.model, 'global_pool', None) == 'avg':
            return 0
        return 1

    @property
    def num_cls_tokens(self) -> int:
        if hasattr(self.model, 'num_cls_tokens'):
            return self.model.num_cls_tokens

        patch_gen = getattr(self.model, 'patch_generator', None)
        if patch_gen is not None:
            return patch_gen.num_cls_tokens
        elif getattr(self.model, 'global_pool', None) == 'avg':
            return 0
        return 1

    @property
    def patch_size(self) -> int:
        if self._patch_size is not None:
            return self._patch_size
        if hasattr(self.model, "patch_size"):
            return self.model.patch_size
        patch_gen = getattr(self.model, "patch_generator", None)
        if patch_gen is not None:
            return patch_gen.patch_size
        return None

    @property
    def max_resolution(self) -> int:
        return self._max_resolution

    @property
    def preferred_resolution(self) -> Resolution:
        return self._preferred_resolution

    @property
    def window_size(self) -> int:
        return self._window_size

    @property
    def min_resolution_step(self) -> int:
        res = self.patch_size
        if self.window_size is not None:
            res *= self.window_size
        return res

    @property
    def blocks(self) -> Iterable[nn.Module]:
        blocks = getattr(self.model, 'blocks', None)
        if blocks is not None:
            return blocks
        return None

    @property
    def embed_dim(self) -> int:
        return self.model.embed_dim

    def make_preprocessor_external(
            self) -> Callable[[torch.Tensor], torch.Tensor]:
        ret = self.input_conditioner
        self.input_conditioner = nn.Identity()
        return ret

    def get_nearest_supported_resolution(self, height: int,
                                         width: int) -> Resolution:
        height = int(
            round(height / self.min_resolution_step) *
            self.min_resolution_step)
        width = int(
            round(width / self.min_resolution_step) * self.min_resolution_step)

        height = max(height, self.min_resolution_step)
        width = max(width, self.min_resolution_step)

        return Resolution(height=height, width=width)

    def switch_to_deploy(self):
        fn = getattr(self.model, 'switch_to_deploy', None)
        if fn is not None:
            fn()

    def forward(
        self,
        x: torch.Tensor,
        feature_fmt: str = 'NLC'
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        '''
        Forward process for model.
        Args:
            x: Input tensor. Unless `make_preprocessor_external` has been called, then the dynamic range of `x` is expected to be `[0, 1]`,
                             otherwise `x` is expected to be mean centered with unit standard deviation.
            feature_format: ['NLC', 'NCHW'] - The output format for the features.
        '''
        res_step = self.min_resolution_step
        if res_step is not None and (x.shape[-2] % res_step != 0
                                     or x.shape[-1] % res_step != 0):
            raise ValueError(
                'The input resolution must be a multiple of `self.min_resolution_step`. '
                '`self.get_nearest_supported_resolution(<height>, <width>) is provided as a convenience API. '
                f'Input: {x.shape[-2:]}, Nearest: {self.get_nearest_supported_resolution(*x.shape[-2:])}'
            )

        x = self.input_conditioner(x)
        y = self.model.forward_features(x)
        ret = self._extract_final(x, y, feature_fmt=feature_fmt)
        return ret

    def _extract_final(self,
                       x: torch.Tensor,
                       y: torch.Tensor,
                       feature_fmt: str = 'NLC'):
        if isinstance(self.model, VisionTransformer):
            patch_gen = getattr(self.model, "patch_generator", None)
            if patch_gen is not None:
                all_summary = y[:, :patch_gen.num_cls_tokens]
                if self.summary_idxs is not None:
                    bb_summary = all_summary[:, self.summary_idxs]
                else:
                    bb_summary = all_summary
                all_feat = y[:, patch_gen.num_skip:]
            elif self.model.global_pool == "avg":
                all_summary = y[:, self.model.num_prefix_tokens:].mean(dim=1)
                bb_summary = all_summary
                all_feat = y
            else:
                all_summary = y[:, 0]
                bb_summary = all_summary
                all_feat = y[:, 1:]
        elif isinstance(self.model, internvl_cl_eradio_model.ERADIO):
            _, f = y
            all_feat = f.flatten(2).transpose(1, 2)
            all_summary = all_feat.mean(dim=1)
            bb_summary = all_summary
        elif isinstance(y, (list, tuple)):
            all_summary, all_feat = y
            bb_summary = all_summary
        else:
            all_summary = y[:, :self.num_cls_tokens]
            if self.summary_idxs is not None and all_summary.shape[1] > 1:
                if all_summary.shape[1] == 1:
                    # Create dummy duplicates
                    all_summary = all_summary.expand(-1, 128, -1)
                bb_summary = all_summary[:, self.summary_idxs]
            else:
                bb_summary = all_summary
            all_feat = y[:, self.num_summary_tokens:]

        all_feat = self.feature_normalizer(all_feat)

        if feature_fmt == 'NCHW':
            fmt_feat = (all_feat.reshape(all_feat.shape[0],
                                         x.shape[-2] // self.patch_size,
                                         x.shape[-1] // self.patch_size,
                                         all_feat.shape[2]).permute(
                                             0, 3, 1, 2))
        elif feature_fmt == 'NLC':
            fmt_feat = all_feat
        else:
            raise ValueError(
                f'Unsupported feature_fmt: {feature_fmt}. Must be one of ["NLC", "NCHW"]'
            )

        ret = RadioOutput(bb_summary.flatten(1), fmt_feat)

        if self.adaptors:
            ret = dict(backbone=ret)
            for name, adaptor in self.adaptors.items():
                if all_summary.ndim == 3:
                    if all_summary.shape[1] == 1:
                        summary = all_summary[:, 0]
                    else:
                        summary = all_summary[:, adaptor.head_idx]
                else:
                    summary = all_summary
                ada_input = AdaptorInput(images=x,
                                         summary=summary.float(),
                                         features=all_feat,
                                         feature_fmt=feature_fmt,
                                         patch_size=self.patch_size)
                v = adaptor(ada_input).to(torch.float32)
                ret[name] = v

        return ret

    def forward_intermediates(
        self,
        x: torch.Tensor,
        indices: Optional[Union[int, List[int], Tuple[int]]] = None,
        return_prefix_tokens: bool = False,
        norm: bool = False,
        stop_early: bool = False,
        output_fmt: str = 'NCHW',
        intermediates_only: bool = False,
        aggregation: Optional[str] = "sparse",
        norm_alpha_scheme: Optional[str] = "post-alpha",
    ) -> List[RadioOutput]:
        """ Forward features that returns intermediates.
        Args:
            x: Input image tensor
            indices: Take last n blocks if int, select matching indices if sequence
            return_prefix_tokens: Return both prefix and spatial intermediate tokens
            norm: Apply norm layer to all intermediates
            stop_early: Stop iterating over blocks when last desired intermediate hit
            output_fmt: Shape of intermediate feature outputs. Options: NCHW, NLC
            intermediates_only: Only return intermediate features
            aggregation: intermediate layer aggregation method (sparse or dense).
                Dense accumulation is done by averaging the features in each group.
            norm_alpha_scheme: apply alpha before ("pre-alpha") or after accumulation ("post-alpha"), or don't normalize ("none")
                Only affects dense aggregation
        Returns:
            List of RadioOutput objects.
        """
        x = self.input_conditioner(x)
        intermediates = self.model.forward_intermediates(
            x,
            indices=indices,
            return_prefix_tokens=return_prefix_tokens,
            norm=norm,
            stop_early=stop_early,
            output_fmt=output_fmt,
            intermediates_only=intermediates_only,
            aggregation=aggregation,
            inter_feature_normalizer=self.inter_feature_normalizer,
            norm_alpha_scheme=norm_alpha_scheme,
        )

        if not intermediates_only:
            final, intermediates = intermediates

        def prepare_summary(summ: Optional[torch.Tensor]):
            if summ is None:
                return summ
            if self.summary_idxs is not None and summ.shape[1] > 1:
                summ = summ[:, self.summary_idxs]
            return summ.flatten(1)

        if return_prefix_tokens:
            radio_outputs = [
                RadioOutput(prepare_summary(summary), features)
                for summary, features in intermediates
            ]
        else:
            radio_outputs = intermediates

        if intermediates_only:
            return radio_outputs
        else:
            final = self._extract_final(x, final, feature_fmt=output_fmt)
            return final, radio_outputs


def create_model_from_args(args) -> nn.Module:
    in_chans = 3
    if args.in_chans is not None:
        in_chans = args.in_chans
    elif args.input_size is not None:
        in_chans = args.input_size[0]

    # Skip weight initialization unless it's explicitly requested.
    weight_init = args.model_kwargs.pop("weight_init", "skip")

    model = create_model(
        args.model,
        pretrained=args.pretrained,
        in_chans=in_chans,
        num_classes=args.num_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=args.drop_block,
        global_pool=args.gp,
        bn_momentum=args.bn_momentum,
        bn_eps=args.bn_eps,
        scriptable=args.torchscript,
        checkpoint_path=args.initial_checkpoint,
        weight_init=weight_init,
        **args.model_kwargs,
    )

    if hasattr(model, 'norm') and not getattr(args, 'model_norm', False):
        model.norm = nn.Identity()

    model.head = nn.Identity()

    if args.cpe_max_size is not None:
        uq_teachers = set(t['name'] for t in args.teachers)
        enable_cpe(
            model,
            args.cpe_max_size,
            num_cls_tokens=len(uq_teachers)
            if args.cls_token_per_teacher else 1,
            register_multiple=getattr(args, 'register_multiple', None),
            num_registers=getattr(args, 'cpe_num_registers', None),
        )

    return model


class RADIOModel(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        *,
        prefix: str = "",
    ) -> None:
        super().__init__()

        RADIOArgs = namedtuple("RADIOArgs", config.args.keys())
        args = RADIOArgs(**config.args)
        self.config = config

        model = create_model_from_args(args)
        input_conditioner: InputConditioner = get_default_conditioner()

        dtype = getattr(args, "dtype", torch.float32)
        if isinstance(dtype, str):
            # Convert the dtype's string representation back to a dtype.
            dtype = getattr(torch, dtype)
        model.to(dtype=dtype)
        input_conditioner.dtype = dtype

        summary_idxs = torch.tensor(
            [
                i for i, t in enumerate(args.teachers)
                if t.get("use_summary", True)
            ],
            dtype=torch.int64,
        )

        adaptor_configs = config.adaptor_configs
        adaptor_names = config.adaptor_names or []

        adaptors = dict()
        for adaptor_name in adaptor_names:
            mlp_config = adaptor_configs[adaptor_name]
            adaptor = GenericAdaptor(args, None, None, mlp_config)
            adaptor.head_idx = mlp_config["head_idx"]
            adaptors[adaptor_name] = adaptor

        feature_normalizer = None
        if config.feature_normalizer_config is not None:
            # Actual normalization values will be restored when loading checkpoint weights.
            feature_normalizer = FeatureNormalizer(
                config.feature_normalizer_config["embed_dim"])

        inter_feature_normalizer = None
        if config.inter_feature_normalizer_config is not None:
            inter_feature_normalizer = IntermediateFeatureNormalizer(
                config.inter_feature_normalizer_config["num_intermediates"],
                config.inter_feature_normalizer_config["embed_dim"],
                rot_per_layer=config.
                inter_feature_normalizer_config["rot_per_layer"],
                dtype=dtype)

        self.radio_model = RADIOModelBase(
            model,
            input_conditioner,
            summary_idxs=summary_idxs,
            patch_size=config.patch_size,
            max_resolution=config.max_resolution,
            window_size=config.vitdet_window_size,
            preferred_resolution=config.preferred_resolution,
            adaptors=adaptors,
            feature_normalizer=feature_normalizer,
            inter_feature_normalizer=inter_feature_normalizer,
        )

    @property
    def adaptors(self) -> nn.ModuleDict:
        return self.radio_model.adaptors

    @property
    def model(self) -> VisionTransformer:
        return self.radio_model.model

    @property
    def input_conditioner(self) -> InputConditioner:
        return self.radio_model.input_conditioner

    @property
    def num_summary_tokens(self) -> int:
        return self.radio_model.num_summary_tokens

    @property
    def patch_size(self) -> int:
        return self.radio_model.patch_size

    @property
    def max_resolution(self) -> int:
        return self.radio_model.max_resolution

    @property
    def preferred_resolution(self) -> Resolution:
        return self.radio_model.preferred_resolution

    @property
    def window_size(self) -> int:
        return self.radio_model.window_size

    @property
    def min_resolution_step(self) -> int:
        return self.radio_model.min_resolution_step

    def make_preprocessor_external(
            self) -> Callable[[torch.Tensor], torch.Tensor]:
        return self.radio_model.make_preprocessor_external()

    def get_nearest_supported_resolution(self, height: int,
                                         width: int) -> Resolution:
        return self.radio_model.get_nearest_supported_resolution(height, width)

    def switch_to_deploy(self):
        return self.radio_model.switch_to_deploy()

    def forward(self, x: torch.Tensor):
        return self.radio_model.forward(x)

    # def get_input_embeddings(self):
    #     return self.embeddings

    # def forward(
    #     self,
    #     pixel_values: Optional[torch.Tensor] = None,
    #     pixel_embeds: Optional[torch.Tensor] = None,
    # ) -> torch.FloatTensor:
    #     if pixel_values is None and pixel_embeds is None:
    #         raise ValueError(
    #             'You have to specify pixel_values or pixel_embeds')

    #     if pixel_embeds is not None:
    #         hidden_states = pixel_embeds
    #     elif pixel_values is not None:
    #         if pixel_values.ndim == 4:
    #             hidden_states = self.embeddings(pixel_values)
    #         else:
    #             raise ValueError(
    #                 f'wrong pixel_values size: {pixel_values.shape}')

    #     encoder_outputs = self.encoder(inputs_embeds=hidden_states)

    #     return encoder_outputs

    # def load_weights(self, weights: Iterable[tuple[str,
    #                                                torch.Tensor]]) -> set[str]:
    #     params_dict = dict(self.named_parameters())
    #     loaded_params: set[str] = set()
    #     for name, loaded_weight in weights:
    #         param = params_dict[name]
    #         weight_loader = getattr(param, "weight_loader",
    #                                 default_weight_loader)
    #         weight_loader(param, loaded_weight)
    #         loaded_params.add(name)
    #     return loaded_params
