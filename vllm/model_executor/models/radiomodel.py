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
from timm.models import VisionTransformer, create_model
from transformers import PretrainedConfig

from vllm.model_executor.layers.quantization import QuantizationConfig

from . import eradio_model
# Import all required modules.
from .adaptor_base import AdaptorBase, AdaptorInput, RadioOutput
from .adaptor_generic import AdaptorBase, GenericAdaptor
from .enable_cpe_support import enable_cpe
from .extra_models import *
# Register extra models
from .extra_timm_models import *
from .feature_normalizer import (FeatureNormalizer,
                                 IntermediateFeatureNormalizer)
#from .radio_model import create_model_from_args
#from .radio_model import RADIOModel as RADIOModelBase, Resolution
from .input_conditioner import InputConditioner, get_default_conditioner


class Resolution(NamedTuple):
    height: int
    width: int


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
        elif isinstance(self.model, eradio_model.ERADIO):
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
