from typing import Optional, Dict, cast

import oneflow as flow

from . import RearrangeMixin, ReduceMixin
from ._einmix import _EinmixMixin

__author__ = 'Tianhe Ren & Depeng Liang'


class Rearrange(RearrangeMixin, flow.nn.Module):
    def forward(self, input):
        return self._apply_recipe(input)


class Reduce(ReduceMixin, flow.nn.Module):
    def forward(self, input):
        return self._apply_recipe(input)


class EinMix(_EinmixMixin, flow.nn.Module):
    def _create_parameters(self, weight_shape, weight_bound, bias_shape, bias_bound):
        self.weight = flow.nn.Parameter(flow.zeros(weight_shape).uniform_(-weight_bound, weight_bound),
                                         requires_grad=True)
        if bias_shape is not None:
            self.bias = flow.nn.Parameter(flow.zeros(bias_shape).uniform_(-bias_bound, bias_bound),
                                           requires_grad=True)
        else:
            self.bias = None

    def _create_rearrange_layers(self,
                                 pre_reshape_pattern: Optional[str],
                                 pre_reshape_lengths: Optional[Dict],
                                 post_reshape_pattern: Optional[str],
                                 post_reshape_lengths: Optional[Dict],
                                 ):
        self.pre_rearrange = None
        if pre_reshape_pattern is not None:
            self.pre_rearrange = Rearrange(pre_reshape_pattern, **cast(dict, pre_reshape_lengths))

        self.post_rearrange = None
        if post_reshape_pattern is not None:
            self.post_rearrange = Rearrange(post_reshape_pattern, **cast(dict, post_reshape_lengths))

    def forward(self, input):
        if self.pre_rearrange is not None:
            input = self.pre_rearrange(input)
        result = flow.einsum(self.einsum_pattern, input, self.weight)
        if self.bias is not None:
            result += self.bias
        if self.post_rearrange is not None:
            result = self.post_rearrange(result)
        return result
