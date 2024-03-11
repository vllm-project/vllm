from typing import Optional, Dict, cast

import chainer

from . import RearrangeMixin, ReduceMixin
from ._einmix import _EinmixMixin

__author__ = 'Alex Rogozhnikov'


class Rearrange(RearrangeMixin, chainer.Link):
    def __call__(self, x):
        return self._apply_recipe(x)


class Reduce(ReduceMixin, chainer.Link):
    def __call__(self, x):
        return self._apply_recipe(x)


class EinMix(_EinmixMixin, chainer.Link):
    def _create_parameters(self, weight_shape, weight_bound, bias_shape, bias_bound):
        uniform = chainer.variable.initializers.Uniform
        with self.init_scope():
            self.weight = chainer.variable.Parameter(uniform(weight_bound), weight_shape)
            if bias_shape is not None:
                self.bias = chainer.variable.Parameter(uniform(bias_bound), bias_shape)
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

    def __call__(self, input):
        if self.pre_rearrange is not None:
            input = self.pre_rearrange(input)
        result = chainer.functions.einsum(self.einsum_pattern, input, self.weight)
        if self.bias is not None:
            result = result + self.bias
        if self.post_rearrange is not None:
            result = self.post_rearrange(result)
        return result
