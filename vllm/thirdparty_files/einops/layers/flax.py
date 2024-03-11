from dataclasses import field
from typing import Optional, Dict, cast

import flax.linen as nn
import jax
import jax.numpy as jnp

from . import RearrangeMixin, ReduceMixin
from ._einmix import _EinmixMixin

__author__ = 'Alex Rogozhnikov'


class Reduce(nn.Module):
    pattern: str
    reduction: str
    sizes: dict = field(default_factory=lambda: {})

    def setup(self):
        self.reducer = ReduceMixin(self.pattern, self.reduction, **self.sizes)

    def __call__(self, input):
        return self.reducer._apply_recipe(input)


class Rearrange(nn.Module):
    pattern: str
    sizes: dict = field(default_factory=lambda: {})

    def setup(self):
        self.rearranger = RearrangeMixin(self.pattern, **self.sizes)

    def __call__(self, input):
        return self.rearranger._apply_recipe(input)


class EinMix(nn.Module, _EinmixMixin):
    pattern: str
    weight_shape: str
    bias_shape: Optional[str] = None
    sizes: dict = field(default_factory=lambda: {})

    def setup(self):
        self.initialize_einmix(
            pattern=self.pattern,
            weight_shape=self.weight_shape,
            bias_shape=self.bias_shape,
            axes_lengths=self.sizes,
        )

    def _create_parameters(self, weight_shape, weight_bound, bias_shape, bias_bound):
        self.weight = self.param("weight", jax.nn.initializers.uniform(weight_bound), weight_shape)

        if bias_shape is not None:
            self.bias = self.param("bias", jax.nn.initializers.uniform(bias_bound), bias_shape)
        else:
            self.bias = None

    def _create_rearrange_layers(self,
                                 pre_reshape_pattern: Optional[str],
                                 pre_reshape_lengths: Optional[Dict],
                                 post_reshape_pattern: Optional[str],
                                 post_reshape_lengths: Optional[Dict]):
        self.pre_rearrange = None
        if pre_reshape_pattern is not None:
            self.pre_rearrange = Rearrange(pre_reshape_pattern, sizes=cast(dict, pre_reshape_lengths))

        self.post_rearrange = None
        if post_reshape_pattern is not None:
            self.post_rearrange = Rearrange(post_reshape_pattern, sizes=cast(dict, post_reshape_lengths))

    def __call__(self, input):
        if self.pre_rearrange is not None:
            input = self.pre_rearrange(input)
        result = jnp.einsum(self.einsum_pattern, input, self.weight)
        if self.bias is not None:
            result += self.bias
        if self.post_rearrange is not None:
            result = self.post_rearrange(result)
        return result
