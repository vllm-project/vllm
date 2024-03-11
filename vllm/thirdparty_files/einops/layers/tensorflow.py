from typing import List, Optional, Dict, cast

import tensorflow as tf
from tensorflow.keras.layers import Layer

from .._backends import UnknownSize
from . import RearrangeMixin, ReduceMixin
from ._einmix import _EinmixMixin
from ..einops import TransformRecipe, _reconstruct_from_shape_uncached

__author__ = 'Alex Rogozhnikov'


def _compute_output_shape(recipe: TransformRecipe, input_shape) -> List[Optional[int]]:
    input_shape = [UnknownSize() if d is None else int(d) for d in input_shape]
    init_shapes, reduced_axes, axes_reordering, added_axes, final_shape = \
        _reconstruct_from_shape_uncached(recipe, input_shape)
    output_shape: List[Optional[int]] = [None if isinstance(d, UnknownSize) else int(d) for d in final_shape]
    return output_shape


class Rearrange(RearrangeMixin, Layer):
    def compute_output_shape(self, input_shape):
        return _compute_output_shape(self.recipe(), input_shape)

    def call(self, inputs):
        return self._apply_recipe(inputs)

    def get_config(self):
        return {'pattern': self.pattern, **self.axes_lengths}


class Reduce(ReduceMixin, Layer):
    def compute_output_shape(self, input_shape):
        return _compute_output_shape(self.recipe(), input_shape)

    def call(self, inputs):
        return self._apply_recipe(inputs)

    def get_config(self):
        return {'pattern': self.pattern, 'reduction': self.reduction, **self.axes_lengths}


class EinMix(_EinmixMixin, Layer):
    def _create_parameters(self, weight_shape, weight_bound, bias_shape, bias_bound):
        self.weight = tf.Variable(tf.random_uniform_initializer(-weight_bound, weight_bound)(shape=weight_shape),
                                  trainable=True)
        if bias_shape is not None:
            self.bias = tf.Variable(tf.random_uniform_initializer(-bias_bound, bias_bound)(shape=bias_shape),
                                    trainable=True)
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

    def build(self, input_shape):
        pass

    def call(self, inputs):
        if self.pre_rearrange is not None:
            inputs = self.pre_rearrange(inputs)
        result = tf.einsum(self.einsum_pattern, inputs, self.weight)
        if self.bias is not None:
            result = result + self.bias
        if self.post_rearrange is not None:
            result = self.post_rearrange(result)
        return result

    def get_config(self):
        return {'pattern': self.pattern,
                'weight_shape': self.weight_shape,
                'bias_shape': self.bias_shape,
                **self.axes_lengths}
