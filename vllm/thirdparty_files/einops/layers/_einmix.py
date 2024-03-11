from typing import Any, List, Optional, Dict

from einops import EinopsError
from einops.parsing import ParsedExpression
import warnings
import string
from ..einops import _product


def _report_axes(axes: set, report_message: str):
    if len(axes) > 0:
        raise EinopsError(report_message.format(axes))


class _EinmixMixin:
    def __init__(self, pattern: str, weight_shape: str, bias_shape: Optional[str]=None, **axes_lengths: Any):
        """
        EinMix - Einstein summation with automated tensor management and axis packing/unpacking.

        EinMix is an advanced tool, helpful tutorial:
        https://github.com/arogozhnikov/einops/blob/master/docs/3-einmix-layer.ipynb

        Imagine taking einsum with two arguments, one of each input, and one - tensor with weights
        >>> einsum('time batch channel_in, channel_in channel_out -> time batch channel_out', input, weight)

        This layer manages weights for you, syntax highlights separate role of weight matrix
        >>> EinMix('time batch channel_in -> time batch channel_out', weight_shape='channel_in channel_out')
        But otherwise it is the same einsum under the hood.

        Simple linear layer with bias term (you have one like that in your framework)
        >>> EinMix('t b cin -> t b cout', weight_shape='cin cout', bias_shape='cout', cin=10, cout=20)
        There is no restriction to mix the last axis. Let's mix along height
        >>> EinMix('h w c-> hout w c', weight_shape='h hout', bias_shape='hout', h=32, hout=32)
        Channel-wise multiplication (like one used in normalizations)
        >>> EinMix('t b c -> t b c', weight_shape='c', c=128)
        Multi-head linear layer (each head is own linear layer):
        >>> EinMix('t b (head cin) -> t b (head cout)', weight_shape='head cin cout', ...)

        ... ah yes, you need to specify all dimensions of weight shape/bias shape in parameters.

        Use cases:
        - when channel dimension is not last, use EinMix, not transposition
        - patch/segment embeddings
        - when need only within-group connections to reduce number of weights and computations
        - perfect as a part of sequential models
        - next-gen MLPs (follow tutorial to learn more)

        Uniform He initialization is applied to weight tensor and encounters for number of elements mixed.

        Parameters
        :param pattern: transformation pattern, left side - dimensions of input, right side - dimensions of output
        :param weight_shape: axes of weight. A tensor of this shape is created, stored, and optimized in a layer
        :param bias_shape: axes of bias added to output. Weights of this shape are created and stored. If `None` (the default), no bias is added.
        :param axes_lengths: dimensions of weight tensor
        """
        super().__init__()
        self.pattern = pattern
        self.weight_shape = weight_shape
        self.bias_shape = bias_shape
        self.axes_lengths = axes_lengths
        self.initialize_einmix(pattern=pattern, weight_shape=weight_shape, bias_shape=bias_shape, axes_lengths=axes_lengths)

    def initialize_einmix(self, pattern: str, weight_shape: str, bias_shape: Optional[str], axes_lengths: dict):
        left_pattern, right_pattern = pattern.split('->')
        left = ParsedExpression(left_pattern)
        right = ParsedExpression(right_pattern)
        weight = ParsedExpression(weight_shape)
        _report_axes(
            set.difference(right.identifiers, {*left.identifiers, *weight.identifiers}),
            'Unrecognized identifiers on the right side of EinMix {}'
        )

        if left.has_ellipsis or right.has_ellipsis or weight.has_ellipsis:
            raise EinopsError('Ellipsis is not supported in EinMix (right now)')
        if any(x.has_non_unitary_anonymous_axes for x in [left, right, weight]):
            raise EinopsError('Anonymous axes (numbers) are not allowed in EinMix')
        if '(' in weight_shape or ')' in weight_shape:
            raise EinopsError(f'Parenthesis is not allowed in weight shape: {weight_shape}')

        pre_reshape_pattern = None
        pre_reshape_lengths = None
        post_reshape_pattern = None
        if any(len(group) != 1 for group in left.composition):
            names: List[str] = []
            for group in left.composition:
                names += group
            composition = ' '.join(names)
            pre_reshape_pattern = f'{left_pattern}->{composition}'
            pre_reshape_lengths = {name: length for name, length in axes_lengths.items() if name in names}

        if any(len(group) != 1 for group in right.composition):
            names = []
            for group in right.composition:
                names += group
            composition = ' '.join(names)
            post_reshape_pattern = f'{composition}->{right_pattern}'

        self._create_rearrange_layers(pre_reshape_pattern, pre_reshape_lengths, post_reshape_pattern, {})

        for axis in weight.identifiers:
            if axis not in axes_lengths:
                raise EinopsError('Dimension {} of weight should be specified'.format(axis))
        _report_axes(
            set.difference(set(axes_lengths), {*left.identifiers, *weight.identifiers}),
            'Axes {} are not used in pattern',
        )
        _report_axes(
            set.difference(weight.identifiers, {*left.identifiers, *right.identifiers}),
            'Weight axes {} are redundant'
        )
        if len(weight.identifiers) == 0:
            warnings.warn('EinMix: weight has no dimensions (means multiplication by a number)')

        _weight_shape = [axes_lengths[axis] for axis, in weight.composition]
        # single output element is a combination of fan_in input elements
        _fan_in = _product([axes_lengths[axis] for axis, in weight.composition if axis not in right.identifiers])
        if bias_shape is not None:
            if not isinstance(bias_shape, str):
                raise EinopsError('bias shape should be string specifying which axes bias depends on')
            bias = ParsedExpression(bias_shape)
            _report_axes(
                set.difference(bias.identifiers, right.identifiers),
                'Bias axes {} not present in output'
            )
            _report_axes(
                set.difference(bias.identifiers, set(axes_lengths)),
                'Sizes not provided for bias axes {}',
            )

            _bias_shape = []
            for axes in right.composition:
                for axis in axes:
                    if axis in bias.identifiers:
                        _bias_shape.append(axes_lengths[axis])
                    else:
                        _bias_shape.append(1)
        else:
            _bias_shape = None

        weight_bound = (3 / _fan_in) ** 0.5
        bias_bound = (1 / _fan_in) ** 0.5
        self._create_parameters(_weight_shape, weight_bound, _bias_shape, bias_bound)

        # rewrite einsum expression with single-letter latin identifiers so that
        # expression will be understood by any framework
        mapped_identifiers = {*left.identifiers, *right.identifiers, *weight.identifiers}
        mapping2letters = {k: letter for letter, k in zip(string.ascii_lowercase, mapped_identifiers)}

        def write_flat(axes: list):
            return ''.join(mapping2letters[axis] for axis in axes)

        self.einsum_pattern: str = '{},{}->{}'.format(
            write_flat(left.flat_axes_order()),
            write_flat(weight.flat_axes_order()),
            write_flat(right.flat_axes_order()),
        )

    def _create_rearrange_layers(self,
                                 pre_reshape_pattern: Optional[str],
                                 pre_reshape_lengths: Optional[Dict],
                                 post_reshape_pattern: Optional[str],
                                 post_reshape_lengths: Optional[Dict]):
        raise NotImplementedError('Should be defined in framework implementations')

    def _create_parameters(self, weight_shape, weight_bound, bias_shape, bias_bound):
        """ Shape and implementations """
        raise NotImplementedError('Should be defined in framework implementations')

    def __repr__(self):
        params = repr(self.pattern)
        params += f", '{self.weight_shape}'"
        if self.bias_shape is not None:
            params += f", '{self.bias_shape}'"
        for axis, length in self.axes_lengths.items():
            params += ', {}={}'.format(axis, length)
        return '{}({})'.format(self.__class__.__name__, params)
