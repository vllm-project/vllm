__author__ = 'Alex Rogozhnikov'

from typing import Any, Dict


from ..einops import TransformRecipe, _apply_recipe, _prepare_recipes_for_all_dims, get_backend
from .. import EinopsError


class RearrangeMixin:
    """
    Rearrange layer behaves identically to einops.rearrange operation.

    :param pattern: str, rearrangement pattern
    :param axes_lengths: any additional specification of dimensions

    See einops.rearrange for source_examples.
    """

    def __init__(self, pattern: str, **axes_lengths: Any) -> None:
        super().__init__()
        self.pattern = pattern
        self.axes_lengths = axes_lengths
        # self._recipe = self.recipe()  # checking parameters
        self._multirecipe = self.multirecipe()
        self._axes_lengths = tuple(self.axes_lengths.items())

    def __repr__(self) -> str:
        params = repr(self.pattern)
        for axis, length in self.axes_lengths.items():
            params += ', {}={}'.format(axis, length)
        return '{}({})'.format(self.__class__.__name__, params)

    def multirecipe(self) -> Dict[int, TransformRecipe]:
        try:
            return _prepare_recipes_for_all_dims(
                self.pattern, operation='rearrange', axes_names=tuple(self.axes_lengths)
            )
        except EinopsError as e:
            raise EinopsError(' Error while preparing {!r}\n {}'.format(self, e))

    def _apply_recipe(self, x):
        backend = get_backend(x)
        return _apply_recipe(
            backend=backend,
            recipe=self._multirecipe[len(x.shape)],
            tensor=x,
            reduction_type='rearrange',
            axes_lengths=self._axes_lengths,
        )

    def __getstate__(self):
        return {'pattern': self.pattern, 'axes_lengths': self.axes_lengths}

    def __setstate__(self, state):
        self.__init__(pattern=state['pattern'], **state['axes_lengths'])


class ReduceMixin:
    """
    Reduce layer behaves identically to einops.reduce operation.

    :param pattern: str, rearrangement pattern
    :param reduction: one of available reductions ('min', 'max', 'sum', 'mean', 'prod'), case-sensitive
    :param axes_lengths: any additional specification of dimensions

    See einops.reduce for source_examples.
    """

    def __init__(self, pattern: str, reduction: str, **axes_lengths: Any):
        super().__init__()
        self.pattern = pattern
        self.reduction = reduction
        self.axes_lengths = axes_lengths
        self._multirecipe = self.multirecipe()
        self._axes_lengths = tuple(self.axes_lengths.items())

    def __repr__(self):
        params = '{!r}, {!r}'.format(self.pattern, self.reduction)
        for axis, length in self.axes_lengths.items():
            params += ', {}={}'.format(axis, length)
        return '{}({})'.format(self.__class__.__name__, params)

    def multirecipe(self) -> Dict[int, TransformRecipe]:
        try:
            return _prepare_recipes_for_all_dims(
                self.pattern, operation=self.reduction, axes_names=tuple(self.axes_lengths)
            )
        except EinopsError as e:
            raise EinopsError(' Error while preparing {!r}\n {}'.format(self, e))

    def _apply_recipe(self, x):
        backend = get_backend(x)
        return _apply_recipe(
            backend=backend,
            recipe=self._multirecipe[len(x.shape)],
            tensor=x,
            reduction_type=self.reduction,
            axes_lengths=self._axes_lengths,
        )

    def __getstate__(self):
        return {'pattern': self.pattern, 'reduction': self.reduction, 'axes_lengths': self.axes_lengths}

    def __setstate__(self, state):
        self.__init__(pattern=state['pattern'], reduction=state['reduction'], **state['axes_lengths'])
