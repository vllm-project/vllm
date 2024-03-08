from functools import lru_cache
from typing import List, Union, TypeVar, Tuple, Sequence

from einops import EinopsError

from einops._backends import get_backend
from einops.parsing import ParsedExpression

Tensor = TypeVar('Tensor')

Shape = Union[Tuple[int, ...], List[int]]


@lru_cache(maxsize=128)
def analyze_pattern(pattern: str, opname: str) -> Tuple[int, int, int]:
    # Maybe some validation of identifiers?
    axes = pattern.split()
    axes_set = set(axes)
    if len(axes) != len(axes_set):
        raise EinopsError(f'Duplicates in axes names in {opname}(..., "{pattern}")')
    if '*' not in axes_set:
        raise EinopsError(f'No *-axis in {opname}(..., "{pattern}")')
    for axis in axes:
        if axis != '*':
            is_valid, reason = ParsedExpression.check_axis_name_return_reason(axis)
            if not is_valid:
                raise EinopsError(f'Invalid axis name {axis} in {opname}(..., "{pattern}")')
    n_axes_before = axes.index('*')
    n_axes_after = len(axes) - n_axes_before - 1
    min_axes = n_axes_before + n_axes_after
    return n_axes_before, n_axes_after, min_axes


def pack(tensors: Sequence[Tensor], pattern: str) -> Tuple[Tensor, List[Shape]]:
    """
    Packs several tensors into one.
    See einops tutorial for introduction into packing (and how it replaces stack and concatenation).

    Parameters:
        tensors: tensors to be packed, can be of different dimensionality
        pattern: pattern that is shared for all inputs and output, e.g. "i j * k" or "batch seq *"

    Returns:
        (packed_tensor, packed_shapes aka PS)

    Example:
    ```python
    >>> from numpy import zeros as Z
    >>> inputs = [Z([2, 3, 5]), Z([2, 3, 7, 5]), Z([2, 3, 7, 9, 5])]
    >>> packed, ps = pack(inputs, 'i j * k')
    >>> packed.shape, ps
    ((2, 3, 71, 5), [(), (7,), (7, 9)])
    ```

    In this example, axes were matched to: i=2, j=3, k=5 based on order (first, second, and last).
    All other axes were 'packed' and concatenated.
    PS (packed shapes) contains information about axes that were matched to '*' in every input.
    Resulting tensor has as many elements as all inputs in total.

    Packing can be reversed with unpack, which additionally needs PS (packed shapes) to reconstruct order.

    ```python
    >>> inputs_unpacked = unpack(packed, ps, 'i j * k')
    >>> [x.shape for x in inputs_unpacked]
    [(2, 3, 5), (2, 3, 7, 5), (2, 3, 7, 9, 5)]
    ```

    Read the tutorial for introduction and application scenarios.
    """
    n_axes_before, n_axes_after, min_axes = analyze_pattern(pattern, 'pack')

    # packing zero tensors is illegal
    backend = get_backend(tensors[0])

    reshaped_tensors: List[Tensor] = []
    packed_shapes: List[Shape] = []
    for i, tensor in enumerate(tensors):
        shape = backend.shape(tensor)
        if len(shape) < min_axes:
            raise EinopsError(f'packed tensor #{i} (enumeration starts with 0) has shape {shape}, '
                              f'while pattern {pattern} assumes at least {min_axes} axes')
        axis_after_packed_axes = len(shape) - n_axes_after
        packed_shapes.append(shape[n_axes_before:axis_after_packed_axes])
        reshaped_tensors.append(
            backend.reshape(tensor, (*shape[:n_axes_before], -1, *shape[axis_after_packed_axes:]))
        )

    return backend.concat(reshaped_tensors, axis=n_axes_before), packed_shapes


def prod(x: Shape) -> int:
    result = 1
    for i in x:
        result *= i
    return result


def unpack(tensor: Tensor, packed_shapes: List[Shape], pattern: str) -> List[Tensor]:
    """
    Unpacks a single tensor into several by splitting over a selected axes.
    See einops tutorial for introduction into packing (and how it replaces stack and concatenation).

    Parameters:
        tensor: tensor to be unpacked
        packed_shapes: packed_shapes (aka PS) is a list of shapes that take place of '*' in each output.
            output will contain a single tensor for every provided shape
        pattern: pattern that is shared for input and all outputs, e.g. "i j * k" or "batch seq *",
            where * designates an axis to be unpacked

    Returns:
        list of tensors

    If framework supports views, results are views to the original tensor.

    Example:
    ```python
    >>> from numpy import zeros as Z
    >>> inputs = [Z([2, 3, 5]), Z([2, 3, 7, 5]), Z([2, 3, 7, 9, 5])]
    >>> packed, ps = pack(inputs, 'i j * k')
    >>> packed.shape, ps
    ((2, 3, 71, 5), [(), (7,), (7, 9)])
    ```

    In this example, axes were matched to: i=2, j=3, k=5 based on order (first, second, and last).
    All other axes were 'packed' and concatenated.
    PS (packed shapes) contains information about axes that were matched to '*' in every input.
    Resulting tensor has as many elements as all inputs in total.

    Packing can be reversed with unpack, which additionally needs PS (packed shapes) to reconstruct order.

    ```python
    >>> inputs_unpacked = unpack(packed, ps, 'i j * k')
    >>> [x.shape for x in inputs_unpacked]
    [(2, 3, 5), (2, 3, 7, 5), (2, 3, 7, 9, 5)]
    ```

    Read the tutorial for introduction and application scenarios.
    """
    n_axes_before, n_axes_after, min_axes = analyze_pattern(pattern, opname='unpack')

    backend = get_backend(tensor)
    input_shape = backend.shape(tensor)
    if len(input_shape) != n_axes_before + 1 + n_axes_after:
        raise EinopsError(f'unpack(..., {pattern}) received input of wrong dim with shape {input_shape}')

    unpacked_axis: int = n_axes_before

    lengths_of_composed_axes: List[int] = [
        -1 if -1 in p_shape else prod(p_shape)
        for p_shape in packed_shapes
    ]

    n_unknown_composed_axes = sum(x == -1 for x in lengths_of_composed_axes)
    if n_unknown_composed_axes > 1:
        raise EinopsError(
            f"unpack(..., {pattern}) received more than one -1 in {packed_shapes} and can't infer dimensions"
        )

    # following manipulations allow to skip some shape verifications
    # and leave it to backends

    # [[], [2, 3], [4], [-1, 5], [6]] < examples of packed_axis
    # split positions when computed should be
    # [0,   1,      7,   11,      N-6 , N ], where N = length of axis
    split_positions = [0] * len(packed_shapes) + [input_shape[unpacked_axis]]
    if n_unknown_composed_axes == 0:
        for i, x in enumerate(lengths_of_composed_axes[:-1]):
            split_positions[i + 1] = split_positions[i] + x
    else:
        unknown_composed_axis: int = lengths_of_composed_axes.index(-1)
        for i in range(unknown_composed_axis):
            split_positions[i + 1] = split_positions[i] + lengths_of_composed_axes[i]
        for j in range(unknown_composed_axis + 1, len(lengths_of_composed_axes))[::-1]:
            split_positions[j] = split_positions[j + 1] - lengths_of_composed_axes[j]

    shape_start = input_shape[:unpacked_axis]
    shape_end = input_shape[unpacked_axis + 1:]
    slice_filler = (slice(None, None),) * unpacked_axis
    try:
        return [
            backend.reshape(
                # shortest way slice arbitrary axis
                tensor[(*slice_filler, slice(split_positions[i], split_positions[i + 1]))],
                (*shape_start, *element_shape, *shape_end)
            )
            for i, element_shape in enumerate(packed_shapes)
        ]
    except BaseException:
        # this hits if there is an error during reshapes, which means passed shapes were incorrect
        raise RuntimeError(f'Error during unpack(..., "{pattern}"): could not split axis of size {split_positions[-1]}'
                           f' into requested {packed_shapes}')
