from typing import List, Tuple, Sequence
from .einops import Tensor, Reduction, EinopsError, _prepare_transformation_recipe, _apply_recipe_array_api
from .packing import analyze_pattern, prod


def reduce(tensor: Tensor, pattern: str, reduction: Reduction, **axes_lengths: int) -> Tensor:
    if isinstance(tensor, list):
        if len(tensor) == 0:
            raise TypeError("Einops can't be applied to an empty list")
        xp = tensor[0].__array_namespace__()
        tensor = xp.stack(tensor)
    else:
        xp = tensor.__array_namespace__()
    try:
        hashable_axes_lengths = tuple(axes_lengths.items())
        recipe = _prepare_transformation_recipe(pattern, reduction, axes_names=tuple(axes_lengths), ndim=tensor.ndim)
        return _apply_recipe_array_api(
            xp,
            recipe=recipe, tensor=tensor, reduction_type=reduction, axes_lengths=hashable_axes_lengths,
        )
    except EinopsError as e:
        message = ' Error while processing {}-reduction pattern "{}".'.format(reduction, pattern)
        if not isinstance(tensor, list):
            message += "\n Input tensor shape: {}. ".format(tensor.shape)
        else:
            message += "\n Input is list. "
        message += "Additional info: {}.".format(axes_lengths)
        raise EinopsError(message + "\n {}".format(e))



def repeat(tensor: Tensor, pattern: str, **axes_lengths) -> Tensor:
    return reduce(tensor, pattern, reduction="repeat", **axes_lengths)


def rearrange(tensor: Tensor, pattern: str, **axes_lengths) -> Tensor:
    return reduce(tensor, pattern, reduction="rearrange", **axes_lengths)


def asnumpy(tensor: Tensor):
    import numpy as np
    return np.from_dlpack(tensor)

Shape = Tuple

def pack(tensors: Sequence[Tensor], pattern: str) -> Tuple[Tensor, List[Shape]]:
    n_axes_before, n_axes_after, min_axes = analyze_pattern(pattern, 'pack')
    xp = tensors[0].__array_namespace__()

    reshaped_tensors: List[Tensor] = []
    packed_shapes: List[Shape] = []
    for i, tensor in enumerate(tensors):
        shape = tensor.shape
        if len(shape) < min_axes:
            raise EinopsError(f'packed tensor #{i} (enumeration starts with 0) has shape {shape}, '
                              f'while pattern {pattern} assumes at least {min_axes} axes')
        axis_after_packed_axes = len(shape) - n_axes_after
        packed_shapes.append(shape[n_axes_before:axis_after_packed_axes])
        reshaped_tensors.append(xp.reshape(tensor, (*shape[:n_axes_before], -1, *shape[axis_after_packed_axes:])))

    return xp.concat(reshaped_tensors, axis=n_axes_before), packed_shapes



def unpack(tensor: Tensor, packed_shapes: List[Shape], pattern: str) -> List[Tensor]:
    xp = tensor.__array_namespace__()
    n_axes_before, n_axes_after, min_axes = analyze_pattern(pattern, opname='unpack')

    # backend = get_backend(tensor)
    input_shape = tensor.shape
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
            xp.reshape(
                # shortest way slice arbitrary axis
                tensor[(*slice_filler, slice(split_positions[i], split_positions[i + 1]), ...)],
                (*shape_start, *element_shape, *shape_end)
            )
            for i, element_shape in enumerate(packed_shapes)
        ]
    except BaseException:
        # this hits if there is an error during reshapes, which means passed shapes were incorrect
        raise RuntimeError(f'Error during unpack(..., "{pattern}"): could not split axis of size {split_positions[-1]}'
                           f' into requested {packed_shapes}')
