from typing import List, TypeVar, Tuple, Sequence

from einops import EinopsError

T = TypeVar('T')

Shape = Tuple[int, ...]


def pack(pattern: str, tensors: Sequence[T]) -> Tuple[T, List[Shape]]:
    axes = pattern.split()
    if len(axes) != len(set(axes)):
        raise EinopsError(f'Duplicates in axes names in pack("{pattern}", ...)')
    if '*' not in axes:
        raise EinopsError(f'No *-axis in pack("{pattern}", ...)')

    # need some validation of identifiers

    n_axes_before = axes.index('*')
    n_axes_after = len(axes) - n_axes_before - 1
    min_axes = n_axes_before + n_axes_after

    xp = tensors[0].__array_namespace__()

    reshaped_tensors: List[T] = []
    packed_shapes: List[Shape] = []
    for i, tensor in enumerate(tensors):
        shape = tensor.shape
        if len(shape) < min_axes:
            raise EinopsError(f'packed tensor #{i} (enumeration starts with 0) has shape {shape}, '
                              f'while pattern {pattern} assumes at least {min_axes} axes')
        axis_after_packed_axes = len(shape) - n_axes_after
        packed_shapes.append(shape[n_axes_before:])
        reshaped_tensors.append(
            xp.reshape(tensor, (*shape[:n_axes_before], -1, *shape[axis_after_packed_axes:]))
        )

    return xp.concat(reshaped_tensors, axis=n_axes_before), packed_shapes


def prod(x: Shape) -> int:
    result = 1
    for i in x:
        result *= i
    return result


def unpack(pattern: str, tensor: T, packed_shapes: List[Shape]) -> List[T]:
    axes = pattern.split()
    if len(axes) != len(set(axes)):
        raise EinopsError(f'Duplicates in axes names in unpack("{pattern}", ...)')
    if '*' not in axes:
        raise EinopsError(f'No *-axis in unpack("{pattern}", ...)')

    # need some validation of identifiers

    input_shape = tensor.shape
    if len(input_shape) != len(axes):
        raise EinopsError(f'unpack({pattern}, ...) received input of wrong dim with shape {input_shape}')

    unpacked_axis = axes.index('*')

    lengths_of_composed_axes: List[int] = [
        -1 if -1 in p_shape else prod(p_shape)
        for p_shape in packed_shapes
    ]

    n_unknown_composed_axes = sum(x == -1 for x in lengths_of_composed_axes)
    if n_unknown_composed_axes > 1:
        raise EinopsError(
            f"unpack({pattern}, ...) received more than one -1 in {packed_shapes} and can't infer dimensions"
        )

    # following manipulations allow to skip some shape verifications
    # and leave them to backends

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
            split_positions[j] = split_positions[j + 1] + lengths_of_composed_axes[j]

    xp = tensor.__array_namespace__()
    shape_start = input_shape[:unpacked_axis]
    shape_end = input_shape[unpacked_axis + 1:]
    slice_filler = (slice(None, None),) * unpacked_axis
    return [
        xp.reshape(
            # shortest way slice arbitrary axis
            tensor[(*slice_filler, slice(split_positions[i], split_positions[i + 1]))],
            (*shape_start, *element_shape, *shape_end)
        )
        for i, element_shape in enumerate(packed_shapes)
    ]


if __name__ == '__main__':
    import numpy.array_api as np

    H = 100
    W = 101
    C = 3

    r = np.zeros((H, W))
    g = np.zeros((H, W))
    b = np.zeros((H, W))
    embeddings = np.zeros((H, W, 32))

    im = np.stack([r, g, b], axis=-1)
    print(im.shape)

    image, shapes = pack('h w *', [r, g, b])
    print(image.shape, shapes)

    print(type(image))
    print(type(im))
    assert np.all(np.equal(image, im))

    images_and_embedding, shapes = pack('h w *', [r, g, b, embeddings])
    print(images_and_embedding.shape, shapes)
    r2, g2, b2, embeddings2 = unpack('h w *', images_and_embedding, shapes)
    assert np.all(np.equal(r, r2))
    assert np.all(np.equal(g, g2))
    assert np.all(np.equal(b, b2))
    assert np.all(np.equal(embeddings, embeddings2))

    print([x.shape for x in unpack('h w *', images_and_embedding, shapes[1:])])

    print('all is fine')
