"""
Specialization of einops for torch.

Unfortunately, torch's jit scripting mechanism isn't strong enough,
and to have scripting supported at least for layers,
a number of additional moves is needed.

Design of main operations (dynamic resolution by lookup) is unlikely
to be implemented by torch.jit.script, 
but torch.compile seems to work with operations just fine.
"""
import warnings
from typing import Dict, List, Tuple

import torch
from einops.einops import TransformRecipe, _reconstruct_from_shape_uncached


class TorchJitBackend:
    """
    Completely static backend that mimics part of normal backend functionality
    but restricted to be within torchscript.
    """

    @staticmethod
    def reduce(x: torch.Tensor, operation: str, reduced_axes: List[int]):
        if operation == "min":
            return x.amin(dim=reduced_axes)
        elif operation == "max":
            return x.amax(dim=reduced_axes)
        elif operation == "sum":
            return x.sum(dim=reduced_axes)
        elif operation == "mean":
            return x.mean(dim=reduced_axes)
        elif operation == "prod":
            for i in list(sorted(reduced_axes))[::-1]:
                x = x.prod(dim=i)
            return x
        else:
            raise NotImplementedError("Unknown reduction ", operation)

    @staticmethod
    def transpose(x, axes: List[int]):
        return x.permute(axes)

    @staticmethod
    def stack_on_zeroth_dimension(tensors: List[torch.Tensor]):
        return torch.stack(tensors)

    @staticmethod
    def tile(x, repeats: List[int]):
        return x.repeat(repeats)

    @staticmethod
    def add_axes(x, n_axes: int, pos2len: Dict[int, int]):
        repeats = [-1] * n_axes
        for axis_position, axis_length in pos2len.items():
            x = torch.unsqueeze(x, axis_position)
            repeats[axis_position] = axis_length
        return x.expand(repeats)

    @staticmethod
    def is_float_type(x):
        return x.dtype in [torch.float16, torch.float32, torch.float64, torch.bfloat16]

    @staticmethod
    def shape(x):
        return x.shape

    @staticmethod
    def reshape(x, shape: List[int]):
        return x.reshape(shape)


# mirrors einops.einops._apply_recipe
def apply_for_scriptable_torch(
    recipe: TransformRecipe, tensor: torch.Tensor, reduction_type: str, axes_dims: List[Tuple[str, int]]
) -> torch.Tensor:
    backend = TorchJitBackend
    (
        init_shapes,
        axes_reordering,
        reduced_axes,
        added_axes,
        final_shapes,
        n_axes_w_added,
    ) = _reconstruct_from_shape_uncached(recipe, backend.shape(tensor), axes_dims=axes_dims)
    if init_shapes is not None:
        tensor = backend.reshape(tensor, init_shapes)
    if axes_reordering is not None:
        tensor = backend.transpose(tensor, axes_reordering)
    if len(reduced_axes) > 0:
        tensor = backend.reduce(tensor, operation=reduction_type, reduced_axes=reduced_axes)
    if len(added_axes) > 0:
        tensor = backend.add_axes(tensor, n_axes=n_axes_w_added, pos2len=added_axes)
    if final_shapes is not None:
        tensor = backend.reshape(tensor, final_shapes)
    return tensor


def allow_ops_in_compiled_graph():
    if hasattr(torch, "__version__") and torch.__version__[0] < "2":
        # torch._dynamo and torch.compile appear in pytorch 2.0
        return
    try:
        from torch._dynamo import allow_in_graph
    except ImportError:
        warnings.warn("allow_ops_in_compiled_graph failed to import torch: ensure pytorch >=2.0", ImportWarning)
        return

    from .einops import rearrange, reduce, repeat, einsum
    from .packing import pack, unpack

    allow_in_graph(rearrange)
    allow_in_graph(reduce)
    allow_in_graph(repeat)
    allow_in_graph(einsum)
    allow_in_graph(pack)
    allow_in_graph(unpack)

    # CF: https://github.com/pytorch/pytorch/blob/2df939aacac68e9621fbd5d876c78d86e72b41e2/torch/_dynamo/__init__.py#L222
    global _ops_were_registered_in_torchdynamo
    _ops_were_registered_in_torchdynamo = True


# module import automatically registers ops in torchdynamo
allow_ops_in_compiled_graph()
