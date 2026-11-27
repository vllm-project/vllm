# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any


def tensors_str_no_data(arg: Any) -> str:
    """Format `arg`, showing tensors in tuples, lists and dicts as metadata only."""
    import torch

    if isinstance(arg, torch.Tensor):
        return (
            f"{type(arg).__name__}(shape={tuple(arg.shape)}, dtype={arg.dtype}, "
            f"device={arg.device})"
        )
    if isinstance(arg, tuple):
        items = ", ".join(tensors_str_no_data(a) for a in arg)
        return f"({items},)" if len(arg) == 1 else f"({items})"
    if isinstance(arg, list):
        return f"[{', '.join(tensors_str_no_data(a) for a in arg)}]"
    if isinstance(arg, dict):
        items = ", ".join(f"{k!r}: {tensors_str_no_data(v)}" for k, v in arg.items())
        return f"{{{items}}}"
    return repr(arg)
