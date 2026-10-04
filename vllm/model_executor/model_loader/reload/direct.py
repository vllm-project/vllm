# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Direct weight reload: checkpoint weights are written into the live parameters.

`start` records where every parameter and buffer lives, `model.load_weights`
runs unchanged so each `weight_loader` `copy_`s into that storage, and `finish`
checks nothing moved. Nothing inspects the model: choosing the mode is the
operator's statement that no post-load processing needs redoing.
"""

import torch

__all__ = ["direct_start", "direct_finish"]

_LIVE = "_direct_reload_live"


def direct_start(model: torch.nn.Module) -> None:
    if _LIVE in model.__dict__:
        raise RuntimeError("direct weight reload already in progress")
    if getattr(model, "_do_torchao_reload", False):
        raise RuntimeError(
            "torchao models re-quantize after loading; use reload_mode=layerwise"
        )
    model.__dict__[_LIVE] = _tensor_layouts(model)


def direct_finish(model: torch.nn.Module) -> None:
    before = model.__dict__.pop(_LIVE)
    after = _tensor_layouts(model)
    moved = [n for n, rec in before.items() if after.get(n) != rec]
    if moved:
        raise RuntimeError(
            f"direct weight reload relocated {', '.join(moved[:5])}"
            f"{', ...' if len(moved) > 5 else ''}: a loader replaced storage "
            "a CUDA graph may have captured, or changed its shape, strides or "
            "dtype."
        )


def _tensor_layouts(model: torch.nn.Module) -> dict[str, tuple]:
    return {
        name: (t.data_ptr(), tuple(t.shape), t.stride(), t.dtype)
        for name, t in (
            *model.named_parameters(remove_duplicate=False),
            *model.named_buffers(remove_duplicate=False),
        )
    }
