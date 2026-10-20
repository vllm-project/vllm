# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any
from weakref import WeakKeyDictionary

from torch import nn

import vllm.distributed.parallel_state as parallel_state

_decoders: WeakKeyDictionary[nn.Module, nn.Module] = WeakKeyDictionary()


def get_layer_group_range(
    start_layer: int, end_layer: int, group_idx: int, num_groups: int
) -> tuple[int, int]:
    """Split the local PP partition into contiguous, nonempty groups."""
    num_layers = end_layer - start_layer
    if not 0 <= group_idx < num_groups <= num_layers:
        raise ValueError("Layer groups must be nonempty and lie within the PP rank.")
    size, remainder = divmod(num_layers, num_groups)
    start = start_layer + group_idx * size + min(group_idx, remainder)
    return start, start + size + (group_idx < remainder)


def get_layered_decoder(model: nn.Module) -> nn.Module:
    """Return the module whose forward iterates `layers[start_layer:end_layer]`.

    Every model that supports PP this way can run local layer groups: a group
    is executed as if it were its own pipeline stage.
    """
    decoder = _decoders.get(model)
    if decoder is None:
        candidates = [
            module
            for module in model.modules()
            if isinstance(getattr(module, "layers", None), nn.ModuleList)
            and isinstance(getattr(module, "start_layer", None), int)
            and isinstance(getattr(module, "end_layer", None), int)
        ]
        if len(candidates) != 1:
            raise ValueError(
                f"Layered prefill requires {type(model).__name__} to contain one "
                "decoder with `layers`, `start_layer` and `end_layer`; found "
                f"{len(candidates)}."
            )
        decoder = _decoders[model] = candidates[0]
    return decoder


class _PPGroupView:
    def __init__(self, group: Any, is_first_rank: bool, is_last_rank: bool):
        self._group = group
        self.is_first_rank = is_first_rank
        self.is_last_rank = is_last_rank

    def __getattr__(self, name: str) -> Any:
        return getattr(self._group, name)


@contextmanager
def layer_group_context(
    model: nn.Module, group_idx: int | None, num_groups: int
) -> Iterator[None]:
    """Run only one local layer group of `model`, or all layers if None.

    The decoder's layer range is narrowed and the PP group reports stage
    boundaries at the group, so the model embeds only before the first group
    and returns intermediate tensors before the last.
    """
    if group_idx is None:
        yield
        return
    decoder = get_layered_decoder(model)
    pp = parallel_state.get_pp_group()
    full_range = decoder.start_layer, decoder.end_layer
    start, end = get_layer_group_range(*full_range, group_idx, num_groups)
    decoder.start_layer, decoder.end_layer = start, end
    parallel_state._PP = _PPGroupView(  # type: ignore[assignment]
        pp,
        is_first_rank=pp.is_first_rank and group_idx == 0,
        is_last_rank=pp.is_last_rank and group_idx == num_groups - 1,
    )
    try:
        yield
    finally:
        decoder.start_layer, decoder.end_layer = full_range
        parallel_state._PP = pp
