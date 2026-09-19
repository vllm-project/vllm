# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The order in which post-load processing visits a model's modules.

Cold start and layerwise reload both consume this and depend on each other, so
it lives beside them rather than in either: `model_loader/utils.py` reaches into
the reload package for its init-time metadata snapshot, and `reload/layerwise.py`
importing back from it would close a cycle.
"""

from collections.abc import Iterator
from enum import Enum, auto

from torch import nn

from vllm.model_executor.layers.attention import is_deferred_attention_layer


class PostLoadPhase(Enum):
    """Which pass of post-load processing a yielded module belongs to."""

    LAYER = auto()
    ATTENTION = auto()
    MODEL = auto()


def iter_post_load_modules(
    model: nn.Module,
) -> Iterator[tuple[str, nn.Module, PostLoadPhase]]:
    """Yield every module, then the deferred attention-like layers, then the model.

    Those layers come up twice because their hook reads weights a sibling may
    have decompressed or repacked during the first pass. The model is offered
    whether or not it carries a hook, so this commits to an order rather than to
    one hook name.
    """
    for name, module in model.named_modules():
        yield name, module, PostLoadPhase.LAYER

    for name, module in model.named_modules():
        if is_deferred_attention_layer(module):
            yield name, module, PostLoadPhase.ATTENTION

    yield "", model, PostLoadPhase.MODEL
