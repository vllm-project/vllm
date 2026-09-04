# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fuser detection for the Transformers modeling backend.

`get_fusers` traces a module class once (see `fx_utils`) and matches it against each
concrete fuser in `fusers`; `Fusers` caches the result per class for a whole model.
At most one match may give the module a different forward, and any number of fusers that
leave the forward alone apply alongside. `base.recursive_replace` then applies them per
instance. RMSNorm-shaped modules the tracer cannot match are warned about.
"""

from collections import UserDict
from typing import TYPE_CHECKING, TypeVar

from cachetools import cached
from torch import nn

from vllm.logger import init_logger
from vllm.model_executor.models.transformers.fusers import (
    AttentionFuser,
    BaseFuser,
    GLUFuser,
    MLAFuser,
    PackedQKVFuser,
    QKVFuser,
    RewriteFuser,
    RMSNormFuser,
)
from vllm.model_executor.models.transformers.fx_utils import trace

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)

F = TypeVar("F", bound=BaseFuser)


def key(module: nn.Module) -> tuple:
    """Cache key for `get_fusers`. Considers module type and its immediate children."""
    return (type(module), tuple(name for name, _ in module.named_children()))


FUSERS: tuple[type[BaseFuser], ...] = (
    MLAFuser,
    GLUFuser,
    QKVFuser,
    PackedQKVFuser,
    RMSNormFuser,
    AttentionFuser,
)
"""Every fuser, in match order: those that redefine the forward first, then those
that leave it alone. A new fuser is added here."""


@cached(cache={}, key=key)
def get_fusers(module: nn.Module) -> list[BaseFuser]:
    """Every fuser that applies to `module`'s class and shape (cached).

    The one that redefines the forward comes first, so applying them in order
    fuses before the rest read or adjust the result."""
    # Projection fusions need >=2 sibling linears; the RMSNorm fusion needs a
    # leaf module (raw tensor math, no submodules). Nothing else can match, and
    # tracing is skipped for it.
    n_linear = sum(isinstance(c, nn.Linear) for c in module.children())
    is_leaf = next(module.children(), None) is None
    graph = trace(module) if n_linear >= 2 or is_leaf else None

    fusers: list[BaseFuser] = []
    for fuser_cls in FUSERS:
        if fuser_cls.redefines_forward and (
            # A forward can only be rewritten from a trace of it, and only once,
            # because every rewrite starts from the original source.
            graph is None or any(fuser.redefines_forward for fuser in fusers)
        ):
            continue
        if (fuser := fuser_cls.match(graph, module)) is None:
            continue
        if isinstance(fuser, RewriteFuser):
            try:
                fuser.update_forward(module)
            except Exception as exc:
                logger.debug(
                    "Attempted to fuse %s using %s but failed "
                    "to update its forward method: %s",
                    type(module),
                    fuser_cls.__name__,
                    exc,
                )
                continue
        fusers.append(fuser)

    # A norm we could not match structurally is left unfused; flag likely misses.
    if (
        module.__class__.__name__.endswith("RMSNorm")
        and graph is not None
        and not any(isinstance(fuser, RMSNormFuser) for fuser in fusers)
    ):
        logger.warning_once(
            "%s looks like an RMSNorm but its computation did not match the "
            "expected pattern, so it was left unfused.",
            module.__class__.__name__,
        )
    return fusers


def get_fuser(module: nn.Module, fuser_cls: type[F]) -> "F | None":
    """The first `fuser_cls` that applies to `module`, if one does."""
    return next((f for f in get_fusers(module) if isinstance(f, fuser_cls)), None)


class Fusers(UserDict):
    """Mapping from module class and shape to fusers, for all fusable modules."""

    def __init__(self, model: nn.Module, vllm_config: "VllmConfig"):
        self.vllm_config = vllm_config
        super().__init__({key(m): get_fusers(m) for m in model.modules()})

    def __getitem__(self, m: nn.Module) -> list[BaseFuser]:
        """The fusers this instance can take, in the order to apply them."""
        fusers = self.data.get(key(m), ())
        return [fuser for fuser in fusers if fuser.validate(m, self.vllm_config)]
