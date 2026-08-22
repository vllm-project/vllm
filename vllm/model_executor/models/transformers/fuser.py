# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fuser detection for the Transformers modeling backend.

`get_fusers` traces a module class once (see `fx_utils`) and matches it against
each concrete fuser in `fusers`; `Fusers` caches the result per class for a
whole model. `base.recursive_replace` then applies the matched fusers per
instance. RMSNorm-shaped modules the tracer cannot match are warned about.
"""

from collections import UserDict
from typing import TYPE_CHECKING, TypeVar

from cachetools import cached
from torch import fx, nn

from vllm.logger import init_logger
from vllm.model_executor.models.transformers.fusers import (
    BaseFuser,
    GLUFuser,
    MLAFuser,
    PackedQKVFuser,
    QKVFuser,
    RewriteFuser,
    RMSNormFuser,
    SinkFuser,
)
from vllm.model_executor.models.transformers.fx_utils import trace

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)

_F = TypeVar("_F", bound=BaseFuser)

FUSER_GROUPS = (
    # Fusers that rewrite what the module computes. They are alternatives, so the
    # first one to match wins.
    (MLAFuser, GLUFuser, QKVFuser, PackedQKVFuser, RMSNormFuser),
    # Sinks are one attribute of an attention module, so they compose with any of
    # the above (e.g. an MLA attention that also has sinks).
    (SinkFuser,),
)
"""Groups of mutually exclusive fusers. At most one fuser per group applies to a
module, so a module can be matched by as many fusers as there are groups."""


def key(module: nn.Module) -> tuple:
    """Cache key for `get_fusers`. Considers module type, its immediate children
    and its own parameters."""
    return (
        type(module),
        tuple(name for name, _ in module.named_children()),
        tuple(name for name, _ in module.named_parameters(recurse=False)),
    )


def _match(
    fuser_cls: type[BaseFuser], graph: fx.Graph, module: nn.Module
) -> BaseFuser | None:
    """Match one fuser class against `module`, compiling the rewritten forward if
    it has one. `None` if the pattern or the rewrite does not apply."""
    fuser = fuser_cls.match(graph, module)
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
            return None
    return fuser


@cached(cache={}, key=key)
def get_fusers(module: nn.Module) -> tuple[BaseFuser, ...]:
    """The fusers for `module`'s class and shape (cached), empty if none match."""
    # Projection fusions and attention sinks need >=2 sibling linears (an attention
    # module has at least a query and an output projection); the RMSNorm fusion needs
    # a leaf module (raw tensor math, no submodules). Nothing else can match, and
    # tracing is skipped for it.
    n_linear = sum(isinstance(c, nn.Linear) for c in module.children())
    is_leaf = next(module.children(), None) is None
    if n_linear < 2 and not is_leaf:
        return ()
    if (graph := trace(module)) is None:
        return ()
    fusers = []
    for group in FUSER_GROUPS:
        for fuser_cls in group:
            if (fuser := _match(fuser_cls, graph, module)) is not None:
                fusers.append(fuser)
                break
    # A norm we could not match structurally is left unfused; flag likely misses.
    if not fusers and module.__class__.__name__.endswith("RMSNorm"):
        logger.warning_once(
            "%s looks like an RMSNorm but its computation did not match the "
            "expected pattern, so it was left unfused.",
            module.__class__.__name__,
        )
    return tuple(fusers)


def get_fuser(module: nn.Module, fuser_cls: type[_F] = BaseFuser) -> "_F | None":
    """The `fuser_cls` matched for `module`, `None` if none was.

    At most one fuser per group applies, so this is unambiguous for any `fuser_cls`
    that does not span groups.
    """
    return next((f for f in get_fusers(module) if isinstance(f, fuser_cls)), None)


class Fusers(UserDict):
    """Mapping from module class and shape to fusers, for all fusable modules."""

    def __init__(self, model: nn.Module, vllm_config: "VllmConfig"):
        self.vllm_config = vllm_config
        super().__init__({key(m): get_fusers(m) for m in model.modules()})

    def __getitem__(self, m: nn.Module) -> tuple[BaseFuser, ...]:
        """The fusers that apply to this instance, in the order to apply them."""
        return tuple(
            fuser
            for fuser in self.data.get(key(m), ())
            if fuser.validate(m, self.vllm_config)
        )
