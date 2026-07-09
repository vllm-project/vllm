# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fuser detection for the Transformers modeling backend.

`get_fuser` traces a module class once (see `fx_utils`) and matches it against
each concrete fuser in `fusers`; `Fusers` caches the result per class for a
whole model. `base.recursive_replace` then applies the matched fuser per
instance. RMSNorm-shaped modules the tracer cannot match are warned about.
"""

from collections import UserDict
from typing import TYPE_CHECKING

from cachetools import cached
from torch import nn

from vllm.logger import init_logger
from vllm.model_executor.models.transformers.fusers import (
    BaseFuser,
    GLUFuser,
    QKVFuser,
    RMSNormFuser,
    StackedFuser,
)
from vllm.model_executor.models.transformers.fx_utils import trace

if TYPE_CHECKING:
    from vllm.config.model import ModelConfig

logger = init_logger(__name__)


@cached(cache={}, key=type)
def get_fuser(module: nn.Module) -> BaseFuser | None:
    """The fuser for `type(module)` (cached per class), or `None` if no match."""
    # Projection fusions need >=2 sibling linears; the RMSNorm fusion needs a
    # leaf module (raw tensor math, no submodules). Nothing else can match, and
    # tracing is skipped for it.
    n_linear = sum(isinstance(c, nn.Linear) for c in module.children())
    is_leaf = next(module.children(), None) is None
    if n_linear < 2 and not is_leaf:
        return None
    if (graph := trace(module)) is None:
        return None
    for fuser_cls in (GLUFuser, QKVFuser, RMSNormFuser):
        if (fuser := fuser_cls.match(graph, module)) is not None:
            if isinstance(fuser, StackedFuser):
                try:
                    fuser.update_forward(module)
                except Exception as exc:
                    # An unrecognised source just means we cannot fuse here.
                    logger.debug(
                        "Could not rewrite %s for fusion: %s", type(module), exc
                    )
                    return None
            return fuser
    # A norm we could not match structurally is left unfused; flag likely misses.
    if module.__class__.__name__.endswith("RMSNorm"):
        logger.warning_once(
            "%s looks like an RMSNorm but its computation did not match the "
            "expected pattern, so it was left unfused.",
            module.__class__.__name__,
        )
    return None


class Fusers(UserDict):
    """Mapping from module class to fuser, for all fusable classes in a model."""

    def __init__(self, model: nn.Module, model_config: "ModelConfig"):
        self.model_config = model_config
        super().__init__({type(m): get_fuser(m) for m in model.modules()})

    def __getitem__(self, m: nn.Module) -> BaseFuser | None:
        fuser = self.data.get(type(m))
        if fuser is not None and fuser.validate(m, self.model_config):
            return fuser
        return None
