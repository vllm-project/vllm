# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The unit a model swaps in: one MonoKernel call standing in for vLLM's path.

A ``MonoOp`` is created once per model instance behind its :class:`MonoSpec`,
then asked per step whether it can run. One call is not one launch: an op is free
to run several launches, or none, behind :meth:`MonoOp.forward`.

``torch.ops.vllm.mono_layer`` is the single registered custom op every model
reaches the layer-level ops through, so a model adds no ``direct_register_custom_op``
of its own. It is opaque to torch.compile, which is the point: the persistent
launch must not be split, reordered, or captured piecewise.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Self

import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.models.common.mono.runtime import MonoRuntime
from vllm.models.common.mono.spec import MonoSpec
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)


class MonoOp(ABC):
    """One model's MonoKernel, gated by its spec and served by a runtime.

    Subclasses set :attr:`spec`, implement :meth:`eligible` for per-step
    conditions the spec cannot express and :meth:`forward` for the call itself.

    Gating runs in tiers, widest first: the spec's ``opt_in`` switch, then the
    spec over ``VllmConfig`` and :meth:`refuse` over the loaded model (a refusal
    means the deployment cannot run the kernel at all, so both honour
    ``on_refusal``), then :meth:`decline` over the loaded model (this instance
    opts out, the rest of the model may not).

    The class attribute :attr:`spec` is what the kernel can do; an instance may
    narrow it to what it actually built (``self.spec = replace(type(self).spec,
    widths=...)``) before calling ``super().__init__``, since a spec is a value.

    Attributes:
        spec: This instance's :class:`MonoSpec`.
        refusals: Why this deployment cannot run the kernel, empty when it can.
        declines: Why this instance opts out, empty when it does not.
        rt: The runtime, created by :meth:`build` only once gating passes.

    """

    spec: MonoSpec

    def __init__(self, vllm_config: VllmConfig) -> None:
        self.vllm_config = vllm_config
        self.refusals = self.spec.refuse(vllm_config) or self.refuse()
        self.declines = [] if self.refusals else self.decline()
        self._rt: MonoRuntime | None = None

    @property
    def rt(self) -> MonoRuntime:
        assert self._rt is not None, f"{self.spec.name} has not been built"
        return self._rt

    @rt.setter
    def rt(self, rt: MonoRuntime) -> None:
        self._rt = rt

    def refuse(self) -> list[str]:
        """Deployment-level refusals only the loaded model can answer.

        Read after :meth:`MonoSpec.refuse` and honouring the same
        ``on_refusal``: a quantisation backend the kernel cannot read is the
        deployment's answer for every layer, not this layer's opt-out.
        """
        return []

    def decline(self) -> list[str]:
        """Reasons this instance opts out, read from the loaded model.

        A :class:`MonoSpec` is deliberately CPU-testable and sees only
        ``VllmConfig``; a quantisation backend chosen at load time, or a module
        a checkpoint does or does not carry, is declined here instead. A decline
        never raises: it is about this layer, not the deployment.
        """
        return []

    @property
    def ok(self) -> bool:
        return not self.refusals and not self.declines and self._rt is not None

    @classmethod
    def create(cls, vllm_config: VllmConfig, **kwargs) -> Self | None:
        """Build the op, or return None when any tier of gating declines it.

        Raises:
            ValueError: When the spec refuses and ``on_refusal`` is ``"raise"``.

        """
        spec = cls.spec
        if not spec.wanted(vllm_config):
            return None
        op = cls(vllm_config, **kwargs)
        if op.refusals:
            why = "; ".join(op.refusals)
            if spec.on_refusal == "raise":
                raise ValueError(f"{spec.name} cannot run this configuration: {why}")
            logger.info_once("%s is off: %s", spec.name, why)
            return None
        if op.declines:
            logger.info_once("%s declined: %s", spec.name, "; ".join(op.declines))
            return None
        op.build()
        return op

    def build(self) -> None:  # noqa: B027
        """Create the runtime and reserve its keys. Called once, after gating."""

    @abstractmethod
    def eligible(self, *args, **kwargs):
        """Whether this step takes the mono path, as a :class:`StepDecision`."""

    @abstractmethod
    def forward(self, *args, **kwargs):
        """Run the step the mono way."""


_ACTIVE: dict[int, MonoOp] = {}


def register_mono_layer_op(op: MonoOp, model_id: int) -> None:
    """Make ``op`` reachable from ``torch.ops.vllm.mono_layer`` for this model."""
    _ACTIVE[model_id] = op


def active_mono_layer_op(model_id: int) -> MonoOp | None:
    return _ACTIVE.get(model_id)


def _mono_layer(
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    layer_idx: int,
    model_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    op = _ACTIVE[model_id]
    return op.forward(positions, hidden_states, residual, layer_idx)


def _mono_layer_fake(
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    layer_idx: int,
    model_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.empty_like(hidden_states), torch.empty_like(residual)


direct_register_custom_op(
    op_name="mono_layer",
    op_func=_mono_layer,
    mutates_args=["hidden_states", "residual"],
    fake_impl=_mono_layer_fake,
)


def mono_layer(
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    layer_idx: int,
    model_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.ops.vllm.mono_layer(
        positions, hidden_states, residual, layer_idx, model_id
    )
