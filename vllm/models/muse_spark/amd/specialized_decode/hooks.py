# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The specialized decode's hooks on ``MuseSpark1xForCausalLM``, kept here so the
model only calls them: ``init_specialized_decode`` in ``__init__`` (after the
config check, which runs before any module allocates weights),
``plan_specialized_decode`` in ``forward``, ``after_specialized_decode_step`` in
``compute_logits``, ``refuse_weight_reload`` and
``materialize_for_specialized_decode`` in ``process_weights_after_loading``. The
model state (``vllm.models.muse_spark.amd.model_state.MuseSparkModelState``)
builds the executor after load.

Nothing here imports the model or FlyDSL at module level."""

from typing import TYPE_CHECKING

import torch

from vllm.models.muse_spark.amd.specialized_decode.runtime import DecodeConfig

if TYPE_CHECKING:
    from vllm.models.muse_spark.amd.specialized_decode.runtime.decode import (
        SpecializedDecode,
        Step,
    )
    from vllm.v1.worker.gpu.model_states.interface import ModelState


class SpecializedDecodeMixin:
    """``self.model`` is the ``MuseSpark1xModel`` the executor runs over."""

    model: torch.nn.Module
    specialized_decode_config: DecodeConfig | None
    specialized_decode: "SpecializedDecode | None"

    def init_specialized_decode(self, config: DecodeConfig | None) -> None:
        """``config``: ``specialized_decode_config(vllm_config)``, None for the
        stock decode."""
        self.specialized_decode_config = config
        self.specialized_decode = None

    def build_specialized_decode(self, vllm_config=None) -> None:
        """From the model state, once the weights are loaded: stage-2 checks,
        peer memory and every kernel build, before memory profiling."""
        from vllm.models.muse_spark.amd.specialized_decode.runtime.decode import (
            SpecializedDecode,
        )

        assert self.specialized_decode_config is not None
        self.specialized_decode = SpecializedDecode(
            self.model, self.specialized_decode_config, vllm_config
        )

    def get_model_state_cls(self) -> "type[ModelState]":
        """Muse Spark's model state, mono on or off (off: ``DefaultModelState``
        with no-op hooks)."""
        from vllm.models.muse_spark.amd.model_state import MuseSparkModelState

        return MuseSparkModelState

    def plan_specialized_decode(
        self, positions: torch.Tensor, intermediate_tensors
    ) -> "Step | None":
        """This step's plan (``Step.layer(i)`` per decoder layer), None for the
        stock forward."""
        decode = getattr(self, "specialized_decode", None)
        return None if decode is None else decode.plan(positions, intermediate_tensors)

    def after_specialized_decode_step(self) -> None:
        """Once a step on every TP rank: the fail-stop poll."""
        decode = getattr(self, "specialized_decode", None)
        if decode is not None:
            decode.after_step()

    def refuse_weight_reload(self) -> None:
        """The kernels hold views of the loaded parameters: once the executor
        exists the weights may not be reloaded or updated."""
        if getattr(self, "specialized_decode", None) is not None:
            raise RuntimeError(
                "Muse Spark specialized decode: reloading or updating weights is "
                "not supported (the kernels read the loaded weights in place); "
                "restart the engine"
            )

    def materialize_for_specialized_decode(self) -> None:
        """With the specialized decode on, makes sure the residual-path constants
        it reads exist: ``load_weights`` materializes them, but a load that never
        calls it (``--load-format dummy``) leaves the gated residuals without
        ``alpha`` / ``beta`` and the router output multiplier unfolded. Only when
        something is missing, so a real load's bf16x3 router planes are not split
        twice. With it off the model is untouched (the stock layers tolerate
        unmaterialized constants)."""
        if getattr(self, "specialized_decode_config", None) is None:
            return
        from vllm.models.muse_spark.common.layers import (
            MuseSpark1xNormPreservingResidualConnection,
            materialize_residual_path_constants,
        )

        assert isinstance(self, torch.nn.Module)
        missing = any(
            (
                isinstance(m, MuseSpark1xNormPreservingResidualConnection)
                and not getattr(m, "_coefficients_ready", False)
            )
            or (
                callable(getattr(m, "fold_router_output_multiplier", None))
                and getattr(m, "router_output_multiplier", None) is not None
                and not getattr(m, "_router_multiplier_folded", False)
            )
            for m in self.modules()
        )
        if missing:
            materialize_residual_path_constants(self)
