# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The specialized decode's hooks on ``Qwen3_5ForCausalLMBase``, kept here so
the model only calls them: ``init_specialized_decode`` in ``__init__`` (the
config check, before any module allocates weights), then
``plan_specialized_decode`` and ``run_specialized_decode`` in ``forward``, and
``after_specialized_decode_step`` in ``compute_logits``. The model state
(``Qwen3_5ModelState``) builds the executor after load."""

from typing import TYPE_CHECKING

import torch

from vllm.models.qwen3_5.amd.specialized_decode.runtime import (
    DecodeConfig,
    specialized_decode_config,
)

if TYPE_CHECKING:
    from vllm.model_executor.models.qwen3_5 import Qwen3_5Model
    from vllm.models.qwen3_5.amd.specialized_decode.runtime.decode import (
        SpecializedDecode,
        Step,
    )


class SpecializedDecodeMixin:
    """``self.model`` is the ``Qwen3_5Model`` the executor runs over."""

    model: "Qwen3_5Model"
    specialized_decode_config: DecodeConfig | None
    specialized_decode: "SpecializedDecode | None"

    def init_specialized_decode(self, vllm_config) -> None:
        """Raises for ``--monokernel`` on a deployment the kernels cannot
        serve."""
        self.specialized_decode_config = specialized_decode_config(vllm_config)
        self.specialized_decode = None

    def build_specialized_decode(self) -> None:
        from vllm.models.qwen3_5.amd.specialized_decode.runtime.decode import (
            SpecializedDecode,
        )

        assert self.specialized_decode_config is not None
        self.specialized_decode = SpecializedDecode(
            self.model, self.specialized_decode_config
        )

    def plan_specialized_decode(
        self, input_ids, positions: torch.Tensor, intermediate_tensors
    ) -> "Step | None":
        """This step's plan, None for the stock forward."""
        decode = self.specialized_decode
        if decode is None:
            return None
        return decode.plan(input_ids, positions, intermediate_tensors)

    def run_specialized_decode(
        self, step: "Step", input_ids, positions, inputs_embeds
    ) -> torch.Tensor:
        decode = self.specialized_decode
        assert decode is not None
        return decode.run(step, input_ids, positions, inputs_embeds)

    def after_specialized_decode_step(self) -> None:
        if self.specialized_decode is not None:
            self.specialized_decode.after_step()
