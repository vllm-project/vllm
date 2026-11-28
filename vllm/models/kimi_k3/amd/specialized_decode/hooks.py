# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The specialized decode's hooks on `KimiLinearForCausalLM`, kept here so the
model only calls them: `init_specialized_decode` in `__init__` (after the config
check, which runs before any module allocates weights),
`keep_specialized_decode_weights` once the decoder layers exist,
`plan_specialized_decode` in `forward` and `after_specialized_decode_step` in
`compute_logits`. The
model state (`model_state.KimiK3ModelState`) builds the executor after load."""

from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger
from vllm.models.kimi_k3.amd.specialized_decode.runtime import DecodeConfig

if TYPE_CHECKING:
    from vllm.models.kimi_k3.amd.specialized_decode.runtime.decode import (
        SpecializedDecode,
        Step,
    )
    from vllm.v1.worker.gpu.model_states.interface import ModelState

logger = init_logger(__name__)


class SpecializedDecodeMixin:
    """`self.model` is the `KimiLinearModel` the executor runs over."""

    model: torch.nn.Module
    specialized_decode_config: DecodeConfig | None
    specialized_decode: "SpecializedDecode | None"

    def init_specialized_decode(self, config: DecodeConfig | None) -> None:
        """`config`: `specialized_decode_config(vllm_config)`, None for the
        stock decode."""
        self.specialized_decode_config = config
        self.specialized_decode = None

    def keep_specialized_decode_weights(self) -> None:
        """After `self.model` is built, before the weights load: with
        `specialized_decode_bf16_copies`, the online-FP8 linears the kernels read
        keep a bf16 copy (`runtime.mark_bf16_copies`)."""
        config = self.specialized_decode_config
        if config is None or not config.bf16_copies:
            return
        from vllm.models.kimi_k3.amd.specialized_decode.runtime import (
            mark_bf16_copies,
        )

        nbytes = mark_bf16_copies(self.model)
        logger.info(
            "Kimi-K3 specialized decode: the layer kernels read bf16 copies of "
            "online-FP8 linears (%.2f GiB on this rank)",
            nbytes / 2**30,
        )

    def build_specialized_decode(self) -> None:
        from vllm.models.kimi_k3.amd.specialized_decode.runtime.decode import (
            SpecializedDecode,
        )

        assert self.specialized_decode_config is not None
        self.specialized_decode = SpecializedDecode(
            self.model, self.specialized_decode_config
        )

    def get_model_state_cls(self) -> "type[ModelState]":
        from vllm.models.kimi_k3.amd.specialized_decode.model_state import (
            KimiK3ModelState,
        )

        return KimiK3ModelState

    def plan_specialized_decode(
        self, positions: torch.Tensor, intermediate_tensors
    ) -> "Step | None":
        """This step's plan (`Step.layer(i)` per decoder layer), None for the
        stock forward."""
        decode = self.specialized_decode
        return None if decode is None else decode.plan(positions, intermediate_tensors)

    def after_specialized_decode_step(self) -> None:
        if self.specialized_decode is not None:
            self.specialized_decode.after_step()
