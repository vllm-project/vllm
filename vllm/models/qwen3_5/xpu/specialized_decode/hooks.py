# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The model-side hooks of this folder, kept here so the model files only call
them:

- ``initialize_specialized_decode``: the fused MoE decode executors, built after
  quantization and before graph capture (``process_weights_after_loading``).
- ``SpecializedDecodeMixin``: the mono decode (``runtime.requested``):
  ``init_specialized_decode`` in the causal LM's ``__init__`` (the config check,
  before any module allocates weights) and ``get_model_state_cls``; the model
  state builds the executor after load (``model_state.Qwen3_5XpuModelState``).
  The backbone's ``forward`` plans each step (``specialized_decode.plan``).
- ``register_drafter``: the MTP predictor, built with the target's executor."""

from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.model_states.interface import ModelState

    from .runtime import DecodeConfig
    from .runtime.decode import SpecializedDecode


@torch.no_grad()
def initialize_specialized_decode(model: torch.nn.Module) -> None:
    from ..model import Qwen3_5SparseMoeBlock
    from .moe import MoEStep

    for module in model.modules():
        if isinstance(module, Qwen3_5SparseMoeBlock):
            module._decode = MoEStep(module)


# MTP predictors of this process (loaded before the target's model state builds
# the executor), run on the mono kernels with the target's
_drafters: list[nn.Module] = []


def register_drafter(predictor: nn.Module, vllm_config) -> None:
    """Called by the MTP predictor's ``__init__``. The target's config check
    already refused unsupported speculative settings."""
    from .runtime import requested

    predictor.specialized_decode = None
    if requested(vllm_config):
        _drafters.append(predictor)


class SpecializedDecodeMixin:
    """On the Qwen3.5 causal LM. ``self.model`` is the ``Qwen3_5Model`` the
    executor runs over."""

    model: nn.Module
    specialized_decode_config: "DecodeConfig | None" = None

    def init_specialized_decode(self, vllm_config) -> None:
        from .runtime import specialized_decode_config

        self.specialized_decode_config = specialized_decode_config(vllm_config)

    def build_specialized_decode(self) -> "SpecializedDecode":
        from .runtime.decode import SpecializedDecode

        assert self.specialized_decode_config is not None
        drafters = list(_drafters)
        decode = SpecializedDecode(self.model, self.specialized_decode_config, drafters)
        self.model.specialized_decode = decode
        for predictor in drafters:
            predictor.specialized_decode = decode
        return decode


def model_state_cls(language_model: nn.Module) -> "type[ModelState]":
    """The model runner state: the mono decode's when it is requested
    (``runtime.requested``), else the hybrid model state vLLM picks for
    Qwen3.5."""
    if getattr(language_model, "specialized_decode_config", None) is None:
        from vllm.v1.worker.gpu.model_states.mamba_hybrid import (
            MambaHybridModelState,
        )

        return MambaHybridModelState
    from ..model_state import Qwen3_5XpuModelState

    return Qwen3_5XpuModelState
