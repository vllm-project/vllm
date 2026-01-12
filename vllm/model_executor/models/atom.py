# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright 2026 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Wrapper around `atom` models"""

from collections.abc import Iterable

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.distributed import get_pp_group, get_tp_group
from vllm.logger import init_logger
from vllm.model_executor.models.interfaces import (
    SupportsPP,
    SupportsQuant,
)
from vllm.model_executor.models.interfaces_base import (
    VllmModel,
    VllmModelForTextGeneration,
)
from vllm.sequence import IntermediateTensors

logger = init_logger(__name__)


class ATOMModelBase(nn.Module, VllmModel, SupportsQuant, SupportsPP):
    def __init_subclass__(cls, *args, **kwargs):
        super().__init_subclass__(*args, **kwargs)

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        logger.info("Using ATOM modeling backend.")

        self.config = vllm_config.model_config.hf_config
        self.text_config = self.config.get_text_config()
        self.cache_config = vllm_config.cache_config
        self.device_config = vllm_config.device_config
        self.model_config = vllm_config.model_config
        self.parallel_config = vllm_config.parallel_config
        self.quant_config = vllm_config.quant_config

        self.pp_group = get_pp_group()
        self.tp_group = get_tp_group()

        # Weights to skip in `self.load_weights`
        self.skip_prefixes: list[str] = []
        self.skip_substrs: list[str] = []
        self.ignore_unexpected_prefixes: list[str] = []
        self.ignore_unexpected_suffixes: list[str] = []

        try:
            import atom  # type: ignore[import-not-found]
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "ATOM modeling backend requires the Python package `atom`, "
                "but it is not installed. Install it before using the "
                "ATOM modeling backend. Here is ATOM: "
                "https://github.com/ROCm/ATOM"
            ) from e

        self.model = atom.prepare_model(config=vllm_config, engine="vllm")
        if self.model is None:
            model_arch = vllm_config.model_config.architectures[0]
            raise ValueError(
                f"The model {model_arch} is not supported by model impl backend atom"
            )

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **model_kwargs,
    ) -> torch.Tensor | IntermediateTensors:
        if not self.pp_group.is_first_rank:
            assert intermediate_tensors is not None
            input_ids = None
            inputs_embeds = intermediate_tensors["hidden_states"]

        hidden_states = self.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            **model_kwargs,
        )

        if not self.pp_group.is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states})

        return hidden_states

    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> set[str]:
        return self.model.load_weights(weights)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        logits = self.model.compute_logits(hidden_states)
        return logits


class ATOMForCausalLM(ATOMModelBase, VllmModelForTextGeneration): ...


class ATOMMoEForCausalLM(ATOMModelBase, VllmModelForTextGeneration): ...
