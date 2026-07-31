# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Intel XPU Kimi-K3 text model interfaces."""

from collections.abc import Iterable

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.sequence import IntermediateTensors


class KimiLinearForCausalLM(nn.Module):
	"""Placeholder for the native Intel XPU Kimi-K3 text model."""

	def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
		super().__init__()
		del vllm_config, prefix
		raise NotImplementedError("Native XPU Kimi-K3 text model is not implemented.")

	def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
		raise NotImplementedError

	def make_empty_intermediate_tensors(
		self,
		batch_size: int,
		dtype: torch.dtype,
		device: torch.device,
	) -> IntermediateTensors:
		raise NotImplementedError

	def forward(
		self,
		input_ids: torch.Tensor | None,
		positions: torch.Tensor,
		intermediate_tensors: IntermediateTensors | None = None,
		inputs_embeds: torch.Tensor | None = None,
		**kwargs: object,
	) -> torch.Tensor | IntermediateTensors:
		raise NotImplementedError

	def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
		raise NotImplementedError

	def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
		raise NotImplementedError

__all__ = ["KimiLinearForCausalLM"]