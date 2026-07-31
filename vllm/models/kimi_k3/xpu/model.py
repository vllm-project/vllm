# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Intel XPU Kimi-K3 multimodal model interfaces."""

import torch.nn as nn

from vllm.config import VllmConfig


class KimiK3ForConditionalGeneration(nn.Module):
	"""Placeholder for the native Intel XPU Kimi-K3 multimodal model."""

	def __init__(self, vllm_config: VllmConfig, prefix: str = "") -> None:
		super().__init__()
		del vllm_config, prefix
		raise NotImplementedError(
			"Native XPU Kimi-K3 multimodal model is not implemented."
		)

	@classmethod
	def get_placeholder_str(cls, modality: str, i: int) -> str | None:
		del i
		if modality == "image":
			return "<|kimi_image_placeholder|>"
		raise ValueError(f"Unsupported modality: {modality}")

__all__ = ["KimiK3ForConditionalGeneration"]