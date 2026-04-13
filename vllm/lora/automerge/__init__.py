# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Auto-merge LoRA: merge single-adapter weights into base at decode time.

When enabled via ``--enable-lora-weight-merge`` (or ``LoRAConfig(
enable_lora_weight_merge=True)``), this module merges the active LoRA
adapter's weights directly into the base model weights, bypassing
Punica/SGMV kernels and using the non-LoRA CUDA graph to reduce TPOT.
Works with any max_loras setting — activates when a batch has a single
adapter, falls back to standard LoRA for multi-adapter or mixed batches.

BF16/FP16/FP32 only. Falls back to standard LoRA path for:
  - Multiple LoRAs in one step
  - Mixed base+LoRA batches
  - Unsupported dtypes (FP8, etc.)
"""

from .state import AutoMergeState, get_state

__all__ = ["AutoMergeState", "get_state"]
