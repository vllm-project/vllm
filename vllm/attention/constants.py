# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Constants related to attention backend selection."""

# Environment variable name used to force selection of a specific
# attention backend by the Attention wrapper
STR_BACKEND_ENV_VAR: str = "VLLM_ATTENTION_BACKEND"

# Possible string values for STR_BACKEND_ENV_VAR environment variable,
# corresponding to available attention backends
STR_FLASHINFER_ATTN_VAL: str = "FLASHINFER"
STR_TORCH_SDPA_ATTN_VAL: str = "TORCH_SDPA"
STR_XFORMERS_ATTN_VAL: str = "XFORMERS"
STR_FLASH_ATTN_VAL: str = "FLASH_ATTN"
STR_INVALID_VAL: str = "INVALID"
