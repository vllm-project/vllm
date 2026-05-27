# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Quest sparse offload attention backend (Phase A).

Importing this package has side effects: it registers
QuestSparseOffloadBackend as the implementation class for
AttentionBackendEnum.CUSTOM. Callers MUST NOT import this package on the
default vLLM path; the backend is opt-in via VLLM_ATTENTION_BACKEND or the
--enable-quest-sparse-offload CLI flag.
"""

# Intentionally no eager imports. See registration.py.
