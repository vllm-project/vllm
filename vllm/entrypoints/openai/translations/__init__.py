# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import warnings

warnings.warn(
    "The 'vllm.entrypoints.openai.translations' module has been renamed to "
    "'vllm.entrypoints.openai.speech_to_text'. Please update your imports. "
    "This backward-compatible alias will be removed in version 0.17+.",
    DeprecationWarning,
    stacklevel=2,
)
