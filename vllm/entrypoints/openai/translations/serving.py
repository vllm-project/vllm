# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import warnings

warnings.warn(
    "'vllm.entrypoints.openai.translations.serving' has been moved to "
    "'vllm.entrypoints.openai.speech_to_text.serving'. Please update your "
    "imports. This backward-compatible alias will be removed in version 0.17+.",
    DeprecationWarning,
    stacklevel=2,
)

from vllm.entrypoints.openai.speech_to_text.serving import *  # noqa: F401,F403,E402
