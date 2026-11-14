# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Backward-compatibility shim for third-party integrations.

`vllm.transformers_utils.tokenizer` was removed in #35024, but
`lm-format-enforcer==0.11.3` (pinned in requirements/common.txt) still imports
`MistralTokenizer` from this module, which breaks the lm-format-enforcer
structured-output backend (see #52614).

Keep this shim until lm-format-enforcer is updated to import from
`vllm.tokenizers.mistral` directly.
"""

import warnings


def __getattr__(name: str):
    # Keep until lm-format-enforcer is updated
    if name == "MistralTokenizer":
        from vllm.tokenizers.mistral import MistralTokenizer

        warnings.warn(
            "`vllm.transformers_utils.tokenizer.MistralTokenizer` "
            "has been moved to `vllm.tokenizers.mistral.MistralTokenizer`. "
            "The old name will be removed in a future version.",
            DeprecationWarning,
            stacklevel=2,
        )

        return MistralTokenizer

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
