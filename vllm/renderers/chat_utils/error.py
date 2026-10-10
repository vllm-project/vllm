# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


class ChatTemplateResolutionError(ValueError):
    """Raised when chat template resolution fails.

    This is a subclass of ValueError for backward compatibility with
    existing exception handlers.
    """
