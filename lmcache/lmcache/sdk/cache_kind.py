# SPDX-License-Identifier: Apache-2.0
"""
Cache kinds for LMCache SDK.
"""

# Future
from __future__ import annotations

# Standard
import enum


class LMCacheSDKCacheKind(enum.Enum):
    """A cacheable tensor family that LMCache exports.

    Each kind is stored under its own namespaced model name and may or may
    not be writable through the SDK. This is a pure value type -- it holds
    no runtime state and never references a live context.
    """

    KV = "kv"
    QUERY = "query"

    def server_model_name(self, model_name: str) -> str:
        """Replace model name with a prefixed model name for this kind.
        Saving between different kinds is differentiated by the model name
        prefix (see vllm_multi_process_adapter.py).

        Args:
            model_name: The original model name.

        Returns:
            The prefixed model name for this kind.
        """
        if self is LMCacheSDKCacheKind.QUERY:
            return f"{model_name}##query"
        return model_name

    def base_model_name(self, model_name: str) -> str:
        """Remove the kind prefix from a model name for registration.

        Args:
            model_name: The prefixed model name.

        Returns:
            The original model name without the kind prefix.
        """
        if self is LMCacheSDKCacheKind.QUERY:
            return model_name.removesuffix("##query")
        return model_name
