# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""NCP-ArchPreview model implementation for pure Hugging Face checkpoints."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .model import NCPOlmo3ForCausalLM


def __getattr__(name: str) -> Any:
    if name == "NCPOlmo3ForCausalLM":
        from .model import NCPOlmo3ForCausalLM

        return NCPOlmo3ForCausalLM
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["NCPOlmo3ForCausalLM"]
