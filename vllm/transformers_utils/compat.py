# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compatibility shims for Transformers symbols that Hub remote code imports."""


def install_remote_code_shims() -> None:
    """Restore Transformers symbols that `trust_remote_code` models import.

    `is_torch_fx_available` was removed in Transformers v5, but modelling files
    hosted on the Hub still import it at module scope, so the Transformers
    backend fails when it instantiates such a model, before any weight is
    loaded and on every device. Transformers
    declined to carry a generic shim (huggingface/transformers#44561), so it is
    restored here when absent. Every Transformers v5 release requires a Torch
    version that ships `torch.fx`, making Torch availability the faithful
    reduction of the removed check.
    """
    import transformers.utils
    from transformers.utils import import_utils

    if hasattr(import_utils, "is_torch_fx_available"):
        return

    def is_torch_fx_available() -> bool:
        return import_utils.is_torch_available()

    import_utils.is_torch_fx_available = is_torch_fx_available
    transformers.utils.is_torch_fx_available = is_torch_fx_available
