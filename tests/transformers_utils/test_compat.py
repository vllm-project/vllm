# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
from transformers.utils import import_utils

from vllm.transformers_utils.compat import install_remote_code_shims


@pytest.mark.parametrize(
    "module", ["transformers.utils.import_utils", "transformers.utils"]
)
def test_remote_code_can_import_is_torch_fx_available(module: str):
    """Hub modelling files import a symbol Transformers v5 removed.

    Hunyuan, DeepSeek MoE and MiniCPM4 all import `is_torch_fx_available` at
    module scope, so without the shim the Transformers backend raises
    `ImportError` while building the model.
    """
    install_remote_code_shims()

    namespace: dict = {}
    exec(f"from {module} import is_torch_fx_available", namespace)

    assert namespace["is_torch_fx_available"]() is True


def test_existing_symbol_is_not_overridden(monkeypatch: pytest.MonkeyPatch):
    sentinel = object()
    monkeypatch.setattr(import_utils, "is_torch_fx_available", sentinel, raising=False)

    install_remote_code_shims()

    assert import_utils.is_torch_fx_available is sentinel
