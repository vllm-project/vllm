# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The MTP depths stay unquantized for NVFP4 checkpoints (BF16 MTP weights)."""

import pytest

from vllm.models.inkling.common import mtp as common_mtp
from vllm.models.inkling.nvidia import mtp as nvidia_mtp


class _QuantConfig:
    def __init__(self, name: str) -> None:
        self._name = name

    def get_name(self) -> str:
        return self._name


@pytest.mark.parametrize("module", [common_mtp, nvidia_mtp])
def test_nvfp4_mtp_depths_unquantized(module):
    assert module._mtp_quant_config(_QuantConfig("modelopt_fp4")) is None


@pytest.mark.parametrize("module", [common_mtp, nvidia_mtp])
@pytest.mark.parametrize("name", ["quark", "modelopt", "fp8", "mxfp4"])
def test_other_quant_configs_pass_through(module, name):
    quant_config = _QuantConfig(name)
    assert module._mtp_quant_config(quant_config) is quant_config


@pytest.mark.parametrize("module", [common_mtp, nvidia_mtp])
def test_unquantized_stays_unquantized(module):
    assert module._mtp_quant_config(None) is None
