# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from vllm.config.ec_transfer import ECTransferConfig


def test_ec_enable_nixl_defaults_false():
    cfg = ECTransferConfig()
    assert cfg.ec_enable_nixl is False


def test_ec_enable_nixl_settable():
    cfg = ECTransferConfig(
        ec_connector="ECCPUConnector", ec_role="ec_both", ec_enable_nixl=True
    )
    assert cfg.ec_enable_nixl is True
