# SPDX-License-Identifier: Apache-2.0

from ..entrypoints.openai.test_oot_registration import (
    run_and_test_dummy_opt_api_server)


def test_distributed_oot(dummy_opt_path: str):
    run_and_test_dummy_opt_api_server(dummy_opt_path, tp=2)
