# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import pytest

from tests.plugins_tests.test_oot_registration_online import (
    run_and_test_dummy_opt_api_server,
)


def test_distributed_oot(dummy_opt_path: str):
    run_and_test_dummy_opt_api_server(dummy_opt_path, tp=2)


@pytest.mark.parametrize("enforce_eager", [True, False])
def test_distributed_oot_pp_raw_input_tokens(dummy_opt_path: str, enforce_eager: bool):
    # The model asserts that input_ids reaches every pipeline stage.
    overrides = {"architectures": ["MyOPTRawInputTokensForCausalLM"]}
    extra_args = ["-pp", "2", "--hf-overrides", json.dumps(overrides)]
    if enforce_eager:
        extra_args.append("--enforce-eager")
    run_and_test_dummy_opt_api_server(
        dummy_opt_path,
        extra_args=extra_args,
        env_dict={"VLLM_USE_V2_MODEL_RUNNER": "1"},
    )
