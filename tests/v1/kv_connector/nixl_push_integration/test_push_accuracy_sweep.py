# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NixlPushConnector (push-mode PD) accuracy sweep, one test per config.

Kept separate from the pull-mode sweep in ../nixl_integration so push coverage
can grow on its own; the per-config runner is shared. Each test runs
run_accuracy_test.sh with its config's environment, as the deleted
config_sweep_accuracy_test.sh did, with no extra flags.
"""

import os
import subprocess
from pathlib import Path

import pytest

# The servers run in the script's own processes; this process starts no engine.
pytestmark = pytest.mark.skip_global_cleanup

RUN_ACCURACY_TEST = (
    Path(__file__).parent.parent / "nixl_integration" / "run_accuracy_test.sh"
)

# PP=2 prefill -> TP=2 decode: each PP stage writes its layer slice into the
# decoder's matching region range (non-MLA and MLA contiguous-slice paths).
PUSH_PP_CONFIG = {
    "KV_CONNECTOR": "NixlPushConnector",
    "PREFILLER_PP_SIZE": "2",
    "PREFILLER_TP_SIZE": "1",
    "DECODER_TP_SIZE": "2",
}


@pytest.mark.parametrize(
    "config",
    [
        pytest.param(
            {**PUSH_PP_CONFIG, "GPU_MEMORY_UTILIZATION": "0.6"},
            id="pp2-tp2-qwen3",
        ),
        pytest.param(
            {
                **PUSH_PP_CONFIG,
                "GPU_MEMORY_UTILIZATION": "0.8",
                "MODEL_NAMES": "deepseek-ai/deepseek-vl2-tiny",
            },
            id="pp2-tp2-mla",
        ),
    ],
)
def test_push_pd_accuracy(config: dict[str, str]):
    subprocess.run(
        ["bash", str(RUN_ACCURACY_TEST)],
        env={**os.environ, **config},
        check=True,
    )
