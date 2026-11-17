# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Full PD-disagg accuracy sweep for 4-GPU pods, one test per config.

The step's environment picks the config group and the serve flags, as the
deleted config_sweep_accuracy_test.sh did:

  DP_EP / HYBRID_SSM / SW_ATTN  the DP-EP, hybrid SSM or sliding-window
                                configs; otherwise the TP configs.
  ATTENTION_BACKEND / ROCM_ATTN / FLASHINFER
                                --attention-backend for every config.
  CROSS_LAYERS_BLOCKS           --enable-cross-layers for every config.

Each test runs run_accuracy_test.sh with its config's environment and the
flags, which starts the prefill and decode servers and the proxy, runs
test_accuracy.py and stops them again. As one test per config, CI can spread
the configs over several jobs.
"""

import os
import subprocess
from pathlib import Path

import pytest

# The servers run in the script's own processes; this process starts no engine.
pytestmark = pytest.mark.skip_global_cleanup

RUN_ACCURACY_TEST = Path(__file__).parent / "run_accuracy_test.sh"

TP_CONFIGS = [
    pytest.param(
        {
            "GPU_MEMORY_UTILIZATION": "0.6",
            "PREFILLER_TP_SIZE": "2",
            "DECODER_TP_SIZE": "2",
        },
        id="p-tp2-d-tp2",
    ),
    pytest.param(
        {
            "GPU_MEMORY_UTILIZATION": "0.6",
            "PREFILLER_TP_SIZE": "1",
            "DECODER_TP_SIZE": "2",
        },
        id="p-tp1-d-tp2",
    ),
    pytest.param(
        {
            "GPU_MEMORY_UTILIZATION": "0.6",
            "PREFILLER_TP_SIZE": "2",
            "DECODER_TP_SIZE": "1",
        },
        id="p-tp2-d-tp1",
    ),
    pytest.param(
        {
            "GPU_MEMORY_UTILIZATION": "0.8",
            "MODEL_NAMES": "deepseek-ai/deepseek-vl2-tiny",
        },
        id="mla-p-tp1-d-tp1",
    ),
    pytest.param(
        {
            "GPU_MEMORY_UTILIZATION": "0.8",
            "PREFILLER_TP_SIZE": "1",
            "DECODER_TP_SIZE": "2",
            "MODEL_NAMES": "deepseek-ai/deepseek-vl2-tiny",
        },
        id="mla-p-tp1-d-tp2",
    ),
    pytest.param(
        {
            "GPU_MEMORY_UTILIZATION": "0.8",
            "PREFILLER_TP_SIZE": "2",
            "DECODER_TP_SIZE": "1",
            "MODEL_NAMES": "deepseek-ai/deepseek-vl2-tiny",
        },
        id="mla-p-tp2-d-tp1",
    ),
]

# The sleep/wake round-trip on the default model. Runs only with the plain TP
# configs (no backend or cross-layer flag), as the separate line in the
# Distributed step did.
SLEEP_WAKE_CONFIG = pytest.param(
    {"GPU_MEMORY_UTILIZATION": "0.6", "SLEEP_WAKE": "1"},
    id="sleep-wake",
)

# The step sets DP_EP, which run_accuracy_test.sh reads from the environment.
DP_EP_CONFIGS = [
    pytest.param(
        {
            "GPU_MEMORY_UTILIZATION": "0.8",
            "PREFILLER_TP_SIZE": "1",
            "DECODER_TP_SIZE": "2",
            "MODEL_NAMES": "deepseek-ai/deepseek-vl2-tiny",
        },
        id="dp-ep-p-tp1-d-dp2",
    ),
    pytest.param(
        {
            "GPU_MEMORY_UTILIZATION": "0.8",
            "PREFILLER_TP_SIZE": "2",
            "DECODER_TP_SIZE": "2",
            "MODEL_NAMES": "deepseek-ai/deepseek-vl2-tiny",
        },
        id="dp-ep-p-tp2-d-dp2",
    ),
]

# HMA is on by default.
GRANITE = {
    "VLLM_SSM_CONV_STATE_LAYOUT": "DS",
    "GPU_MEMORY_UTILIZATION": "0.8",
    "MODEL_NAMES": "ibm-granite/granite-4.0-h-tiny",
    "VLLM_SERVE_EXTRA_ARGS": "--max-model-len,8192,--trust-remote-code",
}
QWEN35 = {
    "VLLM_SSM_CONV_STATE_LAYOUT": "DS",
    "GPU_MEMORY_UTILIZATION": "0.8",
    "MODEL_NAMES": "Qwen/Qwen3.5-0.8B",
}
HYBRID_SSM_CONFIGS = [
    # Mamba2 (Granite 4.0-H)
    pytest.param(GRANITE, id="mamba2-granite-p-tp1-d-tp1"),
    pytest.param(
        {**GRANITE, "PREFILLER_TP_SIZE": "2", "DECODER_TP_SIZE": "2"},
        id="mamba2-granite-p-tp2-d-tp2",
    ),
    pytest.param(
        {**GRANITE, "PREFILLER_TP_SIZE": "2", "DECODER_TP_SIZE": "1"},
        id="mamba2-granite-p-tp2-d-tp1",
    ),
    # GDN (Qwen3.5)
    pytest.param(QWEN35, id="gdn-qwen3.5-p-tp1-d-tp1"),
    pytest.param(
        {**QWEN35, "PREFILLER_TP_SIZE": "1", "DECODER_TP_SIZE": "2"},
        id="gdn-qwen3.5-p-tp1-d-tp2",
    ),
    pytest.param(
        {
            **QWEN35,
            "ENFORCE_EAGER": "0",
            "VLLM_SERVE_EXTRA_ARGS": "--spec-method,mtp,--spec-tokens,1",
        },
        id="gdn-qwen3.5-mtp",
    ),
    # Mamba1 (Jamba)
    pytest.param(
        {
            "VLLM_SSM_CONV_STATE_LAYOUT": "DS",
            "GPU_MEMORY_UTILIZATION": "0.8",
            "MODEL_NAMES": "ai21labs/AI21-Jamba2-3B",
            "VLLM_SERVE_EXTRA_ARGS": "--max-model-len,8192",
        },
        id="mamba1-jamba2-3b",
    ),
]

# gemma3 does not work with FlashInfer.
GEMMA3 = {
    "GPU_MEMORY_UTILIZATION": "0.8",
    "MODEL_NAMES": "google/gemma-3-4b-it",
    "VLLM_SERVE_EXTRA_ARGS": "--max-model-len,8192",
}
SW_ATTN_CONFIGS = [
    pytest.param(GEMMA3, id="sw-gemma3-p-tp1-d-tp1"),
    pytest.param(
        {**GEMMA3, "PREFILLER_TP_SIZE": "1", "DECODER_TP_SIZE": "2"},
        id="sw-gemma3-p-tp1-d-tp2",
    ),
    pytest.param(
        {**GEMMA3, "PREFILLER_TP_SIZE": "2", "DECODER_TP_SIZE": "1"},
        id="sw-gemma3-p-tp2-d-tp1",
    ),
    # Gemma4: sliding window + cross-layer KV sharing
    pytest.param(
        {
            "GPU_MEMORY_UTILIZATION": "0.8",
            "MODEL_NAMES": "google/gemma-4-E2B-it",
            "VLLM_SERVE_EXTRA_ARGS": "--max-model-len,8192",
        },
        id="sw-gemma4-e2b",
    ),
]


def serve_flags(env: dict[str, str]) -> list[str]:
    """The run_accuracy_test.sh flags the step's environment asks for.

    Args:
        env: The step's environment.

    Returns:
        The --attention-backend and --enable-cross-layers flags, applied to
        every config.

    """
    flags = []
    if env.get("ATTENTION_BACKEND"):
        flags = ["--attention-backend", env["ATTENTION_BACKEND"]]
    elif env.get("ROCM_ATTN"):
        flags = ["--attention-backend", "ROCM_ATTN"]
    elif env.get("FLASHINFER"):
        flags = ["--attention-backend", "FLASHINFER"]
    if env.get("CROSS_LAYERS_BLOCKS"):
        flags.append("--enable-cross-layers")
    return flags


def selected_configs(env: dict[str, str]) -> list:
    """The configs the step's environment picks.

    Args:
        env: The step's environment.

    Returns:
        One group's configs, in the order the deleted script ran them.

    """
    if env.get("DP_EP"):
        return DP_EP_CONFIGS
    if env.get("HYBRID_SSM"):
        return HYBRID_SSM_CONFIGS
    if env.get("SW_ATTN"):
        return SW_ATTN_CONFIGS
    if serve_flags(env):
        return TP_CONFIGS
    return TP_CONFIGS + [SLEEP_WAKE_CONFIG]


@pytest.mark.parametrize("config", selected_configs(dict(os.environ)))
def test_pd_accuracy(config: dict[str, str]):
    subprocess.run(
        ["bash", str(RUN_ACCURACY_TEST), *serve_flags(dict(os.environ))],
        env={**os.environ, **config},
        check=True,
    )
