# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Core PD-disagg accuracy sweep: dense, engine-mode and architecture
configs, one test per config.

Sized for a 2-GPU pod so it can run on every PR; the full 4-GPU sweeps
(TP2/TP2, FlashInfer, the wider DP-EP and SW sets, cross-layer, Mamba1) run
nightly only (see .buildkite/test_areas/disaggregated.yaml).

Each test runs run_accuracy_test.sh with its config's environment, which
starts the prefill and decode servers and the proxy, runs test_accuracy.py
and stops them again. As one test per config, CI can spread the configs over
several jobs.

run_accuracy_test.sh spreads engines over the visible GPUs, so with 2 GPUs
every TP>1 config co-locates prefill and decode ranks on the same GPUs.
Shared configs cap GPU_MEMORY_UTILIZATION at 0.4 so two engines fit side by
side; the TP1/TP1 configs keep one dedicated GPU per side at the utilization
the nightly sweep uses for the same model.
"""

import os
import subprocess
from pathlib import Path

import pytest

# The servers run in the script's own processes; this process starts no engine.
pytestmark = pytest.mark.skip_global_cleanup

RUN_ACCURACY_TEST = Path(__file__).parent / "run_accuracy_test.sh"
# Decoders use decode-only CUDA graphs without torch.compile, unless the
# caller's environment sets DECODER_CUDAGRAPH.
DEFAULT_ENV = {"DECODER_CUDAGRAPH": "1"}

DENSE_CONFIGS = [
    # TP shapes and modes on the dense default model (Qwen3-0.6B). TP1/TP1
    # gets one dedicated GPU per side. The hetero-TP configs (P TP1/D TP2 and
    # P TP2/D TP1) share one GPU between the prefiller and one decoder rank
    # (0.35 each), as does push-mode PP prefill (P TP1xPP2, D TP2, sharing
    # both GPUs). DP-attention decode runs the nightly sweep's MoE model
    # (deepseek-vl2-tiny, ~6.5 GB bf16): P TP1 and one D DP2 rank share GPU0,
    # so it needs 0.4 per engine to keep room for KV and CUDA contexts.
    pytest.param(
        {
            "GPU_MEMORY_UTILIZATION": "0.6",
            "PREFILLER_TP_SIZE": "1",
            "DECODER_TP_SIZE": "1",
        },
        id="p-tp1-d-tp1",
    ),
    pytest.param(
        {
            "GPU_MEMORY_UTILIZATION": "0.35",
            "PREFILLER_TP_SIZE": "1",
            "DECODER_TP_SIZE": "2",
        },
        id="p-tp1-d-tp2",
    ),
    pytest.param(
        {
            "GPU_MEMORY_UTILIZATION": "0.35",
            "PREFILLER_TP_SIZE": "2",
            "DECODER_TP_SIZE": "1",
        },
        id="p-tp2-d-tp1",
    ),
    pytest.param(
        {
            "DP_EP": "1",
            "GPU_MEMORY_UTILIZATION": "0.4",
            "PREFILLER_TP_SIZE": "1",
            "DECODER_TP_SIZE": "2",
            "MODEL_NAMES": "deepseek-ai/deepseek-vl2-tiny",
        },
        id="dp-ep-p-tp1-d-dp2",
    ),
    pytest.param(
        {
            "KV_CONNECTOR": "NixlPushConnector",
            "GPU_MEMORY_UTILIZATION": "0.35",
            "PREFILLER_PP_SIZE": "2",
            "PREFILLER_TP_SIZE": "1",
            "DECODER_TP_SIZE": "2",
        },
        id="push-p-pp2-d-tp2",
    ),
]

MODE_CONFIGS = [
    # Engine modes on the dense default model, TP1/TP1 on dedicated GPUs: the
    # sleep/wake round-trip, previously nightly only.
    pytest.param(
        {
            "SLEEP_WAKE": "1",
            "GPU_MEMORY_UTILIZATION": "0.6",
            "PREFILLER_TP_SIZE": "1",
            "DECODER_TP_SIZE": "1",
        },
        id="sleep-wake",
    ),
]

ARCH_CONFIGS = [
    # Architecture coverage at TP1/TP1 on dedicated GPUs: MLA, hybrid SSM
    # conv-state transfer (Mamba2 and GDN), and sliding-window attention
    # combined with cross-layer KV sharing (gemma-4-E2B-it declares
    # num_kv_shared_layers in its HF config). Configs mirror the nightly
    # sweeps; the gemma config was previously unreachable, as no CI step
    # selects SW_ATTN.
    pytest.param(
        {
            "GPU_MEMORY_UTILIZATION": "0.8",
            "MODEL_NAMES": "deepseek-ai/deepseek-vl2-tiny",
        },
        id="mla-deepseek-vl2-tiny",
    ),
    pytest.param(
        {
            "VLLM_SSM_CONV_STATE_LAYOUT": "DS",
            "GPU_MEMORY_UTILIZATION": "0.8",
            "MODEL_NAMES": "ibm-granite/granite-4.0-h-tiny",
            "VLLM_SERVE_EXTRA_ARGS": "--max-model-len,8192,--trust-remote-code",
        },
        id="mamba2-granite-4.0-h-tiny",
    ),
    pytest.param(
        {
            "VLLM_SSM_CONV_STATE_LAYOUT": "DS",
            "GPU_MEMORY_UTILIZATION": "0.8",
            "MODEL_NAMES": "Qwen/Qwen3.5-0.8B",
        },
        id="gdn-qwen3.5-0.8b",
    ),
    pytest.param(
        {
            "GPU_MEMORY_UTILIZATION": "0.8",
            "MODEL_NAMES": "google/gemma-4-E2B-it",
            "VLLM_SERVE_EXTRA_ARGS": "--max-model-len,8192",
        },
        id="sliding-window-gemma-4-e2b",
    ),
]


def run_accuracy_test(config: dict[str, str]):
    """Run run_accuracy_test.sh with one config's environment.

    Args:
        config: Environment variables run_accuracy_test.sh reads, such as the
            model and the prefill and decode parallel sizes.

    """
    subprocess.run(
        ["bash", str(RUN_ACCURACY_TEST)],
        env={**DEFAULT_ENV, **os.environ, **config},
        check=True,
    )


@pytest.mark.parametrize("config", DENSE_CONFIGS)
def test_dense(config: dict[str, str]):
    run_accuracy_test(config)


@pytest.mark.parametrize("config", MODE_CONFIGS)
def test_engine_mode(config: dict[str, str]):
    run_accuracy_test(config)


@pytest.mark.parametrize("config", ARCH_CONFIGS)
def test_architecture(config: dict[str, str]):
    run_accuracy_test(config)
