# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""E2E correctness for the RMSNorm/SiluAndMul + FP8-quant fusion passes.

``tests/compile/fusions_e2e/test_tp1_quant.py`` already asserts that
``rms_quant_fusion`` and ``act_quant_fusion`` fire the expected number of
times on real FP8 models, and ``tests/compile/passes/test_fusion.py`` /
``test_silu_mul_quant_fusion.py`` check synthetic modules numerically.
Neither compares engine output on a real model, which is what
https://github.com/vllm-project/vllm/issues/39428 asks for.
"""

import json

import pytest

from tests.models.registry import _HfExamplesInfo
from tests.utils import (
    compare_two_settings,
    create_new_process_for_each_test,
    multi_gpu_test,
)
from vllm.config import CompilationMode
from vllm.platforms import current_platform

# Smallest dense FP8 checkpoint already used by this directory's suites
# (test_async_tp.py, test_sequence_parallel.py). Its static per-tensor FP8
# scales produce the kFp8StaticTensorSym quant scheme both passes match on,
# and every Llama decoder layer carries two RMSNorm+quant sites (qkv/mlp
# inputs) and one SiluAndMul+quant site (gate/up output).
MODEL_ID = "RedHatAI/Llama-3.2-1B-Instruct-FP8"
MODEL_INFO = _HfExamplesInfo(MODEL_ID)
# Skip at collection time on non-CUDA machines so that even the forked
# process wrapper never spawns (a skip raised inside it is reported as pass).
pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(),
    reason="RMS/Silu + FP8-quant fusion correctness requires CUDA",
)


def _skip_if_unsupported() -> None:
    if not current_platform.is_cuda():
        pytest.skip("RMS/Silu + FP8-quant fusion correctness requires CUDA")

    # CI exercises FP8 fusion on Hopper (sm90); test_sequence_parallel.py
    # skips its FP8 legs below sm90 too. An unknown capability (None) is left
    # to fail in the test body rather than being silently skipped here.
    capability = current_platform.get_device_capability()
    if capability is not None and capability < (9, 0):
        pytest.skip("Per-tensor FP8 fusion correctness is covered on sm90+")


def _server_args(tp_size: int, fuse_quant: bool) -> list[str]:
    compilation_config = {
        "mode": CompilationMode.VLLM_COMPILE,
        "splitting_ops": [],
        "pass_config": {
            "fuse_norm_quant": fuse_quant,
            "fuse_act_quant": fuse_quant,
        },
    }
    return [
        # The fused custom ops handle bfloat16/float16 activations.
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "2048",
        "--max-num-seqs",
        "8",
        "--tensor-parallel-size",
        str(tp_size),
        "--distributed-executor-backend",
        "mp",
        "--compilation_config",
        json.dumps(compilation_config),
    ]


def _compare_fusion_settings(tp_size: int) -> None:
    _skip_if_unsupported()
    MODEL_INFO.check_transformers_version(on_fail="skip")
    MODEL_INFO.check_available_online(on_fail="skip")

    # Both settings pin the flags explicitly. PassConfig fields default to
    # None and are resolved by optimization level, so leaving the baseline
    # implicit risks enabling the passes on both sides and comparing
    # identical configs. The comparison is non-vacuous: on H100,
    # tests/compile/fusions_e2e/test_tp1_quant.py asserts both passes fire
    # (2 match sites/layer for rms_quant, 1/layer for act_quant) on this
    # model family under the same flags.
    compare_two_settings(
        MODEL_ID,
        _server_args(tp_size, fuse_quant=True),
        _server_args(tp_size, fuse_quant=False),
        method="generate",
        force_v1_runner=True,
    )


@create_new_process_for_each_test()
def test_rms_act_quant_fusion_correctness():
    _compare_fusion_settings(tp_size=1)


# Sharding splits the GEMMs the fused quant ops feed, so TP is worth covering
# separately. multi_gpu_test already applies create_new_process_for_each_test.
@multi_gpu_test(num_gpus=2)
def test_rms_act_quant_fusion_correctness_tp2():
    _compare_fusion_settings(tp_size=2)
