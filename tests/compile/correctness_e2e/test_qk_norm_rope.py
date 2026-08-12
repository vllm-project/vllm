# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""E2E correctness for ``QKNormRoPEFusionPass``.

``tests/compile/fusions_e2e`` already asserts the pass fires the expected
number of times, and ``tests/compile/passes/test_qk_norm_rope_fusion.py``
checks a synthetic module numerically. Neither compares engine output on a
real model, which is what https://github.com/vllm-project/vllm/issues/39428
asks for.
"""

import json

from tests.models.registry import _HfExamplesInfo
from tests.utils import (
    compare_two_settings,
    create_new_process_for_each_test,
    multi_gpu_test,
)
from vllm.config import CompilationMode

# Qwen3 RMSNorms Q/K before RoPE (`q_norm`/`k_norm`), which is the pattern the
# pass matches, and its head_dim of 128 is in
# SUPPORTED_FUSED_QK_NORM_ROPE_HEAD_DIMS. Llama-class models have no QK-norm
# and cannot exercise the pass at all.
MODEL_ID = "Qwen/Qwen3-0.6B"
MODEL_INFO = _HfExamplesInfo(MODEL_ID)


def _server_args(tp_size: int, enable_fusion: bool) -> list[str]:
    compilation_config = {
        "mode": CompilationMode.VLLM_COMPILE,
        "splitting_ops": [],
        "pass_config": {"enable_qk_norm_rope_fusion": enable_fusion},
    }
    return [
        # The pass bails on dtypes other than bfloat16/float16.
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
    MODEL_INFO.check_transformers_version(on_fail="skip")
    MODEL_INFO.check_available_online(on_fail="skip")

    # Both settings pin the flag explicitly. PassConfig fields default to None
    # and are resolved by optimization level, so leaving the baseline implicit
    # risks enabling the pass on both sides and comparing identical configs.
    compare_two_settings(
        MODEL_ID,
        _server_args(tp_size, enable_fusion=True),
        _server_args(tp_size, enable_fusion=False),
        method="generate",
        force_v1_runner=True,
    )


@create_new_process_for_each_test()
def test_qk_norm_rope_fusion_correctness():
    _compare_fusion_settings(tp_size=1)


# Sharding splits the heads the pass reasons about, so TP is worth covering
# separately. multi_gpu_test already applies create_new_process_for_each_test.
@multi_gpu_test(num_gpus=2)
def test_qk_norm_rope_fusion_correctness_tp2():
    _compare_fusion_settings(tp_size=2)
