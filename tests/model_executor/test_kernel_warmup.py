# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm.model_executor.warmup import kernel_warmup


@pytest.mark.parametrize("attribute", ["_kv_block_zeroer", "kv_block_zeroer"])
def test_warmup_kv_block_zeroer_supports_runner_attribute_names(attribute):
    zeroer = Mock()
    model_runner = SimpleNamespace(**{attribute: zeroer})
    if attribute == "kv_block_zeroer":
        model_runner._kv_block_zeroer = None

    kernel_warmup._warmup_kv_block_zeroer(model_runner)

    zeroer.warmup.assert_called_once_with()


def test_warmup_kv_block_zeroer_ignores_runner_without_zeroer():
    kernel_warmup._warmup_kv_block_zeroer(SimpleNamespace())


def test_kernel_warmup_routes_model_runner_to_kv_block_zeroer(monkeypatch):
    monkeypatch.setattr(kernel_warmup.envs, "VLLM_USE_DEEP_GEMM", False)
    for warmup_name in (
        "qwen_triton_warmup",
        "deepseek_v4_mhc_warmup",
        "flashinfer_sparse_mla_decode_autotune_warmup",
        "deepseek_v4_sparse_mla_attention_warmup",
    ):
        monkeypatch.setattr(kernel_warmup, warmup_name, Mock())

    route_warmup = Mock(side_effect=RuntimeError("stop after zeroer routing"))
    monkeypatch.setattr(kernel_warmup, "_warmup_kv_block_zeroer", route_warmup)

    model_runner = SimpleNamespace()
    worker = SimpleNamespace(
        model_runner=model_runner,
        use_v2_model_runner=True,
        scheduler_config=SimpleNamespace(max_num_batched_tokens=1),
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(),
            compilation_config=SimpleNamespace(cudagraph_capture_sizes=[]),
        ),
        get_model=Mock(),
    )

    with pytest.raises(RuntimeError, match="stop after zeroer routing"):
        kernel_warmup.kernel_warmup(worker)

    route_warmup.assert_called_once_with(model_runner)
