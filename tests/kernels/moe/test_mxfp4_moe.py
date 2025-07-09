# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib
import importlib.metadata
from dataclasses import dataclass

import pytest
import torch
from packaging import version

QUARK_MXFP4_AVAILABLE = importlib.util.find_spec(
    "quark") is not None and version.parse(
        importlib.metadata.version("amd-quark")) >= version.parse('0.8.99')


@dataclass
class ModelCase:
    model_id: str
    tp: int


@pytest.mark.parametrize('model_case', [
    ModelCase("fxmarty/qwen_1.5-moe-a2.7b-mxfp4", tp=1),
    ModelCase("fxmarty/deepseek_r1_3_layers_mxfp4", tp=8),
    ModelCase("fxmarty/Llama-4-Scout-17B-16E-Instruct-2-layers-mxfp4", tp=1)
])
@pytest.mark.skipif(not QUARK_MXFP4_AVAILABLE,
                    reason="amd-quark>=0.9 is not available")
def test_mxfp4_loading_and_execution_moe(vllm_runner, model_case: ModelCase):
    if torch.cuda.device_count() < model_case.tp:
        pytest.skip(f"This test requires >={model_case.tp} gpus, got only "
                    f"{torch.cuda.device_count()}")

    with vllm_runner(model_case.model_id,
                     tensor_parallel_size=model_case.tp,
                     load_format="dummy") as llm:

        # TODO: llm.apply_model(check_model) currently relies on V0 internals.
        # Re-enable this later.
        # def check_model(model):
        #     layer = model.model.layers[0]

        #     qkv_proj = layer.self_attn.qkv_proj

        #     assert isinstance(qkv_proj.quant_method, QuarkLinearMethod)
        #     assert isinstance(qkv_proj.scheme, QuarkW4A4MXFP4)

        #     assert isinstance(layer.mlp.experts.quant_method,
        #                       QuarkW4A4MXFp4MoEMethod)

        # if model_case.model_id == "fxmarty/qwen_1.5-moe-a2.7b-mxfp4":
        #     llm.apply_model(check_model)

        output = llm.generate_greedy("Today I am in the French Alps and",
                                     max_tokens=20)
        assert output