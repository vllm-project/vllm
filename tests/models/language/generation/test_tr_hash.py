# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from ...registry import HF_EXAMPLE_MODELS
from ...utils import check_outputs_equal


@pytest.mark.cpu_model
def test_tr_hash_routing_and_generation(hf_runner, vllm_runner, enable_pickle):
    model_info = HF_EXAMPLE_MODELS.get_hf_info("TRHashForCausalLM")
    prompts = ["Hello", "The capital of France is", "def add(a, b):\n    return"]
    kwargs = dict(revision=model_info.revision, trust_remote_code=True, dtype="float32")
    with hf_runner(model_info.default, **kwargs) as hf_model:
        hf_outputs = hf_model.generate_greedy(prompts, 12)
        route_tables = [
            layer.mlp.engine.route_table.cpu().clone()
            for layer in hf_model.model.layers
        ]

    def check_routes(worker):
        model = worker.get_model()
        for layer, expected in zip(model.model.layers, route_tables):
            mlp = layer.mlp
            torch.testing.assert_close(mlp.route_table.cpu(), expected, rtol=0, atol=0)
            token_ids = torch.arange(expected.shape[1], device=mlp.route_table.device)
            routes = mlp.route_table[:, token_ids].T.contiguous()
            weights, ids = mlp.route(None, routes, topk=mlp.top_k, renormalize=True)
            torch.testing.assert_close(ids.cpu().long(), expected.T, rtol=0, atol=0)
            torch.testing.assert_close(
                weights.cpu(), torch.full(ids.shape, 0.5), rtol=0, atol=0
            )

    with vllm_runner(model_info.default, max_model_len=256, **kwargs) as vllm_model:
        vllm_model.collective_rpc(check_routes)
        vllm_outputs = vllm_model.generate_greedy(prompts, 12)

    check_outputs_equal(
        outputs_0_lst=hf_outputs,
        outputs_1_lst=vllm_outputs,
        name_0="hf",
        name_1="vllm",
    )
