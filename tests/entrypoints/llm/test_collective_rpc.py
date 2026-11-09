# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from ...utils import create_new_process_for_each_test


@create_new_process_for_each_test()
def test_model_inspection_requires_explicit_call(vllm_runner):
    """Object display shows a summary; explicit inspection returns the model tree."""
    model_name = "vllm-project/tiny-random-LlamaForCausalLM"
    with vllm_runner(
        model_name,
        enforce_eager=True,
        max_model_len=32,
        max_num_seqs=1,
        gpu_memory_utilization=0.1,
    ) as runner:
        llm = runner.llm
        assert repr(llm) == f"LLM(model={model_name!r})"

        inspection = llm.get_model_inspection()
        assert "LlamaForCausalLM" in inspection
        assert "(embed_tokens):" in inspection


@pytest.mark.parametrize("tp_size", [1, 2])
@pytest.mark.parametrize("backend", ["mp", "ray"])
@create_new_process_for_each_test()
def test_collective_rpc(tp_size, backend, monkeypatch, vllm_runner):
    if torch.accelerator.device_count() < tp_size:
        pytest.skip(f"Not enough GPUs for tensor parallelism {tp_size}")
    if tp_size == 1 and backend == "ray":
        pytest.skip("Skip duplicate test case")
    if tp_size == 1:
        backend = None

    # intentionally define the method and class in the test function,
    # to test if they can be serialized and sent to the workers
    def echo_rank(self):
        return self.rank

    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    with vllm_runner(
        "vllm-project/tiny-random-LlamaForCausalLM",
        enforce_eager=True,
        load_format="dummy",
        tensor_parallel_size=tp_size,
        distributed_executor_backend=backend,
    ) as runner:
        assert runner.llm.collective_rpc(echo_rank) == list(range(tp_size))
