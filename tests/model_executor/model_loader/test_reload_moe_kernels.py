# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reloading the same weights must not change what a served MoE model
generates: post-load processing reruns, and the tensors it derives for the
kernels must stay where the captured CUDA graphs read them."""

import pytest
import torch

from tests.models.utils import check_logprobs_close
from vllm.platforms import current_platform

PROMPTS = [
    "The capital of France is",
    "List the first ten prime numbers:",
    "Explain why the sky is blue in two sentences.",
    "Q: A train travels 60 km in 45 minutes. What is its speed in km/h?\nA:",
]


def _gpu_tensor_ptrs(obj: object, path: str, ptrs: dict, seen: set) -> None:
    """Record the address of every GPU tensor reachable from ``obj`` through
    containers and attributes of vLLM objects, such as quant methods, MoE
    kernels and experts."""
    if isinstance(obj, torch.Tensor):
        if obj.is_cuda:
            ptrs[path] = obj.data_ptr()
        return
    if id(obj) in seen:
        return
    seen.add(id(obj))
    if isinstance(obj, torch.nn.Module):
        items = {**vars(obj), **obj._parameters, **obj._buffers, **obj._modules}
    elif isinstance(obj, dict):
        items = dict(obj)
    elif isinstance(obj, (list, tuple)):
        items = dict(enumerate(obj))
    elif type(obj).__module__.startswith("vllm."):
        items = dict(getattr(obj, "__dict__", {}))
    else:
        return
    for name, value in items.items():
        _gpu_tensor_ptrs(value, f"{path}.{name}", ptrs, seen)


def _model_tensor_ptrs(worker) -> dict[str, int]:
    ptrs: dict[str, int] = {}
    _gpu_tensor_ptrs(worker.model_runner.model, "model", ptrs, set())
    return ptrs


@pytest.mark.skipif(
    not current_platform.is_device_capability_family(100),
    reason="TRT-LLM and CuteDSL MoE kernels need SM100",
)
@pytest.mark.parametrize(
    "model,moe_backend,compare_outputs",
    [
        ("openai/gpt-oss-20b", "flashinfer_trtllm", True),
        # NVFP4 CuteDSL greedy outputs differ between two generations on one
        # server, reload or not, so only the addresses are checked
        ("RedHatAI/Qwen3-30B-A3B-NVFP4", "flashinfer_cutedsl", False),
    ],
)
def test_reload_keeps_moe_output(
    model: str, moe_backend: str, compare_outputs: bool, vllm_runner, monkeypatch
) -> None:
    monkeypatch.setenv("VLLM_RELOAD_MODE", "direct")
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    with vllm_runner(
        model,
        max_model_len=1024,
        max_num_seqs=4,
        enable_prefix_caching=False,
        kernel_config={"enable_jit_warmup": False, "moe_backend": moe_backend},
    ) as llm:
        before = llm.generate_greedy_logprobs(PROMPTS, max_tokens=32, num_logprobs=5)
        ptrs = llm.collective_rpc(_model_tensor_ptrs)
        llm.collective_rpc("reload_weights")  # the served checkpoint
        after_ptrs = llm.collective_rpc(_model_tensor_ptrs)
        after = llm.generate_greedy_logprobs(PROMPTS, max_tokens=32, num_logprobs=5)
    moved = [
        sorted(n for n, p in old.items() if new.get(n) != p)
        for old, new in zip(ptrs, after_ptrs)
    ]
    assert all(ptrs) and moved == [[]] * len(ptrs)
    if compare_outputs:
        check_logprobs_close(
            outputs_0_lst=before,
            outputs_1_lst=after,
            name_0="before reload",
            name_1="after reload",
        )


def test_fused_gate_weight_refilled_after_reload():
    from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
    from vllm.model_executor.model_loader.weight_utils import finalize_model_weights

    # Only the attributes `fuse_gate_weights` reads; a full runner needs a
    # MoE config and kernels.
    runner = MoERunner.__new__(MoERunner)
    torch.nn.Module.__init__(runner)
    runner.gate = torch.nn.Linear(8, 4, bias=False)
    runner.shared_expert_gate = torch.nn.Linear(8, 1, bias=False)
    runner._fse_fuse_gate = True
    runner._combined_gate_weight = None
    model = torch.nn.Sequential(runner)

    finalize_model_weights(model)  # cold start
    fused = runner._combined_gate_weight
    with torch.no_grad():
        runner.gate.weight.add_(1.0)
        runner.shared_expert_gate.weight.add_(2.0)
    finalize_model_weights(model)  # reload

    assert runner._combined_gate_weight is fused
    torch.testing.assert_close(
        fused, torch.cat([runner.gate.weight, runner.shared_expert_gate.weight])
    )
