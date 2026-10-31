# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A draft-model weight update reruns the drafter's post-load processing; the
tensors the speculator's CUDA graphs read must keep their addresses."""

import pytest
import torch

PROMPTS = [
    "The capital of France is",
    "List the first ten prime numbers:",
]


def _draft_tensor_ptrs(worker) -> dict[str, int]:
    ptrs = {}
    for name, module in worker.get_draft_model().named_modules():
        tensors = {**vars(module), **module._parameters, **module._buffers}
        for attr, tensor in tensors.items():
            if isinstance(tensor, torch.Tensor) and tensor.device.type != "cpu":
                ptrs[f"{name}.{attr}"] = tensor.data_ptr()
    return ptrs


def _update_draft_weights(worker) -> None:
    """The steps of a draft weight update (`start_draft_weight_update`, the
    engine loading the received weights, `finish_weight_update`), with the
    drafter's own checkpoint as the update."""
    from vllm.model_executor.model_loader import DefaultModelLoader
    from vllm.model_executor.model_loader.reload import finish_reload, start_reload

    model = worker.get_draft_model()
    config = worker.speculative_config.draft_model_config
    loader = DefaultModelLoader(worker.vllm_config.load_config)
    start_reload(model)
    model.load_weights(loader.get_all_weights(config, model))
    finish_reload(model, config, worker.device)


@pytest.mark.parametrize("reload_mode", ["direct", "layerwise"])
def test_draft_weight_update_keeps_draft_tensor_addresses(
    reload_mode: str, vllm_runner, monkeypatch
) -> None:
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    monkeypatch.setenv("VLLM_RELOAD_MODE", reload_mode)
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    with vllm_runner(
        "Qwen/Qwen3-4B",
        speculative_config={
            "method": "dflash",
            "model": "z-lab/Qwen3-4B-DFlash-b16",
            "num_speculative_tokens": 15,
        },
        max_model_len=1024,
        max_num_seqs=4,
    ) as llm:
        # serve first, so the speculator's CUDA graphs are captured
        llm.generate_greedy(PROMPTS, max_tokens=32)
        ptrs = llm.collective_rpc(_draft_tensor_ptrs)
        llm.collective_rpc(_update_draft_weights)
        after_ptrs = llm.collective_rpc(_draft_tensor_ptrs)
    # tokens are not compared: speculative greedy decoding is not bit-exact
    # across steps, so the addresses are the check
    moved = [
        sorted(n for n, p in old.items() if new.get(n) != p)
        for old, new in zip(ptrs, after_ptrs)
    ]
    assert moved == [[]] * len(ptrs)
