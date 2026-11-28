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


def _draft_checkpoint(worker):
    """The draft model, its config and the weights of its own checkpoint."""
    from vllm.config import replace
    from vllm.model_executor.model_loader import DefaultModelLoader

    model = worker.get_draft_model()
    config = worker.speculative_config.draft_model_config
    # Read the checkpoint even when the engine started without it.
    load_config = replace(worker.vllm_config.load_config, load_format="auto")
    weights = DefaultModelLoader(load_config).get_all_weights(config, model)
    return model, config, weights


def _update_draft_weights(worker) -> None:
    """The steps of a draft weight update (`start_draft_weight_update`, the
    engine loading the received weights, `finish_weight_update`), with the
    drafter's own checkpoint as the update."""
    from vllm.model_executor.model_loader.reload import finish_reload, start_reload

    model, config, weights = _draft_checkpoint(worker)
    start_reload(model)
    model.load_weights(weights)
    finish_reload(model, config, worker.device)


@pytest.mark.parametrize("reload_mode", ["direct", "layerwise"])
def test_draft_weight_update_keeps_draft_tensor_addresses(
    reload_mode: str, vllm_runner, monkeypatch
) -> None:
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


def _draft_lm_head_vs_checkpoint(worker) -> tuple[torch.Size, torch.Size, bool]:
    model, _, weights = _draft_checkpoint(worker)
    lm_head = model.lm_head.weight
    expected = dict(weights)["lm_head.weight"]
    same = lm_head.shape == expected.shape and torch.equal(
        lm_head, expected.to(lm_head)
    )
    return lm_head.shape, expected.shape, same


@pytest.mark.parametrize("reload_mode", ["direct", "layerwise"])
def test_dummy_started_reduced_vocab_draft_loads_its_lm_head(
    reload_mode: str, vllm_runner, monkeypatch
) -> None:
    """An EAGLE-3 drafter with a reduced vocabulary has an lm_head of its own.
    After a dummy start it must still have it, so that a draft update loads it."""
    monkeypatch.setenv("VLLM_RELOAD_MODE", reload_mode)
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    with vllm_runner(
        "meta-llama/Llama-3.2-1B-Instruct",
        speculative_config={
            "method": "eagle3",
            "model": "nm-testing/Llama3_2_1B_speculator.eagle3",
            "num_speculative_tokens": 3,
        },
        load_format="dummy",
        max_model_len=1024,
        max_num_seqs=4,
    ) as llm:
        llm.collective_rpc(_update_draft_weights)
        ((shape, expected_shape, same),) = llm.collective_rpc(
            _draft_lm_head_vs_checkpoint
        )
    assert shape == expected_shape
    assert same
