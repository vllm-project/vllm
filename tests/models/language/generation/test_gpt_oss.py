# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import itertools

import pytest
import torch
from torch import nn
from transformers import AutoConfig

from vllm.model_executor.models.gpt_oss import OAIAttention

MODELS = [
    "openai/gpt-oss-20b",
    "nvidia/gpt-oss-puzzle-88B",
]


@pytest.mark.parametrize("model", MODELS)
def test_gpt_oss_sliding_window(vllm_runner, monkeypatch, model: str) -> None:
    """
    Verify that when initializing gpt-oss models, the sliding
    window layers match the model config.
    """
    with monkeypatch.context() as m:
        m.setenv(
            "VLLM_ALLOW_INSECURE_SERIALIZATION", "1"
        )  # Allow dummy model loading for faster tests

        ## extract *expected* sliding window value per layer from hf config
        hf_config = AutoConfig.from_pretrained(model, trust_remote_code=True)
        if hasattr(hf_config, "block_configs"):
            # heterogeneous GptOssPuzzle models
            hf_sliding_windows = {
                layer_idx: block_config.sliding_window
                for layer_idx, block_config in enumerate(hf_config.block_configs)
            }
        else:
            # vanilla homogeneous GptOss models
            hf_sliding_windows = {
                layer_idx: hf_config.sliding_window
                if layer_type == "sliding_attention"
                else None
                for layer_idx, layer_type in enumerate(hf_config.layer_types)
            }

        ## extract *actual* sliding window value per layer from vllm model
        with vllm_runner(
            model,
            load_format="dummy",
            trust_remote_code=True,
            tensor_parallel_size=torch.accelerator.device_count(),
        ) as llm:
            vllm_sliding_windows = llm.apply_model(
                extract_sliding_windows_from_vllm_model
            )
            vllm_sliding_windows = dict(itertools.chain(*vllm_sliding_windows))

        ## verify that the sliding window values match
        vllm_layer_indices = set(vllm_sliding_windows.keys())
        hf_layer_indices = set(hf_sliding_windows.keys())
        assert vllm_layer_indices == hf_layer_indices, (
            f"Layer indices mismatch: {vllm_layer_indices=} != {hf_layer_indices=}"
        )

        mismatches = {
            layer_idx: (hf_sliding_windows[layer_idx], vllm_sliding_windows[layer_idx])
            for layer_idx in hf_sliding_windows
            if hf_sliding_windows[layer_idx] != vllm_sliding_windows[layer_idx]
        }
        assert len(mismatches) == 0, (
            f"Mismatches found: {{layer_idx: (vllm_window, hf_window)}}. {mismatches=}"
        )


def extract_sliding_windows_from_vllm_model(model: nn.Module) -> list[tuple[int, int]]:
    sliding_windows = []
    for module in model.modules():
        if isinstance(module, OAIAttention):
            sliding_windows.append((module.layer_idx, module.attn.sliding_window))
    return sliding_windows
