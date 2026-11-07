# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path

import torch
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

from ....utils import multi_gpu_test
from ...utils import check_logprobs_close


def _save_tiny_random_solar(path: Path) -> str:
    """Save a random tiny Solar checkpoint with the skip connection layout of
    upstage/solar-pro-preview-instruct scaled from 64 to 16 layers.

    Solar is a Llama with block skip connections, so the Llama classes build the
    weights and config without the repo's remote code.
    """
    tokenizer = AutoTokenizer.from_pretrained("upstage/solar-pro-preview-instruct")
    config = LlamaConfig(
        vocab_size=len(tokenizer),
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=16,
        num_attention_heads=4,
        num_key_value_heads=2,
        initializer_range=0.2,
        bskcn_1=[3, 5, 8, 11],
        bskcn_2=[5, 8],
        bskcn_3=[4, 6, 9, 12],
        bskcn_4=[7, 10],
        bskcn_tv=[0.9, 0.8],
    )
    torch.manual_seed(0)
    LlamaForCausalLM(config).to(torch.bfloat16).save_pretrained(path)
    config.architectures = ["SolarForCausalLM"]
    config.save_pretrained(path)
    tokenizer.save_pretrained(path)
    return str(path)


@multi_gpu_test(num_gpus=2)
def test_pipeline_parallel_skip_connections(
    vllm_runner,
    example_prompts,
    monkeypatch,
    tmp_path,
) -> None:
    """Skip connections captured on one pipeline stage and added back on the
    next must carry their hidden states across the stage boundary."""
    # Built in the test process: CPU torch ops in the pytest process before
    # the per-test fork leave libgomp unusable in the child.
    model = _save_tiny_random_solar(tmp_path)
    max_tokens, num_logprobs = 32, 5
    with vllm_runner(model) as vllm_model:
        vllm_outputs_pp_1 = vllm_model.generate_greedy_logprobs(
            example_prompts, max_tokens, num_logprobs
        )

    # Layer 3 is captured on the first stage and added back at layer 4
    monkeypatch.setenv("VLLM_PP_LAYER_PARTITION", "4,12")
    with vllm_runner(model, pipeline_parallel_size=2) as vllm_model:
        vllm_outputs_pp_2 = vllm_model.generate_greedy_logprobs(
            example_prompts, max_tokens, num_logprobs
        )

    check_logprobs_close(
        outputs_0_lst=vllm_outputs_pp_1,
        outputs_1_lst=vllm_outputs_pp_2,
        name_0="vllm_pp_1",
        name_1="vllm_pp_2",
    )
