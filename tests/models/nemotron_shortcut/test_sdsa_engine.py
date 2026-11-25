# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SDSA through the whole engine on a tiny random-weight Nemotron Shortcut, so
no checkpoint is needed: build, compile, the KV cache shared with GDP layers,
chunked prefill, CUDA graphs, TP2 and MTP. The sdsa/ tests check the selection."""

import random

import torch

from tests.conftest import VllmRunner
from tests.models.nemotron_shortcut.utils import (
    make_config,
    realistic_weights,
    save_checkpoint_config,
)
from tests.models.utils import check_logprobs_close
from tests.utils import multi_gpu_marks
from vllm import SamplingParams

pytestmark = multi_gpu_marks(num_gpus=2)

K = 512  # each query attends to its Top-K keys
VOCAB = 32  # few tokens, so that random MTP drafts are often accepted
rng = random.Random(0)
# Two prompts of at most K tokens, and one of 3K tokens, prefilled in chunks.
# The last token id is the image token, so the prompts do not use it.
PROMPTS = [[rng.randrange(VOCAB - 1) for _ in range(n)] for n in (300, K - 8, 3000)]


def refit_weight_buckets(model: torch.nn.Module) -> None:
    """A validated sparse model accepts later buckets without indexer weights."""
    parameters = dict(model.named_parameters())
    indexer = next(name for name in parameters if ".indexer." in name)
    # A replicated norm's local tensor also has the full checkpoint shape at TP2.
    norm = next(name for name in parameters if name.endswith(".input_layernorm.weight"))
    for name in (indexer, norm):
        value = parameters[name].detach().clone()
        assert name in model.load_weights([(name, value)])
        torch.testing.assert_close(parameters[name], value, atol=0, rtol=0)
    assert model.load_weights([]) == set()


def run(path, index_topk, **kwargs) -> tuple[list, dict]:
    """Greedy outputs for PROMPTS at TP2, with SDSA (`index_topk`) or full
    attention (None), and the engine's metrics."""
    with VllmRunner(
        str(path),
        trust_remote_code=True,  # the checkpoint's config is remote code
        load_format="dummy",
        skip_tokenizer_init=True,
        hf_overrides={"text_config": {"index_topk": index_topk}},
        tensor_parallel_size=2,
        dtype="bfloat16",
        max_model_len=4096,
        block_size=None,  # the hybrid model picks it
        enable_chunked_prefill=True,
        max_num_batched_tokens=512,
        limit_mm_per_prompt={"image": 0, "video": 0},
        disable_log_stats=False,
        **kwargs,
    ) as vllm:
        vllm.llm.apply_model(realistic_weights)
        if index_topk is not None:
            vllm.llm.apply_model(refit_weight_buckets)
        params = SamplingParams(
            temperature=0, max_tokens=256, logprobs=5, prompt_logprobs=0
        )
        outputs = vllm.llm.generate([{"prompt_token_ids": p} for p in PROMPTS], params)
        return outputs, {m.name: m for m in vllm.llm.get_metrics()}


def prompt_logprobs(output) -> torch.Tensor:
    """Entry i is the log prob of prompt token i + 1, which sees i + 1 keys."""
    tokens, steps = output.prompt_token_ids[1:], output.prompt_logprobs[1:]
    return torch.tensor([step[token].logprob for token, step in zip(tokens, steps)])


def test_sdsa_engine(tmp_path, monkeypatch) -> None:
    """Up to K keys SDSA keeps every key, so it scores the prompts as full
    attention does, within 0.05 (the kernels differ; measured at most 0.03).
    Past K it attends to K keys, so the long prompt's scores move away, by 0.03
    on average (measured 0.06). MTP drafts do not change greedy decoding, so
    SDSA with drafts picks full attention's tokens on the short prompts, and
    some drafts are accepted."""
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    # apply_model sends realistic_weights to the workers by pickle.
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    # No MoE layers: router near-ties would flip experts between the two runs.
    layers = ["linear_attention"] * 3 + ["full_attention"]
    text = make_config(
        layer_types=layers * 2,
        vocab_size=VOCAB,
        hidden_size=1024,
        num_attention_heads=8,
        head_dim=256,
        # GDP's kernels take 64-dim heads and 128-dim keys.
        gdp_num_heads=8,
        gdp_num_groups=8,
        gdp_head_dim=64,
        gdp_key_head_dim=128,
        max_position_embeddings=4096,
        num_nextn_predict_layers=1,
        tie_word_embeddings=False,
    )
    save_checkpoint_config(tmp_path, text, image_token_id=VOCAB - 1)

    mtp = {"method": "mtp", "num_speculative_tokens": 2}
    sdsa, metrics = run(tmp_path, K, speculative_config=mtp)
    full, _ = run(tmp_path, None)

    for sdsa_prompt, full_prompt in zip(sdsa, full):
        difference = prompt_logprobs(sdsa_prompt) - prompt_logprobs(full_prompt)
        assert difference[:K].abs().max() < 0.05
    difference = prompt_logprobs(sdsa[-1]) - prompt_logprobs(full[-1])
    assert difference[K:].abs().mean() > 0.03
    check_logprobs_close(
        outputs_0_lst=[
            (o.outputs[0].token_ids, "", o.outputs[0].logprobs) for o in sdsa[:2]
        ],
        outputs_1_lst=[
            (o.outputs[0].token_ids, "", o.outputs[0].logprobs) for o in full[:2]
        ],
        name_0="sdsa_mtp",
        name_1="full",
    )
    assert metrics["vllm:spec_decode_num_accepted_tokens"].value > 0
