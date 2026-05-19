# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Smoke tests for HrmTextForCausalLM."""

import pytest

# HRM-Text was added in HF transformers main and is not yet on pypi.
# Skip the entire module if the user's transformers does not have it.
pytest.importorskip("transformers")
HrmTextConfig = pytest.importorskip(
    "transformers.models.hrm_text.configuration_hrm_text"
).HrmTextConfig
HrmTextForCausalLM = pytest.importorskip(
    "transformers.models.hrm_text.modeling_hrm_text"
).HrmTextForCausalLM

from transformers import AutoTokenizer  # noqa: E402

from vllm import LLM  # noqa: E402
from vllm.sampling_params import SamplingParams  # noqa: E402


@pytest.fixture(scope="module")
def tiny_hrm_text_dir(tmp_path_factory):
    """Build a tiny HRM-Text and write a randomly-initialized checkpoint
    to a temp dir, so vLLM can load it from disk.
    """
    config = HrmTextConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,  # per-stack count, will be inflated
        num_attention_heads=4,
        head_dim=16,
        H_cycles=2,
        L_cycles=2,
        L_bp_cycles=[1],
        max_position_embeddings=128,
        rms_norm_eps=1e-6,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
        attention_bias=False,
        mlp_bias=False,
        tie_word_embeddings=True,
        prefix_lm=True,
    )
    model = HrmTextForCausalLM(config)
    out_dir = tmp_path_factory.mktemp("tiny_hrm_text")
    model.save_pretrained(out_dir)
    # Reuse a tiny tokenizer that fits the vocab.
    tok = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
    tok.save_pretrained(out_dir)
    return str(out_dir)


def test_hrm_text_smoke_generation(tiny_hrm_text_dir):
    """Construct the model and generate a few tokens deterministically."""
    llm = LLM(
        model=tiny_hrm_text_dir,
        enforce_eager=True,
        gpu_memory_utilization=0.3,
        max_model_len=64,
        dtype="bfloat16",
    )
    sp = SamplingParams(temperature=0, max_tokens=4)
    out1 = llm.generate(["hello world"], sp)
    out2 = llm.generate(["hello world"], sp)
    assert out1[0].outputs[0].token_ids == out2[0].outputs[0].token_ids


def test_hrm_text_batched_generation(tiny_hrm_text_dir):
    """Batched generate (mixed prefill+decode) must work — regression test
    for the bug where a gated `is_prefilling.all()` causal flip silently
    ran prompts as pure causal in mixed batches."""
    llm = LLM(
        model=tiny_hrm_text_dir,
        enforce_eager=True,
        gpu_memory_utilization=0.3,
        max_model_len=64,
        dtype="bfloat16",
    )
    sp = SamplingParams(temperature=0, max_tokens=4)
    prompts = ["hello world", "the answer is", "lorem ipsum dolor"]
    out_batched = llm.generate(prompts, sp)
    # Determinism: same batch twice -> same outputs.
    out_again = llm.generate(prompts, sp)
    for a, b in zip(out_batched, out_again):
        assert a.outputs[0].token_ids == b.outputs[0].token_ids
    # Per-prompt: results don't depend on batch composition.
    for i, p in enumerate(prompts):
        single = llm.generate([p], sp)
        assert single[0].outputs[0].token_ids == out_batched[i].outputs[0].token_ids
