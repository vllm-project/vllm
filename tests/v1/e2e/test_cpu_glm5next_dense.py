# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3-Flash runs dense on CPU (reference implementation).

Builds a tiny GLM-5.3-Flash checkpoint (KDA + dense-MLA layers, v3.2 sparse
config) with dummy weights and drives prefill + decode through the CPU
engine:

* the sparse indexer is CUDA-only, so the model-config hook must fall a
  v3.2 ``index_topk`` config back to the dense MLA path on CPU;
* the KDA layers execute through the pure-torch reference ops;
* the MLA layers execute through the reference CPU_MLA backend (SDPA
  prefill; decode uses the compiled kernel for DeepSeek-style shapes and
  a plain-SDPA fallback otherwise).

The tiny random weights make generation quality meaningless; the test
asserts the engine produces finite logits and sane token ids across
several decode steps.
"""

import os

import pytest
import torch

from vllm import LLM, SamplingParams
from vllm.platforms import current_platform
from vllm.transformers_utils.configs.glm5_next import Glm5NextConfig

pytestmark = pytest.mark.cpu_model

if not current_platform.is_cpu():
    pytest.skip("skipping CPU-only tests", allow_module_level=True)

os.environ.setdefault("VLLM_CPU_KVCACHE_SPACE", "1")

VOCAB_SIZE = 512
PROMPT_TOKEN_IDS = [10, 11, 12, 13, 14, 15]


def _make_model_dir(tmp_path) -> str:
    """A v3.2 (sparse) config small enough for a fast CPU run."""
    cfg = Glm5NextConfig(
        vocab_size=VOCAB_SIZE,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=4,
        num_attention_heads=4,
        q_lora_rank=64,
        kv_lora_rank=512,
        qk_nope_head_dim=32,
        qk_rope_head_dim=0,  # nope-only latent, like the real checkpoint
        v_head_dim=32,
        mla=True,
        mla_nope=True,
        layer_types=[
            "linear_attention",
            "deepseek_sparse_attention",
            "deepseek_sparse_attention",
            "linear_attention",
        ],
        mlp_layer_types=["dense"] * 4,
        linear_head_dim=32,
        linear_num_heads=4,
        linear_conv_kernel_dim=4,
        linear_lower_bound=-5.0,
        # v3.2 sparse fields: the CPU config hook forces index_topk=None so
        # the dense MLA path runs instead of the CUDA-only sparse indexer.
        index_topk=2048,
        index_n_heads=8,
        index_head_dim=64,
        index_kpool=4,
        mhc=False,
        n_routed_experts=8,
        num_experts_per_token=1,
        num_nextn_predict_layers=0,
        max_position_embeddings=512,
        rms_norm_eps=1e-5,
        logit_scale=1.0,
    )
    cfg.architectures = ["Glm5NextForCausalLM"]
    model_dir = str(tmp_path / "tiny-glm5next")
    cfg.save_pretrained(model_dir)
    return model_dir


def test_glm5next_dense_generate_on_cpu(tmp_path):
    model_dir = _make_model_dir(tmp_path)
    llm = LLM(
        model=model_dir,
        load_format="dummy",
        dtype="float32",
        skip_tokenizer_init=True,
        max_model_len=256,
        enforce_eager=True,
        max_num_seqs=4,
    )
    outputs = llm.generate(
        [{"prompt_token_ids": PROMPT_TOKEN_IDS} for _ in range(2)],
        SamplingParams(max_tokens=8, temperature=0.0, logprobs=5),
    )
    assert len(outputs) == 2
    for out in outputs:
        ids = out.outputs[0].token_ids
        assert len(ids) == 8
        assert all(0 <= t < VOCAB_SIZE for t in ids)
        # dummy weights still produce a full distribution over the vocab
        logprobs = out.outputs[0].logprobs[0]
        assert len(logprobs) == 5
        assert all(torch.isfinite(torch.tensor(v.logprob)) for v in logprobs.values())
