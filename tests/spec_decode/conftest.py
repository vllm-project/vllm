# SPDX-License-Identifier: Apache-2.0
import os
import random
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture(scope="function", autouse=True)
def use_v0_only(monkeypatch):
    """
    Since this module is V0 only, set VLLM_USE_V1=0 for
    all tests in the module.
    """
    monkeypatch.setenv('VLLM_USE_V1', '0')


# Layer skip test fixtures and helpers
TINY_LLAMA_HF = "hf-internal-testing/tiny-random-LlamaForCausalLM"  # CPU-only math
OPT_TINY = "facebook/opt-125m"  # GPU-friendly for vLLM smoke tests


def seed_all(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@pytest.fixture(scope="session", autouse=True)
def _seed_everything():
    seed_all(42)


@pytest.fixture(scope="session")
def opt_tiny_id():
    return OPT_TINY


@pytest.fixture(scope="session")
def hf_tiny_llama_id():
    return TINY_LLAMA_HF


@pytest.fixture(scope="session")
def dummy_target_config():
    # Minimal target_model_config-like object that SpeculativeConfig inspects
    hf_conf = SimpleNamespace(num_hidden_layers=12, model_type="llama")
    return SimpleNamespace(
        hf_config=hf_conf,          # many places read this
        hf_text_config=hf_conf,     # some releases read this instead
        model=TINY_LLAMA_HF,
        # Fields forwarded into draft ModelConfig construction
        tokenizer=None,
        tokenizer_mode="auto",
        trust_remote_code=False,
        allowed_local_media_path="",
        dtype="auto",
        seed=0,
        max_model_len=2048,
        enforce_eager=False,
        max_seq_len_to_capture=8192,
        max_logprobs=20,
        tokenizer_revision=None,
    )


# ---------------- HF math helper (CPU-only) ----------------

@torch.no_grad()
def hf_postnorm_at_k(model_id: str, input_ids: torch.Tensor, k: int,
                     dtype=torch.float32, device="cpu") -> torch.Tensor:
    from transformers import AutoModelForCausalLM
    hf = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype).to(device)
    m = hf.model if hasattr(hf, "model") else hf
    x = m.embed_tokens(input_ids.to(device))
    # Walk to layer k (inclusive)
    layers = getattr(m, "layers", None) or getattr(getattr(m, "decoder", object), "layers", None)
    assert layers is not None, "Unsupported HF model structure for this test."
    for i, layer in enumerate(layers):
        out = layer(hidden_states=x, attention_mask=None)
        x = out[0] if isinstance(out, tuple) else out
        if i >= k:
            break
    norm = getattr(m, "norm", None) or getattr(m, "final_layer_norm", None)
    assert norm is not None, "Unsupported HF model: missing norm."
    return norm(x)


@pytest.fixture
def hf_hidden_at_k():
    return hf_postnorm_at_k


# ------------- helper to reach proposer runner from LLM ----------------

def get_proposer_runner_from_llm(llm) -> object:
    """Return the proposer model runner (your EarlyExitModelRunner) from an LLM."""
    eng = getattr(llm, "llm_engine", None)
    spec = getattr(eng, "spec_worker", None)
    prop = getattr(spec, "proposer_worker", None)
    worker = getattr(prop, "worker", None)
    runner = getattr(worker, "model_runner", None)
    if runner is None:
        raise RuntimeError("Could not locate proposer model_runner")
    return runner
