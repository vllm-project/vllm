# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end smoke test of a tiny random-weight Inkling2 model (TP=1)."""

import json

import pytest
import torch

from vllm import LLM, SamplingParams
from vllm.platforms import current_platform

if not (
    current_platform.is_device_capability_family(100)
    or current_platform.is_device_capability(90)
):
    pytest.skip("Inkling2 requires Hopper or Blackwell", allow_module_level=True)

TINY_CONFIG = {
    "architectures": ["Inkling2ForConditionalGeneration"],
    "model_type": "inkling2",
    "eos_token_id": 1,
    "text_config": {
        "torch_dtype": "bfloat16",
        "hidden_size": 512,
        "num_hidden_layers": 2,
        "local_layer_ids": [0],
        "sliding_window_size": 128,
        "vocab_size": 1024,
        "unpadded_vocab_size": 1000,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "head_dim": 128,
        "n_routed_experts": 16,
        "num_experts_per_tok": 8,
        "intermediate_size": 256,
        "mlp_gate_softcap": 4.0,
        "mlp_up_softcap": 25.0,
    },
    "mtp_config": {
        "hidden_size": 512,
        "num_hidden_layers": 2,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "head_dim": 128,
        "intermediate_size": 512,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1e6,
        "max_position_embeddings": 4096,
        "local_layer_ids": [0],
        "sliding_window_size": 128,
        "block_size": 8,
        "markov_rank": 32,
        "markov_head_type": "rnn_hidden",
        "target_layer_ids": [1, 2],
    },
}


@pytest.mark.parametrize("draft_sample_method", [None, "greedy", "probabilistic"])
def test_tiny_model_generates(tmp_path, draft_sample_method):
    """One local and one full layer load (dummy weights) and decode with the
    paged KV and conv-state caches and CUDA graphs, optionally speculating
    with the DSpark draft (target feature capture, draft KV and graphs).
    Probabilistic drafting caches the draft's bf16 logits for rejection."""
    (tmp_path / "config.json").write_text(json.dumps(TINY_CONFIG))
    llm = LLM(
        speculative_config=(
            None
            if draft_sample_method is None
            else {
                "method": "dspark",
                "num_speculative_tokens": 7,
                "draft_sample_method": draft_sample_method,
            }
        ),
        model=str(tmp_path),
        load_format="dummy",
        skip_tokenizer_init=True,
        tokenizer_mode="hf",
        gpu_memory_utilization=0.035,
        max_model_len=2048,
        max_num_seqs=4,
        limit_mm_per_prompt={"image": 0},
    )
    prompts = [list(range(2, 300)), [5, 5, 5]]
    outputs = llm.generate(
        [{"prompt_token_ids": p} for p in prompts],
        SamplingParams(
            max_tokens=24,
            temperature=1.0 if draft_sample_method == "probabilistic" else 0,
            ignore_eos=True,
            detokenize=False,
        ),
    )
    for output in outputs:
        token_ids = output.outputs[0].token_ids
        assert len(token_ids) == 24
        assert all(0 <= t < 1000 for t in token_ids)


def test_tiny_model_lora_with_quantized_experts(tmp_path):
    """With LoRA enabled, the NVFP4 / MXFP8 experts are no LoRA target: the
    engine starts (its warmup adapter covers every LoRA module) and decodes
    with the attention-only adapters these checkpoints support."""
    (tmp_path / "config.json").write_text(json.dumps(TINY_CONFIG))
    llm = LLM(
        model=str(tmp_path),
        load_format="dummy",
        skip_tokenizer_init=True,
        tokenizer_mode="hf",
        gpu_memory_utilization=0.035,
        max_model_len=2048,
        max_num_seqs=4,
        enable_lora=True,
        max_lora_rank=8,
        limit_mm_per_prompt={"image": 0},
    )
    outputs = llm.generate(
        [{"prompt_token_ids": [5, 5, 5]}],
        SamplingParams(max_tokens=8, temperature=0, ignore_eos=True, detokenize=False),
    )
    assert len(outputs[0].outputs[0].token_ids) == 8


def _chunked_mixed_run(model: str, enforce_eager: bool) -> list[list[float]]:
    llm = LLM(
        model=model,
        load_format="dummy",
        skip_tokenizer_init=True,
        tokenizer_mode="hf",
        gpu_memory_utilization=0.035,
        max_model_len=2048,
        max_num_seqs=16,
        max_num_batched_tokens=256,
        enforce_eager=enforce_eager,
        # The 256-token piecewise graph is captured on 16 verify-sized requests.
        compilation_config={"cudagraph_capture_sizes": [8, 16, 256]},
        limit_mm_per_prompt={"image": 0},
    )
    # The short prompts decode while the long one prefills in 256-token chunks.
    prompts = [[7, 8, 9], [5, 5, 5], list(range(2, 702))]
    outputs = llm.generate(
        [{"prompt_token_ids": p} for p in prompts],
        SamplingParams(
            max_tokens=8,
            temperature=0,
            ignore_eos=True,
            detokenize=False,
            prompt_logprobs=0,
        ),
    )
    del llm
    logprobs = []
    for o in outputs:
        assert o.prompt_logprobs is not None
        logprobs.append(
            [next(iter(d.values())).logprob for d in o.prompt_logprobs[1:] if d]
        )
    return logprobs


def test_piecewise_graphs_match_eager(tmp_path):
    """Piecewise CUDA graphs are captured on a dummy decode-sized batch and
    replayed for any batch of the same token count, so attention must plan
    from the live batch: a chunked prompt prefilling alongside decodes runs
    and matches eager."""
    (tmp_path / "config.json").write_text(json.dumps(TINY_CONFIG))
    graph = _chunked_mixed_run(str(tmp_path), enforce_eager=False)
    eager = _chunked_mixed_run(str(tmp_path), enforce_eager=True)
    for g, e in zip(graph, eager):
        torch.testing.assert_close(torch.tensor(g), torch.tensor(e), rtol=0, atol=1e-2)


# `quantization_config` of the released checkpoints' config.json.
RELEASED_QUANTIZATION_CONFIGS = {
    "NVFP4": {"quant_method": "modelopt", "quant_algo": "NVFP4"},
    "MXFP8": {
        "quant_method": "fp8",
        "activation_scheme": "dynamic",
        "scale_fmt": "ue8m0",
        "weight_block_size": [1, 32],
        "modules_to_not_convert": [
            "model.llm.unembed",
            "model.visual",
            "model.llm.layers.*.attn.*",
        ],
    },
}


@pytest.mark.parametrize("quant_algo", ["NVFP4", "MXFP8"])
def test_released_checkpoint_quant_config(tmp_path, quant_algo):
    """Released checkpoints declare a ModelOpt (NVFP4) or FP8 (MXFP8)
    `quantization_config` in config.json; the per-tensor precisions in
    hf_quant_config.json win."""
    from vllm.config import LoadConfig, ModelConfig
    from vllm.model_executor.model_loader.weight_utils import get_quant_config
    from vllm.models.inkling2.quant_config import Inkling2QuantConfig

    config = {
        **TINY_CONFIG,
        "quantization_config": RELEASED_QUANTIZATION_CONFIGS[quant_algo],
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    experts = "model.llm.layers.{}.mlp.experts.w13_weight"
    mxfp8 = [0, 1] if quant_algo == "MXFP8" else [1]
    groups = [
        {"compute_dtype": "mxfp8", "tensors": [experts.format(i) for i in mxfp8]},
        {"compute_dtype": "bfloat16", "tensors": ["model.llm.embed_tokens.weight"]},
    ]
    if quant_algo == "NVFP4":
        groups.append({"compute_dtype": "nvfp4", "tensors": [experts.format(0)]})
    (tmp_path / "hf_quant_config.json").write_text(
        json.dumps({"quant_method": "inkling2", "model": {"parameter_groups": groups}})
    )
    model_config = ModelConfig(
        model=str(tmp_path), skip_tokenizer_init=True, tokenizer_mode="hf"
    )
    assert model_config.quantization == "inkling2"
    quant_config = get_quant_config(model_config, LoadConfig())
    assert isinstance(quant_config, Inkling2QuantConfig)
    assert [quant_config.is_mxfp8_layer(i) for i in range(2)] == [
        i in mxfp8 for i in range(2)
    ]
