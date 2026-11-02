# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise C4/C128 shard boundaries through real multi-GPU generation."""

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest


def _small_config():
    return {
        "architectures": ["DeepseekV4ForCausalLM"],
        "model_type": "deepseek_v4",
        "hidden_size": 4096,
        "num_hidden_layers": 3,
        "num_attention_heads": 64,
        "num_key_value_heads": 1,
        "head_dim": 512,
        "qk_rope_head_dim": 64,
        "q_lora_rank": 1024,
        "o_lora_rank": 1024,
        "o_groups": 8,
        "index_head_dim": 128,
        "index_n_heads": 64,
        "index_topk": 512,
        "compress_ratios": [0, 4, 128],
        "compress_rope_theta": 160000,
        "sliding_window": 128,
        "n_routed_experts": 8,
        "num_experts_per_tok": 2,
        "norm_topk_prob": True,
        "topk_method": "noaux_tc",
        "hidden_act": "silu",
        "attention_bias": False,
        "attention_dropout": 0.0,
        "tie_word_embeddings": False,
        "n_shared_experts": 1,
        "moe_intermediate_size": 2048,
        "num_hash_layers": 1,
        "num_nextn_predict_layers": 0,
        "expert_dtype": "fp8",
        "vocab_size": 1024,
        "max_position_embeddings": 1048576,
        "hc_mult": 4,
        "hc_eps": 1e-6,
        "hc_sinkhorn_iters": 20,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000,
        "rope_scaling": {
            "type": "yarn",
            "factor": 16,
            "original_max_position_embeddings": 65536,
            "beta_fast": 32,
            "beta_slow": 1,
        },
        "scoring_func": "sqrtsoftplus",
        "routed_scaling_factor": 1.5,
        "swiglu_limit": 10.0,
        "torch_dtype": "bfloat16",
        "quantization_config": {
            "quant_method": "fp8",
            "activation_scheme": "dynamic",
            "fmt": "e4m3",
            "scale_fmt": "ue8m0",
            "weight_block_size": [128, 128],
        },
    }


def _generate(model_path, output_path, pcp, tp=1, ep=False):
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model_path,
        load_format="dummy",
        skip_tokenizer_init=True,
        prefill_context_parallel_size=pcp,
        tensor_parallel_size=tp,
        enable_expert_parallel=ep,
        seed=1234,
        enforce_eager=True,
        enable_prefix_caching=False,
        max_model_len=4096,
        max_num_batched_tokens=520,
        max_num_seqs=8,
        gpu_memory_utilization=0.2,
        kv_cache_dtype="fp8_ds_mla",
        attention_config={"backend": "FLASHMLA_SPARSE"},
        kernel_config={"moe_backend": "triton"},
    )
    lengths = [3, 4, 5, 7, 8, 9, 127, 128, 129, 513, 2305]
    prompts = [{"prompt_token_ids": [10 + i % 100 for i in range(n)]} for n in lengths]
    results = llm.generate(
        prompts,
        SamplingParams(temperature=0, max_tokens=4, ignore_eos=True, logprobs=5),
    )
    records = []
    for result in results:
        completion = result.outputs[0]
        records.append(
            {
                "tokens": completion.token_ids,
                "logprobs": [
                    {str(k): v.logprob for k, v in step.items()}
                    for step in completion.logprobs
                ],
            }
        )
    Path(output_path).write_text(json.dumps(records))


@pytest.mark.distributed
@pytest.mark.parametrize(
    "topologies,expert_dtype",
    [
        (((1, 1, False), (2, 1, False)), "fp8"),
        (((1, 4, True), (2, 2, True)), "fp8"),
        (((1, 4, True), (2, 2, True)), "fp4"),
    ],
    ids=["pcp", "tp_ep", "mxfp4_tp_ep"],
)
def test_pcp_generation_across_compression_boundaries(tmp_path, topologies, expert_dtype):
    """Keep attention dimensions with three layers and eight experts."""
    model = tmp_path / "model"
    model.mkdir()
    config = _small_config()
    config["expert_dtype"] = expert_dtype
    (model / "config.json").write_text(json.dumps(config))
    env = dict(
        os.environ,
        VLLM_USE_V2_MODEL_RUNNER="1",
        VLLM_WORKER_MULTIPROC_METHOD="spawn",
        VLLM_DEEP_GEMM_WARMUP="skip",
    )
    outputs = []
    for pcp, tp, ep in topologies:
        output = tmp_path / f"pcp{pcp}.json"
        log = tmp_path / f"pcp{pcp}.log"
        with log.open("w") as stream:
            result = subprocess.run(
                [
                    sys.executable,
                    __file__,
                    str(model),
                    str(output),
                    str(pcp),
                    str(tp),
                    str(int(ep)),
                ],
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                timeout=900,
            )
        assert result.returncode == 0, log.read_text()
        outputs.append(json.loads(output.read_text()))
    assert len(outputs[0]) == len(outputs[1]) == 11
    # Dummy initialization depends on TP/EP weight partitioning. Only compare
    # logprobs when the topology preserves the initialized weights.
    compare_values = topologies[0][1:] == topologies[1][1:]
    for reference, actual in zip(*outputs):
        assert len(actual["tokens"]) == len(reference["tokens"]) == 4
        if compare_values:
            assert actual["tokens"] == reference["tokens"]
        for expected, observed in zip(reference["logprobs"], actual["logprobs"]):
            assert all(math.isfinite(value) for value in expected.values())
            assert all(math.isfinite(value) for value in observed.values())
            if compare_values:
                assert observed == pytest.approx(expected, abs=1e-5, rel=1e-5)


if __name__ == "__main__":
    _generate(
        sys.argv[1],
        sys.argv[2],
        int(sys.argv[3]),
        int(sys.argv[4]) if len(sys.argv) > 4 else 1,
        bool(int(sys.argv[5])) if len(sys.argv) > 5 else False,
    )
