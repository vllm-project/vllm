# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check a small dummy-weight MiniMax-M3 execution against a pinned baseline.

Run once with --baseline-source and once without it, supplying the first JSON
output as --reference-output. This checks model execution and generated-output
parity, not pretrained model quality. Both runs use eager execution and the
Torch sampler; the attention implementation is the only changed backend.
"""

import argparse
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-source", type=Path)
    parser.add_argument("--reference-output", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"

    from vllm import LLM, SamplingParams
    from vllm.models.minimax_m3.common import sparse_attention as common_impl
    from vllm.models.minimax_m3.common.ops import sparse_attn
    from vllm.models.minimax_m3.nvidia import sparse_attention_msa as msa_impl
    from vllm.transformers_utils.configs.minimax_m3 import (
        MiniMaxM3Config,
        MiniMaxM3TextConfig,
    )

    decode = sparse_attn.minimax_m3_sparse_attn_decode
    if args.baseline_source:
        spec = importlib.util.spec_from_file_location(
            "msa_model_baseline", args.baseline_source
        )
        assert spec is not None and spec.loader is not None
        baseline = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = baseline
        spec.loader.exec_module(baseline)
        decode = baseline.minimax_m3_sparse_attn_decode

    calls = []

    def instrumented(q, kv_cache, topk_idx, block_table, *rest, **kwargs):
        calls.append(
            dict(query_shape=list(q.shape), page_table_shape=list(block_table.shape))
        )
        return decode(q, kv_cache, topk_idx, block_table, *rest, **kwargs)

    common_impl.minimax_m3_sparse_attn_decode = instrumented
    msa_impl.minimax_m3_sparse_attn_decode = instrumented
    config = MiniMaxM3TextConfig(
        vocab_size=256,
        hidden_size=512,
        intermediate_size=256,
        dense_intermediate_size=1024,
        shared_intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=16,
        num_key_value_heads=1,
        head_dim=128,
        max_position_embeddings=1024,
        num_local_experts=4,
        num_experts_per_tok=2,
        num_mtp_modules=0,
        moe_layer_freq=[0, 0],
        sparse_attention_config=dict(
            use_sparse_attention=True,
            sparse_index_dim=128,
            sparse_num_index_heads=1,
            sparse_topk_blocks=16,
            sparse_block_size=128,
            sparse_disable_index_value=[1, 1],
            sparse_score_type="max",
            sparse_init_block=0,
            sparse_local_block=1,
            sparse_attention_freq=[1, 1],
        ),
        architectures=["MiniMaxM3SparseForCausalLM"],
        torch_dtype="bfloat16",
    )
    with tempfile.TemporaryDirectory(prefix="minimax-m3-decode-") as model_dir:
        MiniMaxM3Config(
            text_config=config, architectures=["MiniMaxM3SparseForCausalLM"]
        ).save_pretrained(model_dir)
        llm = LLM(
            model=model_dir,
            skip_tokenizer_init=True,
            load_format="dummy",
            dtype="bfloat16",
            max_model_len=1024,
            max_num_seqs=4,
            max_num_batched_tokens=1024,
            gpu_memory_utilization=0.15,
            enforce_eager=True,
            seed=731,
            distributed_executor_backend="uni",
            disable_log_stats=True,
            block_size=128,
            enable_prefix_caching=False,
            kv_cache_dtype="auto",
        )
        calls.clear()
        try:
            outputs = llm.generate(
                [
                    dict(prompt_token_ids=[1 + i % 200 for i in range(length)])
                    for length in [31, 129, 385]
                ],
                SamplingParams(
                    temperature=0,
                    max_tokens=8,
                    ignore_eos=True,
                    detokenize=False,
                    logprobs=5,
                ),
                use_tqdm=False,
            )
        finally:
            llm.llm_engine.engine_core.shutdown()

    assert len(calls) == 14, "Expected two sparse layers over seven decode steps"
    rows = []
    for output in outputs:
        sequence = output.outputs[0]
        assert sequence.logprobs is not None
        rows.append(
            dict(
                token_ids=list(sequence.token_ids),
                logprobs=[
                    {str(token): value.logprob for token, value in step.items()}
                    for step in sequence.logprobs
                ],
            )
        )
    result = dict(dummy_weights=True, calls=calls, outputs=rows)
    if args.reference_output:
        reference = json.loads(args.reference_output.read_text())
        assert reference["calls"] == calls
        differences = []
        for ref, actual in zip(reference["outputs"], rows, strict=True):
            assert ref["token_ids"] == actual["token_ids"]
            for left, right in zip(ref["logprobs"], actual["logprobs"], strict=True):
                assert left.keys() == right.keys()
                differences.extend(abs(left[token] - right[token]) for token in left)
        maximum = max(differences)
        assert maximum <= 1e-3, maximum
        result["max_logprob_difference"] = maximum
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print("Model execution check passed: 24 output tokens, 14 decode calls")


if __name__ == "__main__":
    main()
