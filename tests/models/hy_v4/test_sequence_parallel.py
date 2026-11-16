# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from functools import partial

import pytest

from tests.models.utils import check_logprobs_close
from vllm import SamplingParams
from vllm.outputs import CompletionOutput
from vllm.transformers_utils.configs.hy_v4 import HYV4Config


def _check_ihc_rows(worker, min_tokens):
    from vllm.distributed import get_tensor_model_parallel_world_size
    from vllm.forward_context import get_forward_context
    from vllm.model_executor.layers.attention.sparse_mla_attention import (
        SparseMLACommonMetadata,
    )

    model = worker.model_runner.model.model
    world = get_tensor_model_parallel_world_size()
    worker.ihc_token_rows = []
    current_rows = 0
    shardq_active = False

    def begin_layer(module, args):
        nonlocal current_rows, shardq_active
        current_rows = args[0].shape[0]
        metadata = get_forward_context().attn_metadata
        assert isinstance(metadata, dict)
        layer_metadata = metadata[module.self_attn.mla_attn.layer_name]
        assert isinstance(layer_metadata, SparseMLACommonMetadata)
        shardq_active = layer_metadata.shardq is not None

    def check_rows(module, args):
        expected = (
            (current_rows + world - 1) // world
            if current_rows >= min_tokens
            else current_rows
        )
        assert args[0].shape[0] == expected, (
            "iHC token rows",
            current_rows,
            expected,
            args[0].shape[0],
        )
        worker.ihc_token_rows.append((current_rows, expected, shardq_active))

    for layer in model.layers:
        layer.register_forward_pre_hook(begin_layer)
        layer.hc_attn_layer.hc_pre.register_forward_pre_hook(check_rows)
        layer.hc_mlp_layer.hc_pre.register_forward_pre_hook(check_rows)


@pytest.mark.parametrize("enforce_eager", [True, False])
def test_ihc_shards_rows_and_preserves_generation(
    vllm_runner, tmp_path, monkeypatch, enforce_eager
):
    """Keep iHC rows local with ShardQ, odd chunks, shared indexers and replay."""
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    config = HYV4Config(
        vocab_size=1024,
        hidden_size=256,
        intermediate_size=512,
        moe_intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=32,
        num_key_value_heads=32,
        q_lora_rank=128,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        index_n_heads=32,
        index_topk=512,
        indexer_types=["full", "shared"],
        layer_types=["deepseek_sparse_attention"] * 2,
        max_position_embeddings=1024,
        num_nextn_predict_layers=0,
    )
    config.architectures = ["HYV4ForCausalLM"]
    config.save_pretrained(tmp_path)
    prompts = [
        {"prompt_token_ids": [2 + i % 997 for i in range(n)]} for n in (769, 17, 3)
    ]
    sampling = SamplingParams(temperature=0, max_tokens=4, ignore_eos=True, logprobs=16)
    outputs: list[list[CompletionOutput]] = []
    for use_shardq in (False, True):
        with vllm_runner(
            str(tmp_path),
            load_format="dummy",
            skip_tokenizer_init=True,
            tensor_parallel_size=2,
            enable_expert_parallel=True,
            dtype="bfloat16",
            enforce_eager=enforce_eager,
            max_model_len=1024,
            max_num_batched_tokens=255,
            max_num_seqs=4,
            enable_chunked_prefill=True,
            enable_prefix_caching=True,
            block_size=64,
            kv_cache_dtype="fp8_ds_mla",
            attention_config={
                "backend": "FLASHMLA_SPARSE",
                "sparse_mla_shard_q": use_shardq,
                "sparse_mla_shard_q_min_tokens": 64,
            },
            gpu_memory_utilization=0.1,
        ) as runner:
            if use_shardq and enforce_eager:
                runner.llm.collective_rpc(partial(_check_ihc_rows, min_tokens=64))
            runs: list[CompletionOutput] = []
            for batch in (prompts, prompts[:1]):
                result = runner.llm.generate(batch, sampling, use_tqdm=False)
                runs.extend(r.outputs[0] for r in result)
            outputs.append(runs)
            if use_shardq and enforce_eager:
                observed = runner.llm.collective_rpc(
                    lambda worker: worker.ihc_token_rows
                )
                assert all(
                    any(full > local and shardq for full, local, shardq in rows)
                    for rows in observed
                ), "iHC SP and attention ShardQ did not run together"
    check_logprobs_close(
        outputs_0_lst=[(list(r.token_ids), r.text, r.logprobs) for r in outputs[0]],
        outputs_1_lst=[(list(r.token_ids), r.text, r.logprobs) for r in outputs[1]],
        name_0="replicated",
        name_1="sequence-parallel",
    )
