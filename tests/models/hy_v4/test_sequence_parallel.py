# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from functools import partial

import pytest

from tests.models.utils import check_logprobs_close
from vllm import SamplingParams
from vllm.outputs import CompletionOutput
from vllm.transformers_utils.configs.hy_v4 import HYV4Config


def _keep_ihc_rows_replicated(worker):
    worker.model_runner.model.model.ihc_sequence_parallel = False


def _check_ihc_rows(worker, sequence_parallel):
    from vllm.distributed import get_tensor_model_parallel_world_size
    from vllm.forward_context import get_forward_context
    from vllm.model_executor.layers.attention.sparse_mla_attention import (
        SparseMLACommonMetadata,
    )

    model = worker.model_runner.model.model
    world = get_tensor_model_parallel_world_size()
    worker.ihc_token_rows = []
    current_rows = 0

    def begin_step(module, args, kwargs):
        nonlocal current_rows
        positions = kwargs["positions"] if "positions" in kwargs else args[1]
        current_rows = positions.shape[-1]

    def check_rows(layer, module, args):
        metadata = get_forward_context().attn_metadata
        assert isinstance(metadata, dict)
        layer_metadata = metadata[layer.self_attn.mla_attn.layer_name]
        assert isinstance(layer_metadata, SparseMLACommonMetadata)
        expected = (
            (current_rows + world - 1) // world
            if sequence_parallel and current_rows >= 1024
            else current_rows
        )
        assert args[0].shape[0] == expected, (
            "iHC token rows",
            current_rows,
            expected,
            args[0].shape[0],
        )
        worker.ihc_token_rows.append(
            (current_rows, expected, layer_metadata.shardq is not None)
        )

    model.register_forward_pre_hook(begin_step, with_kwargs=True)
    for layer in model.layers:
        # The fused boundaries replace the pre layers when the op is built.
        for boundary, hc_layer in (
            (layer.ihc_attn_boundary, layer.hc_attn_layer),
            (layer.ihc_mlp_boundary, layer.hc_mlp_layer),
        ):
            (boundary or hc_layer.hc_pre).register_forward_pre_hook(
                partial(check_rows, layer)
            )


@pytest.mark.parametrize("enforce_eager", [True, False])
def test_ihc_shards_rows_and_preserves_generation(
    vllm_runner, tmp_path, monkeypatch, enforce_eager
):
    """Shard iHC rows with and without ShardQ and preserve generation."""
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
        indexer_types=["full", "shared"],
        layer_types=["deepseek_sparse_attention"] * 2,
        max_position_embeddings=4096,
        num_nextn_predict_layers=0,
    )
    config.architectures = ["HYV4ForCausalLM"]
    config.save_pretrained(tmp_path)
    prompts = [
        {"prompt_token_ids": [2 + i % 997 for i in range(n)]} for n in (3073, 17, 3)
    ]
    sampling = SamplingParams(temperature=0, max_tokens=4, ignore_eos=True, logprobs=16)
    outputs: list[list[CompletionOutput]] = []
    modes = [(False, False), (True, False), (True, True)]
    for sequence_parallel, use_shardq in modes:
        with vllm_runner(
            str(tmp_path),
            load_format="dummy",
            skip_tokenizer_init=True,
            tensor_parallel_size=2,
            enable_expert_parallel=True,
            dtype="bfloat16",
            enforce_eager=enforce_eager,
            max_model_len=4096,
            max_num_batched_tokens=1025,
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
            if not sequence_parallel:
                runner.llm.collective_rpc(_keep_ihc_rows_replicated)
            if enforce_eager:
                runner.llm.collective_rpc(
                    partial(_check_ihc_rows, sequence_parallel=sequence_parallel)
                )
            runs: list[CompletionOutput] = []
            for batch in (prompts, prompts[:1]):
                result = runner.llm.generate(batch, sampling, use_tqdm=False)
                runs.extend(r.outputs[0] for r in result)
            outputs.append(runs)
            if enforce_eager:
                observed = runner.llm.collective_rpc(
                    lambda worker: worker.ihc_token_rows
                )
                assert all(
                    any(
                        (full > local) == sequence_parallel and shardq == use_shardq
                        for full, local, shardq in rows
                        if full >= 1024
                    )
                    for rows in observed
                ), "iHC SP and attention ShardQ were not selected independently"
    for (_, use_shardq), runs in zip(modes[1:], outputs[1:]):
        check_logprobs_close(
            outputs_0_lst=[(list(r.token_ids), r.text, r.logprobs) for r in outputs[0]],
            outputs_1_lst=[(list(r.token_ids), r.text, r.logprobs) for r in runs],
            name_0="replicated",
            name_1=f"sequence-parallel, shardq={use_shardq}",
        )
