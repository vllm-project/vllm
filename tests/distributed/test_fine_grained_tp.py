# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Orthogonal TP x PCP x DP fine-grained vocabulary sharding tests."""

import pytest
import torch
import torch.multiprocessing as mp

from vllm.config import (
    FineGrainedTPConfig,
    ParallelConfig,
    SchedulerConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.distributed import (
    cleanup_dist_env_and_memory,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.utils.network_utils import get_open_port


def test_fine_grained_tp_config_validation():
    """Allowed {1, pcp, dp * pcp} sizes and prompt-logprob capability."""
    for bad_size in (3, 4):
        with pytest.raises(ValueError, match="embedding_tp_size"):
            ParallelConfig(
                data_parallel_size=4,
                prefill_context_parallel_size=2,
                tensor_parallel_size=2,
                distributed_executor_backend="uni",
                fine_grained_tp_config=FineGrainedTPConfig(embedding_tp_size=bad_size),
            )
        with pytest.raises(ValueError, match="lmhead_tp_size"):
            ParallelConfig(
                data_parallel_size=4,
                prefill_context_parallel_size=2,
                tensor_parallel_size=2,
                distributed_executor_backend="uni",
                fine_grained_tp_config=FineGrainedTPConfig(lmhead_tp_size=bad_size),
            )

    with pytest.raises(ValueError, match="embedding_tp_size"):
        ParallelConfig(
            data_parallel_size=4,
            prefill_context_parallel_size=4,
            distributed_executor_backend="uni",
            fine_grained_tp_config=FineGrainedTPConfig(embedding_tp_size=2),
        )

    pcp_sharded = ParallelConfig(
        data_parallel_size=4,
        prefill_context_parallel_size=2,
        tensor_parallel_size=2,
        distributed_executor_backend="uni",
        fine_grained_tp_config=FineGrainedTPConfig(
            embedding_tp_size=8,
            lmhead_tp_size=2,
        ),
    )
    assert pcp_sharded.fine_grained_tp_config.enabled
    assert pcp_sharded.embedding_dp_pcp_factors == (4, 2)
    assert pcp_sharded.lmhead_dp_pcp_factors == (1, 2)
    assert pcp_sharded.supports_prompt_logprobs

    dp_pcp_sharded = ParallelConfig(
        data_parallel_size=4,
        prefill_context_parallel_size=2,
        tensor_parallel_size=2,
        distributed_executor_backend="uni",
        fine_grained_tp_config=FineGrainedTPConfig(
            embedding_tp_size=8,
            lmhead_tp_size=8,
        ),
    )
    assert dp_pcp_sharded.lmhead_dp_pcp_factors == (4, 2)
    assert not dp_pcp_sharded.supports_prompt_logprobs


def _layer_worker(rank, dp, pcp, tp, emb_tp, lm_tp, port, dtype):
    torch.accelerator.set_device_index(rank)
    init_distributed_environment(
        world_size=dp * pcp * tp,
        rank=rank,
        local_rank=rank,
        distributed_init_method=f"tcp://127.0.0.1:{port}",
    )
    dp_rank = rank // (pcp * tp)
    pcp_rank = (rank // tp) % pcp
    config = VllmConfig(
        parallel_config=ParallelConfig(
            data_parallel_size=dp,
            data_parallel_rank=dp_rank,
            prefill_context_parallel_size=pcp,
            tensor_parallel_size=tp,
            enable_expert_parallel=True,
            fine_grained_tp_config=FineGrainedTPConfig(
                embedding_tp_size=emb_tp,
                lmhead_tp_size=lm_tp,
            ),
        ),
        scheduler_config=SchedulerConfig(
            max_model_len=64,
            is_encoder_decoder=False,
            max_num_batched_tokens=64,
            max_num_seqs=8,
        ),
    )
    lm_dp, _ = config.parallel_config.lmhead_dp_pcp_factors
    with set_current_vllm_config(config), torch.device(f"cuda:{rank}"):
        initialize_model_parallel(
            tensor_model_parallel_size=tp, prefill_context_model_parallel_size=pcp
        )
        # Omit prefix on embedding to verify role resolution without prefix strings.
        embedding = VocabParallelEmbedding(131, 32, params_dtype=dtype)
        head = ParallelLMHead(131, 32, params_dtype=dtype, prefix="lm_head")
        replicated_head = ParallelLMHead(
            131, 32, params_dtype=dtype, prefix="aux_head", disable_tp=True
        )
        assert replicated_head.tp_size == 1
        assert embedding.tp_size == tp * emb_tp
        assert head.tp_size == tp * lm_tp

        processor = LogitsProcessor(131)
        values = torch.arange(131 * 32, device=f"cuda:{rank}").reshape(131, 32)
        weights = ((values % 17 - 8) / 16).to(dtype)
        embedding.weight_loader(embedding.weight, weights)
        head.weight_loader(head.weight, weights)

        embed_owner = dp_rank * pcp + pcp_rank
        lm_owner = dp_rank
        tie_modes = (False, True) if emb_tp == lm_tp else (False,)
        if emb_tp != lm_tp:
            with pytest.raises(ValueError, match="mismatched vocabulary shards"):
                head.tie_weights(embedding)

        for tied in tie_modes:
            if tied:
                head.tie_weights(embedding)
                assert head.weight.data_ptr() == embedding.weight.data_ptr()
            for step in range(3):
                n_emb = (embed_owner + step) % 8
                ids = (
                    torch.arange(n_emb, dtype=torch.int64, device=f"cuda:{rank}") * 19
                    + embed_owner
                ) % 131
                expected_emb = weights[ids]
                actual_emb = embedding(ids)
                torch.testing.assert_close(actual_emb, expected_emb, atol=0, rtol=0)

                # Sampling runs after PCP restores per-DP global rows.
                n_lm = (lm_owner + step) % 8
                lm_ids = (
                    torch.arange(n_lm, dtype=torch.int64, device=f"cuda:{rank}") * 23
                    + lm_owner
                ) % 131
                lm_hidden = weights[lm_ids]
                logits = processor(head, lm_hidden)
                torch.testing.assert_close(
                    logits, lm_hidden @ weights.T, atol=0, rtol=0
                )

        torch.manual_seed(7)
        weights = torch.randn(131, 32, dtype=dtype, device=f"cuda:{rank}") * 0.01
        embedding.weight_loader(embedding.weight, weights)
        is_tied = (
            emb_tp == lm_tp and head.weight.data_ptr() == embedding.weight.data_ptr()
        )
        if not is_tied:
            head.weight_loader(head.weight, weights)

        ids = (torch.arange(5, device=f"cuda:{rank}") + embed_owner * 11) % 131
        actual = embedding(ids)
        torch.testing.assert_close(actual, weights[ids], atol=0, rtol=0)

        lm_ids = (torch.arange(5, device=f"cuda:{rank}") + lm_owner * 11) % 131
        lm_hidden = weights[lm_ids]
        reference = (lm_hidden.float() @ weights.float().T).to(dtype)
        full_logits = processor(head, lm_hidden)
        torch.testing.assert_close(full_logits, reference, atol=2e-5, rtol=0.02)

        if lm_dp == 1:
            top1 = processor.get_top_tokens(head, lm_hidden)
            torch.testing.assert_close(top1, full_logits.argmax(dim=-1))
            topk_ids, _ = processor.get_top_k_tokens(head, lm_hidden, k=4)
            torch.testing.assert_close(topk_ids, full_logits.topk(4, dim=-1).indices)

        # Capture and replay CUDA graphs with updated inputs to verify owner indexing.
        graph_ids = torch.tensor(
            [embed_owner % 131, 130, 0], dtype=torch.int64, device=f"cuda:{rank}"
        )
        graph_hidden = weights[
            torch.tensor([lm_owner % 131, 64, 1], device=f"cuda:{rank}")
        ].clone()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            processor(head, graph_hidden)
            embedding(graph_ids)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            captured_emb = embedding(graph_ids)
            captured_logits = processor(head, graph_hidden)
        new_ids = torch.tensor(
            [130, (embed_owner + 1) % 131, 65],
            dtype=torch.int64,
            device=f"cuda:{rank}",
        )
        new_lm_ids = torch.tensor(
            [129, (lm_owner + 3) % 131, 42],
            dtype=torch.int64,
            device=f"cuda:{rank}",
        )
        graph_ids.copy_(new_ids)
        graph_hidden.copy_(weights[new_lm_ids])
        graph.replay()
        torch.testing.assert_close(captured_emb, weights[new_ids], atol=0, rtol=0)
        ref_graph = (weights[new_lm_ids].float() @ weights.float().T).to(dtype)
        torch.testing.assert_close(captured_logits, ref_graph, atol=2e-5, rtol=0.02)
    cleanup_dist_env_and_memory()


@pytest.mark.skipif(
    not torch.accelerator.is_available(), reason="Requires accelerator devices"
)
@pytest.mark.parametrize(
    "dp,pcp,tp,emb_tp,lm_tp",
    [
        (2, 1, 1, 2, 2),
        (4, 1, 1, 4, 1),
        (1, 2, 2, 2, 2),
        (2, 2, 1, 4, 2),
        (2, 2, 2, 4, 4),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fine_grained_tp_orthogonal_layers(dp, pcp, tp, emb_tp, lm_tp, dtype):
    """Cover DPxTP, PCPxTP, untied asymmetric heads, and full DPxPCPxTP."""
    if torch.accelerator.device_count() < dp * pcp * tp:
        pytest.skip(f"Requires {dp * pcp * tp} GPUs")
    mp.spawn(
        _layer_worker,
        args=(dp, pcp, tp, emb_tp, lm_tp, get_open_port(), dtype),
        nprocs=dp * pcp * tp,
    )


def _save_tiny_checkpoint(tmp_path):
    from transformers import DeepseekV2Config, DeepseekV2ForCausalLM

    torch.manual_seed(7)
    config = DeepseekV2Config(
        vocab_size=256,
        hidden_size=128,
        intermediate_size=256,
        moe_intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        q_lora_rank=64,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        n_routed_experts=8,
        n_shared_experts=1,
        num_experts_per_tok=2,
        first_k_dense_replace=1,
        topk_method="greedy",
        n_group=1,
        topk_group=1,
        max_position_embeddings=256,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    model = DeepseekV2ForCausalLM(config)
    model.save_pretrained(tmp_path)
    del model


@pytest.mark.skipif(torch.accelerator.device_count() < 2, reason="Requires >= 2 GPUs")
def test_fine_grained_tp_pcp_checkpoint_generation(tmp_path, vllm_runner):
    """PCP*TP vocabulary sharding preserves tokens and keeps prompt_logprobs."""
    from vllm import SamplingParams

    _save_tiny_checkpoint(tmp_path)
    outputs = []
    for enabled in (False, True):
        fg_cfg = {"embedding_tp_size": 2, "lmhead_tp_size": 2} if enabled else {}
        with vllm_runner(
            str(tmp_path),
            dtype="bfloat16",
            enforce_eager=True,
            max_model_len=128,
            max_num_batched_tokens=64,
            max_num_seqs=8,
            prefill_context_parallel_size=2,
            enable_expert_parallel=True,
            fine_grained_tp_config=fg_cfg,
            skip_tokenizer_init=True,
            gpu_memory_utilization=0.2,
            moe_backend="triton",
        ) as runner:
            requests = [{"prompt_token_ids": [1, 10, 21] * n} for n in (1, 3, 7)]
            result = runner.llm.generate(
                requests,
                SamplingParams(
                    temperature=0,
                    max_tokens=8,
                    prompt_logprobs=1,
                    ignore_eos=True,
                ),
            )
            outputs.append([item.outputs[0].token_ids for item in result])
    assert outputs[0] == outputs[1]


@pytest.mark.skipif(torch.accelerator.device_count() < 8, reason="Requires 8 GPUs")
@pytest.mark.parametrize("cuda_graphs", [False, True])
def test_fine_grained_tp_unbalanced_dp_server(tmp_path, cuda_graphs):
    """Full DPxPCPxTP sharding handles idle DP ranks, SP decode, and graphs."""
    import json
    from concurrent.futures import ThreadPoolExecutor

    import openai

    from tests.utils import RemoteOpenAIServer

    _save_tiny_checkpoint(tmp_path)
    args = [
        "--data-parallel-size",
        "2",
        "--prefill-context-parallel-size",
        "2",
        "--tensor-parallel-size",
        "2",
        "--enable-expert-parallel",
        "--enforce-eager",
        "--skip-tokenizer-init",
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "128",
        "--max-num-batched-tokens",
        "64",
        "--max-num-seqs",
        "8",
        "--gpu-memory-utilization",
        "0.03",
        "--kv-cache-memory-bytes",
        "67108864",
        "--moe-backend",
        "triton",
    ]
    prompts = [[1, 10, 21] * n for n in (1, 3, 7)]
    if cuda_graphs:
        args.remove("--enforce-eager")
        args += [
            "--compilation-config",
            json.dumps(
                {
                    "mode": 0,
                    "cudagraph_mode": "FULL_DECODE_ONLY",
                    "cudagraph_capture_sizes": [1, 2, 4, 8],
                }
            ),
        ]
    outputs = []
    for enabled in (False, True):
        flags = list(args)
        if enabled:
            flags += [
                "--fine-grained-tp-config",
                json.dumps(
                    {
                        "embedding_tp_size": 4,
                        "lmhead_tp_size": 4,
                    }
                ),
            ]
        with (
            RemoteOpenAIServer(str(tmp_path), flags) as server,
            server.get_client() as client,
        ):

            def complete(prompt):
                result = client.completions.create(
                    model=str(tmp_path),
                    prompt=prompt,
                    max_tokens=8,
                    temperature=0,
                    logprobs=5,
                    extra_body={
                        "return_token_ids": True,
                        "ignore_eos": True,
                        "return_tokens_as_token_ids": True,
                    },
                )
                return result.choices[0].model_dump()

            results = [complete(prompt) for prompt in prompts]
            with ThreadPoolExecutor(max_workers=4) as executor:
                results.extend(executor.map(complete, prompts * 2))
            outputs.append(results)
            if enabled:
                with pytest.raises(openai.BadRequestError, match="lmhead_tp_size"):
                    client.completions.create(
                        model=str(tmp_path),
                        prompt=prompts[0],
                        max_tokens=1,
                        extra_body={"prompt_logprobs": 1},
                    )
                complete(prompts[0])
    for baseline, candidate in zip(*outputs):
        assert baseline["token_ids"] == candidate["token_ids"]
        torch.testing.assert_close(
            torch.tensor(baseline["logprobs"]["token_logprobs"]),
            torch.tensor(candidate["logprobs"]["token_logprobs"]),
            atol=0.01,
            rtol=0,
        )
