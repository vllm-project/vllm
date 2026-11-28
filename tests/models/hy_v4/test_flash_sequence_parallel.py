# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from functools import partial
from unittest.mock import patch

import pytest

from tests.models.utils import check_logprobs_close
from vllm import SamplingParams
from vllm.models.hy_v4_flash.configs import HYV4FlashConfig
from vllm.outputs import CompletionOutput
from vllm.v1.worker.gpu_worker import Worker


class ReplicatedFlashWorker(Worker):
    def load_model(self, *, load_dummy_weights: bool = False) -> None:
        from vllm.models.hy_v4_flash.nvidia.model import HYV4FlashModel

        # Disable SP before constructing layers, not just before forward.
        with patch.object(HYV4FlashModel, "supports_ihc_sequence_parallel", False):
            super().load_model(load_dummy_weights=load_dummy_weights)


def _check_reduction_mode(worker, sequence_parallel):
    model = worker.model_runner.model.model
    assert model.ihc_sequence_parallel == sequence_parallel
    assert all(layer.defer_reduction == sequence_parallel for layer in model.layers)


def _check_ihc_rows(worker, sequence_parallel):
    from vllm.distributed import get_tensor_model_parallel_world_size

    model = worker.model_runner.model.model
    world = get_tensor_model_parallel_world_size()
    worker.ihc_token_rows = []
    current_rows = 0

    def begin_step(module, args, kwargs):
        nonlocal current_rows
        positions = kwargs["positions"] if "positions" in kwargs else args[1]
        current_rows = positions.shape[-1]

    def check_rows(module, args):
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
        worker.ihc_token_rows.append((current_rows, expected))

    model.register_forward_pre_hook(begin_step, with_kwargs=True)
    for layer in model.layers:
        for boundary, hc_layer in (
            (layer.ihc_attn_boundary, layer.hc_attn_layer),
            (layer.ihc_mlp_boundary, layer.hc_mlp_layer),
        ):
            (boundary or hc_layer.hc_pre).register_forward_pre_hook(check_rows)


@pytest.mark.parametrize("enforce_eager", [True, False])
def test_flash_ihc_shards_rows_and_preserves_generation(
    vllm_runner, tmp_path, monkeypatch, enforce_eager
):
    """Shard HY4-Flash iHC rows across TP ranks and preserve generation.

    Covers both attention types, the sandwich norms around the deferred
    reductions, Engram injection (layer 2), latent MoE and the dense MLP,
    odd chunked-prefill sizes around the 1024-token threshold, and prefix
    reuse. Runs with expert parallelism; non-EP validation is separate.
    """
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    config = HYV4FlashConfig(
        vocab_size=1024,
        hidden_size=768,
        intermediate_size=2048,
        moe_intermediate_size=512,
        num_hidden_layers=4,
        num_attention_heads=16,
        q_lora_rank=256,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        swa_num_attention_heads=32,
        swa_qk_nope_head_dim=128,
        swa_qk_rope_head_dim=64,
        swa_v_head_dim=128,
        sliding_window=128,
        window_size=[127, 0],
        index_n_heads=16,
        index_head_dim=128,
        index_topk=128,
        compressed_indexer_group_topk=32,
        compressed_indexer_candidate_topk=128,
        n_routed_experts=8,
        n_shared_experts=1,
        num_experts_per_tok=2,
        routed_expert_hidden_size=256,
        layer_types=[
            "deepseek_sparse_attention",
            "sliding_attention",
            "deepseek_sparse_attention",
            "sliding_attention",
        ],
        mlp_layer_types=["sparse", "sparse", "sparse", "dense"],
        engram_layer_ids=[2],
        engram_max_ngram_size=3,
        engram_pad_id=1023,
        engram_layer_head_vocab_sizes={"2": [[512] * 8, [512] * 8]},
        enable_ihc=True,
        hc_mult=4,
        sandwich_norm=True,
        learnable_sink=True,
        gated_mla=True,
        gating_type="elementwise",
        max_position_embeddings=4096,
        num_nextn_predict_layers=0,
    )
    config.architectures = ["HYV4FlashForCausalLM"]
    config.save_pretrained(tmp_path)
    prompts = [
        {"prompt_token_ids": [2 + i % 997 for i in range(n)]} for n in (3073, 17, 3)
    ]
    sampling = SamplingParams(temperature=0, max_tokens=4, ignore_eos=True, logprobs=16)
    outputs: list[list[CompletionOutput]] = []
    modes = [False, True]
    for sequence_parallel in modes:
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
            numa_bind=False,
            attention_config={"backend": "FLASHMLA_SPARSE"},
            gpu_memory_utilization=0.1,
            worker_cls=(
                "auto" if sequence_parallel else f"{__name__}.ReplicatedFlashWorker"
            ),
        ) as runner:
            runner.llm.collective_rpc(
                partial(_check_reduction_mode, sequence_parallel=sequence_parallel)
            )
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
                        (full > local) == sequence_parallel
                        for full, local in rows
                        if full >= 1024
                    )
                    for rows in observed
                ), "iHC rows were not sharded on eligible steps"
    check_logprobs_close(
        outputs_0_lst=[(list(r.token_ids), r.text, r.logprobs) for r in outputs[0]],
        outputs_1_lst=[(list(r.token_ids), r.text, r.logprobs) for r in outputs[1]],
        name_0="replicated",
        name_1="sequence-parallel",
    )
