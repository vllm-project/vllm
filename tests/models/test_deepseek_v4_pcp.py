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


def _generate(model_path, output_path, pcp, tp, ep, mode):
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model_path,
        load_format="dummy",
        skip_tokenizer_init=True,
        prefill_context_parallel_size=pcp,
        tensor_parallel_size=tp,
        enable_expert_parallel=ep,
        seed=1234,
        compilation_config={
            "cudagraph_mode": mode,
            "cudagraph_capture_sizes": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
        },
        enable_prefix_caching=False,
        max_model_len=4096,
        max_num_batched_tokens=512,
        max_num_seqs=8,
        gpu_memory_utilization=0.2,
        kv_cache_dtype="fp8_ds_mla",
        attention_config={"backend": "FLASHMLA_SPARSE"},
        kernel_config={"moe_backend": "triton"},
    )
    assert str(llm.llm_engine.vllm_config.compilation_config.cudagraph_mode) == mode
    # Very short prefills may have no compressed chunks. Longer requests cross
    # C4/C128 boundaries, padded graph buckets and the chunked-prefill budget.
    lengths = [
        1,
        2,
        3,
        4,
        5,
        7,
        8,
        9,
        127,
        128,
        129,
        255,
        256,
        257,
        511,
        512,
        513,
        2305,
    ]
    prompts = [{"prompt_token_ids": [10 + i % 100 for i in range(n)]} for n in lengths]
    results = llm.generate(
        prompts,
        SamplingParams(temperature=0, max_tokens=16, ignore_eos=True, logprobs=5),
    )
    records = []
    for prompt, result in zip(prompts, results):
        assert result.prompt_token_ids == prompt["prompt_token_ids"]
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


@pytest.mark.distributed(num_gpus=4)
@pytest.mark.parametrize(
    "pcp,tp,ep,mode",
    [
        (1, 1, False, "FULL_AND_PIECEWISE"),
        (2, 1, False, "FULL_AND_PIECEWISE"),
        (2, 1, True, "FULL_AND_PIECEWISE"),
        (2, 2, True, "FULL_AND_PIECEWISE"),
        (2, 1, True, "NONE"),
    ],
)
def test_pcp_generation_across_compression_boundaries(tmp_path, pcp, tp, ep, mode):
    """Three layers and eight dummy experts exercise actual capture and replay."""
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps(_small_config()))
    output = tmp_path / "outputs.json"
    log = tmp_path / "engine.log"
    env = dict(
        os.environ,
        VLLM_USE_V2_MODEL_RUNNER="1",
        VLLM_WORKER_MULTIPROC_METHOD="spawn",
        VLLM_DEEP_GEMM_WARMUP="skip",
    )
    with log.open("w") as stream:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "tests.models." + Path(__file__).stem,
                str(model),
                str(output),
                str(pcp),
                str(tp),
                str(int(ep)),
                mode,
            ],
            env=env,
            cwd=Path(__file__).resolve().parents[2],
            stdout=stream,
            stderr=subprocess.STDOUT,
            timeout=900,
        )
    assert result.returncode == 0, log.read_text()
    records = json.loads(output.read_text())
    assert len(records) == 18
    for completion in records:
        assert len(completion["tokens"]) == len(completion["logprobs"]) == 16
        assert all(0 <= token < 1024 for token in completion["tokens"])
        for token, step in zip(completion["tokens"], completion["logprobs"]):
            assert str(token) in step
            assert all(math.isfinite(value) for value in step.values())


def _compressed_slots_worker(rank, world_size, port):
    from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config

    config = VllmConfig(
        parallel_config=ParallelConfig(prefill_context_parallel_size=world_size)
    )
    with set_current_vllm_config(config):
        _check_compressed_slots(rank, world_size, port)


def _check_compressed_slots(rank, world_size, port):
    import torch

    from vllm.distributed import (
        cleanup_dist_env_and_memory,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.v1.attention.backend import CommonAttentionMetadata
    from vllm.v1.attention.backends.mla.compressor_utils import (
        get_compressed_slot_mapping,
    )

    torch.accelerator.set_device_index(rank)
    init_distributed_environment(world_size, rank, f"tcp://127.0.0.1:{port}", rank)
    initialize_model_parallel(prefill_context_model_parallel_size=world_size)
    try:
        query_start = torch.tensor([0, 4], dtype=torch.int32)
        metadata = CommonAttentionMetadata(
            query_start_loc=query_start.cuda(),
            query_start_loc_cpu=query_start,
            seq_lens=torch.tensor([4 * (rank + 1)], dtype=torch.int32, device="cuda"),
            num_reqs=1,
            num_actual_tokens=4,
            max_query_len=4,
            max_seq_len=4 * world_size,
            block_table_tensor=torch.zeros((1, 1), dtype=torch.int32, device="cuda"),
            slot_mapping=torch.arange(4 * world_size, dtype=torch.int64, device="cuda"),
        )
        metadata.slot_mapping[-2:] = -1
        local_buffer = torch.empty(4, dtype=torch.int64, device="cuda")
        gathered = torch.empty(4 * world_size, dtype=torch.int64, device="cuda")

        def build():
            return get_compressed_slot_mapping(
                metadata.num_actual_tokens,
                metadata.slot_mapping,
                metadata.query_start_loc,
                metadata.seq_lens,
                metadata.block_table_tensor.clamp_(min=0),
                32,
                2,
                out=local_buffer,
                pcp_buffer=gathered if world_size > 1 else None,
            )

        slots = build()
        expected = torch.arange(4 * world_size, device="cuda", dtype=torch.int64) // 2
        expected[::2] = -1
        expected[-2:] = -1
        torch.testing.assert_close(slots, expected)
        address = slots.data_ptr()
        replayed = torch.empty_like(slots)
        torch.accelerator.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            replayed.copy_(slots)
        # Build the next batch outside the graph. A newly allocated gather
        # result would leave graph replay reading the previous batch's slots.
        metadata.block_table_tensor.fill_(1)
        next_slots = build()
        assert next_slots.data_ptr() == address
        expected[expected >= 0] += 32
        graph.replay()
        torch.testing.assert_close(replayed, expected)
        # A rank with no cache-writing rows still participates in the gather.
        metadata.slot_mapping[:4] = -1
        if rank == 0 and world_size > 1:
            metadata.query_start_loc.zero_()
            metadata.query_start_loc_cpu.zero_()
            metadata.num_actual_tokens = 0
            metadata.max_query_len = 0
        next_slots = build()
        expected[:4] = -1
        graph.replay()
        torch.testing.assert_close(next_slots, expected)
        torch.testing.assert_close(replayed, expected)
    finally:
        cleanup_dist_env_and_memory()


@pytest.mark.distributed(num_gpus=2)
@pytest.mark.parametrize("world_size", [1, 2])
def test_compressed_slots_preserve_padding_and_graph_storage(world_size):
    import torch.multiprocessing as mp

    from vllm.utils.network_utils import get_open_port

    mp.spawn(
        _compressed_slots_worker,
        args=(world_size, get_open_port()),
        nprocs=world_size,
        join=True,
    )


if __name__ == "__main__":
    _generate(
        sys.argv[1],
        sys.argv[2],
        int(sys.argv[3]),
        int(sys.argv[4]),
        bool(int(sys.argv[5])),
        sys.argv[6],
    )
