# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from tests.utils import multi_gpu_test
from vllm import SamplingParams
from vllm.platforms import current_platform
from vllm.transformers_utils.configs.minimax_m3 import (
    MiniMaxM3Config,
    MiniMaxM3TextConfig,
)

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="ROCm NVFP4 indexer"
)


def _indexer_calls(worker):
    from vllm.models.minimax_m3.amd.ops.index_cp import host
    from vllm.models.minimax_m3.amd.ops.nvfp4_index_decode.kernel_cp_push import CNT_SC

    cp = host.get()
    return None if cp is None else cp.counts[CNT_SC].item()


@pytest.fixture
def minimax_nvfp4_config(tmp_path):
    config = MiniMaxM3TextConfig(
        vocab_size=256,
        hidden_size=256,
        intermediate_size=256,
        dense_intermediate_size=512,
        num_hidden_layers=1,
        moe_layer_freq=[0],
        sparse_attention_config={
            "use_sparse_attention": True,
            "sparse_index_dim": 128,
            "sparse_num_index_heads": 4,
            "sparse_topk_blocks": 16,
            "sparse_block_size": 128,
            "sparse_disable_index_value": [1],
            "sparse_score_type": "max",
            "sparse_init_block": 0,
            "sparse_local_block": 1,
            "sparse_attention_freq": [1],
        },
    )
    MiniMaxM3Config(
        text_config=config,
        architectures=["MiniMaxM3SparseForCausalLM"],
    ).save_pretrained(tmp_path)
    return str(tmp_path)


@pytest.mark.parametrize("num_speculative_tokens", [3, 7])
@multi_gpu_test(num_gpus=4)
def test_dbo_prefill_preserves_context_parallel_verification(
    vllm_runner, minimax_nvfp4_config, monkeypatch, num_speculative_tokens
):
    """Mixed DBO prefills fall back; subsequent verification still uses CP."""
    from vllm.platforms.rocm import on_gfx950

    if not on_gfx950():
        pytest.skip("NVFP4 indexer requires gfx950")
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    with vllm_runner(
        minimax_nvfp4_config,
        load_format="dummy",
        skip_tokenizer_init=True,
        tensor_parallel_size=4,
        max_model_len=2048,
        max_num_batched_tokens=256,
        max_num_seqs=2,
        enable_chunked_prefill=True,
        enable_dbo=True,
        dbo_prefill_token_threshold=128,
        dbo_decode_token_threshold=32768,
        block_size=128,
        kv_cache_dtype="nvfp4",
        kv_cache_memory_bytes=64 << 20,
        gpu_memory_utilization=0.02,
        attention_config={"indexer_kv_dtype": "nvfp4", "indexer_decode_cp": True},
        speculative_config={
            "method": "ngram",
            "num_speculative_tokens": num_speculative_tokens,
            "prompt_lookup_min": 1,
            "prompt_lookup_max": 3,
        },
        compilation_config={"cudagraph_capture_sizes": [4, 8]},
    ) as runner:
        before = runner.llm.collective_rpc(_indexer_calls)
        assert all(count is not None for count in before), (
            "DBO must leave the decode CP indexer available"
        )
        outputs = runner.llm.generate(
            [{"prompt_token_ids": [1, 2, 3] * n} for n in [16, 256]],
            SamplingParams(temperature=0, max_tokens=32, ignore_eos=True),
        )
        assert all(len(output.outputs[0].token_ids) == 32 for output in outputs)
        after = runner.llm.collective_rpc(_indexer_calls)
        assert all(end > start for start, end in zip(before, after))


def _sampling_is_sharded(worker):
    return worker.model_runner.batch_sharder is not None


@multi_gpu_test(num_gpus=4)
def test_batch_sharded_sampling_uses_local_logits(
    vllm_runner, minimax_nvfp4_config, monkeypatch
):
    """Requesting sharded sampling must activate it on the ROCm model."""
    from vllm.platforms.rocm import on_gfx950

    if not on_gfx950():
        pytest.skip("NVFP4 indexer requires gfx950")
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    with vllm_runner(
        minimax_nvfp4_config,
        load_format="dummy",
        skip_tokenizer_init=True,
        tensor_parallel_size=4,
        enable_batch_sharded_sampling=True,
        enable_chunked_prefill=True,
        max_model_len=1024,
        max_num_batched_tokens=256,
        max_num_seqs=4,
        block_size=128,
        kv_cache_dtype="nvfp4",
        kv_cache_memory_bytes=64 << 20,
        gpu_memory_utilization=0.02,
        attention_config={"indexer_kv_dtype": "nvfp4"},
        compilation_config={"cudagraph_capture_sizes": [4]},
    ) as runner:
        assert all(runner.llm.collective_rpc(_sampling_is_sharded)), (
            "Requested batch-sharded sampling must remain enabled"
        )
        outputs = runner.llm.generate(
            [{"prompt_token_ids": [1, 2, 3 + i] * 16} for i in range(4)],
            SamplingParams(temperature=0, max_tokens=16, ignore_eos=True),
        )
        assert all(len(output.outputs[0].token_ids) == 16 for output in outputs)
