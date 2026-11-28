# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end coverage for the MiniMax-M3 FlashInfer MSA indexer on SM90/SM120.

Starts a tiny dummy-weight MiniMax-M3 through `vllm_runner` and checks that
every sparse layer picks the FlashInfer indexer, so its metadata builder is
constructed at engine startup (the path that broke in #67911). On SM90 the
FlashInfer MSA ops need fp8 main and index caches and a GQA group of 8 or 16
(`minimax_m3_use_flashinfer_msa`), so this config uses 8 query heads per KV
head and fp8 caches; with a bf16 index cache the H100 falls back to Triton.
"""

import math

import pytest
from transformers import MiniMaxM3VLConfig

from vllm import SamplingParams
from vllm.platforms import current_platform


def _has_flashinfer_msa_platform() -> bool:
    if not current_platform.is_cuda():
        return False
    from vllm.utils.flashinfer import has_flashinfer_msa, has_flashinfer_msa_sm90

    if current_platform.is_device_capability(90):
        return has_flashinfer_msa_sm90()
    return current_platform.is_device_capability_family(120) and has_flashinfer_msa()


pytestmark = pytest.mark.skipif(
    not _has_flashinfer_msa_platform(),
    reason="The FlashInfer MSA indexer runs on SM90 or SM120/SM121 only",
)

NUM_SPARSE_LAYERS = 2


@pytest.fixture
def flashinfer_indexer_engine(tmp_path, vllm_runner, enable_pickle):
    MiniMaxM3VLConfig(
        architectures=["MiniMaxM3SparseForCausalLM"],
        text_config={
            "num_hidden_layers": 3 + NUM_SPARSE_LAYERS,
            "hidden_size": 128,
            "dense_intermediate_size": 256,
            # SM90 FlashInfer MSA needs a GQA group of 8 or 16.
            "num_attention_heads": 8,
            "num_key_value_heads": 1,
            "head_dim": 128,
            "vocab_size": 256,
            "bos_token_id": 1,
            "eos_token_id": 2,
            "max_position_embeddings": 4096,
            "mlp_layer_types": ["dense"] * (3 + NUM_SPARSE_LAYERS),
            "layer_types": ["full_attention"] * 3
            + ["minimax_m3_sparse"] * NUM_SPARSE_LAYERS,
            "index_n_heads": 1,
            "index_head_dim": 128,
            "index_block_size": 128,
            "index_topk_blocks": 16,
            "index_local_blocks": 1,
        },
    ).save_pretrained(tmp_path)
    with vllm_runner(
        str(tmp_path),
        load_format="dummy",
        skip_tokenizer_init=True,
        dtype="bfloat16",
        max_model_len=4096,
        block_size=128,
        max_num_seqs=1,
        max_num_batched_tokens=4096,
        kv_cache_memory_bytes=64 * 1024 * 1024,
        enforce_eager=True,
        enable_prefix_caching=False,
        # SM90 FlashInfer MSA reads fp8 E4M3 main and index caches only.
        kv_cache_dtype="fp8",
        attention_config={"indexer_kv_dtype": "fp8"},
        generation_config="vllm",
    ) as runner:
        yield runner


def indexer_impls(worker):
    context = worker.vllm_config.compilation_config.static_forward_context
    return [
        [type(layer.indexer.impl).__name__, layer.indexer.impl.local_blocks]
        for _, layer in sorted(context.items())
        if hasattr(layer, "indexer")
    ]


def test_flashinfer_indexer_engine(flashinfer_indexer_engine):
    runner = flashinfer_indexer_engine
    # 2304 tokens span 18 blocks of 128, more than the 16 top-k blocks, so the
    # indexer has to select rather than keep every block.
    for offset in (0, 37):
        tokens = [3 + (i + offset) % 250 for i in range(2304)]
        result = runner.llm.generate(
            [{"prompt_token_ids": tokens}],
            SamplingParams(temperature=0, max_tokens=4, ignore_eos=True, logprobs=5),
            use_tqdm=False,
        )[0]
        assert result.prompt_token_ids == tokens
        output = result.outputs[0]
        assert len(output.token_ids) == 4
        assert all(0 <= token < 256 for token in output.token_ids)
        assert output.logprobs is not None and len(output.logprobs) == 4
        values = [p.logprob for row in output.logprobs for p in row.values()]
        assert all(math.isfinite(value) for value in values)
        assert len(set(values)) > 1

    (impls,) = runner.collective_rpc(indexer_impls)
    assert impls == [["MiniMaxM3IndexerFlashInferImpl", 1]] * NUM_SPARSE_LAYERS
