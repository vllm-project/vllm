# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniMax-M3 index-cache configuration and layer registration."""

import pytest
import regex as re
import torch

from vllm.config import (
    AttentionConfig,
    CacheConfig,
    ModelConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.models.minimax_m3.amd.indexer_msa import MiniMaxM3IndexerMSAImpl
from vllm.models.minimax_m3.amd.model import (
    MiniMaxM3SparseAttention,
    _should_skip_index_topk,
)
from vllm.platforms import current_platform
from vllm.transformers_utils.configs.minimax_m3 import MiniMaxM3Config
from vllm.utils.torch_utils import set_default_torch_dtype

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="MiniMax-M3 AMD model"
)


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    path = tmp_path_factory.mktemp("minimax_m3")
    MiniMaxM3Config(
        architectures=["MiniMaxM3SparseForConditionalGeneration"]
    ).save_pretrained(path)
    return str(path)


@pytest.mark.parametrize(
    ("hf_overrides", "expected_error"),
    [
        (
            {
                "use_index_cache": True,
                "index_topk_freq": 2,
                "index_skip_topk_offset": 1,
            },
            (
                "top-level use_index_cache, index_topk_freq, index_skip_topk_offset "
                'would be ignored. Pass them as --hf-overrides \'{"text_config": '
                '{"use_index_cache": true, "index_topk_freq": 2, '
                '"index_skip_topk_offset": 1}}\'.'
            ),
        ),
        (
            {"use_index_cache": True, "text_config": {"index_topk_freq": 2}},
            (
                "top-level use_index_cache would be ignored. Pass them as "
                '--hf-overrides \'{"text_config": {"use_index_cache": true}}\'.'
            ),
        ),
    ],
    ids=["top_level", "mixed"],
)
def test_top_level_overrides_are_rejected(checkpoint, hf_overrides, expected_error):
    with pytest.raises(ValueError, match=re.escape(expected_error)):
        ModelConfig(checkpoint, hf_overrides=hf_overrides)


def test_text_config_overrides_take_effect(checkpoint):
    text_config = ModelConfig(
        checkpoint,
        hf_overrides={"text_config": {"use_index_cache": True, "index_topk_freq": 4}},
    ).hf_text_config

    # Sparse layers 3..59; every 4th one recomputes the top-k.
    assert [i for i in range(60) if _should_skip_index_topk(text_config, i)] == sorted(
        set(range(3, 60)) - set(range(3, 60, 4))
    )


@pytest.mark.parametrize(
    ("sharing", "frequency", "offset", "layer_id", "skip", "indexer_dtype"),
    [
        (*case, indexer_dtype)
        for case in [
            (False, 4, 0, 4, False),
            (True, 1, 0, 4, False),
            (True, 4, 0, 4, True),
            (True, 4, 1, 4, False),
            (True, 4, 1, 5, True),
        ]
        for indexer_dtype in ("bf16", "fp8")
    ]
    + [pytest.param(True, 4, 0, 4, True, "nvfp4", id="nvfp4-sharing")],
)
def test_index_cache_registration(
    checkpoint, dist_init, indexer_dtype, sharing, frequency, offset, layer_id, skip
):
    model_config = ModelConfig(
        checkpoint,
        dtype="bfloat16",
        max_model_len=128,
        hf_overrides={
            "text_config": {
                "hidden_size": 128,
                "num_attention_heads": 16 if indexer_dtype == "nvfp4" else 1,
                "num_key_value_heads": 1,
                "head_dim": 128,
                "max_position_embeddings": 128,
                "use_index_cache": sharing,
                "index_topk_freq": frequency,
                "index_skip_topk_offset": offset,
            }
        },
    )
    config = model_config.hf_text_config
    config.sparse_attention_config["sparse_num_index_heads"] = 1
    topk = torch.empty((1, 2, 16), dtype=torch.int32, device="cuda")
    table = torch.empty((2, 128), dtype=torch.int32, device="cuda")
    lengths = torch.empty(2, dtype=torch.int32, device="cuda")

    def build(use_sharing):
        config.use_index_cache = use_sharing
        cache_config = CacheConfig(
            block_size=128,
            cache_dtype="nvfp4" if indexer_dtype == "nvfp4" else "auto",
        )
        vllm_config = VllmConfig(
            model_config=model_config,
            cache_config=cache_config,
            attention_config=AttentionConfig(indexer_kv_dtype=indexer_dtype),
        )
        with (
            set_current_vllm_config(vllm_config),
            set_default_torch_dtype(torch.bfloat16),
            torch.device("cuda"),
        ):
            layers = [
                MiniMaxM3SparseAttention(
                    config,
                    layer_id=i,
                    prefix=f"model.layers.{i}.self_attn",
                    cache_config=cache_config,
                    topk_indices_buffer=topk,
                    msa_indexer_impl_cls=(
                        MiniMaxM3IndexerMSAImpl if indexer_dtype == "fp8" else None
                    ),
                    sparse_bt_buffer=table,
                    sparse_ctx_buffer=lengths,
                )
                for i in (3, layer_id)
            ]
        return layers, vllm_config.compilation_config.static_forward_context

    baseline, _ = build(False)
    layers, registered = build(sharing)
    expected = {layer.layer_name for layer in layers}
    expected.add(f"{layers[0].layer_name}.index_cache")
    has_indexer = not skip or indexer_dtype == "nvfp4"
    if has_indexer:
        expected.add(f"{layers[1].layer_name}.index_cache")
    assert set(registered) == expected
    assert layers[0].indexer is not None
    assert layers[1].skip_index_topk == skip
    assert (layers[1].indexer is not None) == has_indexer
    for original, layer in zip(baseline, layers):
        if layer.indexer is not None:
            assert layer.indexer.index_cache.indexer_kv_dtype == indexer_dtype
        assert layer.topk_indices_buffer is topk
        assert layer.sparse_bt_buffer is table
        assert layer.sparse_ctx_buffer is lengths
        layer.load_state_dict(original.state_dict(), strict=True)


@pytest.fixture(params=[False, True], ids=["default", "sharing"])
def index_cache_engine(request, tmp_path, vllm_runner, enable_pickle):
    sharing = request.param
    sparse = [0, 0, 0, 1, 1]
    MiniMaxM3Config(
        architectures=["MiniMaxM3SparseForCausalLM"],
        text_config={
            "num_hidden_layers": 5,
            "hidden_size": 128,
            "dense_intermediate_size": 256,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "head_dim": 128,
            "vocab_size": 256,
            "max_position_embeddings": 4096,
            "moe_layer_freq": [0] * 5,
            "sparse_attention_config": {
                "use_sparse_attention": True,
                "sparse_index_dim": 128,
                "sparse_num_index_heads": 1,
                "sparse_topk_blocks": 16,
                "sparse_block_size": 128,
                "sparse_disable_index_value": sparse,
                "sparse_score_type": "max",
                "sparse_init_block": 0,
                "sparse_local_block": 1,
                "sparse_attention_freq": sparse,
            },
        },
    ).save_pretrained(tmp_path)
    overrides = {"use_index_cache": True, "index_topk_freq": 4} if sharing else {}
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
        attention_config={"indexer_kv_dtype": "bf16"},
        hf_overrides={"text_config": overrides},
        generation_config="vllm",
    ) as runner:
        yield runner, sharing


def index_cache_registration(worker):
    context = worker.vllm_config.compilation_config.static_forward_context
    return sorted(name for name in context if name.endswith(".index_cache"))


def generate_index_cache_requests(runner):
    import math

    from vllm import SamplingParams

    outputs = []
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
        outputs.append(result)
    return outputs


def test_index_cache_engine(index_cache_engine):
    runner, sharing = index_cache_engine
    generate_index_cache_requests(runner)
    (registered,) = runner.collective_rpc(index_cache_registration)
    expected = ["model.layers.3.self_attn.attn.index_cache"]
    if not sharing:
        expected.append("model.layers.4.self_attn.attn.index_cache")
    assert registered == expected
