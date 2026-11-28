# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Normal/boundary MXFP8 prefill and native-cache integration on Blackwell."""

import pytest
import torch
import torch.nn.functional as F

from tests.v1.attention.utils import BatchSpec, create_common_attn_metadata
from vllm.platforms import current_platform

SM100 = current_platform.is_cuda() and current_platform.is_device_capability_family(100)
pytestmark = pytest.mark.skipif(not SM100, reason="MXFP8 prefill requires Blackwell")


@pytest.mark.parametrize(
    "seq_lens,query_lens,expected",
    [
        ([128], [128], 128),
        ([256], [256], 256),
        ([128], [1], 0),
        ([256], [128], 0),
        ([127], [127], 0),
        ([128, 128], [128, 128], 0),
    ],
)
def test_only_position_zero_aligned_prefill_is_eligible(seq_lens, query_lens, expected):
    from vllm.v1.attention.backends.mxfp8_prefill import _full_prefill_tokens

    metadata = create_common_attn_metadata(
        BatchSpec(seq_lens=seq_lens, query_lens=query_lens), 16, torch.device("cpu")
    )
    assert _full_prefill_tokens(metadata) == expected
    assert _full_prefill_tokens(metadata, common_prefix_len=16) == 0
    assert _full_prefill_tokens(metadata.replace(causal=False)) == 0
    assert _full_prefill_tokens(metadata.replace(seq_lens_cpu_upper_bound=None)) == 0


@pytest.mark.parametrize(
    "heads,kv_heads,n",
    [(16, 16, 128), (16, 8, 1024), (32, 8, 1024), (40, 8, 1024), (32, 8, 16384)],
)
def test_normal_and_boundary_phases_preserve_causality(heads, kv_heads, n):
    from vllm.v1.attention.ops.mxfp8_prefill import boundary, bulk

    torch.manual_seed(3248)
    q, k, v = (
        torch.randn(1, n, h, 128, device="cuda", dtype=torch.bfloat16).transpose(1, 2)
        for h in (heads, kv_heads, kv_heads)
    )
    # Run the exposed phases and compare their composition with the public API.
    prepared = boundary.prepare(q, k, v, transposed_u=boundary._uses_gluon(q, k, None))
    partial, lse = bulk.execute(*prepared[:6])
    out = torch.empty_like(partial)
    boundary.merge_boundary(
        prepared, partial, lse, out, boundary._pack_heads(heads, kv_heads)
    )
    assert torch.equal(out, boundary.forward(q, k, v))
    reference = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
    relative = (out.float() - reference.float()).norm() / reference.float().norm()
    assert torch.isfinite(out).all()
    assert relative < 0.06, relative.item()
    cuts = {31, 32, 33, 128, 256, n // 2 + 7, n - 1}
    for cut in sorted(c for c in cuts if c < n):
        scaled = [x.clone() for x in (q, k, v)]
        replaced = [x.clone() for x in (q, k, v)]
        for x, y in zip(scaled, replaced):
            x[..., cut:, :] *= 1024
            y[..., cut:, :] = torch.randn_like(y[..., cut:, :])
        for changed in (scaled, replaced):
            changed_out = boundary.forward(*changed)
            assert torch.equal(out[..., :cut, :], changed_out[..., :cut, :])


def test_boundary_retains_tiny_signal_and_zero_values():
    from vllm.v1.attention.ops.mxfp8_prefill.boundary import forward

    q = torch.ones(1, 1, 128, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.zeros_like(q)
    v[..., 0, 0] = 2**-20
    actual = forward(q, q, v)
    changed = v.clone()
    changed[..., 16, 0] = 1
    assert actual[0, 0, 7, 0] == 2**-23
    assert torch.equal(actual[..., :16, :], forward(q, q, changed)[..., :16, :])
    assert forward(q, q, torch.zeros_like(v)).count_nonzero() == 0


def test_multiple_graph_captures_keep_outputs_alive():
    from vllm.v1.attention.ops.mxfp8_prefill.boundary import forward

    torch.manual_seed(19)
    q = torch.randn(1, 4, 256, 128, device="cuda", dtype=torch.bfloat16)
    k, v = torch.randn_like(q), torch.randn_like(q)
    reference = forward(q, k, v)
    graphs = []
    for _ in range(3):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = forward(q, k, v)
        graphs.append((graph, output))
    for _ in range(4):
        for graph, output in graphs:
            graph.replay()
            torch.accelerator.synchronize()
            assert torch.equal(output, reference)


def _bulk_plan_calls(model):
    from vllm.v1.attention.ops.mxfp8_prefill.bulk import plan

    stats = plan.cache_info()
    return stats.hits + stats.misses


def test_real_engine_executes_mxfp8_prefill_and_native_decode(
    vllm_runner, enable_pickle
):
    from vllm import SamplingParams

    with vllm_runner(
        "Qwen/Qwen3-4B",
        dtype="bfloat16",
        attention_backend="MXFP8_PREFILL",
        max_model_len=512,
        max_num_seqs=1,
        max_num_batched_tokens=512,
        enable_chunked_prefill=False,
        enable_prefix_caching=False,
        enforce_eager=True,
    ) as runner:
        before = runner.apply_model(_bulk_plan_calls)
        outputs = runner.llm.generate(
            [{"prompt_token_ids": [42] * 128}],
            SamplingParams(temperature=0, max_tokens=4, min_tokens=4, logprobs=1),
        )
        after = runner.apply_model(_bulk_plan_calls)
        assert all(end > start for start, end in zip(before, after))
        generated = outputs[0].outputs[0]
        assert len(generated.token_ids) == 4
        assert generated.logprobs is not None
        assert all(
            torch.isfinite(torch.tensor(entry[token].logprob))
            for token, entry in zip(generated.token_ids, generated.logprobs)
        )


def test_native_cache_contents_and_decode_are_preserved(dist_init, tmp_path):
    from transformers import LlamaConfig

    from tests.v1.attention.utils import (
        create_standard_kv_cache_spec,
        create_vllm_config,
    )
    from vllm.config import set_current_vllm_config
    from vllm.model_executor.layers.attention import Attention
    from vllm.v1.attention.backends.flash_attn import (
        FlashAttentionImpl,
        FlashAttentionMetadataBuilder,
    )
    from vllm.v1.attention.backends.mxfp8_prefill import (
        Mxfp8PrefillBackend,
        Mxfp8PrefillImpl,
        Mxfp8PrefillMetadataBuilder,
    )
    from vllm.v1.attention.ops.mxfp8_prefill.boundary import forward

    LlamaConfig(
        architectures=["LlamaForCausalLM"],
        hidden_size=512,
        head_dim=128,
        num_attention_heads=4,
        num_key_value_heads=1,
        num_hidden_layers=1,
        max_position_embeddings=256,
        vocab_size=128,
    ).save_pretrained(tmp_path)
    config = create_vllm_config(
        model_name=str(tmp_path),
        dtype="bfloat16",
        max_model_len=256,
        max_num_seqs=1,
        max_num_batched_tokens=256,
        num_gpu_blocks=16,
    )
    device = torch.device("cuda")
    with set_current_vllm_config(config), torch.device(device):
        layer = Attention(
            4,
            128,
            1.0 / (128**0.5),
            num_kv_heads=1,
            cache_config=config.cache_config,
            prefix="test.mxfp8",
            attn_backend=Mxfp8PrefillBackend,
        )
        assert isinstance(layer.impl, Mxfp8PrefillImpl)
        spec = create_standard_kv_cache_spec(config)
        custom_builder = Mxfp8PrefillMetadataBuilder(
            spec, [layer.layer_name], config, device
        )
        native_builder = FlashAttentionMetadataBuilder(
            spec, [layer.layer_name], config, device
        )
        native = FlashAttentionImpl(4, 128, 1.0 / (128**0.5), 1, None, None, "auto")
        q = torch.randn(128, 4, 128, dtype=torch.bfloat16)
        k = torch.randn(128, 1, 128, dtype=torch.bfloat16)
        v = torch.randn_like(k)
        custom_cache = torch.zeros(16, 1, 16, 256, dtype=torch.bfloat16)
        native_cache = torch.zeros_like(custom_cache)
        common = create_common_attn_metadata(
            BatchSpec(seq_lens=[128], query_lens=[128]),
            16,
            device,
            arange_block_indices=True,
        )
        actual = torch.full((130, 4, 128), -7, dtype=torch.bfloat16)
        for impl, cache in ((layer.impl, custom_cache), (native, native_cache)):
            impl.do_kv_cache_update(layer, k, v, cache, common.slot_mapping)
        layer.impl.forward(
            layer, q, k, v, custom_cache, custom_builder.build(0, common), actual
        )
        expected = forward(*(x.transpose(0, 1).unsqueeze(0) for x in (q, k, v)))
        assert torch.equal(actual[:128], expected.squeeze(0).transpose(0, 1))
        assert torch.all(actual[128:] == -7)
        assert torch.equal(custom_cache, native_cache)
        cached_k, cached_v = custom_cache.transpose(1, 2).split(128, dim=-1)
        assert torch.equal(cached_k.reshape(-1, 1, 128)[:128], k)
        assert torch.equal(cached_v.reshape(-1, 1, 128)[:128], v)
        decode_common = create_common_attn_metadata(
            BatchSpec(seq_lens=[129], query_lens=[1]),
            16,
            device,
            arange_block_indices=True,
        ).replace(slot_mapping=torch.tensor([128], dtype=torch.int64))
        q1, k1, v1 = q[:1].clone(), k[:1].clone(), v[:1].clone()
        custom_out, native_out = torch.empty_like(q1), torch.empty_like(q1)
        for impl, cache in ((layer.impl, custom_cache), (native, native_cache)):
            impl.do_kv_cache_update(layer, k1, v1, cache, decode_common.slot_mapping)
        layer.impl.forward(
            layer,
            q1,
            k1,
            v1,
            custom_cache,
            custom_builder.build(0, decode_common),
            custom_out,
        )
        native.forward(
            layer,
            q1,
            k1,
            v1,
            native_cache,
            native_builder.build(0, decode_common),
            native_out,
        )
        assert torch.equal(custom_out, native_out)
        assert torch.equal(custom_cache, native_cache)
