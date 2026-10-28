# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.models.qwen3_dflash2 import _grouped_conv, _score_edges
from vllm.platforms import current_platform
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator
from vllm.v1.worker.gpu.spec_decode.dflash2.speculator import DFlash2Speculator


@pytest.mark.parametrize("block_size", [5, 8])
def test_grouped_conv_matches_reference(block_size: int):
    torch.manual_seed(0)
    batch, taps, num_groups, group_size = 3, 3, 4, 2
    hidden = torch.randn(batch * block_size, num_groups * group_size)
    delta = torch.randn(batch * block_size, taps, num_groups)
    base = torch.randn(taps, num_groups * group_size)

    actual = _grouped_conv(
        hidden, delta, base, block_size, num_groups, group_size, taps
    )
    hidden_blocks = hidden.view(batch, block_size, num_groups, group_size)
    expected = torch.zeros_like(hidden_blocks)
    base = base.view(taps, num_groups, group_size)
    delta = delta.view(batch, block_size, taps, num_groups)
    for position in range(block_size):
        for tap in range(min(taps, position + 1)):
            expected[:, position] += (
                base[tap] + delta[:, position, tap, :, None]
            ) * hidden_blocks[:, position - tap]

    torch.testing.assert_close(actual, expected.flatten(0, 1).flatten(-2))


@pytest.mark.skipif(not current_platform.is_cuda(), reason="This test requires CUDA")
@pytest.mark.parametrize(
    "rows,width,query,group,taps",
    [
        (8, 5120, 8, 16, 2),
        (64, 5120, 8, 16, 2),
        (256, 5120, 8, 16, 2),
        (15, 143, 5, 11, 3),
    ],
)
@pytest.mark.parametrize("warps", [None, 4, 8, 16])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_beam_conv_residual_norm(rows, width, query, group, taps, warps, dtype):
    """Dtype rounding and request boundaries survive residual/norm fusion."""
    from vllm import _custom_ops as ops
    from vllm.models.beam.nvidia.dflash2_conv_ops import conv_residual_norm

    torch.manual_seed(42)
    hidden = torch.randn(rows, width, device="cuda", dtype=dtype)
    residual = torch.randn_like(hidden)
    weight = torch.randn(width, device="cuda", dtype=hidden.dtype)
    base = torch.randn(taps, width, device="cuda", dtype=hidden.dtype)
    delta = torch.randn(
        rows, 2, taps, width // group, device="cuda", dtype=hidden.dtype
    )[:, 1]
    expected = _grouped_conv(hidden, delta, base, query, width // group, group, taps)
    expected_residual = residual.clone()
    ops.fused_add_rms_norm(expected, expected_residual, weight, 1e-6)
    actual, actual_residual = conv_residual_norm(
        hidden, delta, base, residual, weight, 1e-6, query, group, num_warps=warps
    )
    tolerance = {torch.bfloat16: 0.015625, torch.float16: 0.002, torch.float32: 3e-6}[
        dtype
    ]
    residual_tolerance = 0 if dtype == torch.bfloat16 else tolerance
    torch.testing.assert_close(
        actual_residual,
        expected_residual,
        rtol=residual_tolerance,
        atol=residual_tolerance,
    )
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="This test requires CUDA")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    "batch,num_groups,group_size,block_size,taps",
    [
        (7, 13, 11, 5, 1),
        (7, 13, 11, 5, 3),
        (7, 13, 11, 8, 2),
        (16, 64, 16, 8, 2),
        (16, 160, 16, 8, 2),
    ],
)
def test_grouped_conv_triton_matches_reference(
    dtype: torch.dtype,
    batch: int,
    num_groups: int,
    group_size: int,
    block_size: int,
    taps: int,
):
    torch.manual_seed(0)
    rows = batch * block_size
    hidden = torch.randn(rows, num_groups * group_size, device="cuda", dtype=dtype)
    base = torch.randn(taps, num_groups * group_size, device="cuda", dtype=dtype)
    projected = torch.randn(rows, 2, taps, num_groups, device="cuda", dtype=dtype)
    delta = projected[:, 1]

    actual = _grouped_conv(
        hidden, delta, base, block_size, num_groups, group_size, taps
    )

    hidden_blocks = hidden.float().view(batch, block_size, num_groups, group_size)
    expected = torch.zeros_like(hidden_blocks)
    base_blocks = base.float().view(taps, num_groups, group_size)
    delta_blocks = delta.float().view(batch, block_size, taps, num_groups)
    for position in range(block_size):
        for tap in range(min(taps, position + 1)):
            expected[:, position] += (
                base_blocks[tap] + delta_blocks[:, position, tap, :, None]
            ) * hidden_blocks[:, position - tap]

    torch.testing.assert_close(
        actual,
        expected.flatten(0, 1).flatten(-2).to(dtype),
        rtol=1e-2 if dtype is torch.bfloat16 else 1e-5,
        atol=1e-2 if dtype is torch.bfloat16 else 1e-5,
    )


@pytest.mark.skipif(not current_platform.is_cuda(), reason="This test requires CUDA")
@pytest.mark.parametrize("rows,heads", [(8, 1), (64, 2), (256, 1)])
def test_beam_fused_context_kv(rows, heads):
    """All layer pointers and strided cache pages match the staged context path."""
    from vllm import _custom_ops as ops
    from vllm.models.beam.nvidia.dflash2_context_ops import store_context_kv

    torch.manual_seed(42)
    layers, dim, page = 5, 128, 16
    kv = torch.randn(rows, layers, 2, heads, dim, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(layers, dim, device="cuda", dtype=kv.dtype)
    positions = torch.arange(rows, device="cuda", dtype=torch.int64)
    angles = torch.randn(rows, dim // 2, device="cuda")
    rope = torch.cat((angles.cos(), angles.sin()), -1).to(kv.dtype)
    staged = kv.permute(2, 1, 0, 3, 4).contiguous()
    keys = torch.empty_like(staged[0])
    ops.rms_norm(keys, staged[0], weight, 1e-6)
    ops.rotary_embedding(
        positions.repeat(layers), keys.view(layers * rows, -1), None, dim, rope, True
    )
    values = staged[1]
    storage = torch.randn(
        (rows + page - 1) // page,
        layers,
        heads,
        page,
        2 * dim,
        device="cuda",
        dtype=kv.dtype,
    )
    expected = storage.clone()
    slots = positions.flip(0)[:-1].contiguous()
    slots[::3] = -1
    valid = slots >= 0
    ids = slots[valid]
    for layer in range(layers):
        expected[ids // page, layer, :, ids % page, :dim] = keys[layer, :-1][valid]
        expected[ids // page, layer, :, ids % page, dim:] = values[layer, :-1][valid]
    caches = storage.unbind(1)
    pointers = torch.tensor(
        [c.data_ptr() for c in caches], device="cuda", dtype=torch.int64
    )
    store_context_kv(kv, weight, positions, rope, pointers, slots, caches[0], 1e-6)
    torch.testing.assert_close(storage, expected, rtol=0.01, atol=0.015625)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="This test requires CUDA")
@pytest.mark.parametrize("rows", [1, 8, 65, 2048])
@pytest.mark.parametrize("cache_dtype", ["fp8_e4m3", "fp8_e5m2"])
def test_beam_fp8_context_cache_matches_backend_and_replays(rows, cache_dtype):
    """Quantize rounded context K/V with each layer's live scales and slot mask."""
    from vllm import _custom_ops as ops
    from vllm.models.beam.nvidia.dflash2_context_ops import store_context_kv

    torch.manual_seed(42)
    layers, heads, dim, page = 3, 2, 128, 16
    kv = torch.randn(rows, layers, 2, heads, dim, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(layers, dim, device="cuda", dtype=kv.dtype)
    positions = torch.arange(rows, device="cuda")
    angles = torch.randn(rows, dim // 2, device="cuda")
    rope = torch.cat((angles.cos(), angles.sin()), -1).to(kv.dtype)
    shape = ((rows + page - 1) // page, layers, heads, page, 2 * dim)
    bf16 = torch.zeros(shape, device="cuda", dtype=kv.dtype)
    actual = torch.zeros(shape, device="cuda", dtype=torch.uint8)
    expected = torch.zeros_like(actual)
    dtype = torch.float8_e4m3fn if cache_dtype == "fp8_e4m3" else torch.float8_e5m2
    scales = torch.tensor([[0.01, 0.02], [0.5, 1.5], [2.0, 3.0]], device="cuda")
    scale_ptrs = torch.tensor(
        [s.data_ptr() for pair in scales.unbind() for s in pair.unbind()],
        device="cuda",
    )

    def pointers(storage):
        return torch.tensor([c.data_ptr() for c in storage.unbind(1)], device="cuda")

    slots = positions.flip(0).contiguous()
    slots[1::3] = -1
    args = (kv, weight, positions, rope)
    bf16_ptrs, fp8_ptrs = pointers(bf16), pointers(actual)

    def run():
        store_context_kv(
            *args,
            fp8_ptrs,
            slots,
            actual[:, 0].view(dtype),
            1e-6,
            scale_ptrs,
        )

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for factor in (1.0, 0.5):
        scales.mul_(factor)
        kv.mul_(factor)
        store_context_kv(*args, bf16_ptrs, positions, bf16[:, 0], 1e-6)
        for layer in range(layers):
            source = bf16[:, layer].transpose(1, 2).reshape(-1, heads, 2 * dim)
            cache = expected[:, layer].transpose(1, 2)
            ops.reshape_and_cache_flash(
                source[:rows, :, :dim].contiguous(),
                source[:rows, :, dim:].contiguous(),
                cache[..., :dim],
                cache[..., dim:],
                slots,
                cache_dtype,
                scales[layer, 0],
                scales[layer, 1],
            )
        graph.replay()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_selector_edges_match_sequential_reference():
    torch.manual_seed(1)
    batch, steps, top_k, rank = 2, 4, 3, 5
    vocab = 17
    predecessors = torch.randn(vocab, rank)
    successors = torch.randn(vocab, rank)
    candidate_ids = torch.randint(vocab, (batch, steps, top_k))
    unary = torch.randn(batch, steps, top_k)
    hidden = torch.randn(batch, steps, rank)
    anchors = torch.randint(vocab, (batch,))

    actual = _score_edges(
        predecessors,
        successors,
        candidate_ids,
        unary,
        hidden,
        anchors,
        top_k,
    )
    expected = torch.empty_like(actual)
    for step in range(steps):
        pred = (
            anchors[:, None].expand(-1, top_k)
            if step == 0
            else candidate_ids[:, step - 1]
        )
        expected[:, step] = unary[:, step, None] + torch.einsum(
            "bpr,bcr->bpc",
            predecessors[pred] * hidden[:, step, None],
            successors[candidate_ids[:, step]],
        )

    torch.testing.assert_close(actual, expected)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="This test requires CUDA")
@pytest.mark.parametrize(
    "batch,steps,top_k,rank",
    [(1, 1, 3, 5), (1, 7, 16, 256), (8, 7, 16, 256), (32, 7, 16, 256), (3, 5, 17, 65)],
)
@pytest.mark.parametrize("unary_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("anchor_dtype", [torch.int32, torch.int64])
def test_beam_fused_candidate_edges(
    batch, steps, top_k, rank, unary_dtype, anchor_dtype
):
    """Keep the anchor, predecessor mapping, and BF16 intermediate rounding."""
    from vllm.models.beam.nvidia.dflash2_ops import candidate_edges

    torch.manual_seed(1)
    vocab = 257
    predecessors = torch.randn(vocab, rank, device="cuda", dtype=torch.bfloat16)
    successors = torch.randn_like(predecessors)
    candidates = torch.randint(vocab, (batch, steps, top_k), device="cuda")
    unary = torch.randn(batch, steps, top_k, device="cuda", dtype=unary_dtype)
    hidden = torch.randn(batch, steps, rank, device="cuda", dtype=torch.bfloat16)
    # Production anchors are a strided view of the query token buffer.
    anchors = torch.randint(vocab, (batch * 8,), device="cuda", dtype=anchor_dtype)[::8]
    args = (predecessors, successors, candidates, unary, hidden, anchors, top_k)
    expected = _score_edges(*args)
    actual = candidate_edges(*args)
    torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.015625)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="This test requires CUDA")
@pytest.mark.parametrize("rows", [8, 64, 256])
def test_beam_draft_qk_norm_rope_cache(rows):
    """Fused query preparation preserves BF16 rounding and ignores padded slots."""
    from vllm import _custom_ops as ops
    from vllm.models.beam.nvidia.attention_ops import beam_qk_norm_rope

    torch.manual_seed(42)
    qkv = torch.randn(rows, 896, device="cuda", dtype=torch.bfloat16)
    q, k, v = qkv.split((640, 128, 128), -1)
    qw, kw = torch.randn(2, 128, device="cuda", dtype=q.dtype)
    positions = torch.arange(rows, device="cuda", dtype=torch.int64)
    angles = torch.randn(rows, 64, device="cuda")
    rope = torch.cat((angles.cos(), angles.sin()), -1).to(q.dtype)
    oq, ok = torch.empty_like(q), torch.empty_like(k)
    ops.rms_norm(oq.view(-1, 5, 128), q.view(-1, 5, 128), qw, 1e-6)
    ops.rms_norm(ok.view(-1, 1, 128), k.view(-1, 1, 128), kw, 1e-6)
    ops.rotary_embedding(positions, oq, ok, 128, rope, True)
    cache = torch.randn((rows + 15) // 16, 1, 16, 256, device="cuda", dtype=q.dtype)
    expected_cache = cache.clone()
    slots = positions.flip(0).contiguous()
    slots[::3] = -1
    valid = slots >= 0
    slot_ids = slots[valid]
    expected_cache[slot_ids // 16, 0, slot_ids % 16, :128] = ok[valid]
    expected_cache[slot_ids // 16, 0, slot_ids % 16, 128:] = v[valid]
    actual = beam_qk_norm_rope(
        q,
        k,
        qw,
        kw,
        1e-6,
        positions,
        rope,
        128,
        128,
        value=v,
        kv_cache=cache,
        slot_mapping=slots,
    )
    torch.testing.assert_close(actual, (oq, ok), rtol=0.01, atol=0.015625)
    torch.testing.assert_close(cache, expected_cache, rtol=0.01, atol=0.015625)


def _stub_base(monkeypatch, draft_logits):
    """A DFlashSpeculator.__init__ that allocates only what the base class would.

    The real base class fills draft_logits from draft_logits_spec, so callers
    pass a tensor already in that state.
    """

    def init_base(self, _vllm_config, device):
        self.draft_model_config = SimpleNamespace(
            hf_config=SimpleNamespace(dflash_config={"selector_top_k": 3})
        )
        self.max_num_reqs = 2
        self.num_query_per_req = 5
        self.num_speculative_steps = 4
        self.vocab_size = 17
        self.draft_tokens = torch.empty((2, 4), dtype=torch.int64, device=device)
        self.draft_logits = draft_logits

    monkeypatch.setattr(DFlashSpeculator, "__init__", init_base)


def test_selector_leaves_greedy_drafting_without_proposal_logits(monkeypatch):
    """Greedy is the default, and it caches no proposal distribution.

    The base class allocates draft_logits only for "probabilistic"; verification
    reads `draft_logits is None` to decide whether a distribution is on offer, so
    allocating one here would claim a proposal the walk never sampled from.
    """
    _stub_base(monkeypatch, None)
    speculator = DFlash2Speculator(None, torch.device("cpu"))

    assert speculator.draft_logits is None


def test_selector_asks_for_fp32_proposal_logits():
    """The spec the base class allocates from: fp32, filled -inf.

    Not the head dtype -- rounding selector scores to bf16 moves the argmax of a
    candidate row often enough that the walk and the rejection sampler checking it
    would no longer read the same distribution.
    """
    dtype, fill = DFlash2Speculator.draft_logits_spec(None, None)

    assert dtype is torch.float32
    assert fill == float("-inf")


def _mock_draft_configs(monkeypatch, hf_config, dtype=torch.float32):
    """The current and draft vLLM configs for building a draft model on CPU."""
    mock_current_vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(
            block_size=16,
            user_specified_block_size=False,
            kv_cache_dtype_skip_layers=[],
            cache_dtype="auto",
            sliding_window=None,
            enable_prefix_caching=False,
        ),
        kv_transfer_config=None,
        speculative_config=None,
        attention_config=SimpleNamespace(
            use_non_causal=False,
            backend=None,
            backend_per_kind={},
        ),
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
        ),
        compilation_config=SimpleNamespace(
            compile_custom_ops=False,
            custom_ops="all",
            enabled_custom_ops=set(),
            static_forward_context={},
            mode=0,  # CompilationMode.NONE is 0
        ),
        model_config=SimpleNamespace(
            dtype=dtype,
            is_mm_prefix_lm=False,
            rswa_window=None,
            head_dtype=None,
        ),
        kernel_config=SimpleNamespace(
            linear_backend="auto",
        ),
    )
    monkeypatch.setattr(
        current_platform,
        "get_attn_backend_cls",
        lambda *args, **kwargs: (
            "vllm.v1.attention.backends.cpu_attn.CPUAttentionBackend"
        ),
    )

    class MockGroup:
        rank_in_group = 0
        world_size = 1

    monkeypatch.setattr(
        "vllm.distributed.parallel_state._TP",
        MockGroup(),
    )

    vllm_config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            draft_model_config=SimpleNamespace(
                hf_config=hf_config,
                quantization=None,
            ),
            num_speculative_tokens=4,
            enable_adaptive_verification=False,
        ),
        model_config=SimpleNamespace(
            dtype=dtype,
            is_mm_prefix_lm=False,
            get_total_num_hidden_layers=lambda: 2,
            get_vocab_size=lambda: hf_config.vocab_size,
        ),
        load_config=SimpleNamespace(
            quantization=None,
            quantization_param_path=None,
        ),
    )
    mock_current_vllm_config.speculative_config = vllm_config.speculative_config
    vllm_config.compilation_config = mock_current_vllm_config.compilation_config
    return mock_current_vllm_config, vllm_config


@pytest.mark.skip_global_cleanup
@pytest.mark.parametrize("variant", ["dflash2", "lilicorr", "lilicorr_plain"])
def test_candidate_model_decoder_layer_cls(monkeypatch, variant):
    from vllm.config import set_current_vllm_config
    from vllm.model_executor.models.lilicorr import LiLiCorr
    from vllm.model_executor.models.qwen3_dflash import DFlashQwen3DecoderLayer
    from vllm.model_executor.models.qwen3_dflash2 import (
        DFlash2Qwen3DecoderLayer,
        DFlash2Qwen3Model,
    )

    hf_config = SimpleNamespace(
        vocab_size=1000,
        hidden_size=256,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=2,
        max_position_embeddings=2048,
        rms_norm_eps=1e-6,
        rope_parameters={},
        intermediate_size=512,
        hidden_act="silu",
        dflash_config={
            "selector_rank": 4,
            "selector_top_k": 3,
            "conv_kernel_size": 0 if variant == "lilicorr_plain" else 3,
            "conv_group_size": 0 if variant == "lilicorr_plain" else 2,
            "block_size": 5,
            "lilicorr_candidate_topk": 4,
            "lilicorr_hidden_size": 8,
            "lilicorr_num_layers": 2,
            "lilicorr_num_heads": 2,
            "lilicorr_mlp_ratio": 2.0,
            "lilicorr_factor_dim": 4,
            "lilicorr_vector_eps": 1e-6,
            "lilicorr_logit_scale": 3.0,
            "use_aux_hidden_state": False,
        },
    )
    current_config, vllm_config = _mock_draft_configs(monkeypatch, hf_config)

    # Instantiate the model under meta device to avoid parameter allocation issues
    with set_current_vllm_config(current_config), torch.device("meta"):
        model_cls = DFlash2Qwen3Model if variant == "dflash2" else LiLiCorr
        model = model_cls(vllm_config=vllm_config)

    assert len(model.layers) == 2
    expected = (
        DFlashQwen3DecoderLayer
        if variant == "lilicorr_plain"
        else DFlash2Qwen3DecoderLayer
    )
    assert type(model.layers[0]) is expected


@pytest.mark.skip_global_cleanup
def test_beam_dflash2_loads_every_checkpoint_tensor(monkeypatch):
    """The exported checkpoint's names load into every parameter but the
    embedding and LM head, which alias the target's."""
    from vllm.config import set_current_vllm_config
    from vllm.models.beam import DFlash2BeamForCausalLM

    hidden, heads, kv_heads, head_dim, ffn, vocab, rank = 64, 4, 2, 16, 128, 100, 8
    taps, groups = 2, 4
    hf_config = SimpleNamespace(
        vocab_size=vocab,
        hidden_size=hidden,
        num_hidden_layers=2,
        num_attention_heads=heads,
        num_key_value_heads=kv_heads,
        head_dim=head_dim,
        max_position_embeddings=2048,
        rms_norm_eps=1e-6,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
        intermediate_size=ffn,
        hidden_act="silu",
        is_causal=False,
        layer_types=["sliding_attention"] * 2,
        sliding_window=64,
        dflash_config={
            "block_size": 16,
            "conv_group_size": hidden // groups,
            "conv_kernel_size": taps,
            "final_logit_softcapping": 30,
            "mask_token_id": 99,
            "selector_rank": rank,
            "selector_top_k": 3,
            "target_layer_ids": [0, 1],
        },
    )
    current_config, vllm_config = _mock_draft_configs(monkeypatch, hf_config)
    with set_current_vllm_config(current_config):
        model = DFlash2BeamForCausalLM(vllm_config=vllm_config)

    shapes = {
        "fc.weight": (hidden, 2 * hidden),
        "hidden_norm.weight": (hidden,),
        "norm.weight": (hidden,),
        "candidate_selector.hidden_projection.weight": (rank, hidden),
        "candidate_selector.predecessor_codebook": (vocab, rank),
        "candidate_selector.successor_codebook": (vocab, rank),
    }
    for layer in range(2):
        for conv in ("attention_conv", "mlp_conv"):
            shapes[f"layers.{layer}.{conv}.base_kernel"] = (2, taps, hidden)
            shapes[f"layers.{layer}.{conv}.kernel_projection.weight"] = (
                2 * taps * groups,
                hidden,
            )
        shapes |= {
            f"layers.{layer}.input_layernorm.weight": (hidden,),
            f"layers.{layer}.post_attention_layernorm.weight": (hidden,),
            f"layers.{layer}.self_attn.q_proj.weight": (heads * head_dim, hidden),
            f"layers.{layer}.self_attn.k_proj.weight": (kv_heads * head_dim, hidden),
            f"layers.{layer}.self_attn.v_proj.weight": (kv_heads * head_dim, hidden),
            f"layers.{layer}.self_attn.o_proj.weight": (hidden, heads * head_dim),
            f"layers.{layer}.self_attn.q_norm.weight": (head_dim,),
            f"layers.{layer}.self_attn.k_norm.weight": (head_dim,),
            f"layers.{layer}.mlp.gate_proj.weight": (ffn, hidden),
            f"layers.{layer}.mlp.up_proj.weight": (ffn, hidden),
            f"layers.{layer}.mlp.down_proj.weight": (hidden, ffn),
        }
    torch.manual_seed(0)
    checkpoint = {name: torch.randn(shape) for name, shape in shapes.items()}

    loaded = model.load_weights(iter(checkpoint.items()))

    assert loaded == {name for name, _ in model.named_parameters()}
    context_kv = [
        checkpoint[f"layers.{layer}.self_attn.{proj}_proj.weight"]
        for layer in range(2)
        for proj in ("k", "v")
    ]
    torch.testing.assert_close(model.model._context_kv_weight, torch.cat(context_kv))
    # Numbered after the target's two layers, so the KV cache keys stay distinct.
    assert model.get_draft_kv_cache_layer_names() == [
        "model.layers.2.self_attn.attn",
        "model.layers.3.self_attn.attn",
    ]
    assert model.get_draft_attn_causal() == [False, False]


@pytest.mark.skip_global_cleanup
def test_dflash_fc_runs_fp32_target_hidden_states_in_draft_dtype(monkeypatch):
    """A target with an FP32 residual stream (e.g. Muse Spark) hands FP32 aux
    hidden states to a bf16 drafter."""
    from vllm.config import set_current_vllm_config
    from vllm.model_executor.models.qwen3_dflash import DFlashQwen3ForCausalLM

    hidden = 64
    hf_config = SimpleNamespace(
        vocab_size=100,
        hidden_size=hidden,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=2048,
        rms_norm_eps=1e-6,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
        intermediate_size=128,
        hidden_act="silu",
        dflash_config={"mask_token_id": 99, "target_layer_ids": [0, 1]},
    )
    current_config, vllm_config = _mock_draft_configs(
        monkeypatch, hf_config, dtype=torch.bfloat16
    )
    with set_current_vllm_config(current_config):
        model = DFlashQwen3ForCausalLM(vllm_config=vllm_config)
    fc = model.model.fc
    torch.manual_seed(0)
    torch.nn.init.normal_(fc.weight)
    fc.quant_method.process_weights_after_loading(fc)
    aux_hidden_states = torch.randn(3, 2 * hidden, dtype=torch.float32)

    combined = model.combine_hidden_states(aux_hidden_states)

    assert combined.dtype == torch.bfloat16
    torch.testing.assert_close(combined, fc(aux_hidden_states.to(torch.bfloat16)))


def test_conv_projections_use_draft_quant_config(monkeypatch):
    from torch import nn

    from vllm.distributed import parallel_state
    from vllm.model_executor.layers.quantization import modelopt
    from vllm.model_executor.models.qwen3_dflash import DFlashQwen3DecoderLayer
    from vllm.model_executor.models.qwen3_dflash2 import DFlash2Qwen3DecoderLayer
    from vllm.model_executor.models.utils import AutoWeightsLoader

    monkeypatch.setattr(
        parallel_state, "_TP", SimpleNamespace(rank_in_group=0, world_size=1)
    )
    monkeypatch.setattr(
        DFlashQwen3DecoderLayer,
        "__init__",
        lambda self, *a, **kw: nn.Module.__init__(self),
    )
    # Exercise the real quantized parameter allocation without selecting a GPU kernel.
    monkeypatch.setattr(
        modelopt,
        "select_linear_kernel",
        lambda *a, **kw: SimpleNamespace(input_quant_key=lambda: None),
    )
    quant_config = modelopt.ModelOptNvFp4Config(
        quant_method="W4A16_NVFP4", is_checkpoint_nvfp4_serialized=True
    )
    layer = DFlash2Qwen3DecoderLayer(
        SimpleNamespace(
            speculative_config=SimpleNamespace(num_speculative_tokens=7),
            model_config=SimpleNamespace(dtype=torch.bfloat16),
        ),
        config=SimpleNamespace(
            hidden_size=16,
            dflash_config={"conv_kernel_size": 2, "conv_group_size": 2},
        ),
        layer_idx=0,
        prefix="model.layers.0",
        quant_config=quant_config,
    )
    for name in ("attention_conv", "mlp_conv"):
        module = getattr(layer, name)
        projection = module.kernel_projection
        assert projection.weight.dtype == torch.uint8
        assert projection.weight.shape == (32, 8)
        weight = torch.ones_like(projection.weight)
        AutoWeightsLoader(module).load_weights([("kernel_projection.weight", weight)])
        torch.testing.assert_close(projection.weight, weight)
        assert module.base_kernel.dtype == torch.bfloat16


def test_context_kv_uses_quantized_projection_fallback(monkeypatch):
    from torch import nn

    from vllm.model_executor.models import qwen3_dflash

    class Projection(nn.Module):
        def __init__(self, packed_weight):
            super().__init__()
            self.register_buffer("packed_weight", packed_weight)
            self.quant_method = object()
            self.calls = 0

        def forward(self, hidden_states):
            self.calls += 1
            return torch.nn.functional.linear(hidden_states, self.packed_weight), None

    monkeypatch.setattr(
        qwen3_dflash.ops,
        "rms_norm",
        lambda output, hidden_states, weight, eps: output.copy_(hidden_states),
    )
    context_states = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    projections = [
        Projection(
            torch.tensor(
                [
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 1.0],
                    [1.0, -1.0],
                ]
            )
        ),
        Projection(
            torch.tensor(
                [
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [2.0, 0.0],
                    [0.0, 2.0],
                    [-1.0, 0.0],
                    [0.0, -1.0],
                ]
            )
        ),
    ]
    model = SimpleNamespace(
        hidden_norm=SimpleNamespace(weight=nn.Parameter(torch.ones(2))),
        _rms_norm_eps=1e-6,
    )
    layers_attn = [
        SimpleNamespace(
            qkv_proj=projection,
            q_size=2,
            k_norm=SimpleNamespace(weight=nn.Parameter(torch.ones(2))),
        )
        for projection in projections
    ]
    qwen3_dflash.DFlashQwen3Model._build_context_kv_buffers(
        model,
        layers_attn,
        has_bias=False,
    )
    assert model._fused_kv_weight is None
    assert all(not hasattr(projection, "weight") for projection in projections)

    actual_k, actual_v = qwen3_dflash.DFlashQwen3Model._project_context_kv(
        model,
        context_states,
        num_ctx=2,
        num_layers=2,
        num_kv_heads=1,
        head_dim=2,
    )

    expected_k = torch.tensor(
        [[[1.0, 2.0], [3.0, 4.0]], [[2.0, 4.0], [6.0, 8.0]]]
    ).unsqueeze(2)
    expected_v = torch.tensor(
        [[[3.0, -1.0], [7.0, -1.0]], [[-1.0, -2.0], [-3.0, -4.0]]]
    ).unsqueeze(2)
    torch.testing.assert_close(actual_k, expected_k)
    torch.testing.assert_close(actual_v, expected_v)
    assert [projection.calls for projection in projections] == [1, 1]


def _stub_prepare_inputs_speculator(monkeypatch, speculative_config):
    from vllm.v1.worker.gpu.spec_decode.dflash import speculator as dflash_speculator

    calls = []
    monkeypatch.setattr(
        dflash_speculator,
        "prepare_dflash_inputs",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    speculator = object.__new__(DFlashSpeculator)
    speculator.speculative_config = speculative_config
    speculator.num_query_per_req = 6
    speculator.num_speculative_steps = 5
    speculator.max_num_reqs = 4
    speculator.max_num_tokens = 4096
    speculator.max_model_len = 8192
    speculator.parallel_drafting_token_id = 3
    speculator.sample_from_anchor = False
    speculator.input_buffers = SimpleNamespace(
        input_ids=torch.zeros(4096, dtype=torch.int32)
    )
    speculator.block_tables = SimpleNamespace(
        slot_mappings={0: torch.zeros(4096, dtype=torch.int64)},
        input_block_tables={0: torch.zeros(4, 32, dtype=torch.int32)},
        kernel_block_sizes={0: 256},
        cp_rank=0,
        cp_size=1,
        cp_interleave=1,
    )
    speculator.context_positions = torch.zeros(4096, dtype=torch.int64)
    speculator._context_slot_mappings = [torch.zeros(4096, dtype=torch.int64)]
    speculator.sample_indices = torch.zeros(20, dtype=torch.int64)
    speculator.sample_pos = torch.zeros(20, dtype=torch.int64)
    speculator.sample_idx_mapping = torch.zeros(20, dtype=torch.int32)
    speculator.temperature = torch.zeros(4, dtype=torch.float32)
    speculator.seeds = torch.zeros(4, dtype=torch.int64)
    speculator.draft_kv_cache_group_ids = [0]
    return speculator, calls


def _warmup_buckets(calls, num_query_per_req):
    buckets = set()
    for args, _ in calls:
        span = int(args[9].num_scheduled_tokens.max())
        max_tokens_per_req = span + num_query_per_req
        buckets.add(min(256, 1 << (max_tokens_per_req - 1).bit_length()))
    return buckets


def test_dflash_warmup_covers_reachable_block_size_buckets(monkeypatch):
    """BLOCK_SIZE follows the largest per-request span in the batch, which the
    worker warmup's tiny synthetic steps never grow; every bucket the wrapper
    can pick must be compiled here or the first large prefill chunk JITs live.
    """
    speculator, calls = _stub_prepare_inputs_speculator(monkeypatch, SimpleNamespace())

    speculator.warmup_prepare_inputs()

    # num_query_per_req=6 queries are appended per request, so the smallest
    # reachable span+query is 7 and buckets below 8 cannot occur.
    assert _warmup_buckets(calls, 6) == {8, 16, 32, 64, 128, 256}
    assert all("context_only" not in kwargs for _, kwargs in calls)


def test_dflash_warmup_covers_prefill_only_buckets(monkeypatch):
    """A prefill-only producer (#56957) launches with context_only=True and no
    appended query tokens, making every bucket from 1 up reachable."""
    speculator, calls = _stub_prepare_inputs_speculator(
        monkeypatch,
        SimpleNamespace(is_dspark_prefill_only=lambda: True),
    )

    speculator.warmup_prepare_inputs()

    assert _warmup_buckets(calls, 0) == {1, 2, 4, 8, 16, 32, 64, 128, 256}
    assert all(kwargs.get("context_only") is True for _, kwargs in calls)
