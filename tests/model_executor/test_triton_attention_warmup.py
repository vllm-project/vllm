# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

from vllm.model_executor.warmup import triton_attention_warmup as warmup
from vllm.v1.attention.backend import AttentionType
from vllm.v1.kv_cache_interface import KVQuantMode


def _make_key(num_kv_heads: int = 4) -> warmup.TritonUnifiedAttentionWarmupKey:
    return warmup.TritonUnifiedAttentionWarmupKey(
        num_query_heads=16,
        num_kv_heads=num_kv_heads,
        head_size=128,
        block_size=16,
        q_dtype=torch.float16,
        cache_dtype=torch.float16,
        kv_quant_mode=KVQuantMode.NONE,
        scale=0.125,
        sliding_window=(-1, -1),
        softcap=0.0,
        use_alibi=False,
        use_alibi_sqrt=False,
        use_sinks=False,
        chunk_lookback=-1,
        use_td=False,
    )


def test_warmup_shapes_follow_cudagraph_3d_threshold() -> None:
    key = _make_key(num_kv_heads=4)

    shapes = warmup._warmup_shapes(
        key, cudagraph_batch_sizes=(1, 2, 4, 8, 16, 32, 40, 64)
    )

    assert shapes[0] == warmup._WarmupShape("prefill_2d", (16,), (16,), False)
    decode_shapes = [
        (shape.name, len(shape.query_lens), shape.use_3d) for shape in shapes[1:]
    ]
    assert decode_shapes == [
        ("decode_3d", 1, True),
        ("decode_3d", 2, True),
        ("decode_3d", 4, True),
        ("decode_3d", 8, True),
        ("decode_3d", 16, True),
        ("decode_3d", 32, True),
        ("decode_2d", 40, False),
    ]


def test_iter_triton_unified_attention_warmup_keys_dedupes_layers(
    monkeypatch,
) -> None:
    class TritonBackend:
        @staticmethod
        def get_name() -> str:
            return "TRITON_ATTN"

    impl = SimpleNamespace(
        attn_type=AttentionType.DECODER,
        kv_cache_dtype="auto",
        num_heads=16,
        num_kv_heads=4,
        head_size=128,
        scale=0.125,
        sliding_window=(-1, -1),
        logits_soft_cap=0.0,
        alibi_slopes=None,
        use_alibi_sqrt=False,
        sinks=None,
        chunk_lookback=-1,
        use_td=False,
    )
    group = SimpleNamespace(
        backend=TritonBackend(),
        layer_names=["layer.0", "layer.1"],
        kv_cache_spec=SimpleNamespace(block_size=16, dtype=torch.float16),
    )
    runner = SimpleNamespace(
        attn_groups=[[group]],
        vllm_config=SimpleNamespace(),
        model_config=SimpleNamespace(dtype=torch.float16),
    )
    monkeypatch.setattr(
        warmup,
        "get_layers_from_vllm_config",
        lambda *_args: {
            "layer.0": SimpleNamespace(impl=impl),
            "layer.1": SimpleNamespace(impl=impl),
        },
    )

    keys = warmup._iter_triton_unified_attention_warmup_keys(runner)

    assert keys == [_make_key(num_kv_heads=4)]


def test_triton_unified_attention_warmup_dispatches_synthetic_calls(
    monkeypatch,
) -> None:
    key = _make_key(num_kv_heads=4)
    calls: list[
        tuple[warmup.TritonUnifiedAttentionWarmupKey, torch.device, tuple[int, ...]]
    ] = []
    runner = SimpleNamespace(
        is_pooling_model=False,
        attn_groups=[[object()]],
        device=torch.device("cuda"),
        cudagraph_batch_sizes=[1, 2, 4, 8],
    )

    monkeypatch.setattr(warmup.current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(warmup.current_platform, "is_xpu", lambda: False)
    monkeypatch.setattr(
        warmup, "_iter_triton_unified_attention_warmup_keys", lambda _runner: [key]
    )
    monkeypatch.setattr(
        warmup,
        "_warmup_unified_attention_key",
        lambda key, device, cudagraph_batch_sizes=(): calls.append(
            (key, device, cudagraph_batch_sizes)
        ),
    )

    warmup.triton_unified_attention_warmup(runner)

    assert calls == [(key, torch.device("cuda"), (1, 2, 4, 8))]


def test_triton_unified_attention_warmup_skips_pooling_runner(monkeypatch) -> None:
    calls = 0
    runner = SimpleNamespace(is_pooling_model=True, attn_groups=[[object()]])

    def fail_if_called(*_args, **_kwargs):
        nonlocal calls
        calls += 1

    monkeypatch.setattr(
        warmup, "_iter_triton_unified_attention_warmup_keys", fail_if_called
    )

    warmup.triton_unified_attention_warmup(runner)

    assert calls == 0
