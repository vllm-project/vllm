# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import ANY

import torch

from vllm.model_executor.warmup import triton_attention_warmup as warmup
from vllm.v1.attention.backend import AttentionType
from vllm.v1.kv_cache_interface import KVQuantMode


def _make_key(
    num_kv_heads: int = 4,
    head_size: int = 128,
    cache_dtype: torch.dtype = torch.float16,
    kv_quant_mode: KVQuantMode = KVQuantMode.NONE,
    sliding_window: tuple[int, int] = (-1, -1),
) -> warmup.TritonUnifiedAttentionWarmupKey:
    return warmup.TritonUnifiedAttentionWarmupKey(
        num_query_heads=16,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        block_size=16,
        q_dtype=torch.float16,
        cache_dtype=cache_dtype,
        kv_quant_mode=kv_quant_mode,
        scale=0.125,
        sliding_window=sliding_window,
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


def test_warmup_shapes_add_nvfp4_raw_current_2d_representatives() -> None:
    key = _make_key(
        num_kv_heads=2,
        head_size=256,
        cache_dtype=torch.uint8,
        kv_quant_mode=KVQuantMode.NVFP4,
    )

    shapes = warmup._warmup_shapes(
        key, cudagraph_batch_sizes=(1, 2, 4, 8, 16, 32, 40, 64)
    )

    raw_current_shapes = [shape for shape in shapes if shape.use_raw_current]
    assert raw_current_shapes == [
        warmup._WarmupShape("raw_current_2d", (16,), (32,), False, True),
        warmup._WarmupShape("raw_current_2d", (16, 16), (32, 32), False, True),
        warmup._WarmupShape("raw_current_2d", (16,) * 16, (32,) * 16, False, True),
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


def test_iter_triton_unified_attention_warmup_keys_includes_nvfp4(
    monkeypatch,
) -> None:
    class TritonBackend:
        @staticmethod
        def get_name() -> str:
            return "TRITON_ATTN"

    impl = SimpleNamespace(
        attn_type=AttentionType.DECODER,
        kv_cache_dtype="nvfp4",
        num_heads=16,
        num_kv_heads=2,
        head_size=256,
        scale=0.125,
        sliding_window=(1024, 1024),
        logits_soft_cap=0.0,
        alibi_slopes=None,
        use_alibi_sqrt=False,
        sinks=None,
        chunk_lookback=-1,
        use_td=False,
    )
    group = SimpleNamespace(
        backend=TritonBackend(),
        layer_names=["layer.0"],
        kv_cache_spec=SimpleNamespace(block_size=16, dtype=torch.uint8),
    )
    runner = SimpleNamespace(
        attn_groups=[[group]],
        vllm_config=SimpleNamespace(),
        model_config=SimpleNamespace(dtype=torch.float16),
    )
    monkeypatch.setattr(
        warmup,
        "get_layers_from_vllm_config",
        lambda *_args: {"layer.0": SimpleNamespace(impl=impl)},
    )

    keys = warmup._iter_triton_unified_attention_warmup_keys(runner)

    assert keys == [
        _make_key(
            num_kv_heads=2,
            head_size=256,
            cache_dtype=torch.uint8,
            kv_quant_mode=KVQuantMode.NVFP4,
            sliding_window=(1024, 1024),
        )
    ]


def test_allocate_kv_cache_tensors_uses_nvfp4_data_and_scale_views() -> None:
    key = _make_key(
        num_kv_heads=2,
        head_size=256,
        cache_dtype=torch.uint8,
        kv_quant_mode=KVQuantMode.NVFP4,
    )
    shape = warmup._WarmupShape("decode_3d", (1, 1), (32, 32), True)

    k, v, k_scale_cache, v_scale_cache = warmup._allocate_kv_cache_tensors(
        key, shape, torch.device("cpu")
    )

    assert k.shape == (2, 16, 2, 128)
    assert v.shape == k.shape
    assert k.dtype == torch.uint8
    assert v.dtype == torch.uint8
    assert k_scale_cache is not None
    assert v_scale_cache is not None
    assert k_scale_cache.shape == (2, 16, 2, 16)
    assert v_scale_cache.shape == k_scale_cache.shape
    assert k_scale_cache.dtype == torch.uint8
    assert v_scale_cache.dtype == torch.uint8


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


def test_warmup_unified_attention_key_passes_nvfp4_scales(monkeypatch) -> None:
    key = _make_key(
        num_kv_heads=2,
        head_size=256,
        cache_dtype=torch.uint8,
        kv_quant_mode=KVQuantMode.NVFP4,
    )
    calls: list[dict] = []

    import vllm.v1.attention.ops.triton_unified_attention as unified_attention_module

    monkeypatch.setattr(
        warmup,
        "_warmup_shapes",
        lambda _key, _cudagraph_batch_sizes=(): (
            warmup._WarmupShape("decode_3d", (1,), (32,), True),
        ),
    )
    monkeypatch.setattr(
        unified_attention_module,
        "unified_attention",
        lambda **kwargs: calls.append(kwargs),
    )

    warmup._warmup_unified_attention_key(key, torch.device("cpu"))

    assert len(calls) == 1
    assert calls[0] == {
        "q": ANY,
        "k": ANY,
        "v": ANY,
        "out": ANY,
        "cu_seqlens_q": ANY,
        "max_seqlen_q": 1,
        "seqused_k": ANY,
        "max_seqlen_k": 32,
        "softmax_scale": 0.125,
        "causal": True,
        "window_size": (-1, -1),
        "block_table": ANY,
        "softcap": 0.0,
        "q_descale": None,
        "k_descale": ANY,
        "v_descale": ANY,
        "seq_threshold_3D": 64,
        "num_par_softmax_segments": ANY,
        "softmax_segm_output": ANY,
        "softmax_segm_max": ANY,
        "softmax_segm_expsum": ANY,
        "alibi_slopes": None,
        "sinks": None,
        "use_alibi_sqrt": False,
        "kv_quant_mode": KVQuantMode.NVFP4,
        "k_scale_cache": ANY,
        "v_scale_cache": ANY,
        "raw_k": None,
        "raw_v": None,
        "chunk_lookback": -1,
        "use_td": False,
    }
    assert calls[0]["k"].dtype == torch.uint8
    assert calls[0]["v"].dtype == torch.uint8
    assert calls[0]["k_scale_cache"].dtype == torch.uint8
    assert calls[0]["v_scale_cache"].dtype == torch.uint8
    assert calls[0]["k_descale"].shape == (1,)
    assert calls[0]["v_descale"].shape == (1,)
    assert calls[0]["softmax_segm_output"].shape == (64, 16, 16, 256)
    assert calls[0]["softmax_segm_max"].shape == (64, 16, 16)
    assert calls[0]["softmax_segm_expsum"].shape == (64, 16, 16)


def test_warmup_unified_attention_key_passes_nvfp4_raw_current_tensors(
    monkeypatch,
) -> None:
    key = _make_key(
        num_kv_heads=2,
        head_size=256,
        cache_dtype=torch.uint8,
        kv_quant_mode=KVQuantMode.NVFP4,
    )
    calls: list[dict] = []

    import vllm.v1.attention.ops.triton_unified_attention as unified_attention_module

    monkeypatch.setattr(
        warmup,
        "_warmup_shapes",
        lambda _key, _cudagraph_batch_sizes=(): (
            warmup._WarmupShape("raw_current_2d", (16, 16), (32, 32), False, True),
        ),
    )
    monkeypatch.setattr(
        unified_attention_module,
        "unified_attention",
        lambda **kwargs: calls.append(kwargs),
    )

    warmup._warmup_unified_attention_key(key, torch.device("cpu"))

    assert len(calls) == 1
    assert calls[0]["max_seqlen_q"] == 16
    assert calls[0]["max_seqlen_k"] == 32
    assert calls[0]["seq_threshold_3D"] is None
    assert calls[0]["num_par_softmax_segments"] is None
    assert calls[0]["softmax_segm_output"] is None
    assert calls[0]["raw_k"] is not None
    assert calls[0]["raw_v"] is not None
    assert calls[0]["raw_k"].shape == (32, 2, 256)
    assert calls[0]["raw_v"].shape == (32, 2, 256)
    assert calls[0]["raw_k"].dtype == torch.float16
    assert calls[0]["raw_v"].dtype == torch.float16
    assert calls[0]["block_table"].shape == (2, 16)
    assert calls[0]["block_table"].stride(0) == 16
    assert calls[0]["kv_quant_mode"] == KVQuantMode.NVFP4


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
