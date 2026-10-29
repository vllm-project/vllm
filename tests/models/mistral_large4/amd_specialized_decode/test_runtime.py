# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests of the Mistral Large 4 specialized decode: the switch, the startup
refusals, the kernel variant per layer and the per-step routing."""

from types import SimpleNamespace

import pytest
import torch

from vllm.config import CompilationMode
from vllm.models.mistral.common.layer_spec import get_layer_specs
from vllm.models.mistral_large4.amd.specialized_decode.runtime.spec import (
    DECODE_CONFIG,
    specialized_decode_config,
)

N_LAYERS = 72


def ml4_text_config():
    """The released checkpoint's layer pattern (params.json, as the config parser
    expands it)."""
    nope = [i % 4 == 3 for i in range(N_LAYERS)]
    sources = [35 if i >= 39 and i % 4 == 3 else i for i in range(N_LAYERS)]
    sharing = [
        "shared" if i >= 17 and i % 4 in (1, 2) else "full" for i in range(N_LAYERS)
    ]
    return SimpleNamespace(
        num_hidden_layers=N_LAYERS,
        nope_layer_flags=nope,
        kv_source_layers=sources,
        index_sharing=sharing,
        index_topk=2048,
        indexer_nope_layer_flags=list(nope),
        n_routed_experts=128,
        first_k_dense_replace=1,
        moe_layer_freq=1,
    )


def vllm_config(**overrides):
    """A supported deployment; ``overrides`` (dotted names) change one field."""
    cfg = SimpleNamespace(
        parallel_config=SimpleNamespace(
            tensor_parallel_size=4, pipeline_parallel_size=1, data_parallel_size=1,
            decode_context_parallel_size=1, prefill_context_parallel_size=1,
            use_ubatching=False, enable_expert_parallel=False, enable_eplb=False,
        ),
        cache_config=SimpleNamespace(cache_dtype="auto", block_size=16, kv_offloading_size=None),  # noqa: E501
        lora_config=None,
        kv_transfer_config=None,
        use_v2_model_runner=True,
        speculative_config=None,
        model_config=SimpleNamespace(enable_sleep_mode=False, quantization="fp8"),
        weight_transfer_config=None,
        offload_config=SimpleNamespace(
            uva=SimpleNamespace(cpu_offload_gb=0),
            prefetch=SimpleNamespace(offload_group_size=0),
        ),
        compilation_config=SimpleNamespace(mode=CompilationMode.NONE),
        aux_output_config=SimpleNamespace(enable_return_routed_experts=False),
        kernel_config=SimpleNamespace(),
        additional_config={"decode_backend": "latency_specialized"},
    )  # fmt: skip
    for name, value in overrides.items():
        obj = cfg
        *path, last = name.split(".")
        for p in path:
            obj = getattr(obj, p)
        setattr(obj, last, value)
    return cfg


@pytest.fixture(autouse=True)
def a4w4(monkeypatch):
    monkeypatch.setenv("VLLM_FP8_MOE_REQUANT", "mxfp4")
    monkeypatch.setenv("VLLM_FP8_MOE_REQUANT_ACT", "a4w4")


def test_off_by_default():
    assert specialized_decode_config(vllm_config(additional_config={})) is None
    assert specialized_decode_config(vllm_config(additional_config=None)) is None


def test_switches():
    assert specialized_decode_config(vllm_config()) is DECODE_CONFIG
    cfg = vllm_config(additional_config={}, **{"kernel_config.monokernel": True})
    assert specialized_decode_config(cfg) is DECODE_CONFIG
    with pytest.raises(ValueError, match="decode_backend must be"):
        specialized_decode_config(
            vllm_config(additional_config={"decode_backend": "x"})
        )


def test_refusal_lists_every_reason(monkeypatch):
    monkeypatch.setenv("VLLM_FP8_MOE_REQUANT_ACT", "a8w4")
    cfg = vllm_config(**{
        "parallel_config.tensor_parallel_size": 8,
        "cache_config.cache_dtype": "fp8_ds_mla",
        "speculative_config": object(),
        "use_v2_model_runner": False,
    })  # fmt: skip
    with pytest.raises(ValueError) as err:
        specialized_decode_config(cfg)
    msg = str(err.value)
    for part in (
        "tensor parallel size 4", "bf16 or fp8 (e4m3) KV cache", "speculative decoding",
        "Model Runner V2", "a4w4",
    ):  # fmt: skip
        assert part in msg, msg


@pytest.mark.parametrize(
    "field, value",
    [
        ("parallel_config.pipeline_parallel_size", 2),
        ("parallel_config.data_parallel_size", 2),
        ("parallel_config.decode_context_parallel_size", 4),
        ("parallel_config.use_ubatching", True),
        ("parallel_config.enable_expert_parallel", True),
        ("cache_config.block_size", 8),
        ("lora_config", object()),
        ("model_config.enable_sleep_mode", True),
        ("compilation_config.mode", CompilationMode.STOCK_TORCH_COMPILE),
    ],
)
def test_default_deny(field, value):
    assert DECODE_CONFIG.refusal_reasons(vllm_config(**{field: value}))


def test_widths_are_graph_capture_sizes():
    # vLLM pads decode batches to its capture sizes 1, 2, 4, 8, ...; every width
    # the kernels serve must be one, or FULL-graph replays would never use them
    assert DECODE_CONFIG.widths == (1, 2, 4)


def test_supported_has_no_reasons():
    assert DECODE_CONFIG.refusal_reasons(vllm_config()) == []
    native = vllm_config(**{"model_config.quantization": "quark"})
    assert DECODE_CONFIG.refusal_reasons(native) == []  # experts checked after load


def _fake_layers(text_config):
    return [
        SimpleNamespace(self_attn=SimpleNamespace(spec=s))
        for s in get_layer_specs(text_config)
    ]


def test_layer_kinds_of_the_checkpoint():
    pytest.importorskip("flydsl")  # weights.py imports the kernel helpers
    from vllm.models.mistral_large4.amd.specialized_decode.runtime.weights import (
        LayerKind,
        moe_kinds,
    )

    kinds = moe_kinds(_fake_layers(ml4_text_config()))
    assert len(kinds) == N_LAYERS - 1
    counts = {k: kinds.count(k) for k in set(kinds)}
    assert counts == {
        LayerKind.ROPE: 25, LayerKind.ROPE_SHARED: 28,
        LayerKind.NOPE: 9, LayerKind.NOPE_KVS: 9,
    }  # fmt: skip
    assert kinds[39 - 1] == LayerKind.NOPE_KVS and kinds[35 - 1] == LayerKind.NOPE
    assert kinds[17 - 1] == kinds[18 - 1] == LayerKind.ROPE_SHARED
    assert kinds[16 - 1] == LayerKind.ROPE


def test_layer_mix_refused():
    pytest.importorskip("flydsl")
    from vllm.models.mistral_large4.amd.specialized_decode.mono.common.runtime.consensus import (  # noqa: E501
        MonoUnsupported,
    )
    from vllm.models.mistral_large4.amd.specialized_decode.runtime.weights import (
        moe_kinds,
    )

    dense = ml4_text_config()
    dense.first_k_dense_replace = 3
    with pytest.raises(MonoUnsupported, match="layer mix"):
        moe_kinds(_fake_layers(dense))
    shared_first = ml4_text_config()
    shared_first.index_sharing = ["full", "shared"] + ["full"] * (N_LAYERS - 2)
    with pytest.raises(MonoUnsupported, match="shared top-k"):
        moe_kinds(_fake_layers(shared_first))


def _context(rows=1, prefills=0, max_q=1, ubatch=None):
    md = SimpleNamespace(
        num_prefills=prefills,
        max_query_len=max_q,
        block_table=torch.zeros(4, 2048, dtype=torch.int32),
    )
    return SimpleNamespace(
        ubatch_slices=ubatch,
        attn_metadata={"L1": md},
        slot_mapping={"L1": torch.zeros(rows, dtype=torch.int64)},
    )


def test_step_routing():
    from vllm.models.mistral_large4.amd.specialized_decode.runtime.decode import (
        step_refusal,
    )

    pos = torch.zeros(1, dtype=torch.int64)
    widths = DECODE_CONFIG.widths
    assert step_refusal(_context(), "L1", pos, widths) is None
    for rows in widths:
        row_pos = torch.zeros(rows, dtype=torch.int64)
        assert step_refusal(_context(rows=rows), "L1", row_pos, widths) is None
    assert step_refusal(
        _context(rows=3), "L1", torch.zeros(3, dtype=torch.int64), widths
    )
    assert step_refusal(
        _context(rows=16), "L1", torch.zeros(16, dtype=torch.int64), widths
    )
    assert step_refusal(_context(prefills=1), "L1", pos, widths)
    assert step_refusal(_context(max_q=4), "L1", pos, widths)
    assert step_refusal(_context(ubatch=object()), "L1", pos, widths)
    assert step_refusal(_context(), "L1", pos.int(), widths)
    no_md = _context()
    no_md.attn_metadata = None
    assert step_refusal(no_md, "L1", pos, widths)


@pytest.mark.parametrize(
    "cache_dtype, supported",
    [("auto", True), ("bfloat16", True), ("fp8", True), ("fp8_e4m3", True),
     ("fp8_e5m2", False)],
)  # fmt: skip
def test_kv_cache_dtype(cache_dtype, supported):
    reasons = DECODE_CONFIG.refusal_reasons(
        vllm_config(**{"cache_config.cache_dtype": cache_dtype})
    )
    assert reasons == ([] if supported else ["requires a bf16 or fp8 (e4m3) KV cache"])


def _bound_layers(*caches, k_scale=1.0, kv_cache_dtype="fp8"):
    """MonoRunner.moe entries as `_kv_cache_dtype` / `_kv_scale_refusal` read them."""
    return SimpleNamespace(
        moe=[
            SimpleNamespace(
                attn=SimpleNamespace(
                    attn=SimpleNamespace(
                        kv_cache=kv,
                        kv_cache_dtype=kv_cache_dtype,
                        _k_scale_float=k_scale,
                    )
                )
            )
            for kv in caches
        ]
    )


@pytest.mark.parametrize(
    "kv_cache_dtype, k_scale, refused",
    [("fp8", 1.0, False), ("fp8", 0.5, True), ("fp8_e4m3", 2.0, True),
     # a bf16 cache ignores the (unused) KV scale
     ("auto", 0.5, False)],
)  # fmt: skip
def test_kv_scale_refusal(kv_cache_dtype, k_scale, refused):
    """The kernel reads and writes fp8 unscaled; refused when the runner is
    built (after weight loading), not on the first decode step."""
    from vllm.models.mistral_large4.amd.specialized_decode.runtime.runner import (
        _kv_scale_refusal,
    )

    bound = _bound_layers(
        torch.zeros(1), k_scale=k_scale, kv_cache_dtype=kv_cache_dtype
    )
    reason = _kv_scale_refusal(bound.moe)
    assert (reason is not None) == refused
    if refused:
        assert "KV scales" in reason


@pytest.mark.parametrize(
    "dtypes, expected",
    [
        ((torch.bfloat16, torch.bfloat16), "bf16"),
        # vLLM allocates an fp8 cache as uint8 and views it as fp8 per call
        ((torch.uint8, torch.uint8), "fp8"),
        ((torch.float8_e4m3fn,), "fp8"),
        ((torch.bfloat16, torch.uint8), "unsupported latent KV cache dtypes"),
        ((torch.float16,), "unsupported latent KV cache dtypes"),
    ],
)
def test_runner_kv_cache_dtype(dtypes, expected):
    from vllm.models.mistral_large4.amd.specialized_decode.runtime.runner import (
        MonoRunner,
    )

    caches = [torch.zeros(2, 16, 576, dtype=dtype) for dtype in dtypes]
    bound = _bound_layers(*caches)
    if expected in ("bf16", "fp8"):
        assert MonoRunner._kv_cache_dtype(bound) == expected
    else:
        with pytest.raises(ValueError, match=expected):
            MonoRunner._kv_cache_dtype(bound)


def test_cache_refusals_latent_dtype():
    from vllm.models.mistral_large4.amd.specialized_decode.runtime.guards import (
        INDEX_ROW,
        LATENT,
        cache_refusals,
    )

    def model(dtype):
        layer = SimpleNamespace(
            self_attn=SimpleNamespace(
                spec=SimpleNamespace(layer_idx=1),
                attn=SimpleNamespace(kv_cache=torch.zeros(4, 16, LATENT, dtype=dtype)),
                indexer=SimpleNamespace(
                    k_cache=SimpleNamespace(
                        kv_cache=torch.zeros(4, 16, INDEX_ROW, dtype=torch.uint8)
                    )
                ),
            )
        )
        return SimpleNamespace(layers=[None, layer])

    for dtype in (torch.bfloat16, torch.uint8, torch.float8_e4m3fn):
        assert cache_refusals(model(dtype), 2048) == [], dtype
    assert cache_refusals(model(torch.float16), 2048)
