# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host side of --monokernel: config and weight refusals, per-step routing,
the per-forward plan, cache binding and the fail-stop gate."""

from collections import Counter
from types import SimpleNamespace

import pytest
import torch

from vllm.config import CompilationMode, CUDAGraphMode, KernelConfig
from vllm.model_executor.layers.fused_moe.experts.rocm_aiter_moe import (
    AiterMxfp4Experts,
)
from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import Mxfp4MoeBackend
from vllm.models.minimax_m3.amd.specialized_decode.runtime import decode as runtime
from vllm.models.minimax_m3.amd.specialized_decode.runtime.caches import CacheBinding
from vllm.models.minimax_m3.amd.specialized_decode.runtime.executor import (
    FfnExecutor,
)
from vllm.models.minimax_m3.amd.specialized_decode.runtime.routing import decide_step
from vllm.models.minimax_m3.amd.specialized_decode.runtime.spec import (
    DECODE_CONFIG,
    DecodeConfig,
    specialized_decode_config,
)
from vllm.models.minimax_m3.amd.specialized_decode.runtime.weights import (
    EXPERTS,
    HIDDEN,
    INTER,
    MonoLayerWeights,
)
from vllm.models.minimax_m3.common.sparse_attention import (
    MiniMaxM3SparseDecodeMetadata,
    MiniMaxM3SparseMetadata,
)


def _metadata(rows, *, decode_tokens=None, prefills=0, query_len=1, padded=0):
    """A decode-first step: ``decode_tokens`` rows of requests ``query_len``
    rows each (an EAGLE3 verify step: 1 + num_speculative_tokens)."""
    decode_tokens = rows if decode_tokens is None else decode_tokens
    requests = decode_tokens // query_len
    metadata = MiniMaxM3SparseMetadata(
        seq_lens=torch.ones(requests, dtype=torch.int32),
        max_seq_len=1,
        slot_mapping=torch.arange(rows),
        num_actual_tokens=decode_tokens,
        num_decodes=requests,
        num_decode_tokens=decode_tokens,
        num_prefills=prefills,
        num_prefill_tokens=prefills,
        decode=MiniMaxM3SparseDecodeMetadata(
            seq_lens=torch.ones(requests, dtype=torch.int32),
            block_table=torch.zeros(requests, 1, dtype=torch.int32),
            decode_query_len=query_len,
        ),
    )
    if padded:
        metadata.slot_mapping[rows - padded :] = -1
    return metadata


NONE, FULL = CUDAGraphMode.NONE, CUDAGraphMode.FULL


@pytest.mark.parametrize("rows", [1, 4, 16])
def test_layer_handoffs_match_stage_order(monkeypatch, rows):
    """Trace the actual kernel without a launch: every mailbox consumer must
    follow its declared writer, including the fused projection stages.
    """
    from vllm.models.minimax_m3.amd.specialized_decode.mono.layer.kernels.post_attn import (  # noqa: E501
        K4_ABI,
        build_post_attn_kernel,
    )
    from vllm.models.minimax_m3.amd.specialized_decode.mono.layer.layout import (
        mailbox_regions,
    )
    from vllm.models.minimax_m3.amd.specialized_decode.mono.layer.support.plan.check import (  # noqa: E501
        check,
    )
    from vllm.models.minimax_m3.amd.specialized_decode.mono.layer.support.plan.trace import (  # noqa: E501
        recording,
    )

    monkeypatch.setenv("ARCH", "gfx950")
    monkeypatch.setenv("COMPILE_ONLY", "1")
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
    launch = build_post_attn_kernel(
        4,
        128**-0.5,
        1e-6,
        2.0,
        1.0,
        7.0,
        0,
        1,
        tokens=rows,
        fuse_k1=True,
        debug=True,
    )
    with recording() as record:
        launch(*K4_ABI.zeros())
    assert record.stages, "a cached compilation would bypass the contract check"
    check(record, mailbox_regions(rows, 1, True))


@pytest.mark.parametrize(
    "rows, mode, metadata, mono",
    [
        (1, NONE, _metadata(1), True),
        (16, NONE, _metadata(16), True),
        (2, FULL, _metadata(2), True),
        (4, NONE, _metadata(4, decode_tokens=2, padded=2), True),
        (17, NONE, _metadata(1), False),
        (1, CUDAGraphMode.PIECEWISE, _metadata(1), False),
        (1, NONE, None, False),
        (1, NONE, _metadata(1, prefills=1), False),
        (1, NONE, _metadata(1, decode_tokens=2), False),
        # EAGLE3 verify steps (4 rows a request): rows like any other
        (4, FULL, _metadata(4, query_len=4), True),
        (8, NONE, _metadata(8, query_len=4), True),
        (12, FULL, _metadata(12, query_len=4), True),
        (16, FULL, _metadata(16, query_len=4), True),
        (16, FULL, _metadata(16, decode_tokens=12, query_len=4, padded=4), True),
        (20, FULL, _metadata(20, query_len=4), False),
        (8, NONE, _metadata(8, prefills=1, query_len=4), False),
        (8, CUDAGraphMode.PIECEWISE, _metadata(8, query_len=4), False),
    ],
)
def test_step_routing(rows, mode, metadata, mono):
    assert decide_step(DECODE_CONFIG, rows, mode, metadata).use_mono is mono


def _vllm_config(layers=60, **parallel):
    values = dict(
        tensor_parallel_size=4,
        pipeline_parallel_size=1,
        data_parallel_size=1,
        decode_context_parallel_size=1,
        prefill_context_parallel_size=1,
        enable_expert_parallel=False,
        enable_eplb=False,
        use_ubatching=False,
    )
    values.update(parallel)
    return SimpleNamespace(
        parallel_config=SimpleNamespace(**values),
        cache_config=SimpleNamespace(
            cache_dtype="fp8",
            block_size=128,
            enable_prefix_caching=False,
            kv_offloading_size=None,
        ),
        attention_config=SimpleNamespace(
            resolve_indexer_kv_dtype=lambda _default: "fp8"
        ),
        compilation_config=SimpleNamespace(mode=CompilationMode.NONE),
        kernel_config=KernelConfig(moe_backend="aiter", monokernel=True),
        lora_config=None,
        kv_transfer_config=SimpleNamespace(kv_connector=None),
        aux_output_config=SimpleNamespace(enable_return_routed_experts=False),
        speculative_config=None,
        weight_transfer_config=None,
        offload_config=SimpleNamespace(
            uva=SimpleNamespace(cpu_offload_gb=0),
            prefetch=SimpleNamespace(offload_group_size=0),
        ),
        use_v2_model_runner=True,
        model_config=SimpleNamespace(
            enable_sleep_mode=False,
            hf_text_config=SimpleNamespace(
                use_index_cache=False,
                num_hidden_layers=layers,
                sparse_attention_config={"sparse_attention_freq": [0] * 3 + [1] * 57},
            ),
        ),
    )


def test_config_refusals_report_every_reason():
    assert specialized_decode_config(_vllm_config()).last_layer == 59
    # a truncated checkpoint: the sparse layers end at its last layer
    assert specialized_decode_config(_vllm_config(layers=5)).last_layer == 4

    config = _vllm_config(tensor_parallel_size=2, enable_expert_parallel=True)
    config.cache_config.block_size = 64
    config.aux_output_config.enable_return_routed_experts = True
    config.model_config.hf_text_config.sparse_attention_config["sparse_attention_freq"][
        2
    ] = 1
    with pytest.raises(ValueError) as error:
        specialized_decode_config(config)
    for reason in (
        "tensor parallel",
        "expert parallelism",
        "block size",
        "routed-expert capture",
        "sparse attention on layers 3",
    ):
        assert reason in str(error.value)
    with pytest.raises(ValueError, match="sparse attention on layers 3"):
        specialized_decode_config(_vllm_config(layers=3))


@pytest.mark.parametrize("method", ["eagle3", "ngram", "mtp", "draft_model"])
def test_only_eagle3_speculation_is_accepted(method):
    config = _vllm_config()
    config.speculative_config = SimpleNamespace(method=method)
    if method == "eagle3":
        assert specialized_decode_config(config) is not None
    else:
        with pytest.raises(ValueError, match="only EAGLE3"):
            specialized_decode_config(config)


def _parameter(shape, dtype, *, meta=True):
    return torch.empty(shape, dtype=dtype, device="meta" if meta else "cpu")


def _layer():
    w13 = _parameter((EXPERTS, 2 * INTER, HIDDEN // 2), torch.float4_e2m1fn_x2)
    w2 = _parameter((EXPERTS, HIDDEN, INTER // 2), torch.float4_e2m1fn_x2)
    w13.is_shuffled = True
    w2.is_shuffled = True
    fused_experts = object.__new__(AiterMxfp4Experts)
    fused_experts.moe_config = SimpleNamespace(
        use_mori_kernels=False,
        use_mxfp4_w4a4_dsv4=False,
    )
    experts = SimpleNamespace(
        top_k=4,
        renormalize=True,
        scoring_func="sigmoid",
        swiglu_alpha=1.702,
        swiglu_beta=1.0,
        swiglu_limit=7.0,
        quant_method=SimpleNamespace(
            mxfp4_backend=Mxfp4MoeBackend.AITER_MXFP4_MXFP4,
            moe_kernel=SimpleNamespace(fused_experts=fused_experts),
        ),
        w13_weight=w13,
        w2_weight=w2,
        w13_weight_scale=_parameter((EXPERTS, 2 * INTER, HIDDEN // 32), torch.uint8),
        w2_weight_scale=_parameter((EXPERTS, HIDDEN, INTER // 32), torch.uint8),
    )
    moe = SimpleNamespace(
        is_fused_shared_expert_enabled=True,
        use_aiter_moe_fse=True,
        experts=SimpleNamespace(routed_experts=experts),
        gate=SimpleNamespace(weight=_parameter((128, HIDDEN), torch.float32)),
        e_score_correction_bias=_parameter((128,), torch.float32),
        routed_scaling_factor=2.0,
        n_shared_experts=1,
    )
    qkv = type("MinimaxM3QKVParallelLinearWithIndexer", (), {})()
    qkv.weight = _parameter((2560, HIDDEN), torch.bfloat16, meta=False)
    output = type("RowParallelLinear", (), {})()
    output.weight = _parameter((HIDDEN, 2048), torch.bfloat16, meta=False)
    head_norm = SimpleNamespace(weight=_parameter((128,), torch.bfloat16))
    attn = SimpleNamespace(
        qkv_proj=qkv,
        o_proj=output,
        q_norm=head_norm,
        k_norm=head_norm,
        index_q_norm=head_norm,
        index_k_norm=head_norm,
    )
    norms = []
    for _ in range(2):
        norm = type("MiniMAXGemmaRMSNorm", (), {})()
        norm.weight = _parameter((HIDDEN,), torch.bfloat16)
        norm.variance_epsilon = 1e-6
        norms.append(norm)
    return SimpleNamespace(
        layer_id=3,
        self_attn=attn,
        block_sparse_moe=moe,
        input_layernorm=norms[0],
        post_attention_layernorm=norms[1],
    )


def test_weight_views_are_zero_copy():
    layer = _layer()
    moe = layer.block_sparse_moe
    moe.gate.weight = _parameter((128, HIDDEN), torch.float32, meta=False)
    layer.post_attention_layernorm.weight = _parameter(
        (HIDDEN,), torch.bfloat16, meta=False
    )
    weights = MonoLayerWeights.from_layer(layer, require_cuda=False)
    assert (
        weights.post_norm.data_ptr() == layer.post_attention_layernorm.weight.data_ptr()
    )
    assert weights.router.data_ptr() == moe.gate.weight.data_ptr()
    assert weights.router.dtype == torch.float32
    assert weights.w13.dtype == torch.float4_e2m1fn_x2


def _experts(layer):
    return layer.block_sparse_moe.experts.routed_experts


WEIGHT_REFUSALS = {
    "AITER_MXFP4_MXFP4": lambda x: setattr(
        _experts(x).quant_method, "mxfp4_backend", Mxfp4MoeBackend.AITER_MXFP4_BF16
    ),
    "effective BF16": lambda x: setattr(
        _experts(x).quant_method.moe_kernel.fused_experts.moe_config,
        "use_mori_kernels",
        True,
    ),
    "router": lambda x: setattr(
        x.block_sparse_moe.gate, "weight", _parameter((128, HIDDEN), torch.bfloat16)
    ),
    "routed scale": lambda x: setattr(x.block_sparse_moe, "routed_scaling_factor", 1),
}


@pytest.mark.parametrize("match", WEIGHT_REFUSALS)
def test_weight_refusals(match):
    layer = _layer()
    WEIGHT_REFUSALS[match](layer)
    with pytest.raises(ValueError, match=match):
        MonoLayerWeights.from_layer(layer, require_cuda=False)


class _Executor:
    uses_caches = True

    def __init__(self):
        self.fault_word = torch.zeros(4, dtype=torch.int32)
        self.slots: torch.Tensor | None = None
        self.binds: list[int] = []
        self.bind_failure: Exception | None = None
        self.calls: Counter = Counter()

    def layer(self, layer_id):
        return lambda _positions, hidden, residual: (hidden, residual)

    def bind(self, layer_id, binding):
        if self.bind_failure is not None:
            raise self.bind_failure
        self.binds.append(layer_id)

    def begin_step(self, slots):
        self.slots = slots

    def after_step(self):
        self.calls["after"] += 1

    def abort(self):
        self.calls["abort"] += 1

    def close(self):
        self.calls["close"] += 1


def _no_checks(monkeypatch, refusals=()):
    monkeypatch.setattr(runtime, "model_refusals", lambda model, config: list(refusals))
    monkeypatch.setattr(
        runtime, "MonoLayerWeights", SimpleNamespace(from_layer=lambda layer: layer)
    )
    monkeypatch.setattr(runtime, "tp_agree", lambda ok: ok)


@pytest.fixture
def decode(monkeypatch):
    _no_checks(monkeypatch)
    layers = [
        SimpleNamespace(layer_id=i, self_attn=SimpleNamespace(layer_name=f"layer.{i}"))
        for i in (3, 4)
    ]
    executor = _Executor()
    instance = runtime.SpecializedDecode(
        SimpleNamespace(layers=layers), DecodeConfig(last_layer=4), executor
    )
    return instance, executor


def test_plan_maps_every_sparse_layer_once_per_forward(decode, monkeypatch):
    instance, executor = decode
    context = SimpleNamespace(
        attn_metadata={"layer.3": _metadata(1)},
        cudagraph_runtime_mode=NONE,
    )
    monkeypatch.setattr(runtime, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(runtime, "get_forward_context", lambda: context)

    assert instance.plan(torch.zeros(1, dtype=torch.int64), object()) is None
    step = instance.plan(torch.zeros(1, dtype=torch.int64), None)
    assert step is not None and step.launched
    assert [step.layer(i) is not None for i in (2, 3, 4, 5)] == [
        False,
        True,
        True,
        False,
    ]
    assert (instance.mono_steps, instance.stock_steps) == (1, 1)
    assert executor.slots is not None and executor.slots.numel() == 1


@pytest.mark.parametrize(
    "refusals, peers_agree, match",
    [(["layer 3 is odd"], True, "layer 3 is odd"), ([], False, "another TP rank")],
)
def test_refusal_is_rank_uniform_and_precedes_the_executor(
    monkeypatch, refusals, peers_agree, match
):
    _no_checks(monkeypatch, refusals)
    monkeypatch.setattr(runtime, "tp_agree", lambda ok: ok and peers_agree)
    built: list = []
    with pytest.raises(ValueError, match=match):
        runtime.SpecializedDecode(
            SimpleNamespace(layers=[]), DecodeConfig(), built.append
        )
    assert not built


def _binding() -> CacheBinding:
    return CacheBinding(*(torch.empty(1) for _ in range(8)))


def test_cache_bind_rebinds_only_replaced_caches(decode, monkeypatch):
    instance, executor = decode
    bindings = {3: _binding()}
    monkeypatch.setattr(runtime, "checked_cache_bindings", lambda layers: bindings)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    instance.bind_caches()
    instance.bind_caches()
    assert executor.binds == [3]
    bindings[3] = _binding()
    instance.bind_caches()
    assert executor.binds == [3, 3]

    published = dict(instance._cache_identities)
    bindings[3] = _binding()
    executor.bind_failure = ValueError("injected")
    with pytest.raises(RuntimeError, match="rank-uniform.*cache bind") as error:
        instance.bind_caches()
    assert isinstance(error.value.__cause__, ValueError)
    assert instance._cache_identities == published


def _bare_decode(executor) -> runtime.SpecializedDecode:
    decode = object.__new__(runtime.SpecializedDecode)
    decode.executor = executor
    decode._closed = False
    return decode


LAUNCHED = runtime.Step(3, (lambda *args: args[1:],))


@pytest.mark.parametrize("step", [None, runtime.Step(3, (None,)), LAUNCHED])
def test_every_forward_asserts_the_error_word_is_clear(step):
    """Stock and mono steps alike, so a word set between steps (a host abort)
    stops the next step whichever path it takes."""
    executor = _Executor()
    decode = _bare_decode(executor)
    decode.after_step(step)
    assert executor.calls["after"] == (step is LAUNCHED)
    executor.fault_word[0] = 1
    with pytest.raises(RuntimeError, match="error word set"):
        decode.after_step(step)


@pytest.mark.parametrize("step, aborts", [(None, 0), (LAUNCHED, 1)])
def test_layer_exception_marks_peers_only_after_a_launch(step, aborts):
    executor = _Executor()
    decode = _bare_decode(executor)
    with pytest.raises(ValueError, match="layer"):
        decode.after_step(step, ValueError("layer"))
    assert executor.calls == Counter(abort=aborts)


def test_close_is_idempotent_and_stops_reading_the_word():
    executor = _Executor()
    decode = _bare_decode(executor)
    decode.close()
    decode.close()
    assert executor.calls == Counter(close=1)
    executor.fault_word[0] = 1
    decode.after_step(None)


def test_abort_never_raises():
    executor = object.__new__(FfnExecutor)
    executor.rank = 1

    def broken():
        raise RuntimeError("peer view")

    executor.peer = SimpleNamespace(
        bytes=torch.zeros(16, dtype=torch.uint8),
        addresses=SimpleNamespace(tolist=broken),
    )
    executor.abort()
