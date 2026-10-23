# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Muse Spark's specialized decode on CPU: the switch and its phases, the
three refusal stages (config, loaded weights, KV caches), plan() routing
(DSpark verify included) and the builds a kernel layer launches, mono-off
inertness and the fail-stop vote. The kernel modules (``mono.layer``,
``mono.k1``) are stubbed."""

import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

pytest.importorskip("aiter")

from vllm.config import CompilationMode, CUDAGraphMode  # noqa: E402
from vllm.model_executor.layers.fused_moe.activation import MoEActivation  # noqa: E402
from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (  # noqa: E402
    Mxfp4MoeBackend,
)
from vllm.models.muse_spark.amd import model as model_mod  # noqa: E402
from vllm.models.muse_spark.amd.model import (  # noqa: E402
    MuseSpark1xForCausalLM,
    MuseSpark1xModel,
    MuseSpark1xMoE,
)
from vllm.models.muse_spark.amd.muse_moe_runner import (  # noqa: E402
    MuseSparkQuarkOCPMXFusedMoE,
)
from vllm.models.muse_spark.amd.specialized_decode import runtime  # noqa: E402
from vllm.models.muse_spark.amd.specialized_decode.mono import spec  # noqa: E402
from vllm.models.muse_spark.amd.specialized_decode.runtime import (  # noqa: E402
    DecodeConfig,
    moe_mode,
    refuse_config,
    specialized_decode_config,
)
from vllm.models.muse_spark.amd.specialized_decode.runtime import (  # noqa: E402
    decode as decode_mod,
)
from vllm.models.muse_spark.amd.specialized_decode.runtime import (  # noqa: E402
    layers as layers_mod,
)
from vllm.models.muse_spark.amd.specialized_decode.runtime import (  # noqa: E402
    runner as runner_mod,
)
from vllm.models.muse_spark.amd.specialized_decode.runtime.decode import (  # noqa: E402
    SpecializedDecode,
    Step,
    step_reason,
)
from vllm.models.muse_spark.amd.specialized_decode.runtime.guards import (  # noqa: E402
    FULL,
    SLIDING,
    FailStop,
    kernel_layers,
    refuse_caches,
    refuse_model,
)
from vllm.models.muse_spark.amd.specialized_decode.runtime.layers import (  # noqa: E402
    KernelLayer,
    Phases,
)
from vllm.models.muse_spark.common.layers import (  # noqa: E402
    MuseSpark1xNormPreservingResidualConnection,
    MuseSpark1xResidualStream,
    MuseSpark1xRMSNorm,
    MuseSpark1xRMSNormWithInputScale,
)
from vllm.v1.attention.backends.registry import AttentionBackendEnum  # noqa: E402
from vllm.v1.attention.backends.rocm_aiter_fa import (  # noqa: E402
    AiterFlashAttentionImpl,
)

from .conftest import dspark, vllm_config  # noqa: E402
from .reference import FakeMrv2  # noqa: E402

BF, F32, U8 = torch.bfloat16, torch.float32, torch.uint8
ALL_WIDTHS = (1, 2, 3, 4, 5, 6, 7, 8)


# --------------------------------------------------------------------------
# Stage 1: the switch and the config


def test_auto_runs_the_stock_decode(monkeypatch):
    """Without the switch nothing is checked, even on an unsupported deployment."""

    def fail(_):
        raise AssertionError("checked")

    monkeypatch.setattr(runtime, "refuse_config", fail)
    monkeypatch.setattr(runtime, "unsupported", fail)
    cfg = vllm_config(
        additional_config={}, **{"parallel_config.tensor_parallel_size": 4}
    )
    assert specialized_decode_config(cfg) is None
    cfg.additional_config = {"decode_backend": "auto"}
    assert specialized_decode_config(cfg) is None
    cfg.additional_config = None
    assert runtime.decode_backend(cfg) == "auto"
    assert not runtime.requested(cfg)
    assert specialized_decode_config(cfg) is None


def test_monokernel_flag_enables(on_mi355x):
    cfg = vllm_config(additional_config={}, **{"kernel_config.monokernel": True})
    assert runtime.requested(cfg)
    assert specialized_decode_config(cfg) == DecodeConfig(q=1, widths=ALL_WIDTHS)


@pytest.mark.parametrize("monokernel", [False, True])
def test_an_unknown_backend_is_refused(monokernel):
    cfg = vllm_config(
        additional_config={"decode_backend": "fast"},
        **{"kernel_config.monokernel": monokernel},
    )
    with pytest.raises(ValueError, match="decode_backend must be"):
        specialized_decode_config(cfg)


def test_supported_configs(on_mi355x, monkeypatch):
    assert specialized_decode_config(vllm_config()) == DecodeConfig(1, ALL_WIDTHS)
    # widths the scheduler can never batch are not built
    cfg = vllm_config(**{"scheduler_config.max_num_seqs": 2})
    assert specialized_decode_config(cfg) == DecodeConfig(1, (1, 2))
    # DSpark, K = 7: one request a launch
    cfg = vllm_config(speculative_config=dspark())
    assert specialized_decode_config(cfg) == DecodeConfig(8, (8,))
    cfg = vllm_config(
        speculative_config=dspark(), **{"scheduler_config.max_num_seqs": 1}
    )
    assert specialized_decode_config(cfg) == DecodeConfig(8, (8,))
    # breakable CUDA graphs are not a refusal
    monkeypatch.setenv("VLLM_USE_BREAKABLE_CUDAGRAPH", "1")
    assert specialized_decode_config(vllm_config()) is not None
    # FULL_DECODE_ONLY too
    cfg = vllm_config(
        **{"compilation_config.cudagraph_mode": CUDAGraphMode.FULL_DECODE_ONLY}
    )
    assert specialized_decode_config(cfg) is not None
    with pytest.raises(ValueError, match="num_speculative_tokens 3"):
        specialized_decode_config(
            vllm_config(speculative_config=dspark(num_speculative_tokens=3))
        )


def test_expert_parallelism_selects_the_tep8_build(on_mi355x):
    # attention stays TP8; the experts go expert-parallel (linear placement)
    assert moe_mode(vllm_config()) == "tp"
    ep = vllm_config(**{"parallel_config.enable_expert_parallel": True})
    assert moe_mode(ep) == "ep"
    assert refuse_config(ep) == []
    assert specialized_decode_config(ep) == DecodeConfig(1, ALL_WIDTHS)
    # EP still needs TP8 attention and no data parallelism
    for over, reason in (
        ({"parallel_config.tensor_parallel_size": 4}, "TP 4"),
        ({"parallel_config.data_parallel_size": 2}, "data parallelism"),
        ({"parallel_config.enable_eplb": True}, "EPLB"),
    ):
        bad = vllm_config(**{"parallel_config.enable_expert_parallel": True, **over})
        assert any(reason in r for r in refuse_config(bad)), over


def _text(**over):
    from .conftest import muse_text_config

    return muse_text_config(**over)


_KINDS = _text().layer_types
_REFUSALS = [
    # 1-9 parallelism
    ({"parallel_config.tensor_parallel_size": 4}, {}, "TP 4 (TP8 only)"),
    ({"parallel_config.pipeline_parallel_size": 2}, {}, "pipeline parallelism"),
    ({"parallel_config.data_parallel_size": 2}, {}, "data parallelism"),
    (
        {
            "parallel_config.enable_expert_parallel": True,
            "parallel_config.expert_placement_strategy": "round_robin",
        },
        {},
        "non-linear expert placement",
    ),
    ({"parallel_config.enable_eplb": True}, {}, "EPLB"),
    ({"parallel_config.decode_context_parallel_size": 8}, {}, "decode context"),
    ({"parallel_config.prefill_context_parallel_size": 2}, {}, "prefill context"),
    ({"parallel_config.use_ubatching": True}, {}, "DBO"),
    ({"parallel_config.use_sequence_parallel_moe": True}, {}, "sequence-parallel"),
    # 10-16 features
    ({"lora_config": object()}, {}, "LoRA"),
    ({"kv_transfer_config": object()}, {}, "KV transfer / KV offloading"),
    (
        {"aux_output_config.enable_return_routed_experts": True},
        {},
        "routed-expert capture",
    ),
    ({"model_config.enable_sleep_mode": True}, {}, "sleep mode"),
    ({"offload_config.uva.cpu_offload_gb": 4}, {}, "CPU weight offload"),
    ({"offload_config.prefetch.offload_group_size": 2}, {}, "CPU weight offload"),
    ({"weight_transfer_config": object()}, {}, "weight updates"),
    ({"model_config.enable_prompt_embeds": True}, {}, "prompt embeds"),
    # 17-20 speculative decoding
    ({"speculative_config": dspark(method="eagle3")}, {}, "method 'eagle3'"),
    (
        {"speculative_config": dspark(num_speculative_tokens=3)},
        {},
        "num_speculative_tokens 3 (0 or 7 only)",
    ),
    (
        {"speculative_config": dspark(enable_adaptive_verification=True)},
        {},
        "adaptive verification",
    ),
    ({"speculative_config": dspark(dynamic=True)}, {}, "dynamic speculative"),
    # 21-23 runner and graphs
    ({"use_v2_model_runner": False}, {}, "V1 model runner"),
    (
        {"compilation_config.mode": CompilationMode.STOCK_TORCH_COMPILE},
        {},
        "torch.compile",
    ),
    (
        {"compilation_config.cudagraph_mode": CUDAGraphMode.PIECEWISE},
        {},
        "PIECEWISE decode CUDA graphs",
    ),
    # 24-28 dtypes, KV cache, attention
    ({"model_config.dtype": torch.float16}, {}, "model dtype torch.float16"),
    ({"cache_config.cache_dtype": "fp8"}, {}, "KV cache dtype 'fp8'"),
    ({"cache_config.block_size": 32}, {}, "KV block size 32"),
    (
        {"attention_config.backend": AttentionBackendEnum.TRITON_ATTN},
        {},
        "attention backend TRITON_ATTN",
    ),
    ({"attention_config.backend": None}, {}, "attention backend auto"),
    (
        {
            "attention_config.backend_per_kind": {
                "sliding_window": AttentionBackendEnum.TRITON_ATTN
            }
        },
        {},
        "attention backend sliding_window=TRITON_ATTN",
    ),
    ({}, {"shuffle": True}, "the AITER shuffled KV cache layout"),
    # 29-31 MoE env, quantization
    ({}, {"VLLM_ROCM_MUSE_MOE_REDUCE_DTYPE": "fp16"}, "REDUCE_DTYPE='fp16'"),
    ({"quant_config": None}, {}, "quantization 'none'"),
    (
        {"quant_config": SimpleNamespace(get_name=lambda: "fp8")},
        {},
        "quantization 'fp8'",
    ),
    # 32-36 the model
    ({"model_config.hf_text_config.hidden_size": 4096}, {}, "model shapes"),
    ({"model_config.hf_text_config.num_shared_experts": 1}, {}, "model shapes"),
    (
        {"model_config.hf_text_config.gated_delta_net": SimpleNamespace(enable=True)},
        {},
        "gated_delta_net layers",
    ),
    (
        {"model_config.hf_text_config.mlp_layer_types": ["dense"] + ["sparse"] * 61},
        {},
        "dense MLP layers",
    ),
    (
        {"model_config.hf_text_config.layer_types": ["chunked_attention"] + _KINDS[1:]},
        {},
        "layer type 'chunked_attention'",
    ),
    (
        {
            "model_config.hf_text_config.layer_rope_theta": [500000.0] * 62,
        },
        {},
        "RoPE on full-attention layers",
    ),
    # 38-39
    ({}, {"VLLM_MOE_SKIP_PADDING": "0"}, "VLLM_MOE_SKIP_PADDING=0"),
    (
        {"parallel_config.enable_batch_sharded_sampling": True},
        {},
        "batch-sharded sampling",
    ),
]


@pytest.mark.parametrize(
    "over, env, reason",
    _REFUSALS,
    ids=[f"{i}-{r[2][:24]}" for i, r in enumerate(_REFUSALS)],
)
def test_refusals(on_mi355x, monkeypatch, over, env, reason):
    from vllm._aiter_ops import rocm_aiter_ops

    for k, v in env.items():
        if k == "shuffle":
            monkeypatch.setattr(
                rocm_aiter_ops,
                "is_shuffle_kv_cache_enabled",
                classmethod(lambda cls: True),
            )
        else:
            monkeypatch.setenv(k, v)
    reasons = runtime.refuse_config(vllm_config(**over))
    assert len(reasons) == 1 and reason in reasons[0], reasons


def test_platform_reasons_off_rocm(monkeypatch):
    from vllm.platforms import current_platform

    monkeypatch.setattr(current_platform, "is_rocm", lambda: False)
    assert runtime.platform_reasons() == ["this platform (ROCm only)"]


def test_every_refusal_is_reported_at_once(monkeypatch):
    monkeypatch.setattr(
        runtime, "platform_reasons", lambda: ["this GPU (gfx950 / MI355X only)"]
    )
    monkeypatch.setenv("VLLM_MOE_SKIP_PADDING", "1")
    cfg = vllm_config(
        **{
            "parallel_config.tensor_parallel_size": 4,
            "use_v2_model_runner": False,
            "lora_config": object(),
            "compilation_config.cudagraph_mode": CUDAGraphMode.PIECEWISE,
        }
    )
    with pytest.raises(ValueError) as e:
        specialized_decode_config(cfg)
    msg = str(e.value)
    assert msg.startswith("Muse Spark specialized decode (--monokernel")
    for reason in (
        "TP 4",
        "V1 model runner",
        "this GPU",
        "LoRA",
        "PIECEWISE decode CUDA graphs",
        "decode_backend=auto without --monokernel",
    ):
        assert reason in msg


# --------------------------------------------------------------------------
# Stage 2: the loaded model


def _meta(*shape, dtype=BF):
    return torch.empty(*shape, dtype=dtype, device="meta")


def _norm(n: int, eps: float, *, weight=True, post=False, before=False):
    cls = MuseSpark1xRMSNormWithInputScale if before else MuseSpark1xRMSNorm
    norm = cls(
        n,
        eps=eps,
        has_weight=weight,
        is_post_norm=post,
        zero_centered_gamma=True,
        post_norm_gain_center_type="one",
        residual_id=None if post else 3,
    )
    if weight:
        norm.weight = nn.Parameter(torch.zeros(n, dtype=BF), requires_grad=False)
    return norm


def _connection(ready=True):
    conn = MuseSpark1xNormPreservingResidualConnection(spec.HIDDEN, temperature=0.3)
    conn.gate.data.zero_()
    if ready:
        conn.materialize_coefficients()
    return conn


def _moe(backend=Mxfp4MoeBackend.AITER_MXFP4_MXFP4, quantized=True):
    """A ``MuseSpark1xMoE`` with Muse's loaded tensors (meta) and its quant
    method: AITER MXFP4 x MXFP4 by default."""
    moe = MuseSpark1xMoE.__new__(MuseSpark1xMoE)
    nn.Module.__init__(moe)
    if quantized:
        method = MuseSparkQuarkOCPMXFusedMoE.__new__(MuseSparkQuarkOCPMXFusedMoE)
        method.mxfp4_backend = backend
    else:
        method = SimpleNamespace()  # the stock BF16 experts
    w13, w2 = _meta(256, 1024, 2048, dtype=U8), _meta(256, 4096, 256, dtype=U8)
    w13.is_shuffled = w2.is_shuffled = True
    moe.experts = SimpleNamespace(
        routed_experts=SimpleNamespace(
            quant_method=method,
            w13_weight=w13,
            w13_weight_scale=_meta(256, 1024, 128, dtype=U8),
            w2_weight=w2,
            w2_weight_scale=_meta(256, 4096, 16, dtype=U8),
            expert_map=None,
            activation=MoEActivation.SILU,
        )
    )
    moe.router = SimpleNamespace(weight=_meta(256, 8192, dtype=F32))
    # as MuseSpark1xMoE.fold_router_output_multiplier leaves it: the multiplier
    # stays set, the flag records the fold
    moe.router_output_multiplier = 0.1767
    moe._router_multiplier_folded = True
    moe.expert_bias = _meta(256, dtype=F32)
    moe.route_scale = 1.0
    moe.is_sequence_parallel = False
    moe.pooler = SimpleNamespace(weight=_meta(4096, 8192))
    moe.pre_expert_norm = _norm(4096, 1e-5)
    moe.post_expert_norm = _norm(4096, 1e-8, post=True, before=True)
    moe.unpooler = SimpleNamespace(weight=_meta(8192, 4096))
    return moe


def _impl(kind: str):
    """The checkpoint's ``AiterFlashAttentionImpl`` options for a ``kind``
    layer (model.py: sliding window 2048, scale 0.125 x qk_scale_factor)."""
    impl = AiterFlashAttentionImpl.__new__(AiterFlashAttentionImpl)
    impl.sliding_window = (2047, 0) if kind == SLIDING else (-1, -1)
    impl.sinks = impl.alibi_slopes = impl.kv_sharing_target_layer_name = None
    impl.logits_soft_cap = 0.0
    impl.num_queries_per_kv = 8
    impl.scale = 64**-0.5 * 5.4730064863875
    return impl


class _Attn:
    """``MuseSpark1xAttention`` as the kernel layers see it (a ``kind``
    layer: RoPE and the window on sliding layers)."""

    def __init__(self, i: int, kv=None, kind=SLIDING):
        self.o_proj = SimpleNamespace(weight=_meta(8192, 1024))
        self.qkv_proj = SimpleNamespace(weight=_meta(2304, 8192))
        self.num_heads, self.head_dim, self.num_kv_heads = 16, 64, 2
        self.fused_qk_norm_rope_kv_insert = True
        self.pre_qkv_proj_norm = _norm(8192, 1e-5)
        self.qk_norm = SimpleNamespace(has_weight=False)
        self.o_norm = SimpleNamespace(has_weight=False)
        self.fuse_o_norm_gate, self.rms_norm_eps = True, 1e-5
        self.use_attn_temperature_tuning = False
        self.rotary_emb = (
            SimpleNamespace(
                is_neox_style=False,
                rotary_dim=64,
                head_size=64,
                cos_sin_cache=torch.zeros(1024, 64),
            )
            if kind == SLIDING
            else None
        )
        self.attn = SimpleNamespace(
            impl=_impl(kind),
            layer_name=f"model.layers.{i}.self_attn.attn",
            kv_cache=torch.tensor([]) if kv is None else kv,
        )
        self.calls: list = []

    def __call__(self, positions, hidden_states, skip_o_proj=False):
        assert skip_o_proj
        self.calls.append((positions, hidden_states))
        return torch.zeros(hidden_states.size(0), spec.CORE, dtype=BF)

    def front(self, positions, hidden_states):
        self.calls.append(("front", positions, hidden_states))
        return torch.zeros(hidden_states.size(0), spec.QKV_ROWS, dtype=BF)

    def attend(self, qkv, gate=True):
        self.calls.append(("attend", qkv, gate))
        return torch.zeros(qkv.size(0), spec.CORE, dtype=BF)


def _kind(i: int) -> str:
    return FULL if _KINDS[i] == "full_attention" else SLIDING


def _layer(i: int, moe=None, kv=None):
    return SimpleNamespace(
        layer_idx=i,
        self_attn=_Attn(i, kv, _kind(i)),
        feed_forward=_moe() if moe is None else moe,
        add_attn=_connection(),
        add_ffn=_connection(),
        post_attn_norm=_norm(8192, 1e-8, weight=False, post=True),
        pre_ffn_norm=_norm(8192, 1e-5),
        post_ffn_norm=_norm(8192, 1e-8, post=True, before=True),
    )


def _backbone(n=4, unquantized=(0,), kinds=None):
    kinds = kinds or _KINDS
    layers = [
        _layer(i, _moe(quantized=False) if i in unquantized else None) for i in range(n)
    ]
    return SimpleNamespace(
        layers=layers,
        start_layer=0,
        end_layer=n,
        config=SimpleNamespace(layer_types=kinds[:n]),
    )


def test_refuse_model_accepts_the_checkpoint_layout():
    bb = _backbone()
    layers = kernel_layers(bb)
    assert layers == {1: FULL, 2: SLIDING, 3: SLIDING}
    assert refuse_model(bb, layers) == []
    # the fp4 / e8m0 dtypes of a later aiter are accepted too
    if hasattr(torch, "float4_e2m1fn_x2"):
        X = bb.layers[2].feed_forward.experts.routed_experts
        X.w13_weight = _meta(256, 1024, 2048, dtype=torch.float4_e2m1fn_x2)
        X.w13_weight.is_shuffled = True
        X.w2_weight_scale = _meta(256, 4096, 16, dtype=torch.float8_e8m0fnu)
        assert refuse_model(bb, layers) == []


def _to_ep(bb, rank, kind="mask"):
    """Every kernel layer's experts as TEP8 loads them on TP rank ``rank``: 32
    whole experts (w13 / w2 at the full intermediate) and the linear expert
    map, as AITER's 0 / 1 mask (``kind="mask"``, a trailing sentinel) or vLLM's
    global -> local ids (``kind="ids"``)."""
    if kind == "mask":
        emap = torch.zeros(257, dtype=torch.int64)
        emap[32 * rank : 32 * rank + 32] = 1
    else:
        emap = torch.full((256,), -1, dtype=torch.int64)
        emap[32 * rank : 32 * rank + 32] = torch.arange(32)
    for L in bb.layers:
        X = L.feed_forward.experts.routed_experts
        X.w13_weight = _meta(32, 8192, 2048, dtype=U8)
        X.w13_weight_scale = _meta(32, 8192, 128, dtype=U8)
        X.w2_weight = _meta(32, 4096, 2048, dtype=U8)
        X.w2_weight_scale = _meta(32, 4096, 128, dtype=U8)
        X.w13_weight.is_shuffled = X.w2_weight.is_shuffled = True
        X.expert_map = emap


@pytest.mark.parametrize("kind", ["mask", "ids"])
def test_refuse_model_accepts_the_tep8_layout(kind, monkeypatch):
    monkeypatch.setattr(
        "vllm.distributed.get_tp_group", lambda: SimpleNamespace(rank_in_group=5)
    )
    bb = _backbone()
    layers = kernel_layers(bb)
    _to_ep(bb, 5, kind)
    assert refuse_model(bb, layers, moe="ep") == []
    # the TP8 kernel reads neither the shapes nor the map
    assert refuse_model(bb, layers, moe="tp") != []


@pytest.mark.parametrize(
    "what, reason",
    [
        ("tp weights", "layer 1 w13: (256, 1024, 2048)"),
        ("no map", "the expert map is not this rank's linear 32"),
        ("other rank's map", "the expert map is not this rank's linear 32"),
        ("round-robin map", "the expert map is not this rank's linear 32"),
    ],
)
def test_refuse_model_names_every_tep8_mismatch(what, reason, monkeypatch):
    monkeypatch.setattr(
        "vllm.distributed.get_tp_group", lambda: SimpleNamespace(rank_in_group=5)
    )
    bb = _backbone()
    layers = kernel_layers(bb)
    _to_ep(bb, 5)
    X = bb.layers[1].feed_forward.experts.routed_experts
    if what == "tp weights":
        X.w13_weight = _meta(256, 1024, 2048, dtype=U8)
        X.w13_weight.is_shuffled = True
    elif what == "no map":
        X.expert_map = None
    elif what == "other rank's map":
        X.expert_map = torch.zeros(257, dtype=torch.int64)
        X.expert_map[32 * 4 : 32 * 4 + 32] = 1
    else:  # rank 5 of a round-robin placement: experts 5, 13, 21, ...
        X.expert_map = torch.zeros(257, dtype=torch.int64)
        X.expert_map[5:256:8] = 1
    reasons = [r for r in refuse_model(bb, layers, moe="ep") if r.startswith("layer 1")]
    assert len(reasons) == 1 and reason in reasons[0], reasons


def _break(bb, what):
    L = bb.layers[2]
    F, A = L.feed_forward, L.self_attn
    X = F.experts.routed_experts
    if what == "multiplier":
        F._router_multiplier_folded = False
    elif what == "coefficients":
        L.add_attn = _connection(ready=False)
    elif what == "o_proj":
        A.o_proj.weight = _meta(8192, 2048)
    elif what == "o_proj strided":
        A.o_proj.weight = _meta(1024, 8192).t()
    elif what == "w13 shuffle":
        X.w13_weight = _meta(256, 1024, 2048, dtype=U8)
    elif what == "insert":
        A.fused_qk_norm_rope_kv_insert = False
    elif what == "impl":
        A.attn.impl = SimpleNamespace()
    elif what == "expert map":
        X.expert_map = torch.arange(256)
    elif what == "router dtype":
        F.router.weight = _meta(256, 8192)
    elif what == "missing pooler":
        del F.pooler
    elif what == "pre_ffn eps":
        L.pre_ffn_norm.variance_epsilon = 1e-6
    elif what == "post_expert side":
        F.post_expert_norm = _norm(4096, 1e-8, post=True, before=False)
    elif what == "post_attn weight":
        L.post_attn_norm = _norm(8192, 1e-8, post=True)
    elif what == "activation":
        X.activation = MoEActivation.GELU
    elif what == "route scale":
        F.route_scale = 2.5
    elif what == "heads":
        A.num_kv_heads = 1
    elif what == "pre_expert variance size":
        F.pre_expert_norm.variance_size_override = 2048
    elif what == "post_attn variance size":
        L.post_attn_norm.variance_size_override = 4096
    elif what == "prev add_ffn":
        bb.layers[1].add_ffn = _connection(ready=False)
    else:
        raise AssertionError(what)


@pytest.mark.parametrize(
    "what, reason",
    [
        ("multiplier", "layer 2: router output multiplier not folded"),
        ("coefficients", "layer 2: add_attn coefficients not materialized"),
        (
            "o_proj",
            "layer 2 o_proj: (8192, 2048) torch.bfloat16, kernel reads contiguous "
            "(8192, 1024) torch.bfloat16",
        ),
        ("o_proj strided", "layer 2 o_proj: (8192, 1024) torch.bfloat16 strided"),
        ("w13 shuffle", "layer 2: w13 not AITER-shuffled"),
        ("insert", "layer 2: the fused QK-norm/RoPE/KV insert is off"),
        ("impl", "layer 2: attention impl SimpleNamespace (ROCM_AITER_FA only)"),
        ("expert map", "layer 2: an expert map (EP)"),
        ("router dtype", "layer 2 router: (256, 8192) torch.bfloat16"),
        ("missing pooler", "layer 2 pooler: missing"),
        ("pre_ffn eps", "layer 2: pre_ffn_norm (gain 1, eps 1e-5, weight after)"),
        (
            "post_expert side",
            "layer 2: post_expert_norm (gain 1, eps 1e-8, weight first)",
        ),
        ("post_attn weight", "layer 2: post_attn_norm (weightless, eps 1e-8)"),
        ("activation", "layer 2: expert activation"),
        ("route scale", "layer 2: route_scale 2.5"),
        ("heads", "layer 2: attention heads"),
        (
            "pre_expert variance size",
            "layer 2: pre_expert_norm (gain 1, eps 1e-5, weight after)",
        ),
        ("post_attn variance size", "layer 2: post_attn_norm (weightless, eps 1e-8)"),
        ("prev add_ffn", "layer 1: add_ffn coefficients not materialized"),
    ],
)
def test_refuse_model_names_every_mismatch(what, reason):
    bb = _backbone()
    layers = kernel_layers(bb)
    _break(bb, what)
    reasons = refuse_model(bb, layers)
    assert len(reasons) == 1 and reasons[0].startswith(reason), reasons


# --------------------------------------------------------------------------
# Stage 3: the KV caches


def _cfg_mode(mode):
    return SimpleNamespace(compilation_config=SimpleNamespace(cudagraph_mode=mode))


def _kv(blocks=4, block=16, dtype=BF, width=128):
    return _meta(blocks, 2, block, width, dtype=dtype)


def test_refuse_caches_accepts_real_layouts():
    bb = _backbone()
    layers = kernel_layers(bb)
    # one pool shared by every layer of a KV group (aliasing), ~11 GB a view
    pool = _kv(blocks=(11 << 30) // (2 * 16 * 128 * 2))
    for i in layers:
        bb.layers[i].self_attn.attn.kv_cache = pool
    assert refuse_caches(bb, layers, _cfg_mode(CUDAGraphMode.FULL_DECODE_ONLY)) == []
    assert refuse_caches(bb, layers, _cfg_mode(CUDAGraphMode.NONE)) == []


@pytest.mark.parametrize(
    "kv, mode, reason",
    [
        (_kv(dtype=torch.float8_e4m3fn), None, "layer 2: KV cache torch.float8_e4m3fn"),
        (_kv(dtype=torch.float16), None, "layer 2: KV cache torch.float16 (bf16"),
        (_meta(4, 1, 16, 128), None, "layer 2: KV cache shape (4, 1, 16, 128)"),
        (_meta(4, 2, 16, 64), None, "layer 2: KV cache shape (4, 2, 16, 64)"),
        (_kv(block=32), None, "layer 2: KV kernel block 32 (--block-size 16 only)"),
        (_meta(4, 2, 16, 256)[..., ::2], None, "layer 2: KV cache not dense"),
        (None, CUDAGraphMode.PIECEWISE, "PIECEWISE decode CUDA graphs"),
    ],
)
def test_refuse_caches(kv, mode, reason):
    bb = _backbone()
    layers = kernel_layers(bb)
    for i in layers:
        bb.layers[i].self_attn.attn.kv_cache = _kv()
    if kv is not None:
        bb.layers[2].self_attn.attn.kv_cache = kv
    reasons = refuse_caches(
        bb, layers, _cfg_mode(mode or CUDAGraphMode.FULL_AND_PIECEWISE)
    )
    assert len(reasons) == 1 and reasons[0].startswith(reason), reasons


# --------------------------------------------------------------------------
# plan(): which steps run the kernels


def _md(width: int, q: int, **over):
    m = SimpleNamespace(
        decode_metadata=SimpleNamespace(max_query_len=q),
        num_prefills=0,
        num_extends=0,
        num_decode_tokens=width,
        num_decodes=width // q,
    )
    for k, v in over.items():
        setattr(m, k, v)
    return m


NAMES = {FULL: "full.attn", SLIDING: "sliding.attn"}


def _metadata(width: int, q: int, full=None, **over):
    """Both KV kinds' metadata of a step: identical unless ``full`` is given."""
    return {
        "sliding.attn": _md(width, q, **over),
        "full.attn": _md(width, q, **(full or over)),
        "other": object(),
    }


@pytest.mark.parametrize(
    "width, q, md, why",
    [
        (1, 1, _metadata(1, 1), None),
        (8, 1, _metadata(8, 1), None),
        (8, 8, _metadata(8, 8), None),  # one DSpark verify of 8 rows
        (4, 1, _metadata(4, 1, num_prefills=1), "not pure decode"),
        (4, 1, _metadata(4, 1, num_extends=1), "not pure decode"),
        (4, 1, _metadata(4, 1, full={"num_prefills": 1}), "not pure decode"),
        (
            8,
            8,
            _metadata(8, 8, decode_metadata=SimpleNamespace(max_query_len=4)),
            "not uniform",
        ),  # fmt: skip
        (4, 1, _metadata(4, 1, num_decode_tokens=3), "not uniform / padded width"),
        (4, 1, _metadata(4, 1, num_decodes=3), "not uniform / padded width"),
        (4, 1, _metadata(4, 1, decode_metadata=None), "no decode metadata"),
        (4, 1, {"sliding.attn": _md(4, 1)}, "no decode metadata"),
        (4, 1, None, "no metadata"),
        (4, 1, [_metadata(2, 1), _metadata(2, 1)], "no metadata"),  # DBO
    ],
)
def test_step_reason(width, q, md, why):
    assert step_reason(md, NAMES, width, q) == why


class _Kernel:
    def __call__(self, step, positions, x):
        return "kernel", step.width, x


def _decode(monkeypatch, mode=CUDAGraphMode.NONE, md=None, q=1, is_padding=True):
    """An executor with kernels on layers 1 and 2 at its widths, bound to a KV
    cache, under a forward context of ``mode`` (MRV2 captures FULL graphs and
    warms up with NONE)."""
    decode = object.__new__(SpecializedDecode)
    widths = (8,) if q == 8 else (1, 2, 4, 8)
    decode.config = DecodeConfig(q=q, widths=widths)
    decode.names = NAMES
    decode.runner = SimpleNamespace(no_pad=torch.zeros(16, dtype=U8))
    decode.kernels = {w: {1: _Kernel(), 2: _Kernel()} for w in widths}
    decode.reads_kv = False  # Phase A: no metadata read (_check_metadata)
    decode._kv = (1, 2)
    monkeypatch.setattr(SpecializedDecode, "_kv_key", lambda self: (1, 2))
    context = SimpleNamespace(
        cudagraph_runtime_mode=mode,
        attn_metadata=_metadata(4, q) if md is None else md,
        slot_mapping={"full.attn": torch.zeros(4, dtype=torch.int64)},
        is_padding=torch.ones(8, dtype=torch.bool) if is_padding else None,
    )
    monkeypatch.setattr(decode_mod, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(decode_mod, "get_forward_context", lambda: context)
    return decode, context


@pytest.mark.parametrize("mode", [CUDAGraphMode.NONE, CUDAGraphMode.FULL])
def test_plan_runs_the_kernels_of_its_width(monkeypatch, mode):
    decode, ctx = _decode(monkeypatch, mode)
    step = decode.plan(torch.zeros(4), None)
    assert step is not None and step.width == 4
    assert step.kernels is decode.kernels[4] and step.attn_metadata is ctx.attn_metadata
    assert step.pad is ctx.is_padding
    assert step.layer(1)(None, "x") == ("kernel", 4, "x")
    assert step.layer(0) is None and step.layer(3) is None
    # M-RoPE positions [3, T]: the width is the last dim
    assert decode.plan(torch.zeros(3, 4), None) is not None


def test_plan_runs_a_dspark_verify_step(monkeypatch):
    decode, _ = _decode(monkeypatch, md=_metadata(8, 8), q=8)
    step = decode.plan(torch.zeros(8), None)
    assert step is not None and step.width == 8
    decode, _ = _decode(monkeypatch, md=_metadata(4, 4), q=8)
    assert decode.plan(torch.zeros(4), None) is None


@pytest.mark.parametrize(
    "mode, tokens, md, intermediate",
    [
        (CUDAGraphMode.PIECEWISE, 4, None, None),
        (CUDAGraphMode.NONE, 16, _metadata(16, 1), None),  # no kernels that wide
        (CUDAGraphMode.NONE, 3, _metadata(3, 1), None),  # no width 3 built here
        (CUDAGraphMode.NONE, 4, _metadata(4, 1, num_prefills=1), None),
        (CUDAGraphMode.FULL, 4, _metadata(4, 1, num_extends=1), None),
        (CUDAGraphMode.NONE, 4, None, object()),  # a pipeline stage's input
        (CUDAGraphMode.NONE, 4, {}, None),  # a profile / dummy run
    ],
)
def test_plan_runs_other_steps_stock(monkeypatch, mode, tokens, md, intermediate):
    decode, _ = _decode(monkeypatch, mode, md)
    assert decode.plan(torch.zeros(tokens), intermediate) is None


# --------------------------------------------------------------------------
# The layer kernel and the layer loop


class _Runner:
    """A ``MonoRunner`` as the layer kernel sees it, recording the step bumps."""

    def __init__(self, calls, first_layer=1):
        self.calls, self.first_layer, self.fault = calls, first_layer, {}
        self.scratch = torch.zeros(8, dtype=U8)
        self.epoch = torch.zeros(1, dtype=torch.int32)
        self.health = torch.zeros(8, dtype=torch.int32)
        self.peer_table = torch.zeros(8, dtype=torch.int64)
        self.alarm, self.rank = 1234, 3

    def step_begin(self):
        self.calls.append("step_begin")
        self.epoch.add_(1)

    launch_args = runner_mod.MonoRunner.launch_args
    peer_args = runner_mod.MonoRunner.peer_args


def _stream(layer, width=2):
    return MuseSpark1xResidualStream(
        stream=torch.zeros(width, spec.HIDDEN),
        branch=torch.zeros(width, spec.HIDDEN, dtype=BF),
        post_norm=layer.post_ffn_norm,
        connection=layer.add_ffn,
    )


# --------------------------------------------------------------------------
# The model hooks and the model state


def _causal_lm(config=None, decode=None):
    m = object.__new__(MuseSpark1xForCausalLM)
    nn.Module.__init__(m)
    m.init_specialized_decode(config)
    m.specialized_decode = decode
    return m


# --------------------------------------------------------------------------
# Fail-stop, the runner, the epoch


class _Event:
    def record(self):
        pass

    def query(self):
        return True


def _failstop(monkeypatch, others_failed=lambda step: False):
    real_zeros = torch.zeros
    monkeypatch.setattr(
        torch, "zeros", lambda *a, pin_memory=False, **k: real_zeros(*a, **k)
    )
    monkeypatch.setattr(torch.cuda, "Event", _Event)
    capturing = {"on": False}
    monkeypatch.setattr(
        torch.cuda, "is_current_stream_capturing", lambda: capturing["on"]
    )
    votes = []

    def all_reduce(t, group=None):
        votes.append(t.clone())
        if others_failed(len(votes)):
            t[5] = 1

    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)
    failed = real_zeros(1, dtype=torch.int32)
    return FailStop(failed, rank=2, world=8, group="gloo"), failed, capturing, votes


def test_failstop_votes_and_raises(monkeypatch):
    stop, failed, capturing, votes = _failstop(monkeypatch)
    for _ in range(FailStop.CHECK_EVERY):
        stop.after_step()
    assert len(votes) == 1 and not votes[0].any()
    # a capture runs compute_logits without a step: not counted
    capturing["on"] = True
    for _ in range(3 * FailStop.CHECK_EVERY):
        stop.after_step()
    capturing["on"] = False
    failed[0] = 1  # this rank gave up a wait
    for _ in range(FailStop.CHECK_EVERY - 1):
        stop.after_step()
    with pytest.raises(RuntimeError, match=r"TP ranks \[2\] gave up"):
        stop.after_step()
    assert len(votes) == 2 and int(votes[1][2]) == 1


# --------------------------------------------------------------------------
# The executor's lifecycle (build, KV binds, self-test, capture), single rank


class _ExecRunner:
    """``MonoRunner`` without peer memory (a single CPU rank)."""

    def __init__(self, device):
        from vllm.models.muse_spark.amd.specialized_decode.mono.abi import (
            HEALTH_WORDS,
            STARTUP_WAIT_SECONDS,
            WAIT_BUDGET,
        )

        self.scratch = torch.zeros(8, dtype=U8)
        self.epoch = torch.zeros(1, dtype=torch.int32)
        self.health = torch.zeros(HEALTH_WORDS, dtype=torch.int32)
        self.health[WAIT_BUDGET] = STARTUP_WAIT_SECONDS
        self.no_pad = torch.zeros(runner_mod.NO_PAD_BYTES, dtype=U8)
        self.peer_table = torch.zeros(8, dtype=torch.int64)
        self.rank, self.alarm, self.fault, self.first_layer = 0, 0, {}, -1
        self.steps, self.epoch_mark = 0, None

    step_begin = runner_mod.MonoRunner.step_begin
    set_wait_budget = runner_mod.MonoRunner.set_wait_budget
    reset_count = runner_mod.MonoRunner.reset_count
    launch_args = runner_mod.MonoRunner.launch_args
    peer_args = runner_mod.MonoRunner.peer_args


def _executor(monkeypatch, fill=0.0, compile_error=None, extra=None, q=1):
    """A ``SpecializedDecode`` over a 4-layer fake backbone (kernel layers 1..3)
    at widths 1..3 (8 under DSpark), with the kernel modules, the TP group and
    the fail-stop stubbed. ``fill``: what the stub launches write into
    ``branch_out`` (and K1's ``qkvo``); ``extra``: more ``additional_config``
    (the phases)."""
    from vllm.models.muse_spark.amd.specialized_decode.mono.abi import FAILED

    bb = _backbone()
    bb.parameters = lambda: iter([torch.zeros(1)])
    compiled, launches, logs = [], [], []

    def compile_build(key):
        if compile_error is not None:
            raise compile_error
        compiled.append(key)

    def launch(key, args):
        tokens = key.tokens
        pad = args["pad"][:tokens]
        launches.append((key, int(args["epoch"][0]), bool(pad.all()), args))
        assert set(args) == set(spec.K2_ARGS_BY_CORE[key.core])
        args["stream_out"].zero_()
        args["branch_out"].fill_(0.0 if bool(pad.all()) else fill)

    def launch_k1(key, args):
        pad = args["pad"][: key.tokens]
        launches.append((key, int(args["epoch"][0]), bool(pad.all()), args))
        assert set(args) == set(spec.K1_ARGS)
        args["stream_out"].zero_()
        args["qkvo"].fill_(0.0 if bool(pad.all()) else fill)

    tp = SimpleNamespace(cpu_group=None, world_size=1, rank_in_group=0)
    monkeypatch.setattr(decode_mod, "get_tp_group", lambda: tp)
    monkeypatch.setattr(decode_mod, "MonoRunner", _ExecRunner)
    monkeypatch.setattr(decode_mod, "compile_build", compile_build)
    monkeypatch.setattr(layers_mod, "k2_launch", launch)
    monkeypatch.setattr(layers_mod, "k1_launch", launch_k1)
    monkeypatch.setattr(
        decode_mod, "FailStop", lambda failed, rank, world, group: failed
    )
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda *a: None)
    monkeypatch.setattr(decode_mod.logger, "info", lambda msg, *a: logs.append(a))
    cfg = vllm_config()
    cfg.additional_config.update(extra or {})
    widths = (1, 2, 3) if q == 1 else (8,)
    decode = SpecializedDecode(bb, DecodeConfig(q, widths), cfg)
    assert decode.failstop.data_ptr() == decode.runner.health[FAILED:].data_ptr()
    return decode, bb, compiled, launches, logs


def _bind(bb):
    full, sliding = (
        torch.zeros(2, 2, 16, 128, dtype=BF),
        torch.zeros(3, 2, 16, 128, dtype=BF),
    )
    for i in (1, 2, 3):
        bb.layers[i].self_attn.attn.kv_cache = full if i == 1 else sliding


def test_executor_refuses_before_any_collective(monkeypatch):
    with pytest.raises(ValueError, match="compiling the kernels failed"):
        _executor(monkeypatch, compile_error=RuntimeError("no gfx950"))
    monkeypatch.setattr(
        decode_mod, "refuse_model", lambda bb, layers, *a: ["layer 2: w13 not shuffled"]
    )
    monkeypatch.setattr(
        decode_mod,
        "MonoRunner",
        lambda device: pytest.fail("peer memory before the weights' vote"),
    )
    with pytest.raises(ValueError, match="the loaded weights: layer 2: w13"):
        SpecializedDecode(_backbone(), DecodeConfig(1, (1,)), vllm_config())


# --------------------------------------------------------------------------
# Mono off: the model's edits are inert (the stock decode, call for call)


def _recording_layers(n: int, calls: list):
    """``n`` stock decoder-layer stand-ins: each records its call and returns
    a fresh residual stream (so the next layer's input is identifiable)."""

    def make(i):
        def stock(*args, **kwargs):
            out = _stream(_layer(i))
            calls.append((i, args, kwargs, out))
            return out

        return stock

    return [make(i) for i in range(n)]


def _stock_backbone(layers, fuse_across_layers: bool):
    bb = object.__new__(MuseSpark1xModel)
    nn.Module.__init__(bb)
    bb.layers = layers
    bb.start_layer, bb.end_layer = 0, len(layers)
    bb.residual_dtype, bb.default_dtype = F32, BF
    bb.fuse_across_layers = fuse_across_layers
    bb.aux_hidden_state_layers = ()
    bb.norm = lambda x, out_dtype: ("norm", x)
    return bb


@pytest.mark.parametrize("fuse_across_layers", [True, False])
def test_mono_off_layer_loop_calls_exactly_the_stock_layers(
    monkeypatch, fuse_across_layers
):
    """With no step (the switch off), or a step whose layers all run stock,
    every decoder layer is called once, in order, exactly as the stock loop
    calls it: ``layer(positions, x)`` with the previous layer's output (or its
    unfused boundary), no keyword, and the final norm sees the last one."""
    monkeypatch.setattr(
        model_mod,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )
    unfused: list = []

    def unfuse(x):
        unfused.append((x, torch.zeros(2, spec.HIDDEN)))
        return unfused[-1][1]

    monkeypatch.setattr(model_mod, "norm_then_gated_residual", unfuse)
    monkeypatch.setattr(
        model_mod,
        "norm_residual_then_norm",
        lambda x, norm, out_dtype: (("final", x), None),
    )
    positions = torch.arange(2)
    embeds = torch.ones(2, spec.HIDDEN, dtype=BF)
    n = 4
    for step in (None, Step(2, {}, {}, {}, torch.zeros(16, dtype=U8))):
        calls: list = []
        unfused.clear()
        bb = _stock_backbone(_recording_layers(n, calls), fuse_across_layers)
        out = bb.forward(None, positions, None, inputs_embeds=embeds, step=step)
        assert [c[0] for c in calls] == list(range(n))
        for _, args, kwargs, _ in calls:
            assert kwargs == {} and len(args) == 2 and args[0] is positions
        first = calls[0][1][1]
        assert first.dtype == F32 and torch.equal(first, embeds.float())
        # what layer i returned is what layer i + 1 (or the final norm) got
        outs = [unfused[i][0] for i in range(len(unfused))]
        if fuse_across_layers:
            assert unfused == []
            for i in range(1, n):
                assert calls[i][1][1] is calls[i - 1][3]
            assert out == ("final", calls[-1][3])
        else:
            assert len(outs) == n
            assert all(o is c[3] for o, c in zip(outs, calls))
            for i in range(1, n):
                assert calls[i][1][1] is unfused[i - 1][1]
            assert out == ("norm", unfused[-1][1])


def test_mono_off_model_hooks_are_inert(monkeypatch):
    """The CausalLM's hooks with the switch off: forward hands the model no
    step, compute_logits polls nothing, process_weights_after_loading neither
    refuses nor materializes, and the attention runs o_proj by default."""
    from vllm.models.muse_spark.amd.model import MuseSpark1xAttention
    from vllm.models.muse_spark.common import layers as common_layers

    off = _causal_lm()  # specialized_decode_config None, no executor
    seen = []

    def model(*args, **kwargs):
        seen.append((args, kwargs))
        return "hidden"

    off.model = model
    pos = torch.zeros(4)
    assert off.forward("ids", pos) == "hidden"
    assert seen == [(("ids", pos, None, None), {"step": None})]
    off.lm_head = SimpleNamespace(weight=torch.zeros(1, dtype=BF))
    logits = torch.zeros(1, 3)
    off.logits_processor = lambda head, hidden: logits
    off.mul_tanh_soft_cap = None
    off.metap_output_multiplier = 1.0
    off.output_soft_cap_temp = None
    assert off.compute_logits(torch.zeros(1, 4)) is logits
    monkeypatch.setattr(
        common_layers,
        "materialize_residual_path_constants",
        lambda m: pytest.fail("materialized with the switch off"),
    )
    off.process_weights_after_loading()  # no error, nothing materialized

    # attention: o_proj runs unless skip_o_proj is asked for (default False)
    attn = object.__new__(MuseSpark1xAttention)
    nn.Module.__init__(attn)
    attn.num_heads, attn.head_dim = 2, 4
    attn.q_size, attn.kv_size = 8, 0
    attn.qkv_proj = lambda h: (h, None)
    attn._qk_norm_rope_kv_insert = lambda qkv, positions: (qkv, None, None, None)
    attn.fused_qk_norm_rope_kv_insert = True
    attn.attn = lambda q, k, v: q
    attn.fuse_o_norm_gate = False
    attn.use_o_norm = False
    attn.use_o_gate = False
    o_calls = []

    def o_proj(x):
        o_calls.append(x)
        return x * 2, None

    attn.o_proj = o_proj
    h = torch.ones(3, 8)
    assert torch.equal(attn(torch.zeros(3), h), h * 2) and len(o_calls) == 1
    assert torch.equal(attn(torch.zeros(3), h, skip_o_proj=True), h)
    assert len(o_calls) == 1


def test_importing_the_model_loads_no_kernel_module():
    """model.py imports only the switch and the hooks: no FlyDSL kernel module
    (mono.layer, the stages, the sync helpers) and no executor at import."""
    import subprocess

    code = (
        "import sys\n"
        "import vllm.models.muse_spark.amd.model\n"
        "pkg = 'vllm.models.muse_spark.amd.specialized_decode'\n"
        "bad = [m for m in sys.modules if m.startswith(pkg + '.mono.')\n"
        "       and m not in (pkg + '.mono.spec', pkg + '.mono.abi')]\n"
        "bad += [m for m in (pkg + '.runtime.decode', pkg + '.runtime.layers')\n"
        "        if m in sys.modules]\n"
        "assert not bad, bad\n"
    )
    r = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=300
    )
    assert r.returncode == 0, r.stderr[-4000:]


# --------------------------------------------------------------------------
# Phases B and C: the switch, the builds, compile dispatch, the per-layer
# binding, the refusals they add and the metadata invariants

K1_SPLIT = {"mono_front": "k1", "mono_core": "split"}


def test_phases_default_to_phase_a_and_are_validated():
    cfg = vllm_config()
    assert layers_mod.decode_phases(cfg) == Phases("stock", "tensor")
    cfg.additional_config = None
    assert layers_mod.decode_phases(cfg) == Phases()
    cfg.additional_config = {"mono_front": "k1", "mono_core": "gated"}
    assert layers_mod.decode_phases(cfg) == Phases("k1", "gated")
    for bad in ({"mono_front": "K1"}, {"mono_core": "fused"}):
        cfg.additional_config = bad
        with pytest.raises(ValueError, match="decode_backend=auto"):
            layers_mod.decode_phases(cfg)
    # only mono_front / mono_core are read: any other key is ignored
    cfg.additional_config = {"mono_core": "split", "mono_split_preset": "short"}
    assert layers_mod.decode_phases(cfg) == Phases(core="split")
    # the split core serves q = 1 only; DSpark verify keeps the gated core load
    split = Phases(core="split")
    assert split.k2_core(1) == "split" and split.k2_core(8) == "gated"
    assert split.reads_kv(1) and not split.reads_kv(8)
    assert Phases(front="k1").reads_kv(8)
    assert not Phases(core="gated").reads_kv(1) and not Phases().reads_kv(1)


@pytest.mark.parametrize("q", [1, 8])
@pytest.mark.parametrize(
    "front, core", [(f, c) for f in ("stock", "k1") for c in spec.K2_CORES]
)
def test_builds_per_width_kind_and_q(front, core, q):
    """K1Build(w, rope=sliding) with K1 on; K2Build(w, core, window=sliding) for
    the split core at q = 1, the gated core at q = 8 (DESIGN_BC 3.2)."""
    bb = _backbone()
    runner = _Runner([])
    phases = Phases(front, core)
    for w in (1, 5, 8) if q == 1 else (8,):
        for i in (1, 2):
            kind = _kind(i)
            k = KernelLayer(bb.layers[i], bb.layers[i - 1], runner, w, kind, q, phases)
            want_core = "gated" if core == "split" and q == 8 else core
            k2 = spec.K2Build(
                tokens=w,
                core=want_core,
                window=want_core == "split" and kind == SLIDING,
            )
            assert k.key() == k2 and k.core == want_core
            if front == "k1":
                k1 = spec.K1Build(tokens=w, rope=kind == SLIDING)
                assert k.k1_key() == k1 and k.builds() == [k1, k2]
            else:
                assert k.k1_key() is None and k.builds() == [k2]
    with pytest.raises(ValueError, match="attention kind"):
        KernelLayer(bb.layers[1], bb.layers[0], runner, 1, "global")


def _mrv2(bb):
    """MRV2's persistent buffers with two KV groups: layer 1 full, layers 2
    and 3 sliding (one group)."""
    name = lambda i: bb.layers[i].self_attn.attn.layer_name  # noqa: E731
    return FakeMrv2({"full": [name(1)], "sw": [name(2), name(3)]}, {"sw"})


def _break_bc(bb, what):
    L, P = bb.layers[2], bb.layers[1]
    A = L.self_attn
    if what == "qkv shape":
        A.qkv_proj.weight = _meta(2304, 4096)
    elif what == "qkv quantized":
        A.qkv_proj.weight_scale = _meta(2304, dtype=F32)
    elif what == "pre_qkv eps":
        A.pre_qkv_proj_norm.variance_epsilon = 1e-6
    elif what == "prev post_ffn eps":
        P.post_ffn_norm.variance_epsilon = 1e-6
    elif what == "prev add_ffn dtype":
        P.add_ffn.alpha = torch.zeros(spec.HIDDEN, dtype=BF)
    elif what == "qk_norm weight":
        A.qk_norm.has_weight = True
    elif what == "temperature":
        A.use_attn_temperature_tuning = True
    elif what == "window":
        A.attn.impl.sliding_window = (4095, 0)
    elif what == "neox":
        A.rotary_emb.is_neox_style = True
    elif what == "short table":
        A.rotary_emb.cos_sin_cache = torch.zeros(512, 64)
    elif what == "full rope":
        bb.layers[1].self_attn.rotary_emb = bb.layers[3].self_attn.rotary_emb
    elif what == "sinks":
        A.attn.impl.sinks = torch.zeros(16)
    elif what == "scale":
        A.attn.impl.scale = 0.125
    elif what == "gqa":
        A.attn.impl.num_queries_per_kv = 16
    else:
        raise AssertionError(what)


_BC_REFUSALS = [
    # what, the phases it needs, reason
    ("qkv shape", ("k1", "tensor"), "layer 2 qkv_proj: (2304, 4096)"),
    ("qkv quantized", ("k1", "tensor"), "layer 2: qkv_proj quantized"),
    (
        "pre_qkv eps",
        ("k1", "tensor"),
        "layer 2: pre_qkv_proj_norm (gain 1, eps 1e-5, weight after)",
    ),
    (
        "prev post_ffn eps",
        ("k1", "tensor"),
        "layer 2: previous post_ffn_norm (gain 1, eps 1e-8, weight first)",
    ),
    (
        "prev add_ffn dtype",
        ("k1", "tensor"),
        "layer 2: previous add_ffn coefficients not materialized",
    ),
    ("qk_norm weight", ("stock", "gated"), "layer 2: QK / output norm"),
    ("temperature", ("k1", "tensor"), "layer 2: QK / output norm"),
    ("window", ("stock", "split"), "layer 2: sliding attention (window / RoPE)"),
    ("neox", ("k1", "tensor"), "layer 2: sliding attention (window / RoPE)"),
    ("short table", ("k1", "tensor"), "layer 2: sliding attention (window / RoPE)"),
    ("full rope", ("stock", "split"), "layer 1: full attention (window / RoPE)"),
    ("sinks", ("stock", "split"), "layer 2: attention impl options"),
    ("scale", ("k1", "tensor"), "layer 2: attention impl options"),
    ("gqa", ("stock", "split"), "layer 2: attention impl options"),
]


@pytest.mark.parametrize("what, phases, reason", _BC_REFUSALS)
def test_phase_bc_refusals_name_exactly_their_reason(what, phases, reason):
    """DESIGN_BC 8.2: each row gives exactly its reason under the phases that
    read it, and none with Phase A (the default) -- Phase A serves what it
    served."""
    bb = _backbone()
    layers = kernel_layers(bb)
    for front, core in (("k1", "split"), ("k1", "gated"), ("stock", "gated")):
        assert refuse_model(bb, layers, front, core, 1024) == [], (front, core)
    _break_bc(bb, what)
    reasons = refuse_model(bb, layers, *phases, 1024)
    assert len(reasons) == 1 and reasons[0].startswith(reason), reasons
    reasons = refuse_model(bb, layers, "k1", "split", 1024)
    assert len(reasons) == 1 and reasons[0].startswith(reason), reasons
    assert refuse_model(bb, layers) == []


def _cache_cfg(block=16, mode=CUDAGraphMode.FULL_AND_PIECEWISE):
    return SimpleNamespace(
        compilation_config=SimpleNamespace(cudagraph_mode=mode),
        cache_config=SimpleNamespace(block_size=block),
    )


@pytest.mark.parametrize(
    "kv, block, reason",
    [
        (None, 32, "layer 2: KV manager block 32 != kernel block 16"),
        (
            _meta(4, 2, 16, 132)[..., :128],
            16,
            "layer 2: KV rows not 16-byte aligned / not contiguous in the head dim",
        ),
        (
            _meta(1, 2, 16, 2**26)[..., :128],
            16,
            "layer 2: KV page exceeds the 32-bit window",
        ),
    ],
)
def test_refuse_caches_phase_bc(kv, block, reason):
    """DESIGN_BC 8.3: only kernels that read the cache (K1, the split core)
    refuse these."""
    bb = _backbone()
    layers = kernel_layers(bb)
    for i in layers:
        bb.layers[i].self_attn.attn.kv_cache = _kv()
    assert refuse_caches(bb, layers, _cache_cfg(), reads_kv=True) == []
    if kv is not None:
        bb.layers[2].self_attn.attn.kv_cache = kv
    reasons = refuse_caches(bb, layers, _cache_cfg(block), reads_kv=True)
    if block != 16:
        assert len(reasons) == 3 and all(" != kernel block 16" in r for r in reasons)
    else:
        assert len(reasons) == 1, reasons
    assert reasons[-1 if block == 16 else 1].startswith(reason), reasons
    assert refuse_caches(bb, layers, _cache_cfg(block)) == []


def _meta_decode(monkeypatch, phases=None, q=1):
    """An executor over the fake backbone with MRV2's persistent buffers
    (FakeMrv2: groups full / sw) under a FULL capture's forward context."""
    phases = phases or Phases("k1", "split")
    bb = _backbone()
    mrv2 = _mrv2(bb)
    decode = object.__new__(SpecializedDecode)
    decode.backbone, decode.layers = bb, kernel_layers(bb)
    decode.config = DecodeConfig(q=q, widths=(2, 8))
    decode.vllm_config = vllm_config()
    decode.vllm_config.scheduler_config.max_num_seqs = 8
    decode.names = {
        FULL: bb.layers[1].self_attn.attn.layer_name,
        SLIDING: bb.layers[2].self_attn.attn.layer_name,
    }
    decode.runner = SimpleNamespace(no_pad=torch.zeros(16, dtype=U8))
    decode.kernels = {w: {1: _Kernel()} for w in (2, 8)}
    decode.reads_kv = phases.reads_kv(q)
    decode._kv, decode._tables, decode._step_ptrs = (1, 2), {}, None
    monkeypatch.setattr(SpecializedDecode, "_kv_key", lambda self: (1, 2))
    ctx = SimpleNamespace(cudagraph_runtime_mode=CUDAGraphMode.FULL)
    monkeypatch.setattr(decode_mod, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(decode_mod, "get_forward_context", lambda: ctx)

    def plan(view):
        md = dict(view.attn_metadata)
        for m in md.values():
            m.num_decode_tokens, m.num_decodes = view.width, view.width // q
        ctx.attn_metadata, ctx.slot_mapping = md, view.slot_mapping
        ctx.is_padding = view.pad
        return decode.plan(view.positions, None)

    return decode, bb, mrv2, plan


def test_plan_checks_the_metadata_pointers(monkeypatch):
    """Every plan() passes the persistent buffers the first one did (the
    capture of each width, then eager steps); a moved table, slot mapping,
    seq_lens, positions or pad mask raises (DESIGN_BC 8.3)."""
    decode, bb, mrv2, plan = _meta_decode(monkeypatch)
    pages = {"full": [[3, 4, 5], [6]], "sw": [[7, 8, 9], [10]]}
    assert plan(mrv2.capture(8)) is not None  # the width-8 capture records
    assert plan(mrv2.capture(2)) is not None
    for _ in range(3):
        assert plan(mrv2.step([40, 9], pages, 2)) is not None
    names = [bb.layers[i].self_attn.attn.layer_name for i in (1, 2, 3)]
    assert sorted(decode._tables) == sorted(names)
    view = mrv2.step([40, 9], pages, 2)
    moved = {
        "table": lambda v: setattr(
            v.attn_metadata[names[2]],
            "block_table",
            v.attn_metadata[names[2]].block_table.clone(),
        ),
        "slots": lambda v: v.slot_mapping.__setitem__(
            names[0], v.slot_mapping[names[0]].clone()
        ),
        "seq_lens": lambda v: setattr(
            v.attn_metadata[names[1]],
            "seq_lens",
            v.attn_metadata[names[1]].seq_lens.clone(),
        ),
        "positions": lambda v: setattr(v, "positions", v.positions.clone()),
        "pad": lambda v: setattr(v, "pad", v.pad.clone()),
    }
    for what, move in moved.items():
        view = mrv2.step([40, 9], pages, 2)
        view.slot_mapping = dict(view.slot_mapping)
        move(view)
        with pytest.raises(RuntimeError, match="moved off MRV2's persistent"):
            plan(view)
    # a stride change (another table shape) too
    view = mrv2.step([40, 9], pages, 2)
    md = view.attn_metadata[names[0]]
    md.block_table = torch.zeros(8, 128, dtype=torch.int32)
    with pytest.raises(RuntimeError, match="moved off"):
        plan(view)


def test_executor_dspark_split_runs_gated(monkeypatch):
    """DSpark (q = 8): the split core falls back to the gated core load; K1 stays."""
    decode, bb, compiled, launches, _ = _executor(monkeypatch, extra=K1_SPLIT, q=8)
    assert compiled == [
        spec.K1Build(tokens=8, rope=False),
        spec.K1Build(tokens=8, rope=True),
        spec.K2Build(tokens=8, core="gated"),
    ]
    _bind(bb)
    decode.prepare_kv()
    # the kinds share K2(gated) but not K1: both kinds' first layers run
    assert [type(k).__name__ for k, *_ in launches] == ["K1Build", "K2Build"] * 4


# --------------------------------------------------------------------------
# The attention's front / attend split (mono off: forward unchanged)


def _recording_attention(monkeypatch, fused=True):
    """A ``MuseSpark1xAttention`` whose every callee records itself (16 heads
    of 4 dims, o_gate on): the order and the arguments of what forward runs."""
    from vllm.models.muse_spark.amd.model import MuseSpark1xAttention

    attn = object.__new__(MuseSpark1xAttention)
    nn.Module.__init__(attn)
    rec: list = []
    attn.num_heads, attn.head_dim, attn.num_kv_heads = 4, 4, 1
    attn.q_size, attn.kv_size = 16, 4
    attn.use_o_gate, attn.use_o_norm, attn.fuse_o_norm_gate = True, True, True
    attn.use_rope, attn.rotary_emb = False, None
    attn.fused_qk_norm_rope_kv_insert = fused
    attn.qk_norm = None
    attn.use_attn_temperature_tuning = False
    attn.rms_norm_eps = 1e-5

    def qkv_proj(h):
        rec.append(("qkv_proj", h))
        return h * 3, None

    def core(q, k, v):
        rec.append(("attn", q, k, v))
        return q * 5

    def o_proj(x):
        rec.append(("o_proj", x))
        return x * 7, None

    def insert(qkv, *a):
        rec.append(("insert", qkv, *a))
        qkv.add_(1)

    attn.qkv_proj, attn.attn, attn.o_proj = qkv_proj, core, o_proj
    attn.attn.layer_name = "L"
    monkeypatch.setattr(model_mod, "rocm_fused_qk_norm_rope_kv_insert", insert)

    def gated(x, g, *a, eps):
        rec.append(("gate", x, g, a, eps))
        return x + g

    monkeypatch.setattr(model_mod, "rms_norm_gated", gated)
    return attn, rec


def test_unfused_forward_is_unchanged(monkeypatch):
    """Without the fused insert (the draft, other configs): k / v reach the
    attention layer, and front / attend refuse."""
    attn, rec = _recording_attention(monkeypatch, fused=False)
    h, pos = torch.arange(3 * 40.0).view(3, 40), torch.arange(3)
    out = attn(pos, h)
    assert [r[0] for r in rec] == ["qkv_proj", "attn", "gate", "o_proj"]
    _, q, k, v = rec[1]
    qkv = h * 3
    assert torch.equal(k, qkv[:, 16:20]) and torch.equal(v, qkv[:, 20:24])
    assert out.shape == (3, 16)
    with pytest.raises(AssertionError):
        attn.front(pos, h)
    with pytest.raises(AssertionError):
        attn.attend(h)
