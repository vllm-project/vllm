# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi-K3's specialized decode on CPU: the switch, its widths and layers, its
refusals (config, loaded weights, caches) and plan() routing."""

from types import SimpleNamespace

import pytest
import torch

from vllm.config import (
    CompilationMode,
    CUDAGraphMode,
    KVTransferConfig,
    ModelConfig,
    ParallelConfig,
    SpeculativeConfig,
    VllmConfig,
)
from vllm.models.kimi_k3.amd.specialized_decode import runtime
from vllm.models.kimi_k3.amd.specialized_decode.runtime import (
    FFN,
    WHOLE,
    DecodeConfig,
    layer_kinds,
    specialized_decode_config,
)
from vllm.models.kimi_k3.amd.specialized_decode.runtime import decode as decode_mod
from vllm.models.kimi_k3.amd.specialized_decode.runtime.decode import (
    SpecializedDecode,
    step_reason,
)
from vllm.models.kimi_k3.amd.specialized_decode.runtime.guards import (
    cache_reasons,
    mla_weight_reasons,
    weight_reasons,
)


def _vllm_config(**over) -> SimpleNamespace:
    """A supported config: TP8, DCP8 (interleave 1), 7 speculative tokens,
    max_num_seqs 1, an fp8 KV cache, FULL graphs, MRV2.
    ``over`` sets dotted paths, e.g. ``{"parallel_config.data_parallel_size": 2}``."""
    spec = SimpleNamespace(
        num_speculative_tokens=7,
        enable_adaptive_verification=False,
        dynamic=False,
    )
    spec.uses_dynamic_speculative_decoding = lambda: spec.dynamic
    c = SimpleNamespace(
        additional_config={"decode_backend": "latency_specialized"},
        parallel_config=SimpleNamespace(
            tensor_parallel_size=8,
            pipeline_parallel_size=1,
            data_parallel_size=1,
            enable_expert_parallel=False,
            enable_eplb=False,
            decode_context_parallel_size=8,
            cp_kv_cache_interleave_size=1,
            prefill_context_parallel_size=1,
            use_ubatching=False,
        ),
        model_config=SimpleNamespace(
            dtype=torch.bfloat16,
            enable_sleep_mode=False,
            hf_text_config=SimpleNamespace(
                hidden_size=7168,
                num_experts=896,
                attn_res_block_size=12,
                linear_attn_config={
                    "num_heads": 96,
                    "head_dim": 128,
                    "short_conv_kernel_size": 4,
                },
            ),
        ),
        cache_config=SimpleNamespace(
            cache_dtype="fp8",
            mamba_cache_dtype="auto",
            use_replayssm=False,
            use_kda_recoverssm=False,
        ),
        lora_config=None,
        kv_transfer_config=None,
        aux_output_config=SimpleNamespace(enable_return_routed_experts=False),
        offload_config=SimpleNamespace(
            uva=SimpleNamespace(cpu_offload_gb=0),
            prefetch=SimpleNamespace(offload_group_size=0),
        ),
        weight_transfer_config=None,
        speculative_config=spec,
        scheduler_config=SimpleNamespace(max_num_seqs=1),
        use_v2_model_runner=True,
        compilation_config=SimpleNamespace(
            mode=CompilationMode.VLLM_COMPILE,
            cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
        ),
    )
    for path, v in over.items():
        obj = c
        *head, last = path.split(".")
        for h in head:
            obj = getattr(obj, h)
        setattr(obj, last, v)
    return c


@pytest.fixture(autouse=True)
def fp4_bmm(monkeypatch):
    monkeypatch.setenv("VLLM_ROCM_USE_AITER_FP4BMM", "1")


@pytest.fixture
def on_mi355x(monkeypatch):
    monkeypatch.setattr(runtime, "platform_reasons", lambda: [])


@pytest.mark.parametrize(
    "k, want",
    [
        (7, (8, 16)),
        (3, (4, 8, 12, 16)),
        (1, tuple(range(2, 17, 2))),
        (15, (16,)),
    ],
)
def test_widths_are_whole_requests_up_to_max_rows(k, want):
    assert DecodeConfig(q=k + 1).widths == want


def _k3_text_config():
    """Kimi-K3's layer pattern: 93 layers, MLA every 4th, AttnRes blocks of 12,
    layer 0 dense."""
    mla = {i for i in range(93) if i % 4 == 3} | {92}
    return SimpleNamespace(
        num_hidden_layers=93,
        attn_res_block_size=12,
        num_experts=896,
        first_k_dense_replace=1,
        moe_layer_freq=1,
        is_kda_layer=lambda i: i not in mla,
    )


def test_kernel_layers_follow_the_k3_pattern():
    kinds = layer_kinds(_k3_text_config())
    assert list(kinds) == list(range(1, 93))
    assert kinds[1] == WHOLE and kinds[12] == WHOLE
    assert kinds[3] == FFN and kinds[92] == FFN


def test_auto_runs_the_stock_decode():
    """Without the switch nothing is checked, even on an unsupported deployment."""
    cfg = _vllm_config(
        additional_config={}, **{"parallel_config.tensor_parallel_size": 4}
    )
    assert specialized_decode_config(cfg) is None
    cfg.additional_config = {"decode_backend": "auto"}
    assert specialized_decode_config(cfg) is None


def test_an_unknown_backend_is_refused():
    cfg = _vllm_config(additional_config={"decode_backend": "fast"})
    with pytest.raises(ValueError, match="decode_backend must be"):
        specialized_decode_config(cfg)


def test_supported_config_is_accepted(supported_vllm_config):
    assert specialized_decode_config(supported_vllm_config) == DecodeConfig(q=8)


@pytest.fixture
def supported_vllm_config(tmp_path, monkeypatch, on_mi355x):
    from vllm.transformers_utils.configs.kimi_linear import KimiLinearConfig

    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    # CUDA must not auto-enable a process-wide setting in this ROCm fixture.
    monkeypatch.setenv("VLLM_USE_BREAKABLE_CUDAGRAPH", "0")
    KimiLinearConfig(
        architectures=["KimiLinearForCausalLM"],
        hidden_size=7168,
        num_experts=896,
        attn_res_block_size=12,
        linear_attn_config={
            "kda_layers": [1],
            "full_attn_layers": [2],
            "num_heads": 96,
            "head_dim": 128,
            "short_conv_kernel_size": 4,
        },
    ).save_pretrained(tmp_path)
    cfg = VllmConfig(
        model_config=ModelConfig(
            model=str(tmp_path), dtype=torch.bfloat16, max_model_len=1024, seed=0
        ),
        # Config-only tests do not need executor auto-detection of local GPUs.
        parallel_config=ParallelConfig(
            tensor_parallel_size=8, distributed_executor_backend="mp"
        ),
        additional_config={"decode_backend": "latency_specialized"},
        speculative_config=SpeculativeConfig(method="ngram", num_speculative_tokens=7),
    )
    # a served DCP1 step: one request, an fp8 KV cache (M1c)
    cfg.scheduler_config.max_num_seqs = 1
    cfg.cache_config.cache_dtype = "fp8"
    return cfg


@pytest.mark.parametrize("read_mode", [False, True])
@pytest.mark.parametrize("backend", ["rdma", "xgmi"])
def test_moriio_consumer_is_supported(supported_vllm_config, read_mode, backend):
    supported_vllm_config.kv_transfer_config = KVTransferConfig(
        kv_connector="MoRIIOConnector",
        kv_role="kv_consumer",
        kv_connector_extra_config={"read_mode": read_mode, "backend": backend},
    )
    assert specialized_decode_config(supported_vllm_config) == DecodeConfig(q=8)


@pytest.mark.parametrize(
    "overrides",
    [
        {"kv_connector": "NixlConnector"},
        {"kv_connector": "MultiConnector"},
        {"kv_connector_module_path": "custom.connector"},
        {"kv_role": "kv_producer"},
        {"kv_role": "kv_both"},
    ],
)
def test_unsupported_kv_transfer_is_refused(supported_vllm_config, overrides):
    supported_vllm_config.kv_transfer_config = KVTransferConfig(
        **(
            {
                "kv_connector": "MoRIIOConnector",
                "kv_role": "kv_consumer",
                "kv_connector_extra_config": {"read_mode": True},
            }
            | overrides
        )
    )
    with pytest.raises(ValueError, match="KV transfer connectors"):
        specialized_decode_config(supported_vllm_config)


@pytest.mark.parametrize("dcp", [1, 8])
@pytest.mark.parametrize("interleave", [1, 16])
def test_any_interleave_is_accepted(on_mi355x, dcp, interleave):
    """M1a writes the slots vLLM maps, M1b masks a request's empty shards."""
    cfg = _vllm_config(
        **{
            "parallel_config.decode_context_parallel_size": dcp,
            "parallel_config.cp_kv_cache_interleave_size": interleave,
        }
    )
    assert specialized_decode_config(cfg) == DecodeConfig(q=8, interleave=interleave)


@pytest.mark.parametrize(
    "over, env, why",
    [
        ({"cache_config.cache_dtype": "auto"}, "1", "an fp8 KV cache"),
        ({}, "0", "VLLM_ROCM_USE_AITER_FP4BMM=1"),
    ],
)
def test_mla_without_m1_runs_vllm_ops(on_mi355x, monkeypatch, over, env, why):
    """What only M1 needs is no refusal: the MLA layers run vLLM's ops, K2."""
    monkeypatch.setenv("VLLM_ROCM_USE_AITER_FP4BMM", env)
    cfg = _vllm_config(**over)
    assert runtime.unsupported(cfg) == []
    reasons = runtime.m1_reasons(cfg)
    assert len(reasons) == 1 and why in reasons[0]
    assert specialized_decode_config(cfg) == DecodeConfig(q=8, m1=False)


@pytest.mark.parametrize(
    "over, reason",
    [
        ({"parallel_config.tensor_parallel_size": 4}, "TP 4"),
        ({"parallel_config.pipeline_parallel_size": 2}, "pipeline parallelism"),
        ({"parallel_config.decode_context_parallel_size": 2}, "other than 1 or 8"),
        ({"parallel_config.decode_context_parallel_size": 4}, "other than 1 or 8"),
        (
            {
                "kv_transfer_config": KVTransferConfig(
                    kv_connector="NixlConnector", kv_role="kv_consumer"
                )
            },
            "KV transfer connectors",
        ),
        ({"speculative_config": None}, "without speculative decoding"),
        ({"speculative_config.num_speculative_tokens": 16}, "more than 15"),
        ({"speculative_config.enable_adaptive_verification": True}, "adaptive"),
        ({"speculative_config.dynamic": True}, "dynamic speculative"),
        ({"cache_config.mamba_cache_dtype": "float32"}, "KDA cache dtypes"),
        ({"cache_config.use_replayssm": True}, "ReplaySSM"),
        ({"use_v2_model_runner": False}, "V1 model runner"),
        (
            {"compilation_config.mode": CompilationMode.STOCK_TORCH_COMPILE},
            "torch.compile",
        ),
        ({"model_config.hf_text_config.hidden_size": 4096}, "model shapes"),
    ],
)
def test_refusals(on_mi355x, over, reason):
    reasons = runtime.unsupported(_vllm_config(**over))
    assert len(reasons) == 1 and reason in reasons[0], reasons


def _step(dcp: int, seqs: int, k: int) -> SimpleNamespace:
    return _vllm_config(
        **{
            "parallel_config.decode_context_parallel_size": dcp,
            "scheduler_config.max_num_seqs": seqs,
            "speculative_config.num_speculative_tokens": k,
        }
    )


@pytest.mark.parametrize(
    "dcp, seqs, k",
    [
        *[(8, s, k) for s, k in [(1, 1), (1, 3), (1, 7), (1, 15), (2, 1), (2, 3)]],
        *[(8, s, k) for s, k in [(2, 7), (4, 1), (4, 3), (8, 1)]],
        *[(1, s, k) for s, k in [(1, 3), (1, 7), (2, 3), (2, 7)]],
    ],
)
def test_served_steps_are_accepted(on_mi355x, dcp, seqs, k):
    assert runtime.unsupported(_step(dcp, seqs, k)) == []


@pytest.mark.parametrize(
    "dcp, seqs, k, reason",
    [
        (8, 3, 1, "max_num_seqs 3 at 1"),
        (8, 16, 1, "max_num_seqs 16 at 1"),
        (8, 4, 7, "max_num_seqs 4 at 7"),
        (8, 8, 3, "max_num_seqs 8 at 3"),
        (8, 256, 7, "max_num_seqs 256 at 7"),
        (1, 2, 15, "max_num_seqs 2 at 15"),
        # 3 requests at qlen 4 (12 rows): M1c's 80 splits do not share out
        (1, 4, 3, "DCP1 runs M1c only"),
        # no M1c at qlen 2 / 16: AITER's asm decode
        (1, 1, 1, "DCP1 runs M1c only"),
        (1, 8, 1, "DCP1 runs M1c only"),
        (1, 1, 15, "DCP1 runs M1c only"),
    ],
)
def test_unserved_steps_are_refused(on_mi355x, dcp, seqs, k, reason):
    reasons = runtime.unsupported(_step(dcp, seqs, k))
    assert len(reasons) == 1 and reason in reasons[0], reasons


def test_dcp1_without_m1_is_refused(on_mi355x):
    """DCP1 runs M1c, which needs M1 (at DCP8 vLLM's ops run instead)."""
    cfg = _vllm_config(
        **{
            "parallel_config.decode_context_parallel_size": 1,
            "cache_config.cache_dtype": "auto",
        }
    )
    reasons = runtime.unsupported(cfg)
    assert len(reasons) == 1 and "DCP1 without M1 (an fp8 KV cache)" in reasons[0]


def test_every_refusal_is_reported_at_once(monkeypatch):
    monkeypatch.setattr(
        runtime, "platform_reasons", lambda: ["this GPU (gfx950 / MI355X only)"]
    )
    cfg = _vllm_config(
        **{"parallel_config.tensor_parallel_size": 4, "use_v2_model_runner": False}
    )
    with pytest.raises(ValueError) as e:
        specialized_decode_config(cfg)
    for reason in ("TP 4", "V1 model runner", "this GPU", "decode_backend=auto"):
        assert reason in str(e.value)


def test_piecewise_graphs_only_warn(on_mi355x):
    """They are not refused: plan() runs every PIECEWISE step stock."""
    cfg = _vllm_config(**{"compilation_config.cudagraph_mode": CUDAGraphMode.PIECEWISE})
    assert runtime.piecewise_graphs_only(cfg)
    assert specialized_decode_config(cfg) is not None


def test_cache_refusal_names_the_cache():
    pytest.importorskip("flydsl")
    conv = torch.zeros(2, 4, 3, dtype=torch.float32)
    rstate = torch.zeros(2, 12, 128, 128, dtype=torch.bfloat16)
    reasons = cache_reasons({"layers.5.attn": (conv, rstate)})
    assert len(reasons) == 2 and all(r.startswith("layers.5.attn:") for r in reasons)


def test_cache_refusal_checks_the_mla_cache():
    pytest.importorskip("flydsl")
    ok = torch.zeros(4, 16, 576, dtype=torch.uint8)
    assert cache_reasons({"layers.3.attn": ok}) == []
    strided = torch.zeros(4, 32, 576, dtype=torch.uint8)[:, ::2]
    bf16 = torch.zeros(4, 16, 576, dtype=torch.bfloat16)
    for bad in (strided, bf16):
        reasons = cache_reasons({"layers.3.attn": bad})
        assert len(reasons) == 1 and reasons[0].startswith("layers.3.attn:")


def _kda_metadata(**md):
    """One request of 4 tokens unless overridden."""
    m = SimpleNamespace(
        spec_sequence_masks=torch.ones(1, dtype=torch.bool),
        num_prefills=0,
        num_decodes=0,
        num_spec_decodes=1,
        spec_state_indices_tensor=torch.zeros(1, 4, dtype=torch.int32),
        uniform_spec_sequence_length=4,
    )
    for k, v in md.items():
        setattr(m, k, v)
    return {"kda": m}


@pytest.mark.parametrize(
    "width, md, why",
    [
        (4, {}, None),
        (8, {"num_spec_decodes": 2}, None),
        (4, {"num_prefills": 1}, "not pure decode"),
        (4, {"num_decodes": 1}, "not pure decode"),
        (4, {"spec_sequence_masks": None}, "not spec-verify"),
        (8, {}, "not uniform"),
        (4, {"uniform_spec_sequence_length": None}, "not uniform"),
    ],
)
def test_steps_must_be_pure_uniform_spec_verify(width, md, why):
    assert step_reason(_kda_metadata(**md), "kda", width) == why


def test_steps_need_metadata():
    assert step_reason(None, "kda", 4) == "no metadata"


@pytest.mark.parametrize(
    "mla, why",
    [
        ({}, None),
        (None, "no MLA decode metadata"),
        ({"decode": None}, "no MLA decode metadata"),
        ({"num_prefills": 1}, "MLA: not pure decode"),
        ({"num_decode_tokens": 2}, "MLA: not pure decode"),
        ({"decode": SimpleNamespace(max_qo_len=2)}, "MLA: not uniform"),
        (
            {"decode": SimpleNamespace(max_qo_len=4, use_gluon_verify=True)},
            "MLA: Gluon decode",
        ),
        (
            {"decode": SimpleNamespace(max_qo_len=4, use_gluon_decode=True)},
            "MLA: Gluon decode",
        ),
    ],
)
def test_steps_check_the_mla_decode(mla, why):
    """One request of 4 tokens: the MLA layers' metadata must agree."""
    md = _kda_metadata()
    if mla is not None:
        a = SimpleNamespace(
            decode=SimpleNamespace(max_qo_len=4), num_prefills=0, num_decode_tokens=4
        )
        for k, v in mla.items():
            setattr(a, k, v)
        md["mla"] = a
    assert step_reason(md, "kda", 4, "mla") == why


class _Kernel:
    def launch(self, step, positions, hidden_states, residual, prefix_delta):
        return "kernel", step.width


def _decode(monkeypatch, mode=CUDAGraphMode.FULL, md=None):
    """An executor with kernels on layers 1 and 2 at widths 4 and 8, bound to a
    KV cache, under a forward context of ``mode``."""
    decode = object.__new__(SpecializedDecode)
    decode.config = DecodeConfig(q=4)
    decode.runner = SimpleNamespace(kda_prefix="kda", mla_prefix="")
    decode.kernels = {w: {1: _Kernel(), 2: _Kernel()} for w in (4, 8)}
    decode._kv = (1, 2)
    monkeypatch.setattr(SpecializedDecode, "_kv_key", lambda self: (1, 2))
    context = SimpleNamespace(
        cudagraph_runtime_mode=mode,
        attn_metadata=_kda_metadata() if md is None else md,
    )
    monkeypatch.setattr(decode_mod, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(decode_mod, "get_forward_context", lambda: context)
    return decode


def test_plan_runs_the_kernels_of_its_width(monkeypatch):
    step = _decode(monkeypatch).plan(torch.zeros(4), None)
    assert step is not None and step.width == 4
    assert step.layer(1)(None, None, None, None) == ("kernel", 4)
    assert step.layer(0) is None and step.layer(3) is None


@pytest.mark.parametrize(
    "mode, tokens, md, intermediate",
    [
        (CUDAGraphMode.PIECEWISE, 4, None, None),
        (CUDAGraphMode.FULL, 16, None, None),  # no kernels that wide
        (CUDAGraphMode.FULL, 4, _kda_metadata(num_prefills=1), None),
        (CUDAGraphMode.FULL, 4, None, object()),  # a pipeline stage's input
    ],
)
def test_plan_runs_other_steps_stock(monkeypatch, mode, tokens, md, intermediate):
    decode = _decode(monkeypatch, mode, md)
    assert decode.plan(torch.zeros(tokens), intermediate) is None


def test_plan_runs_stock_before_a_kv_cache_is_bound(monkeypatch):
    decode = _decode(monkeypatch)
    decode._kv = None
    assert decode.plan(torch.zeros(4), None) is None


def test_plan_refuses_a_kv_cache_bound_behind_its_back(monkeypatch):
    decode = _decode(monkeypatch)
    monkeypatch.setattr(SpecializedDecode, "_kv_key", lambda self: (3, 4))
    with pytest.raises(RuntimeError, match="without prepare_kv"):
        decode.plan(torch.zeros(4), None)


def test_weight_views_reject_every_mismatch():
    """weight_reasons names each layer whose weights the kernels cannot read as
    they are, and a MoE backend that lays them out otherwise."""
    pytest.importorskip("flydsl")
    from vllm.models.kimi_k3.amd.specialized_decode.mono.stages import moe

    bad_moe = SimpleNamespace(
        use_latent_moe=True,
        shared_experts=object(),
        routed_scaling_factor=2.0,
        experts=SimpleNamespace(quant_method=SimpleNamespace()),
    )
    no_latent = SimpleNamespace(use_latent_moe=False, shared_experts=None)
    layers = [None] + [SimpleNamespace(mlp=no_latent)] * 92
    layers[5] = SimpleNamespace(mlp=bad_moe)
    model = SimpleNamespace(
        config=_k3_text_config(), layers=layers, start_layer=0, end_layer=93
    )
    reasons = weight_reasons(model)
    assert "layer 1: needs a latent MoE with shared experts" in reasons
    assert "layer 5: routed_scaling_factor 2.0" in reasons
    assert any(r.startswith("layer 5: MoE backend must be") for r in reasons)
    assert moe.E == 896


def test_attend_hands_m1a_padded_query_to_forward_mqa():
    from vllm.models.kimi_k3.amd.specialized_decode.runtime import mla
    from vllm.v1.attention.backends.mla.rocm_aiter_mla import AiterMLAHelper

    pad = AiterMLAHelper.get_mla_padded_q
    q = torch.randn(4, 96, 8)
    padded = pad(96, q, target_heads=128)  # what M1a writes
    other = torch.randn(4, 96, 8)
    seen = {}

    def forward_mqa(q, kv, md, attn):
        assert q.shape[1] == 96
        seen["q"] = AiterMLAHelper.get_mla_padded_q(96, q, target_heads=128)
        seen["other"] = AiterMLAHelper.get_mla_padded_q(96, other, target_heads=128)
        return None, None

    impl = SimpleNamespace(_decode_num_heads=96, forward_mqa=forward_mqa)
    wrapper = SimpleNamespace(mla_attn=SimpleNamespace(impl=impl))
    mla.attend(wrapper, padded, torch.zeros(4, dtype=torch.uint8), None)
    assert seen["q"] is padded  # no copy
    assert torch.equal(seen["other"], pad(96, other, target_heads=128))
    assert AiterMLAHelper.get_mla_padded_q is pad  # restored


def test_qkv_gate_weights_split_the_merged_gate():
    """KimiK3MergedQKVGateLinear (vLLM's) stacks [q_a | kv_a | gate]: M1a
    reads the two row slices in place."""
    from vllm.models.kimi_k3.amd.specialized_decode.runtime import mla

    w = torch.randn(2112 + 1536, 8)
    wrapper = SimpleNamespace(
        fused_qkv_a_proj=SimpleNamespace(weight=w), g_proj=None,
        _qkv_gate_fused=True, q_lora_rank=1536, qk_rope_head_dim=64,
        mla_attn=SimpleNamespace(kv_lora_rank=512),
    )  # fmt: skip
    qkv, g = mla.qkv_gate_weights(wrapper)
    assert qkv.shape == (2112, 8) and g.shape == (1536, 8)
    assert qkv.is_contiguous() and g.is_contiguous()
    assert g.data_ptr() == w[2112:].data_ptr()
    wrapper._qkv_gate_fused = False
    assert mla.qkv_gate_weights(wrapper) == (w, None)
    gw = torch.randn(1536, 8)
    wrapper.g_proj = SimpleNamespace(weight=gw)
    assert mla.qkv_gate_weights(wrapper) == (w, gw)


def test_m1c_falls_back_where_its_splits_dont_share_out():
    """M1c's KV splits share out evenly over a width's requests or the width
    runs the asm decode: qlen 4's 80 splits over 12 rows (3 requests) don't."""
    pytest.importorskip("flydsl")
    from vllm.models.kimi_k3.amd.specialized_decode.runtime.layers import (
        m1c_at,
        m1c_splits,
    )

    k3 = SimpleNamespace(m1c=True, num_spec=3)
    assert [w for w in (4, 8, 12, 16) if m1c_at(k3, w)] == [4, 8, 16]
    assert m1c_splits(k3, 8) == 80 and m1c_splits(k3, 12) == 0
    k7 = SimpleNamespace(m1c=True, num_spec=7)
    assert all(m1c_at(k7, w) for w in (8, 16))
    off = SimpleNamespace(m1c=False, num_spec=3)
    assert not m1c_at(off, 4) and m1c_splits(off, 4) == 0


def test_mla_weight_views_fall_back_per_model():
    """An MLA layer M1 cannot read is no weight refusal: mla_weight_reasons
    names it, and the MLA layers run vLLM's ops."""
    no_latent = SimpleNamespace(use_latent_moe=False, shared_experts=None)
    config = _k3_text_config()
    ffn = [i for i, kind in runtime.layer_kinds(config).items() if kind == "ffn"]
    w = SimpleNamespace(
        num_heads=16, q_lora_rank=1536, qk_nope_head_dim=128, qk_rope_head_dim=64,
        v_head_dim=128, mla_attn=SimpleNamespace(kv_lora_rank=512),
    )  # fmt: skip
    layers = [
        SimpleNamespace(mlp=no_latent, self_attn=SimpleNamespace(mla_attn=w))
    ] * 93
    model = SimpleNamespace(config=config, layers=layers, start_layer=0, end_layer=93)
    assert not any("MLA" in r for r in weight_reasons(model))
    reasons = mla_weight_reasons(model)
    assert ffn and len(reasons) == len(ffn)
    assert (
        reasons[0] == f"layer {ffn[0]}: MLA heads/q_lora/kv_lora/nope/rope/v "
        "(16, 1536, 512, 128, 64, 128)"
    )
