# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host side of --monokernel for Qwen3.8: the config refusals (every reason
in one error), the widths, and the per-step routing."""

from types import SimpleNamespace

import pytest
import torch

from vllm.config import CompilationMode, CUDAGraphMode
from vllm.models.qwen3_5.amd.specialized_decode import runtime
from vllm.models.qwen3_5.amd.specialized_decode.mono import layout as L
from vllm.models.qwen3_5.amd.specialized_decode.runtime.routing import step_reason
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(),
    reason="the Qwen3.8 specialized decode is ROCm only",
)


def _text_config(**over):
    values = dict(zip(L.SHAPE_NAMES, L.SHAPES))
    values.update(norm_topk_prob=True, num_hidden_layers=92)
    values.update(over)
    return SimpleNamespace(**values)


def _vllm_config(monokernel=True, **parallel):
    values = dict(
        tensor_parallel_size=8,
        pipeline_parallel_size=1,
        data_parallel_size=1,
        decode_context_parallel_size=1,
        prefill_context_parallel_size=1,
        enable_expert_parallel=False,
        enable_eplb=False,
        use_ubatching=False,
        use_sequence_parallel_moe=False,
    )
    values.update(parallel)
    return SimpleNamespace(
        kernel_config=SimpleNamespace(monokernel=monokernel),
        parallel_config=SimpleNamespace(**values),
        cache_config=SimpleNamespace(
            mamba_cache_mode="none",
            mamba_cache_dtype="auto",
            mamba_ssm_cache_dtype="float32",
            cache_dtype="fp8",
        ),
        compilation_config=SimpleNamespace(
            mode=CompilationMode.VLLM_COMPILE, cudagraph_mode=CUDAGraphMode.FULL
        ),
        lora_config=None,
        kv_transfer_config=None,
        aux_output_config=SimpleNamespace(enable_return_routed_experts=False),
        speculative_config=None,
        weight_transfer_config=None,
        offload_config=SimpleNamespace(
            uva=SimpleNamespace(cpu_offload_gb=0),
            prefetch=SimpleNamespace(offload_group_size=0),
        ),
        use_v2_model_runner=True,
        additional_config={},
        model_config=SimpleNamespace(
            architecture="Qwen3_5MoeForCausalLM",
            dtype=torch.bfloat16,
            enable_sleep_mode=False,
            hf_text_config=_text_config(),
        ),
    )


@pytest.fixture
def supported_platform(monkeypatch):
    monkeypatch.setenv("VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS", "1")
    monkeypatch.setattr(runtime, "platform_reasons", lambda: [])


def test_off_without_monokernel():
    # nothing else is read: every deployment runs vLLM's decode
    config = SimpleNamespace(kernel_config=SimpleNamespace(monokernel=False))
    assert runtime.specialized_decode_config(config) is None


def test_supported_config(supported_platform):
    config = runtime.specialized_decode_config(_vllm_config())
    assert config == runtime.DecodeConfig()
    assert config.widths == tuple(range(1, 17))


def test_refusals_report_every_reason(supported_platform, monkeypatch):
    monkeypatch.setenv("VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS", "0")
    config = _vllm_config(tensor_parallel_size=4, enable_expert_parallel=True)
    config.model_config.architecture = "Qwen3_5MoeForConditionalGeneration"
    config.cache_config.mamba_cache_mode = "align"
    config.cache_config.mamba_ssm_cache_dtype = "auto"
    config.speculative_config = SimpleNamespace(method="mtp")
    config.lora_config = SimpleNamespace()
    config.use_v2_model_runner = False
    config.model_config.hf_text_config = _text_config(
        hidden_size=4096, norm_topk_prob=False, num_hidden_layers=160
    )
    with pytest.raises(ValueError) as e:
        runtime.specialized_decode_config(config)
    msg = str(e.value)
    for why in (
        "architecture Qwen3_5MoeForConditionalGeneration",
        "TP 4 (TP8 only)",
        "expert parallelism",
        "LoRA",
        "speculative decoding",
        "GDN prefix caching (mamba_cache_mode 'align'",
        "GDN cache dtypes",
        "an unfused shared expert",
        "the V1 model runner",
        "model shapes (4096,",
        "router weights without top-k renormalization",
        "160 decoder layers (at most 128)",
        "without --monokernel, vLLM's decode runs",
    ):
        assert why in msg, (why, msg)


@pytest.mark.parametrize(
    "over, why",
    [
        (dict(pipeline_parallel_size=2), "pipeline parallelism"),
        (dict(data_parallel_size=2), "data parallelism"),
        (dict(enable_eplb=True), "EPLB"),
        (dict(decode_context_parallel_size=2), "decode context parallelism"),
        (dict(prefill_context_parallel_size=2), "prefill context parallelism"),
        (dict(use_ubatching=True), "DBO"),
        (dict(use_sequence_parallel_moe=True), "sequence-parallel MoE"),
    ],
)
def test_parallel_refusals(supported_platform, over, why):
    got = runtime.unsupported(_vllm_config(**over))
    assert len(got) == 1 and why in got[0], got


def test_feature_refusals(supported_platform):
    def reasons(edit):
        config = _vllm_config()
        edit(config)
        return runtime.unsupported(config)

    assert reasons(lambda c: None) == []
    cases = {
        "KV transfer": lambda c: setattr(c, "kv_transfer_config", object()),
        "routed-expert capture": lambda c: setattr(
            c.aux_output_config, "enable_return_routed_experts", True
        ),
        "sleep mode": lambda c: setattr(c.model_config, "enable_sleep_mode", True),
        "CPU weight offload": lambda c: setattr(
            c.offload_config.uva, "cpu_offload_gb", 1
        ),
        "weight updates": lambda c: setattr(c, "weight_transfer_config", object()),
        "torch.compile": lambda c: setattr(
            c.compilation_config, "mode", CompilationMode.STOCK_TORCH_COMPILE
        ),
        "layer scale": lambda c: setattr(
            c.model_config, "hf_text_config", _text_config(layer_scale=True)
        ),
        "model shapes": lambda c: setattr(
            c.model_config, "hf_text_config", _text_config(hidden_act="gelu")
        ),
    }
    for why, edit in cases.items():
        got = reasons(edit)
        assert len(got) == 1 and why in got[0], (why, got)


def test_kv_cache_dtype_is_vllms(supported_platform):
    # the full-attention layers run vLLM's attention: any KV cache dtype
    for dtype in ("auto", "bfloat16", "fp8", "fp8_e4m3"):
        config = _vllm_config()
        config.cache_config.cache_dtype = dtype
        assert runtime.unsupported(config) == [], dtype


def test_piecewise_graphs_warn(supported_platform):
    config = _vllm_config()
    assert not runtime.piecewise_graphs_only(config)
    config.compilation_config.cudagraph_mode = CUDAGraphMode.PIECEWISE
    assert runtime.piecewise_graphs_only(config)
    assert runtime.specialized_decode_config(config) is not None
    config.compilation_config.cudagraph_mode = CUDAGraphMode.NONE
    assert not runtime.piecewise_graphs_only(config)


def _gdn_metadata(decodes=4, prefills=0, spec=0, indices=None):
    return SimpleNamespace(
        num_prefills=prefills,
        num_decodes=decodes,
        num_spec_decodes=spec,
        spec_sequence_masks=None if spec == 0 else torch.ones(1, dtype=torch.bool),
        non_spec_state_indices_tensor=(
            torch.arange(decodes, dtype=torch.int32) if indices is None else indices
        ),
    )


@pytest.mark.parametrize(
    "md, width, why",
    [
        ({"gdn": _gdn_metadata(4)}, 4, None),
        # a FULL graph's pad rows: the indices cover the padded width
        ({"gdn": _gdn_metadata(3, indices=torch.zeros(8, dtype=torch.int32))}, 8, None),
        (None, 4, "no metadata"),
        ({}, 4, "not pure decode"),
        ({"gdn": _gdn_metadata(3, prefills=1)}, 4, "not pure decode"),
        ({"gdn": _gdn_metadata(2, spec=2)}, 4, "spec-verify"),
        (
            {"gdn": _gdn_metadata(0, indices=torch.zeros(4, dtype=torch.int32))},
            4,
            "no decode",
        ),
        (
            {"gdn": _gdn_metadata(4, indices=torch.arange(4))},
            4,
            "no int32 state indices",
        ),
        ({"gdn": _gdn_metadata(4)}, 8, "state indices do not cover the rows"),
        (
            {"gdn": _gdn_metadata(4, indices=torch.zeros(8, dtype=torch.int32)[::2])},
            4,
            "state indices do not cover the rows",
        ),
    ],
)
def test_step_reason(md, width, why):
    assert step_reason(md, "gdn", width) == why


def test_mxfp6_projections_config(supported_platform):
    """``monokernel_projections`` selects the MXFP6 projections, which serve
    up to ``Q6_MAX`` rows, and an unknown value is refused with the others."""
    config = _vllm_config()
    config.additional_config = {runtime.PROJECTIONS_KEY: "mxfp6"}
    got = runtime.specialized_decode_config(config)
    assert got == runtime.DecodeConfig(max_tokens=L.Q6_MAX, fp6=True)
    assert got.widths == tuple(range(1, L.Q6_MAX + 1))
    config.additional_config = {runtime.PROJECTIONS_KEY: "auto"}
    assert runtime.specialized_decode_config(config) == runtime.DecodeConfig()
    config = _vllm_config(tensor_parallel_size=4)
    config.additional_config = {runtime.PROJECTIONS_KEY: "fp6"}
    with pytest.raises(ValueError) as e:
        runtime.specialized_decode_config(config)
    for why in ("TP 4 (TP8 only)", "monokernel_projections 'fp6'"):
        assert why in str(e.value), (why, str(e.value))


def _fake_layer(i: int, kind: str, fp6: bool) -> SimpleNamespace:
    """A decoder layer with what ``Launcher``'s build keys read."""
    norm = SimpleNamespace(variance_epsilon=1e-6, eps=1e-6)

    def lin():
        return SimpleNamespace(
            quant_method=None,
            bias=None,
            specialized_decode_fp6=(None, None) if fp6 else None,
        )

    mlp = SimpleNamespace(specialized_decode_rm=object() if fp6 else None)
    layer = SimpleNamespace(
        layer_idx=i,
        layer_type=kind,
        input_layernorm=norm,
        post_attention_layernorm=norm,
        mlp=mlp,
    )
    if kind == "linear_attention":
        layer.linear_attn = SimpleNamespace(
            in_proj_qkvz=lin(), out_proj=lin(), norm=norm
        )
    else:
        rope = SimpleNamespace(
            rotary_dim=64, cos_sin_cache=torch.empty(0, dtype=torch.bfloat16)
        )
        layer.self_attn = SimpleNamespace(
            qkv_proj=lin(), o_proj=lin(), q_norm=norm, rotary_emb=rope
        )
    return layer


@pytest.mark.parametrize("fp6", [False, True])
def test_builds(fp6: bool, monkeypatch):
    """The builds the executor compiles: with the MXFP6 projections, q6 / qp
    GDN builds, o6 layer builds with rm up to RM_MAX rows, and both attention
    front builds (with and without the KV cache write) of every width but
    the first layer's."""
    from vllm.models.qwen3_5.amd.specialized_decode.mono import afront, blocks
    from vllm.models.qwen3_5.amd.specialized_decode.runtime import decode, layers

    # Without the MXFP6 copies the fake linears are online MXFP4 on the ASM
    # FP4 GEMM, as the stock flags serve them.
    monkeypatch.setattr(layers, "mxfp4_shuffled", lambda lin: None if fp6 else True)
    kinds = ["linear_attention"] * 3 + ["full_attention"]
    model = SimpleNamespace(
        layers=[_fake_layer(i, k, fp6) for i, k in enumerate(kinds)]
    )
    sd = decode.SpecializedDecode.__new__(decode.SpecializedDecode)
    sd.model = model
    width = L.Q6_MAX if fp6 else L.MAX_TOKENS
    sd.config = runtime.DecodeConfig(max_tokens=width, fp6=fp6)
    sd.launch = layers.Launcher.__new__(layers.Launcher)
    keys = [k for _, k in sd._builds()]
    gdn = [k for k in keys if isinstance(k, blocks.GdnBuild)]
    lay = [k for k in keys if isinstance(k, blocks.LayerBuild)]
    af = [k for k in keys if isinstance(k, afront.AFBuild)]
    assert {k.tokens for k in gdn} == set(range(1, width + 1))
    assert all(k.q6 == fp6 and k.qp == fp6 for k in gdn)
    assert all(k.o6 == fp6 and k.oq != fp6 for k in lay)
    assert {(k.tokens, k.og) for k in lay if k.rm} == (
        {(s, og) for s in range(1, blocks.RM_MAX + 1) for og in (False, True)}
        if fp6
        else set()
    )
    assert {(k.tokens, k.kv) for k in af} == (
        {(s, kv) for s in range(1, width + 1) for kv in (False, True)} if fp6 else set()
    )
