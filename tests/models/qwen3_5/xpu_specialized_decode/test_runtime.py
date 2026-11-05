# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The Qwen3.5 XPU mono decode on CPU: the switch, its refusals (config, loaded
weights, caches, TP agreement), the model state it selects and plan() routing."""

from types import SimpleNamespace

import pytest
import torch

from vllm.config import CompilationMode, CUDAGraphMode
from vllm.models.qwen3_5.xpu.specialized_decode import hooks, runtime
from vllm.models.qwen3_5.xpu.specialized_decode.runtime import (
    SHAPES,
    DecodeConfig,
    specialized_decode_config,
    unsupported,
)
from vllm.models.qwen3_5.xpu.specialized_decode.runtime import decode as decode_mod
from vllm.models.qwen3_5.xpu.specialized_decode.runtime.decode import (
    draft_kind,
    step_kind,
)
from vllm.models.qwen3_5.xpu.specialized_decode.runtime.guards import bind_reasons


class _Ops:
    """``torch.ops._xpu_C`` with the mono decode kernels."""

    def __init__(self, xe2=True, max_ctx=512 * 1024):
        self.xe2, self.max_ctx = xe2, max_ctx

    def __getattr__(self, name):  # the other ops: presence only
        if name not in runtime.OPS:
            raise AttributeError(name)

        def called(*args):
            raise AssertionError(f"{name} called")

        return called

    def mono_decode_supported(self, device):
        return self.xe2

    def mono_max_context(self):
        return self.max_ctx


@pytest.fixture(autouse=True)
def _xpu(monkeypatch):
    """Run the platform check as on an XPU with the kernels built."""
    from vllm.platforms import current_platform

    monkeypatch.setattr(current_platform, "is_xpu", lambda: True)


def _spec(k=3, method="mtp"):
    s = SimpleNamespace(
        method=method, num_speculative_tokens=k, enable_adaptive_verification=False
    )
    s.uses_dynamic_speculative_decoding = lambda: False
    return s


def _vllm_config(**over) -> SimpleNamespace:
    """A supported config: Qwen3.6-35B-A3B shapes, TP2, fp16, online fp8
    per-tensor, block 64, FULL_AND_PIECEWISE graphs, MRV2, --monokernel.
    ``over`` sets dotted paths, e.g. ``{"parallel_config.data_parallel_size": 2}``."""
    c = SimpleNamespace(
        kernel_config=SimpleNamespace(monokernel=True),
        model_config=SimpleNamespace(
            architectures=["Qwen3_5MoeForConditionalGeneration"],
            dtype=torch.float16,
            quantization="fp8_per_tensor",
            hf_text_config=SimpleNamespace(**SHAPES),
            max_model_len=40960,
            enable_sleep_mode=False,
            enforce_eager=False,
        ),
        quant_config=object(),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=2,
            pipeline_parallel_size=1,
            data_parallel_size=1,
            enable_expert_parallel=False,
            enable_eplb=False,
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
            use_ubatching=False,
            enable_batch_sharded_sampling=None,
        ),
        cache_config=SimpleNamespace(
            cache_dtype="auto",
            block_size=64,
            mamba_cache_dtype="auto",
            mamba_ssm_cache_dtype="auto",
        ),
        compilation_config=SimpleNamespace(
            mode=CompilationMode.NONE,
            cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
        ),
        offload_config=SimpleNamespace(
            uva=SimpleNamespace(cpu_offload_gb=0),
            prefetch=SimpleNamespace(offload_group_size=0),
        ),
        aux_output_config=SimpleNamespace(enable_return_routed_experts=False),
        lora_config=None,
        kv_transfer_config=None,
        weight_transfer_config=None,
        speculative_config=None,
        use_v2_model_runner=True,
    )
    for path, value in over.items():
        obj = c
        *head, last = path.split(".")
        for h in head:
            obj = getattr(obj, h)
        setattr(obj, last, value)
    return c


def _reasons(cfg, **kw) -> list[str]:
    return unsupported(cfg, ops=_Ops(**kw))


def test_served_deployments(monkeypatch):
    monkeypatch.setattr(runtime, "_ops", lambda ops: _Ops())
    assert specialized_decode_config(_vllm_config()) == DecodeConfig(q=1)
    for k in (1, 2, 3, 7):
        cfg = _vllm_config(speculative_config=_spec(k))
        assert specialized_decode_config(cfg) == DecodeConfig(q=k + 1)
    graphs = _vllm_config(
        **{"compilation_config.cudagraph_mode": CUDAGraphMode.FULL_DECODE_ONLY}
    )
    assert _reasons(graphs) == []
    # bf16 activations with a bf16 (auto or explicit) KV cache and conv state
    bf16 = {"model_config.dtype": torch.bfloat16}
    assert _reasons(_vllm_config(**bf16)) == []
    explicit = {
        "cache_config.cache_dtype": "bfloat16",
        "cache_config.mamba_cache_dtype": "bfloat16",
    }
    assert _reasons(_vllm_config(**bf16, **explicit)) == []


def test_switch_off_checks_nothing():
    """Without --monokernel the config is not even looked at."""
    cfg = SimpleNamespace(kernel_config=SimpleNamespace(monokernel=False))
    assert specialized_decode_config(cfg) is None


@pytest.mark.parametrize(
    "over,needle",
    [
        ({"parallel_config.tensor_parallel_size": 1}, "TP 1"),
        ({"parallel_config.tensor_parallel_size": 4}, "TP 4"),
        ({"parallel_config.pipeline_parallel_size": 2}, "pipeline"),
        ({"parallel_config.data_parallel_size": 2}, "data parallelism"),
        ({"parallel_config.enable_expert_parallel": True}, "expert parallelism"),
        ({"parallel_config.enable_eplb": True}, "EPLB"),
        ({"parallel_config.decode_context_parallel_size": 2}, "decode context"),
        ({"parallel_config.use_ubatching": True}, "DBO"),
        ({"parallel_config.enable_batch_sharded_sampling": True}, "batch-sharded"),
        ({"lora_config": object()}, "LoRA"),
        ({"kv_transfer_config": object()}, "KV transfer"),
        ({"weight_transfer_config": object()}, "weight updates"),
        ({"model_config.enable_sleep_mode": True}, "sleep mode"),
        ({"offload_config.uva.cpu_offload_gb": 4}, "CPU weight offload"),
        ({"model_config.dtype": torch.float32}, "dtype"),
        # an fp16 / bf16 mix of activations and KV cache / conv state
        ({"cache_config.cache_dtype": "bfloat16"}, "KV cache dtype"),
        (
            {
                "model_config.dtype": torch.bfloat16,
                "cache_config.cache_dtype": "float16",
            },
            "KV cache dtype",
        ),
        ({"cache_config.mamba_cache_dtype": "bfloat16"}, "conv state dtype"),
        ({"model_config.quantization": None, "quant_config": None}, "quantization"),
        ({"model_config.quantization": "fp8_per_block"}, "quantization"),
        ({"cache_config.cache_dtype": "fp8"}, "KV cache dtype"),
        ({"cache_config.block_size": 16}, "block size"),
        ({"cache_config.mamba_ssm_cache_dtype": "float16"}, "recurrent state"),
        ({"model_config.max_model_len": 1 << 20}, "max_model_len"),
        ({"model_config.architectures": ["Qwen3_5ForCausalLM"]}, "architecture"),
        ({"model_config.hf_text_config.hidden_size": 4096}, "hidden_size"),
        ({"model_config.hf_text_config.num_experts": 512}, "num_experts"),
        ({"speculative_config": _spec(1, method="eagle3")}, "eagle3"),
        ({"speculative_config": _spec(8)}, "speculative tokens"),
        ({"use_v2_model_runner": False}, "V1 model runner"),
        (
            {"compilation_config.mode": CompilationMode.STOCK_TORCH_COMPILE},
            "torch.compile",
        ),
        ({"compilation_config.cudagraph_mode": CUDAGraphMode.PIECEWISE}, "FULL"),
        ({"compilation_config.cudagraph_mode": CUDAGraphMode.NONE}, "FULL"),
        ({"model_config.enforce_eager": True}, "FULL"),
    ],
)
def test_refusals(over, needle):
    reasons = _reasons(_vllm_config(**over))
    assert any(needle in r for r in reasons), reasons


def test_every_reason_at_once(monkeypatch):
    cfg = _vllm_config(
        **{
            "parallel_config.tensor_parallel_size": 1,
            "cache_config.cache_dtype": "fp8",
            "use_v2_model_runner": False,
        }
    )
    monkeypatch.setattr(runtime, "_ops", lambda ops: _Ops(xe2=False))
    with pytest.raises(ValueError) as e:
        specialized_decode_config(cfg)
    msg = str(e.value)
    for needle in ("TP 1", "KV cache dtype", "V1 model runner", "Xe2"):
        assert needle in msg, msg


def test_platform_refusals(monkeypatch):
    from vllm.platforms import current_platform

    assert any("Xe2" in r for r in _reasons(_vllm_config(), xe2=False))
    assert (
        "without the mono decode kernels"
        in unsupported(_vllm_config(), ops=object())[-1]
    )
    # an older build without one of the ops
    old = SimpleNamespace(
        **{n: getattr(_Ops(), n) for n in runtime.OPS if n != "mono_shard_topk"}
    )
    assert "missing mono_shard_topk)" in unsupported(_vllm_config(), ops=old)[-1]
    monkeypatch.setattr(current_platform, "is_xpu", lambda: False)
    assert unsupported(_vllm_config(), ops=_Ops())[-1] == "this platform (XPU only)"


def test_peer_refusal_raises_on_every_rank(monkeypatch):
    """One rank refusing (its loaded weights, its caches) raises on all."""
    monkeypatch.setattr(decode_mod, "tp_agree", lambda ok, group: False)
    monkeypatch.setattr(
        decode_mod, "get_tp_group", lambda: SimpleNamespace(cpu_group=None)
    )
    with pytest.raises(ValueError, match="another TP rank refused"):
        decode_mod.agree([], "the KV caches")
    with pytest.raises(ValueError, match="fp8 scales"):
        decode_mod.agree(["fp8 scales other than per-tensor"], "the loaded weights")


def test_bind_refusal_is_a_reason():
    def bind():
        raise RuntimeError("mono: kv_cache must be [blocks, 1, bs, 512] fp16\nmore")

    assert bind_reasons(bind) == ["mono: kv_cache must be [blocks, 1, bs, 512] fp16"]
    assert bind_reasons(lambda: None) == []


def test_model_state_selection():
    from vllm.models.qwen3_5.xpu.model_state import Qwen3_5XpuModelState
    from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState

    off = SimpleNamespace(specialized_decode_config=None)
    on = SimpleNamespace(specialized_decode_config=DecodeConfig(q=2))
    assert hooks.model_state_cls(off) is MambaHybridModelState
    assert hooks.model_state_cls(on) is Qwen3_5XpuModelState


def test_drafter_registered_only_with_monokernel(monkeypatch):
    monkeypatch.setattr(hooks, "_drafters", [])
    p = torch.nn.Module()
    hooks.register_drafter(
        p, SimpleNamespace(kernel_config=SimpleNamespace(monokernel=False))
    )
    assert hooks._drafters == [] and p.specialized_decode is None
    hooks.register_drafter(
        p, SimpleNamespace(kernel_config=SimpleNamespace(monokernel=True))
    )
    assert hooks._drafters == [p]


def _gdn(prefills=0, decodes=0, spec=0, q=1):
    s = torch.zeros(max(spec, 1), q, dtype=torch.int32) if spec else None
    return SimpleNamespace(
        num_prefills=prefills,
        num_decodes=decodes,
        num_spec_decodes=spec,
        spec_state_indices_tensor=s,
        uniform_spec_sequence_length=q if spec else None,
    )


def _attn(max_query_len=1, prefill_tokens=0, decode_tokens=0):
    return SimpleNamespace(
        max_query_len=max_query_len,
        num_prefill_tokens=prefill_tokens,
        num_decode_tokens=decode_tokens,
    )


def _md(g, a):
    return {"gdn": g, "attn": a}


@pytest.mark.parametrize(
    "md,rows,q,kind",
    [
        (_md(_gdn(decodes=4), _attn()), 4, 1, 1),  # pure decode
        (_md(_gdn(decodes=3), _attn()), 4, 1, 1),  # padded to the graph's 4
        (_md(_gdn(prefills=1, decodes=3), _attn(7)), 16, 1, None),  # mixed
        (_md(_gdn(decodes=4), _attn(2)), 8, 1, None),
        (_md(_gdn(spec=2, q=4), _attn(4)), 8, 4, 4),  # verify 2 x 4
        (_md(_gdn(spec=1, q=4), _attn(4)), 8, 4, 4),  # padded requests
        (_md(_gdn(spec=2, q=4), _attn(4)), 6, 4, None),  # not whole requests
        (_md(_gdn(decodes=1, spec=1, q=4), _attn(4)), 8, 4, None),
        (_md(_gdn(prefills=1, spec=1, q=4), _attn(9)), 16, 4, None),
        ("not a dict", 4, 1, None),
    ],
)
def test_step_kind(md, rows, q, kind):
    assert step_kind(md, "gdn", "attn", rows, q) == kind


def test_draft_kind():
    assert draft_kind({"d": _attn(4)}, "d", 8, 4) == 4  # first pass of 2 x 4
    assert draft_kind({"d": _attn(1)}, "d", 2, 4) == 1  # later passes
    assert draft_kind({"d": _attn(4)}, "d", 6, 4) is None
    assert draft_kind({"d": _attn(4, prefill_tokens=3)}, "d", 8, 4) is None
    assert draft_kind({"d": _attn(9)}, "d", 16, 4) is None


def test_draft_positions():
    from vllm.models.qwen3_5.xpu.specialized_decode.runtime.sampling import (
        draft_positions,
    )

    # two requests (rows 0-2: 2 drafts, rows 3-4: 1 draft), 3 candidates and
    # the -inf column; a draft is looked up in the previous row's candidates
    idx = torch.tensor(
        [
            [10, 11, 12, -1],
            [20, 21, 22, -1],
            [30, 31, 32, -1],
            [40, 41, 42, -1],
            [50, 51, 52, -1],
        ]
    )
    drafts = torch.tensor([7, 11, 22, 9, 40])
    assert draft_positions(idx, drafts).tolist()[1:3] == [1, 2]
    assert draft_positions(idx, drafts)[4] == 0
    # not a candidate of the verifying row (20 is row 1's, not row 0's)
    assert draft_positions(idx, torch.tensor([7, 20, 22, 9, 40]))[1] == 3
    # placeholders stay placeholders
    pos = draft_positions(idx, torch.tensor([7, -1, -1, 9, 40]))
    assert pos[1:3].tolist() == [-1, -1]


def test_shared_expert_layout(monkeypatch):
    """The mono layers read the shared expert as vLLM's online quantization
    leaves it (the [K, N] view of a row-major [N, K] weight). Under
    --monokernel the fused MoE builds no plan and never replaces those weights
    with its row-major [K, N] copies (its steps run the stock MoE); without
    --monokernel it is unchanged."""
    import sys

    from vllm.models.qwen3_5.xpu.specialized_decode import moe

    shared_w13 = torch.randn(512, 2048).t()
    weights = (torch.zeros(1), torch.zeros(1, 4, 8), shared_w13)
    monkeypatch.setattr(moe, "get_weights", lambda block: weights)
    prepared: list = []

    def prepare(block):
        prepared.append(block)
        return weights[:2] + (shared_w13.contiguous(),) + (None,) * 6

    monkeypatch.setattr(moe, "prepare_weights", prepare)
    fused = SimpleNamespace(
        supports=lambda *a: True,
        XpuFusedMoeWithSharedExpert=lambda **kw: SimpleNamespace(device="xpu"),
    )
    monkeypatch.setitem(
        sys.modules, "vllm_xpu_kernels.moe_shared_fused_interface", fused
    )
    config = SimpleNamespace(in_dtype=torch.float16, hidden_dim=8, experts_per_token=2)

    def block(monokernel):
        return SimpleNamespace(
            experts=SimpleNamespace(moe_config=config), monokernel=monokernel
        )

    step = moe.MoEStep(block(True))
    assert step.plan is None and not prepared
    assert step.run(torch.zeros(1, 8, dtype=torch.float16)) is None  # stock MoE
    assert weights[2] is shared_w13

    step = moe.MoEStep(block(False))
    assert step.plan is not None and len(prepared) == 1
