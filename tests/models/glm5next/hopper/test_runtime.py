# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3-Flash's specialized decode on Hopper, on the CPU: the SM90 refusals
(GB300's unchanged), the Hopper policy table, the dispatch to the Hopper
executor, plan() on synthetic metadata, the stage-module fallback, and the
executor's layer loop on fake stages against the stock decoder and MTP layer
forwards. Two TP ranks are simulated: a TP-partial output is half the full one
and an all-reduce doubles it, so a missed or repeated reduction shows."""

import subprocess
import sys
import types
from collections import Counter
from types import SimpleNamespace

import pytest
import torch

from vllm.config import CompilationMode, CUDAGraphMode
from vllm.models.glm5next.common import mtp as mtp_mod
from vllm.models.glm5next.common.model import Glm5NextDecoderLayer
from vllm.models.glm5next.common.mtp import Glm5NextMultiTokenPredictorLayer
from vllm.models.glm5next.nvidia.specialized_decode import runtime
from vllm.models.glm5next.nvidia.specialized_decode.cutedsl.policy import batch_table
from vllm.models.glm5next.nvidia.specialized_decode.hopper import decode as hopper
from vllm.models.glm5next.nvidia.specialized_decode.hopper import steps
from vllm.models.glm5next.nvidia.specialized_decode.policy import (
    DEFAULT,
    HOPPER,
    Policy,
)
from vllm.models.glm5next.nvidia.specialized_decode.runtime import (
    DecodeConfig,
    specialized_decode_config,
    unsupported,
)
from vllm.models.glm5next.nvidia.specialized_decode.runtime import (
    decode as decode_mod,
)
from vllm.models.glm5next.nvidia.specialized_decode.runtime import draft as draft_mod
from vllm.platforms import current_platform
from vllm.utils import import_utils

PACKAGE = "vllm.models.glm5next.nvidia.specialized_decode.hopper"
STAGE_MODULES = ("kda", "dsa", "moe", "dense", "boundary", "logits")
TP = 2  # simulated: an all-reduce sums TP equal partials
H, N = 8, 4  # hidden size, mHC streams
F64 = torch.float64


# --- Refusals -----------------------------------------------------------------


def _vllm_config(**over):
    """A supported Hopper deployment (the recipe): TP4, MTP5, an auto (BF16) KV
    cache, FULL decode graphs. `over` sets dotted paths."""
    c = SimpleNamespace(
        additional_config={"decode_backend": "latency_specialized"},
        parallel_config=SimpleNamespace(
            tensor_parallel_size=4,
            pipeline_parallel_size=1,
            data_parallel_size=1,
            enable_expert_parallel=False,
            use_sequence_parallel_moe=False,
            use_ubatching=False,
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
        ),
        model_config=SimpleNamespace(
            dtype=torch.bfloat16,
            quantization="fp8",
            enable_sleep_mode=False,
            max_model_len=16384,
            hf_text_config=SimpleNamespace(**runtime.GEOMETRY, mhc=True),
        ),
        cache_config=SimpleNamespace(
            cache_dtype="auto",
            mamba_cache_dtype="auto",
            mamba_ssm_cache_dtype="auto",
            use_replayssm=False,
            use_kda_recoverssm=False,
            kv_cache_dtype_skip_layers=[],
        ),
        quant_config=SimpleNamespace(
            is_checkpoint_fp8_serialized=True, weight_block_size=[128, 128]
        ),
        attention_config=SimpleNamespace(
            backend=None, backend_per_kind={}, hisparse_config=None
        ),
        lora_config=None,
        speculative_config=SimpleNamespace(method="mtp", num_speculative_tokens=5),
        compilation_config=SimpleNamespace(
            mode=CompilationMode.NONE,
            cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
            cudagraph_capture_sizes=[1, 2, 4, 6, 8, 12, 18, 24, 32, 48],
        ),
    )
    for path, value in over.items():
        obj = c
        *head, last = path.split(".")
        for name in head:
            obj = getattr(obj, name)
        setattr(obj, last, value)
    return c


@pytest.fixture
def gpu(monkeypatch):
    """`gpu(capability)` makes the platform report that GPU family."""

    def select(capability: int):
        monkeypatch.setattr(
            current_platform,
            "is_device_capability_family",
            lambda c, device_id=0: c // 10 == capability // 10,
        )
        monkeypatch.setattr(import_utils, "has_cutedsl", lambda: True)

    return select


def test_hopper_recipe_is_supported(gpu):
    gpu(90)
    assert unsupported(_vllm_config()) == []
    assert unsupported(_vllm_config(**{"cache_config.cache_dtype": "bfloat16"})) == []
    assert unsupported(_vllm_config(speculative_config=None)) == []
    config = specialized_decode_config(_vllm_config())
    assert config is not None and config.q == 6
    assert config.capture_sizes == (1, 2, 4, 6, 8, 12, 18, 24, 32, 48)


@pytest.mark.parametrize(
    "over, reason",
    [
        (
            {"cache_config.cache_dtype": "fp8"},
            "on SM90, KV cache dtypes other than BF16",
        ),
        ({"speculative_config.num_speculative_tokens": 6}, "more than 5 speculative"),
        ({"speculative_config.method": "eagle3"}, "speculative methods but MTP"),
        ({"parallel_config.tensor_parallel_size": 8}, "TP other than 4"),
        ({"parallel_config.pipeline_parallel_size": 2}, "pipeline parallelism"),
        ({"parallel_config.data_parallel_size": 2}, "data parallelism"),
        ({"parallel_config.enable_expert_parallel": True}, "expert parallelism"),
        ({"parallel_config.use_ubatching": True}, "DBO"),
        ({"parallel_config.decode_context_parallel_size": 2}, "decode context"),
        ({"parallel_config.prefill_context_parallel_size": 2}, "prefill context"),
        ({"parallel_config.use_sequence_parallel_moe": True}, "sequence-parallel"),
        ({"model_config.enable_sleep_mode": True}, "sleep mode"),
        ({"lora_config": object()}, "LoRA"),
        (
            {"compilation_config.mode": CompilationMode.STOCK_TORCH_COMPILE},
            "torch.compile",
        ),
        (
            {"compilation_config.cudagraph_mode": CUDAGraphMode.NONE},
            "FULL decode CUDA graphs",
        ),
        (
            {"compilation_config.cudagraph_mode": CUDAGraphMode.PIECEWISE},
            "FULL decode CUDA graphs",
        ),
        ({"cache_config.mamba_ssm_cache_dtype": "bfloat16"}, "KDA cache dtypes"),
        ({"cache_config.use_replayssm": True}, "ReplaySSM"),
        ({"cache_config.use_kda_recoverssm": True}, "ReplaySSM"),
        ({"model_config.hf_text_config.hidden_size": 6144}, "model geometry"),
        ({"model_config.hf_text_config.mhc": False}, "without mHC"),
    ],
)
def test_hopper_refusals(gpu, over, reason):
    gpu(90)
    reasons = unsupported(_vllm_config(**over))
    assert len(reasons) == 1 and reason in reasons[0], reasons


def test_every_reason_at_once(gpu):
    gpu(90)
    config = _vllm_config(
        **{
            "cache_config.cache_dtype": "fp8",
            "speculative_config.num_speculative_tokens": 7,
            "parallel_config.tensor_parallel_size": 2,
            "compilation_config.cudagraph_mode": CUDAGraphMode.NONE,
        }
    )
    reasons = unsupported(config)
    assert len(reasons) == 4, reasons
    with pytest.raises(ValueError, match="does not support") as error:
        specialized_decode_config(config)
    for reason in reasons:
        assert reason in str(error.value)


def test_gb300_refusals_unchanged(gpu):
    """GB300 keeps an FP8 KV cache and up to 7 speculative tokens."""
    gpu(100)
    fp8 = {"cache_config.cache_dtype": "fp8"}
    assert unsupported(_vllm_config(**fp8)) == []
    seven = {**fp8, "speculative_config.num_speculative_tokens": 7}
    assert unsupported(_vllm_config(**seven)) == []
    eight = {**fp8, "speculative_config.num_speculative_tokens": 8}
    assert unsupported(_vllm_config(**eight)) == ["more than 7 speculative tokens"]
    assert unsupported(_vllm_config()) == ["KV cache dtypes other than FP8 E4M3"]


@pytest.mark.parametrize("capability", [80, 89, 120])
def test_other_gpus_refused(gpu, capability):
    gpu(capability)
    reasons = unsupported(_vllm_config())
    assert "this GPU (SM100/103, or SM90)" in reasons


def test_switch_off_imports_no_hopper_code():
    """On Hopper, decode_backend=auto keeps the stock decode: importing the
    model and reading the config loads no Hopper module."""
    code = "\n".join(
        [
            "import sys",
            "from types import SimpleNamespace",
            "from vllm.platforms import current_platform",
            "current_platform.is_device_capability_family = (",
            "    lambda c, device_id=0: c // 10 == 9)",
            "import vllm.models.glm5next.common.model",
            "import vllm.models.glm5next.common.mtp",
            "from vllm.models.glm5next.nvidia.specialized_decode.runtime import (",
            "    specialized_decode_config)",
            "for extra in ({}, {'decode_backend': 'auto'}, None):",
            "    config = SimpleNamespace(additional_config=extra)",
            "    assert specialized_decode_config(config) is None",
            f"loaded = [m for m in sys.modules if m.startswith('{PACKAGE}')]",
            "assert not loaded, loaded",
        ]
    )
    subprocess.run([sys.executable, "-c", code], check=True)


# --- Policy -------------------------------------------------------------------


def test_hopper_policy_table():
    policy = Policy(None, HOPPER)
    assert set(policy.tables) == set(HOPPER)
    for stage, table in policy.tables.items():
        assert sorted(table) == list(range(1, 33)), stage
        assert set(table.values()) == {"auto"}, stage
    assert policy("moe", 24) == "auto" and policy("moe", 33) is None


def test_hopper_policy_overrides():
    policy = Policy({"moe": "", "boundary": "1-16:lamport1,17-32:lamport2"}, HOPPER)
    assert not policy.enabled("moe") and policy("moe", 6) is None
    assert policy("boundary", 12) == "lamport1" and policy("boundary", 24) == "lamport2"
    assert policy("kda", 6) == "auto"
    with pytest.raises(ValueError, match="Unknown"):
        Policy({"router": "32-64:split_k"}, HOPPER)  # a GB300 stage
    with pytest.raises(ValueError, match="Unknown"):
        Policy({"draft_boundary": "1-4:auto"})  # a Hopper stage on GB300


def test_gb300_policy_unchanged():
    assert Policy().tables == {s: batch_table(v) for s, v in DEFAULT.items()}
    assert "draft_boundary" not in DEFAULT and "router" not in HOPPER


# --- Fake model ---------------------------------------------------------------


def _rand(seed, *shape, scale=0.5):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=g, dtype=F64) * scale


def _rms(x):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)


def _pre(residual, fn, scale, base, norm_weight):
    """A stand-in for the mHC pre: mixes, then the normed layer input."""
    mix = torch.softmax(residual.mean(-1) * fn.double(), dim=-1)
    post = torch.sigmoid(residual.mean(-1, keepdim=True) * scale.double()[0])
    comb = torch.softmax(mix.unsqueeze(-1) + base.double().view(N, N), dim=-1)
    return post, comb, _rms((mix.unsqueeze(-1) * residual).sum(1)) * norm_weight


def _post(x, residual, post, comb):
    return comb.transpose(-1, -2) @ residual + post * x.unsqueeze(1)


def _fused(x, residual, post, comb, fn, scale, base, norm_weight):
    residual = _post(x, residual, post, comb)
    return (residual, *_pre(residual, fn, scale, base, norm_weight))


class _Norm:
    variance_epsilon = 1e-6

    def __init__(self, scale):
        self.weight = torch.full((H,), scale, dtype=F64)

    def __call__(self, x, residual=None):
        if residual is None:
            return _rms(x) * self.weight
        total = x + residual
        return _rms(total) * self.weight, total


class _Attention:
    """A stock attention layer: o_proj reduces unless told not to."""

    def __init__(self, prefix, seed):
        self.prefix = prefix
        self.weight = _rand(seed, H, H)
        self.o_proj = SimpleNamespace(reduce_results=True)
        self.kv_cache = (f"cache of {prefix}",)
        # The MLA layer's KV cache: the runner's placeholder until bound.
        backend = SimpleNamespace(get_name=lambda: hopper.DSA_BACKEND)
        mla = SimpleNamespace(kv_cache=torch.tensor([]), attn_backend=backend)
        self.mla_attn = SimpleNamespace(mla_attn=mla, skip_topk=False)

    def fn(self, x):
        return torch.tanh(x @ self.weight)

    def __call__(self, hidden_states, positions):
        out = self.fn(hidden_states)
        return out if self.o_proj.reduce_results else out / TP


class _Mlp:
    def __init__(self, moe, seed):
        self.moe = moe
        self.prefix = f"mlp{seed}"
        self.weight = _rand(seed, H, H)
        if moe:
            config = SimpleNamespace(skip_final_all_reduce=False)
            self.experts = SimpleNamespace(moe_config=config)
        else:
            self.down_proj = SimpleNamespace(reduce_results=True)

    def fn(self, x):
        return torch.relu(x @ self.weight) - 0.25 * x

    def __call__(self, x, **kwargs):
        if self.moe:
            reduced = not self.experts.moe_config.skip_final_all_reduce
        else:
            reduced = self.down_proj.reduce_results
        return self.fn(x) if reduced else self.fn(x) / TP


class _Layer:
    """The fields of `Glm5NextDecoderLayer` its forward and the executor read,
    with stand-in mHC kernels."""

    hidden_size, n, mhc, is_mtp_layer, is_sequence_parallel = H, N, True, False, False
    rms_norm_eps = hc_eps = 1e-6
    mhc_post_mult_value, mhc_sinkhorn_iterations = 2.0, 20
    hc_attn_fn_split = hc_ffn_fn_split = None

    def __init__(self, idx, kind, moe, total):
        self.layer_idx, self.layer_kind, self.num_hidden_layers = idx, kind, total
        self.self_attn = _Attention(f"model.layers.{idx}.self_attn", 10 + idx)
        self.mlp, self._mlp_is_moe = _Mlp(moe, 50 + idx), moe
        self.input_layernorm, self.post_attention_layernorm = _Norm(1.25), _Norm(0.75)
        # BF16-exact values: the Hopper boundary reads a BF16 copy.
        self.hc_attn_fn = torch.tensor([0.5, 0.25, 1.0, 2.0], dtype=F64) * (idx + 1)
        self.hc_ffn_fn = torch.tensor([1.0, 0.5, 0.75, 1.5], dtype=F64) * (idx + 1)
        self.hc_attn_scale = torch.tensor([0.5, 1.0, 1.0], dtype=F64)
        self.hc_ffn_scale = torch.tensor([0.25, 1.0, 1.0], dtype=F64)
        self.hc_attn_base = torch.arange(N * N, dtype=F64) / 8
        self.hc_ffn_base = -torch.arange(N * N, dtype=F64) / 16

    def hc_pre(self, x, fn, scale, base, norm_weight=None, norm_eps=0.0):
        return _pre(x, fn, scale, base, norm_weight)

    def hc_post(self, x, residual, post, comb):
        return _post(x, residual, post, comb)

    def hc_fused_post_pre(
        self, x, residual, post, comb, fn, scale, base, norm_weight=None, **kwargs
    ):
        return _fused(x, residual, post, comb, fn, scale, base, norm_weight)


# KDA, KDA, DSA, KDA, DSA layers; layer 0 dense, the rest MoE.
KINDS = ("kda", "kda", "mla", "kda", "mla")


def _backbone():
    layers = [_Layer(i, k, i > 0, len(KINDS)) for i, k in enumerate(KINDS)]
    return SimpleNamespace(
        layers=layers,
        config=SimpleNamespace(num_experts_per_tok=8),
        aux_hidden_state_layers=(),
        norm=_Norm(1.5),
    )


def _stock_backbone(backbone, x, positions):
    """`Glm5NextModel.forward` without PP / SP / aux layers."""
    state = (x, None, None, None)
    for layer in backbone.layers:
        state = Glm5NextDecoderLayer.forward(layer, positions, *state)
    return backbone.norm(state[0])


def _fake_modules(present):
    """Fake `hopper.<module>` stage modules with the documented interfaces;
    each computes its layer's stock function, TP-partial."""
    calls: Counter = Counter()

    class HopperKdaLayer:
        def __init__(self, attention, batches, q):
            if q > 8:
                raise ValueError("q above 8")
            self.attention, self.batches, self.q = attention, set(batches), q
            self.prepared = False

        def prepare_caches(self, caches):
            assert caches is self.attention.kv_cache
            self.prepared = True

        def supports(self, tokens, q):
            return self.prepared and q == self.q and tokens in self.batches

        def run(self, x, metadata, q, reduce=False):
            assert metadata.prefix == self.attention.prefix and q == self.q
            calls["kda"] += 1
            out = self.attention.fn(x)
            return out if reduce else out / TP

    class HopperDsaLayer:
        def __init__(self, attention, tokens, q):
            self.attention, self.batches, self.q = attention, set(tokens), q
            self.prepared = 0

        def prepare_caches(self, *_):
            self.prepared += 1

        def supports(self, tokens, q):
            return q == self.q and tokens in self.batches

        def run(self, x, positions, metadata, q):
            assert positions.shape[-1] == x.shape[0] and q == self.q
            assert isinstance(metadata, dict)
            calls["dsa"] += 1
            return self.attention.fn(x) / TP

    def mlp_stage(name, moe):
        class Stage:
            @classmethod
            def of(cls, mlp, *config):
                assert len(config) == moe  # the MoE stage takes the text config
                return cls(mlp) if mlp.moe == moe else None

            def __init__(self, mlp):
                self.mlp = mlp
                self.prepared: list[int] = []

            def supports(self, tokens):
                return 1 <= tokens <= 32

            def prepare(self, tokens):
                self.prepared = list(tokens)

            def __call__(self, x):
                assert x.shape[0] in self.prepared
                calls[name] += 1
                return self.mlp.fn(x) / TP

        Stage.__name__ = name
        return Stage

    class HopperBoundary:
        def __init__(self, **kwargs):
            self.kwargs, self.modes, self.joined = kwargs, kwargs["modes"], 0

        def supports(self, tokens):
            return tokens in self.modes

        def __call__(self, x, residual, post, comb, fn, scale, base, norm_weight):
            assert fn.dtype == torch.bfloat16 and residual.is_contiguous()
            calls["boundary"] += 1
            return _fused(TP * x, residual, post, comb, fn, scale, base, norm_weight)

        def join(self):
            self.joined += 1

    class HopperDraftAllReduceNorm:
        def __init__(self, hidden_size, max_num_tokens, top_k, eps, device):
            self.max_num_tokens = max_num_tokens

        def supports(self, tokens):
            return tokens <= self.max_num_tokens

        def __call__(self, x, residual, norm_weight):
            calls["draft_boundary"] += 1
            total = TP * x + residual
            return _rms(total) * norm_weight, total

    class HopperLogitsHead:
        def __init__(self, weight, tokens):
            self.weight, self.tokens = weight, set(tokens)

        def supports(self, rows):
            return rows in self.tokens and rows <= 4

        def __call__(self, x):
            calls["logits_head"] += 1
            return x @ self.weight.T

    class HopperLogitsGather:
        def __init__(self, columns, capacity, device):
            self.capacity = capacity

        def __call__(self, logits):
            calls["logits_gather"] += 1
            return torch.cat([logits] * TP, dim=1)

    contents: dict[str, dict[str, object]] = {
        "kda": {"HopperKdaLayer": HopperKdaLayer},
        "dsa": {"HopperDsaLayer": HopperDsaLayer},
        "moe": {"HopperMoe": mlp_stage("moe", True)},
        "dense": {"HopperDenseMlp": mlp_stage("dense_mlp", False)},
        "boundary": {
            "HopperBoundary": HopperBoundary,
            "HopperDraftAllReduceNorm": HopperDraftAllReduceNorm,
            "DEFAULT_MODES": {m: ("lamport", 1 + (m > 16)) for m in range(1, 33)},
            "parse_mode": lambda s: (s.rstrip("0123456789"), int(s[-1])),
        },
        "logits": {
            "HopperLogitsHead": HopperLogitsHead,
            "HopperLogitsGather": HopperLogitsGather,
        },
    }
    modules: dict[str, types.ModuleType | None] = {}
    for name in STAGE_MODULES:
        if name in present:
            module = types.ModuleType(f"{PACKAGE}.{name}")
            module.__dict__.update(contents[name])
            modules[name] = module
        else:
            modules[name] = None  # import raises ModuleNotFoundError
    return modules, calls


@pytest.fixture
def stages(monkeypatch):
    """`stages(present)` installs fake stage modules (absent ones import as
    missing) and returns their call counter."""
    monkeypatch.setattr(decode_mod, "tensor_model_parallel_all_reduce", _reduce)
    monkeypatch.setattr(hopper, "tensor_model_parallel_all_reduce", _reduce)

    def install(present=STAGE_MODULES):
        modules, calls = _fake_modules(present)
        for name, module in modules.items():
            monkeypatch.setitem(sys.modules, f"{PACKAGE}.{name}", module)
        return calls

    return install


def _reduce(x):
    return TP * x


def _config(q=6, policy=(), sizes=(1, 2, 3, 4, 6, 8, 12, 18, 24, 32, 36, 48)):
    return DecodeConfig(
        max_context=4096, q=q, capture_sizes=tuple(sizes), policy=tuple(policy)
    )


def _kda_md(prefix, tokens, q, prefill=False):
    if q == 1:
        fields = dict(num_decodes=tokens, num_decode_tokens=tokens, num_spec_decodes=0)
    else:
        fields = dict(
            num_decodes=0,
            num_decode_tokens=0,
            num_spec_decodes=tokens // q,
            num_spec_decode_tokens=tokens,
            spec_state_indices_tensor=torch.zeros(tokens // q, q, dtype=torch.int32),
        )
    return SimpleNamespace(prefix=prefix, num_prefills=int(prefill), **fields)


def _mla_md(tokens, q, prefill=False):
    return SimpleNamespace(
        num_prefills=int(prefill),
        num_decodes=tokens // q,
        num_decode_tokens=tokens,
    )


def _metadata(backbone, tokens, q, prefill=False):
    by_layer = {}
    for layer in backbone.layers:
        prefix = layer.self_attn.prefix
        if layer.layer_kind == "kda":
            by_layer[prefix] = _kda_md(prefix, tokens, q, prefill)
        else:
            by_layer[prefix] = _mla_md(tokens, q, prefill)
    return by_layer


def _build(backbone, config):
    decode = hopper.HopperSpecializedDecode(backbone, config)
    decode.prepare_kv()
    return decode


# --- Executor -----------------------------------------------------------------

ALL_HOPPER = {"kda": 3, "dsa": 2, "moe": 4, "dense_mlp": 1, "boundary": 9}


@pytest.mark.parametrize(
    "present, policy, hopper_calls",
    [
        (STAGE_MODULES, {}, ALL_HOPPER),
        # Sublayers reduce their own outputs; the boundary runs stock.
        (("kda", "dsa", "moe", "dense"), {}, {**ALL_HOPPER, "boundary": 0}),
        # Stock sublayers leave their outputs TP-partial for the boundary.
        (("boundary",), {}, {"boundary": 9}),
        (("boundary", "moe"), {}, {"boundary": 9, "moe": 4}),
        (STAGE_MODULES, {"moe": "", "kda": ""}, {**ALL_HOPPER, "moe": 0, "kda": 0}),
        (STAGE_MODULES, {"boundary": "1-6:auto"}, {**ALL_HOPPER, "boundary": 0}),
        (STAGE_MODULES, {"dsa": "12:auto"}, ALL_HOPPER),
        (STAGE_MODULES, {"dense_mlp": "1-8:auto"}, {**ALL_HOPPER, "dense_mlp": 0}),
    ],
)
def test_executor_matches_stock(stages, present, policy, hopper_calls):
    """Each combination of Hopper and stock stages gives the stock layers'
    output, with every stage on the path the policy picks."""
    calls = stages(present)
    backbone = _backbone()
    decode = _build(backbone, _config(policy=policy.items()))
    tokens, q = 12, 6
    x = _rand(0, tokens, H)
    positions = torch.arange(tokens)
    step = decode_mod.Step(q, positions, _metadata(backbone, tokens, q))
    out = decode.run(step, None, x)
    torch.testing.assert_close(
        out, _stock_backbone(backbone, x, positions), rtol=0, atol=0
    )
    assert {k: v for k, v in calls.items() if v} == {
        k: v for k, v in hopper_calls.items() if v
    }
    # The executor's own counts: Hopper calls / all calls per stage.
    counts = {name: decode.calls[name, True] for name, _ in decode.calls}
    assert {k: v for k, v in counts.items() if v} == {
        k: v for k, v in hopper_calls.items() if v
    }
    # Every attention stage was compiled for the bound caches.
    for stages in decode.attn_stages.values():
        assert all(stage.kernel.prepared for stage in stages)
    # Stock layers are left as they were.
    for layer in backbone.layers:
        assert layer.self_attn.o_proj.reduce_results
        if layer._mlp_is_moe:
            assert not layer.mlp.experts.moe_config.skip_final_all_reduce
        else:
            assert layer.mlp.down_proj.reduce_results
    if decode.ar_mhc is not None:
        assert decode.ar_mhc.joined == 1
        assert decode.ar_mhc.kwargs["max_num_tokens"] == max(decode.ar_mhc.modes)


def test_boundary_modes(stages):
    stages()
    decode = _build(_backbone(), _config(policy={"boundary": "1-6:lamport3"}.items()))
    assert decode.ar_mhc.modes == {6: ("lamport", 3)}
    decode = _build(_backbone(), _config(q=1))
    assert decode.ar_mhc.modes[8] == ("lamport", 1)
    assert decode.ar_mhc.modes[24] == ("lamport", 2)
    assert 36 not in decode.ar_mhc.modes  # above MAX_TOKENS


def test_plain_decode_matches_stock(stages):
    calls = stages()
    backbone = _backbone()
    decode = _build(backbone, _config(q=1))
    assert decode.batches == [1, 2, 3, 4, 6, 8, 12, 18, 24, 32]
    x = _rand(1, 3, H)
    step = decode_mod.Step(1, torch.arange(3), _metadata(backbone, 3, 1))
    out = decode.run(step, None, x)
    torch.testing.assert_close(
        out, _stock_backbone(backbone, x, torch.arange(3)), rtol=0, atol=0
    )
    assert calls["kda"] == 3 and calls["boundary"] == 9


def test_no_stage_module_runs_stock(stages):
    """Switched on with no stage module (or every stage off): plan() leaves
    every step to the stock forward."""
    stages(present=())
    decode = _build(_backbone(), _config())
    assert not decode.active
    assert decode.plan(torch.arange(12), None) is None
    stages()
    off = {stage: "" for stage in HOPPER}
    decode = _build(_backbone(), _config(policy=off.items()))
    assert not decode.active


def test_stage_declines_its_layer(stages):
    """A stage that declines its layer (constructor ValueError, `of` None) runs
    stock for that layer only."""
    stages()
    decode = _build(_backbone(), _config(q=9, sizes=(9, 18, 27)))
    assert all(s[0].name == "dsa" for s in decode.attn_stages.values())
    assert len(decode.attn_stages) == 2


def test_dsa_needs_its_backend(stages):
    """DSA layers on another sparse MLA backend run stock."""
    stages()
    backbone = _backbone()
    other = SimpleNamespace(get_name=lambda: "FLASHMLA_SPARSE")
    backbone.layers[2].self_attn.mla_attn.mla_attn.attn_backend = other
    decode = _build(backbone, _config())
    dsa = [i for i, s in decode.attn_stages.items() if s[0].name == "dsa"]
    assert dsa == [4]


def test_stage_class_fallback(monkeypatch):
    assert steps.stage_class("not_a_stage", "Anything") is None

    def broken(path):
        raise ModuleNotFoundError("No module named 'cutlass_x'", name="cutlass_x")

    monkeypatch.setattr(steps.importlib, "import_module", broken)
    with pytest.raises(ModuleNotFoundError):
        steps.stage_class("moe", "HopperMoe")


# --- plan() -------------------------------------------------------------------


@pytest.fixture
def capture(monkeypatch):
    """`capture(metadata, mode)` makes plan() see a CUDA graph capture of
    `metadata` in `mode`."""
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    monkeypatch.setattr(decode_mod, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(draft_mod, "is_forward_context_available", lambda: True)

    def set_context(metadata, mode=CUDAGraphMode.FULL):
        context = SimpleNamespace(cudagraph_runtime_mode=mode, attn_metadata=metadata)
        monkeypatch.setattr(decode_mod, "get_forward_context", lambda: context)
        monkeypatch.setattr(draft_mod, "get_forward_context", lambda: context)

    return set_context


@pytest.mark.parametrize("tokens", [6, 12, 18, 24])
def test_plan_verify(stages, capture, tokens):
    stages()
    backbone = _backbone()
    decode = _build(backbone, _config())
    capture(_metadata(backbone, tokens, 6))
    step = decode.plan(torch.arange(tokens), None)
    assert step is not None and step.q == 6


def test_plan_runs_stock(stages, capture, monkeypatch):
    stages()
    backbone = _backbone()
    decode = _build(backbone, _config(q=6))
    positions = torch.arange(12)
    capture(_metadata(backbone, 12, 6, prefill=True))
    assert decode.plan(positions, None) is None  # prompt rows
    capture(_metadata(backbone, 12, 6), CUDAGraphMode.PIECEWISE)
    assert decode.plan(positions, None) is None
    capture(_metadata(backbone, 36, 6))
    assert decode.plan(torch.arange(36), None) is None  # above 32 tokens
    capture(_metadata(backbone, 12, 6))
    assert decode.plan(positions, {"hidden_states": None}) is None  # PP input
    mixed = _metadata(backbone, 12, 6)
    mixed[backbone.layers[0].self_attn.prefix].num_spec_decode_tokens = 11
    capture(mixed)
    assert decode.plan(positions, None) is None  # not every request verifies q
    capture(_metadata(backbone, 12, 6))
    assert decode.plan(positions, None) is not None
    backbone.aux_hidden_state_layers = (2,)
    assert decode.plan(positions, None) is None  # EAGLE-3 style aux layers
    backbone.aux_hidden_state_layers = ()
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    assert decode.plan(positions, None) is None  # eager: warmup, profiling


def test_plan_needs_a_hopper_stage(stages, capture):
    stages()
    backbone = _backbone()
    only_6 = {stage: "6:auto" for stage in HOPPER}
    decode = _build(backbone, _config(policy=only_6.items()))
    capture(_metadata(backbone, 12, 6))
    assert decode.plan(torch.arange(12), None) is None
    capture(_metadata(backbone, 6, 6))
    assert decode.plan(torch.arange(6), None) is not None


# --- Dispatch -----------------------------------------------------------------


def test_dispatch(stages, monkeypatch):
    stages()
    monkeypatch.setattr(runtime, "is_hopper", lambda: True)
    decode = decode_mod.SpecializedDecode.for_target(_backbone(), _config())
    assert type(decode) is hopper.HopperSpecializedDecode
    built = []
    monkeypatch.setattr(
        decode_mod.SpecializedDecode, "__init__", lambda self, *a: built.append(a)
    )
    monkeypatch.setattr(runtime, "is_hopper", lambda: False)
    decode = decode_mod.SpecializedDecode.for_target("backbone", "config")
    assert type(decode) is decode_mod.SpecializedDecode and built


# --- Logits -------------------------------------------------------------------


class _Processor:
    """The fields of `LogitsProcessor` the logits stages read."""

    logits_as_input, head_dtype, soft_cap, scale = False, None, None, 1.0

    def __init__(self, vocab):
        self.org_vocab_size = vocab
        self.calls: Counter = Counter()

    def __call__(self, lm_head, hidden_states, skip_gather=False):
        assert skip_gather
        self.calls["head"] += 1
        return hidden_states @ lm_head.weight.T

    def _gather_logits(self, logits):
        self.calls["gather"] += 1
        return torch.cat([logits] * TP, dim=1)


def test_logits(stages, monkeypatch):
    """The head GEMM (here up to 4 rows) and the gather (up to 8 rows, by
    the policy) each run where they serve; above both the stock processor
    runs."""
    calls = stages()
    decode = _build(_backbone(), _config(policy={"logits_gather": "1-8:auto"}.items()))
    vocab = 10
    lm_head = SimpleNamespace(weight=_rand(7, 6, H).bfloat16())  # a padded shard
    processor = _Processor(vocab)
    decode.build_logits(processor, lm_head)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    for rows in (3, 6, 12):
        calls.clear()
        processor.calls.clear()
        h = _rand(8, rows, H).bfloat16()
        logits = decode.compute_logits(processor, lm_head, h)
        if rows > 8:
            assert logits is None and not calls and not processor.calls
            continue
        expected = torch.cat([h @ lm_head.weight.T] * TP, dim=1)[:, :vocab]
        torch.testing.assert_close(logits, expected, rtol=0, atol=0)
        head = rows <= 4
        assert calls["logits_head"] == head and processor.calls["head"] == (not head)
        assert calls["logits_gather"] == 1 and processor.calls["gather"] == 0


# --- MTP draft ----------------------------------------------------------------


class _MtpBlock:
    hidden_size, n, mhc, is_mtp_layer, is_sequence_parallel = H, N, False, True, False
    layer_idx, layer_kind, _mlp_is_moe = len(KINDS), "mla", True

    def __init__(self):
        self.self_attn = _Attention("mtp.self_attn", 90)
        self.mlp = _Mlp(True, 91)
        self.input_layernorm, self.post_attention_layernorm = _Norm(1.1), _Norm(0.9)

    def __call__(self, positions, hidden_states, residual=None):
        return Glm5NextDecoderLayer.forward(self, positions, hidden_states, residual)


def _eh_norm(positions, embeds, previous, enorm_weight, hnorm_weight, eps):
    return torch.cat([_rms(embeds) * enorm_weight, _rms(previous) * hnorm_weight], -1)


def _predictor():
    mtp = SimpleNamespace(
        config=SimpleNamespace(num_experts_per_tok=8),
        enorm=_Norm(1.2),
        hnorm=_Norm(0.8),
        eh_proj=_Linear(_rand(92, 2 * H, H)),
        shared_head=SimpleNamespace(norm=_Norm(1.3)),
        mtp_block=_MtpBlock(),
    )
    shard = SimpleNamespace(
        num_added_elements=0, org_vocab_start_index=0, org_vocab_end_index=4
    )
    embed = SimpleNamespace(weight=torch.zeros(4, H, dtype=F64), shard_indices=shard)
    return SimpleNamespace(_mtp_layers=[mtp], embed_tokens=embed), mtp


class _Linear:
    def __init__(self, weight):
        self.weight = weight

    def __call__(self, x):
        return x @ self.weight


@pytest.mark.parametrize("q, tokens", [(6, 12), (1, 2), (1, 4)])
def test_draft_matches_stock(stages, monkeypatch, q, tokens):
    """Draft prefills (q tokens a request, only the last rows go on) and draft
    decodes on the Hopper stages give the stock MTP layer's output."""
    calls = stages()
    monkeypatch.setattr(draft_mod, "fused_eh_norm", _eh_norm)
    monkeypatch.setattr(mtp_mod, "fused_eh_norm", _eh_norm)
    predictor, mtp = _predictor()
    draft = hopper.HopperSpecializedDraft(predictor, _config(q=6))
    assert draft.active and not draft.fused_embedding
    embeds, previous = _rand(3, tokens, H), _rand(4, tokens, H)
    positions = torch.arange(tokens)
    rows = torch.arange(q - 1, tokens, q) if q > 1 else None
    step = decode_mod.Step(q, positions, {})
    out, _ = draft.run(mtp, step, None, embeds, previous, rows)
    stock, _ = Glm5NextMultiTokenPredictorLayer.forward(
        mtp, None, positions, previous, embeds
    )
    keep = rows if rows is not None else positions
    # The executor's MoE sees only the sampled rows of a draft prefill.
    torch.testing.assert_close(out[keep], stock[keep], rtol=0, atol=1e-12)
    assert calls["dsa"] == 1 and calls["moe"] == 1 and calls["draft_boundary"] == 2
    assert mtp.mtp_block.self_attn.o_proj.reduce_results


def test_draft_plan(stages, capture):
    stages()
    predictor, _ = _predictor()
    draft = hopper.HopperSpecializedDraft(predictor, _config(q=6))
    md = {"mtp.self_attn": _mla_md(12, 6)}
    capture(md)
    step = draft.plan(torch.arange(12))
    assert step is not None and step.q == 6
    capture({"mtp.self_attn": _mla_md(3, 1)})
    step = draft.plan(torch.arange(3))
    assert step is not None and step.q == 1
    capture({"mtp.self_attn": _mla_md(36, 6)})
    assert draft.plan(torch.arange(36)) is None


def test_draft_prepares_bound_caches(stages, monkeypatch):
    """The draft, built before the KV caches are bound, compiles its attention
    stages on the first eager forward after each binding."""
    stages()
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    predictor, mtp = _predictor()
    draft = hopper.HopperSpecializedDraft(predictor, _config(q=6))
    kernels = [stage.kernel for stage in draft.attn_stages[mtp.mtp_block.layer_idx]]
    assert len(kernels) == 2  # q = 6 (draft prefill) and q = 1 (draft decode)
    assert draft.plan(torch.arange(6)) is None
    assert [k.prepared for k in kernels] == [0, 0]  # the profiling run: unbound
    mla = mtp.mtp_block.self_attn.mla_attn.mla_attn
    mla.kv_cache = torch.ones(4)
    draft.plan(torch.arange(6))
    draft.plan(torch.arange(6))
    assert [k.prepared for k in kernels] == [1, 1]
    mla.kv_cache = torch.ones(8)  # the real cache after the profiling one
    draft.plan(torch.arange(64))  # any eager forward, also above the token cap
    assert [k.prepared for k in kernels] == [2, 2]


# --- The stage modules' interfaces --------------------------------------------

# How the executor calls each stage (`hopper.decode`, `hopper.steps`):
# (module, class, [(member, positional args, keyword args)]).
CALLS = [
    (
        "moe",
        "HopperMoe",
        [
            ("of", ("moe", "config"), {}),
            ("supports", (1,), {}),
            ("prepare", ([1],), {}),
        ],
    ),
    ("dense", "HopperDenseMlp", [("of", ("mlp",), {}), ("supports", (1,), {})]),
    (
        "kda",
        "HopperKdaLayer",
        [
            (None, ("attention", [6], 6), {}),
            ("supports", (6, 6), {}),
            ("prepare_caches", ("caches",), {}),
            ("run", ("x", "md", 6), {"reduce": True}),
        ],
    ),
    (
        "dsa",
        "HopperDsaLayer",
        [
            (None, ("attention", [6], 6), {}),
            ("supports", (6, 6), {}),
            ("prepare_caches", (None,), {}),
            ("run", ("x", "positions", {}, 6), {}),
        ],
    ),
    (
        "boundary",
        "HopperBoundary",
        [
            (
                None,
                (),
                dict(
                    hidden_size=4096,
                    max_num_tokens=32,
                    rms_eps=1e-6,
                    hc_eps=1e-6,
                    post_mult=2.0,
                    sinkhorn_repeat=20,
                    norm_eps=1e-5,
                    device="cuda",
                    modes={},
                ),
            ),
            ("supports", (6,), {}),
            ("join", (), {}),
            (
                "__call__",
                ("x", "residual", "post", "comb", "fn", "scale", "base", "w"),
                {},
            ),
        ],
    ),
    (
        "boundary",
        "HopperDraftAllReduceNorm",
        [
            (
                None,
                (),
                dict(
                    hidden_size=4096,
                    max_num_tokens=32,
                    top_k=8,
                    eps=1e-5,
                    device="cuda",
                ),
            ),
            ("supports", (6,), {}),
            ("__call__", ("x", "residual", "weight"), {}),
            ("prepare_embedding", ("weight", 0, 100, 1e-5), {}),
            ("embed_norm", ("ids", "positions", "prev", "weight", "e", "h"), {}),
        ],
    ),
    (
        "logits",
        "HopperLogitsHead",
        [(None, ("weight", [1]), {}), ("supports", (1,), {}), ("__call__", ("x",), {})],
    ),
    (
        "logits",
        "HopperLogitsGather",
        [
            (None, (), dict(columns=1, capacity=1, device="cuda")),
            ("__call__", ("logits",), {}),
        ],
    ),
]


@pytest.mark.parametrize("module, name, calls", CALLS, ids=[c[1] for c in CALLS])
def test_stage_interface(module, name, calls):
    """The stage modules of this build take the executor's calls (skips a
    stage whose module is not part of the build)."""
    import inspect

    cls = steps.stage_class(module, name)
    if cls is None:
        pytest.skip(f"no hopper.{module} in this build")
    for member, args, kwargs in calls:
        if member is None:
            inspect.signature(cls).bind(*args, **kwargs)
        elif member == "of":
            inspect.signature(cls.of).bind(*args, **kwargs)
        else:
            inspect.signature(getattr(cls, member)).bind("self", *args, **kwargs)


def test_boundary_mode_helpers():
    found = steps.stage_class("boundary", "parse_mode")
    if found is None:
        pytest.skip("no hopper.boundary in this build")
    from vllm.models.glm5next.nvidia.specialized_decode.hopper import boundary

    assert boundary.parse_mode("lamport2") == ("lamport", 2)
    assert set(boundary.DEFAULT_MODES) <= set(range(1, hopper.MAX_TOKENS + 1))


def test_draft_stages_have_their_own_entries(stages):
    """The draft's DSA and MoE switch apart from the backbone's."""
    stages()
    predictor, mtp = _predictor()
    index = mtp.mtp_block.layer_idx
    off = {"draft_dsa": "", "draft_moe": ""}
    draft = hopper.HopperSpecializedDraft(predictor, _config(q=6, policy=off.items()))
    assert not draft.attn_stages and not draft.ffn_stages and draft.active
    backbone_off = {"dsa": "", "moe": ""}
    draft = hopper.HopperSpecializedDraft(
        predictor, _config(q=6, policy=backbone_off.items())
    )
    assert index in draft.attn_stages and index in draft.ffn_stages
    decode = _build(_backbone(), _config(policy=off.items()))
    assert {s[0].name for s in decode.attn_stages.values()} == {"kda", "dsa"}
