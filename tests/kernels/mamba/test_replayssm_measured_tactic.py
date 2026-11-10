# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GB10 FP16 profile forwarding; real numerical kernels have separate tests."""

import pytest
import torch

from vllm.model_executor.layers.mamba.ops import ssu_dispatch

TUNING_ENVS = (
    "FLASHINFER_SSU_HEADS_PER_CTA",
    "FLASHINFER_SSU_PRECOMPUTE_NUM_WARPS",
    "FLASHINFER_SSU_MAIN_CTA_PER_SM",
    "FLASHINFER_SSU_MAIN_PIPELINE_STAGES",
)
SPLIT = dict(
    algorithm="two-kernel",
    d_split=2,
    main_pipeline_stages=2,
    main_ctas_per_sm=5,
    precompute_heads_per_cta=1,
)
MONO1 = dict(algorithm="monolith", d_split=1)
MONO2 = dict(algorithm="monolith", d_split=2)
SPLIT_SHAPES = {
    *((1, t, mode) for t in range(1, 7) for mode in ("philox", "shared_seed")),
    *((2, t, "philox") for t in range(1, 7)),
    (2, 5, "shared_seed"),
    (2, 6, "shared_seed"),
}
MONO1_SHAPES = {
    (b, t, mode) for b in (1, 2) for t in (7, 8) for mode in ("philox", "shared_seed")
}
MONO2_SHAPES = {(3, 5, "philox")}
POINTS = [
    (b, t, mode)
    for b in range(1, 5)
    for t in range(1, 9)
    if b * t <= 16
    for mode in ("philox", "shared_seed")
]


def _expected(point):
    if point in SPLIT_SHAPES:
        return SPLIT
    if point in MONO1_SHAPES:
        return MONO1
    if point in MONO2_SHAPES:
        return MONO2
    return {}


def _arguments(batch=1, tokens=8, policy="philox"):
    # Match the real fused producer's interleaved row pitch. Singleton axes may
    # have arbitrary unused strides after the ordinary-decode unsqueeze.
    projected = torch.empty(batch * tokens, 18560, dtype=torch.bfloat16)
    x = projected[:, 8192:16384].view(batch, tokens, 128, 64)
    ring = 16 + tokens
    return dict(
        state=torch.empty(batch + 2, 128, 64, 128, dtype=torch.float16),
        x=x,
        dt=projected[:, 18432:].view(batch, tokens, 128, 1).expand_as(x),
        A=torch.empty(128)[:, None, None].expand(128, 64, 128),
        B=projected[:, 16384:17408].view(batch, tokens, 8, 128),
        C=projected[:, 17408:18432].view(batch, tokens, 8, 128),
        out=torch.empty(batch, tokens, 128, 64, dtype=x.dtype),
        x_cache=torch.empty(batch + 2, 128, ring, 64, dtype=x.dtype),
        B_cache=torch.empty(batch + 2, 8, ring, 128, dtype=x.dtype),
        dt_cache=torch.empty(batch + 2, 128, ring),
        ring_start=torch.zeros(batch + 2, dtype=torch.int32),
        prev_num_accepted_tokens=torch.zeros(batch + 2, dtype=torch.int32),
        D=torch.empty(128, dtype=x.dtype)[:, None].expand(128, 64),
        dt_bias=torch.empty(128, dtype=x.dtype)[:, None].expand(128, 64),
        dt_softplus=True,
        state_batch_indices=torch.arange(1, batch + 1, dtype=torch.int32),
        scratch=(torch.empty(1), torch.empty(1), torch.empty(1)),
        enable_stochastic_rounding=True,
        stochastic_rounding_philox_rounds=5,
        stochastic_rounding_rng=policy,
        use_gb10_profile=True,
    )


@pytest.fixture
def calls(monkeypatch):
    for name in TUNING_ENVS:
        monkeypatch.delenv(name, raising=False)
    seen = []

    def kernel(*args, **kwargs):
        seen.append(kwargs)
        return args[11]

    monkeypatch.setattr(ssu_dispatch, "_flashinfer_replayssm_kernel", kernel)
    monkeypatch.setattr(
        ssu_dispatch, "_flashinfer_replayssm_main_controls_supported", True
    )
    return seen


def _invoke(kwargs, calls):
    with torch.random.fork_rng(devices=[]):
        ssu_dispatch.selective_state_update_replayssm_flashinfer(**kwargs)
    return calls[-1]


@pytest.mark.parametrize("batch,tokens,policy", POINTS)
def test_measured_replayssm_profile_grid(calls, batch, tokens, policy):
    kwargs = _arguments(batch, tokens, policy)
    actual = _invoke(kwargs, calls)
    expected = _expected((batch, tokens, policy))
    controls = {k: actual[k] for k in SPLIT if k in actual}
    assert controls == expected
    assert actual["philox_rounds"] == 5
    assert actual["rand_seed"].dtype == torch.int64
    assert actual["rand_seed"].shape == (1,)
    assert actual["cb_scaled"] is kwargs["scratch"][0]
    assert actual.get("sr_rng", "philox") == policy
    assert ("sr_rng" in actual) == (policy == "shared_seed")


def test_measured_replayssm_profile_scope_count():
    assert len(POINTS) == 50
    assert len(SPLIT_SHAPES | MONO1_SHAPES | MONO2_SHAPES) == 29


@pytest.mark.parametrize(
    "setting",
    [
        "disabled",
        "RN",
        "round10",
        "PDL",
        "varlen",
        "window",
        "bf16",
        "fp8",
        "int8",
        "state_layout",
        "input_layout",
        "output_layout",
        "indices_layout",
        "scaled_state",
        "no_scratch",
        "long_T",
        "large_B",
    ],
)
def test_measured_replayssm_unmatched_stays_auto(calls, setting):
    kwargs = _arguments(2, 8)
    if setting == "disabled":
        # Covers cached hardware/owner eligibility and a nonfused producer.
        kwargs["use_gb10_profile"] = False
    elif setting == "RN":
        kwargs["enable_stochastic_rounding"] = False
    elif setting == "round10":
        kwargs["stochastic_rounding_philox_rounds"] = 10
    elif setting == "PDL":
        kwargs["enable_pdl"] = True
    elif setting == "varlen":
        kwargs["cu_seqlens"] = torch.tensor([0, 8, 16], dtype=torch.int32)
        kwargs["max_seqlen"] = 8
    elif setting == "window":
        kwargs["x_cache"] = kwargs["x_cache"][:, :, :23]
    elif setting in ("bf16", "fp8", "int8"):
        dtype = {
            "bf16": torch.bfloat16,
            "fp8": torch.float8_e4m3fn,
            "int8": torch.int8,
        }[setting]
        kwargs["state"] = torch.empty(kwargs["state"].shape, dtype=dtype)
    elif setting == "state_layout":
        kwargs["state"] = (
            kwargs["state"].transpose(-1, -2).contiguous().transpose(-1, -2)
        )
    elif setting == "input_layout":
        kwargs["x"] = kwargs["x"].contiguous()
    elif setting == "output_layout":
        kwargs["out"] = torch.empty(2, 8, 128, 128, dtype=torch.bfloat16)[..., :64]
    elif setting == "indices_layout":
        kwargs["state_batch_indices"] = torch.arange(4, dtype=torch.int32)[::2]
    elif setting == "scaled_state":
        kwargs["state_scale"] = torch.empty(4, 128, 64)
    elif setting == "no_scratch":
        kwargs["scratch"] = None
    elif setting == "long_T":
        kwargs = _arguments(1, 9)
    elif setting == "large_B":
        kwargs = _arguments(5, 1)
    actual = _invoke(kwargs, calls)
    assert not any(name in actual for name in SPLIT)


@pytest.mark.parametrize("name", TUNING_ENVS)
def test_measured_replayssm_respects_explicit_native_env(calls, monkeypatch, name):
    monkeypatch.setenv(name, "0")  # Presence preserves the explicit user path.
    actual = _invoke(_arguments(1, 4), calls)
    assert not any(key in actual for key in SPLIT)


@pytest.mark.parametrize("batch,tokens", [(1, 4), (1, 8), (3, 5)])
def test_measured_replayssm_old_api_preserves_legal_choices(
    calls, monkeypatch, batch, tokens
):
    monkeypatch.setattr(
        ssu_dispatch, "_flashinfer_replayssm_main_controls_supported", False
    )
    actual = _invoke(_arguments(batch, tokens), calls)
    expected = {} if tokens == 4 else (MONO1 if batch == 1 else MONO2)
    assert {k: actual[k] for k in SPLIT if k in actual} == expected


@pytest.mark.parametrize("batch", [1, 2])
def test_measured_replayssm_regular_decode_normalization(calls, batch):
    kwargs = _arguments(batch, 1)
    for name in ("x", "dt", "B", "C", "out"):
        kwargs[name] = kwargs[name].squeeze(1)
    kwargs["state_batch_indices"] = kwargs["state_batch_indices"][:, None]
    actual = _invoke(kwargs, calls)
    assert {k: actual[k] for k in SPLIT if k in actual} == SPLIT
