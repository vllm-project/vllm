# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""AttnRes split (VLLM_K3_ATTNRES_SPLIT): bank partial + prefix combine.

The split must reproduce the fused ``attn_res`` kernel: same output (up to the
softmax summation order), same in-place prefix update and block write. The
layer-level test chains stub decoder layers through the real
``forward_attn_residual`` so the cross-layer hand-off (one sweep feeding this
layer's pre-MLP mix and the next layer's pre-attention mix) is covered too.

On ROCm the checks run on the GPU. Elsewhere they run the Triton kernels in a
fresh interpreter process (``TRITON_INTERPRET=1``): ``vllm.triton_utils``
disables Triton on CPU-only hosts at import time, so it has to be re-enabled
before anything imports vLLM.
"""

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _enable_interpreted_triton() -> None:
    os.environ["TRITON_INTERPRET"] = "1"
    from triton.backends import backends

    for name, backend in backends.items():
        if name != "cpu" and backend.driver is not None:
            backend.driver.is_active = staticmethod(lambda: True)
            break
    import vllm.platforms

    vllm.platforms.vllm_version_matches_substr = lambda _: False


def _rand(*shape: int, device: str, padding: int = 0) -> torch.Tensor:
    storage = torch.randn(
        *shape[:-1], shape[-1] + padding, device=device, dtype=torch.bfloat16
    )
    return storage[..., : shape[-1]]


def _weights(hidden_size: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    norm_weight = 1 + 0.1 * torch.randn(
        hidden_size, device=device, dtype=torch.bfloat16
    )
    qk_weight = (
        torch.randn(hidden_size, device=device, dtype=torch.bfloat16) / hidden_size**0.5
    )
    return norm_weight, qk_weight


def _reference(prefix, blocks, norm_weight, qk_weight, num_blocks, eps):
    hidden_size = prefix.shape[-1]
    values = torch.cat((blocks[:, :num_blocks], prefix.unsqueeze(1)), dim=1).float()
    keys = torch.nn.functional.rms_norm(
        values, (hidden_size,), norm_weight.float(), eps
    )
    probs = (keys @ qk_weight.float()).softmax(dim=-1)
    return torch.matmul(probs.unsqueeze(1), values).squeeze(1)


# (num_tokens, num_blocks, hidden_size, has_delta, write_block, output_norm, quant)
_COMBINE_CASES = [
    (1, 1, 128, False, False, False, False),
    (5, 3, 96, True, False, True, False),
    (4, 2, 128, False, True, True, False),
    (3, 4, 128, True, True, True, False),
    (6, 8, 64, True, False, True, True),
]


def check_combine_matches_fused(device: str) -> None:
    from vllm.models.kimi_k3.amd.ops.attn_res import (
        attn_res,
        attn_res_combine,
        attn_res_partial,
        attn_res_partial_alloc,
    )

    torch.manual_seed(0)
    eps, output_eps = 1e-5, 2e-5
    for case in _COMBINE_CASES:
        tokens, num_blocks, hidden, has_delta, write, out_norm, quant = case
        prefix = _rand(tokens, hidden, device=device, padding=3)
        delta = _rand(tokens, hidden, device=device, padding=5) if has_delta else None
        blocks = _rand(tokens, num_blocks + 2, hidden, device=device, padding=7)
        norm_weight, qk_weight = _weights(hidden, device)
        output_norm_weight = _weights(hidden, device)[0] if out_norm else None
        write_idx = num_blocks if write else -1
        quant_dtype = torch.float8_e4m3fn if quant else None

        fused_prefix, fused_blocks = prefix.clone(), blocks.clone()
        split_prefix, split_blocks = prefix.clone(), blocks.clone()
        args = (norm_weight, qk_weight, output_norm_weight)
        expected = attn_res(
            fused_prefix,
            delta,
            fused_blocks,
            *args,
            num_blocks,
            write_idx,
            eps,
            output_eps,
            quant_dtype=quant_dtype,
        )
        partial, _ = attn_res_partial_alloc(split_blocks, dual=False)
        attn_res_partial(split_blocks, num_blocks, norm_weight, qk_weight, eps, partial)
        actual = attn_res_combine(
            split_prefix,
            delta,
            split_blocks,
            *args,
            partial,
            write_idx,
            eps,
            output_eps,
            quant_dtype=quant_dtype,
        )

        # Side effects are computed identically, so they match bit for bit.
        torch.testing.assert_close(split_prefix, fused_prefix, atol=0, rtol=0)
        torch.testing.assert_close(split_blocks, fused_blocks, atol=0, rtol=0)
        if quant:
            (out_e, scale_e), (out_a, scale_a) = expected, actual
            assert out_a.dtype == quant_dtype and scale_a.shape == (tokens, 1)
            torch.testing.assert_close(scale_a, scale_e, atol=0, rtol=1e-2)
            deq_e, deq_a = out_e.float() * scale_e, out_a.float() * scale_a
            # One FP8 step either way when the summation order flips a rounding.
            assert ((deq_a - deq_e).abs() <= 0.07 * deq_e.abs() + scale_e).all()
            continue
        assert actual.shape == prefix.shape and actual.is_contiguous()
        torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
        if not out_norm:
            reference = _reference(
                split_prefix, split_blocks, norm_weight, qk_weight, num_blocks, eps
            )
            torch.testing.assert_close(actual.float(), reference, atol=3e-2, rtol=3e-2)


def check_dual_partial_matches_single(device: str) -> None:
    from vllm.models.kimi_k3.amd.ops.attn_res import (
        attn_res_partial,
        attn_res_partial_alloc,
    )

    torch.manual_seed(1)
    tokens, num_blocks, hidden = 5, 3, 96
    blocks = _rand(tokens, 6, hidden, device=device, padding=4)
    norm_a, qk_a = _weights(hidden, device)
    norm_b, qk_b = _weights(hidden, device)
    out_a, out_b = attn_res_partial_alloc(blocks, dual=True)
    assert out_b is not None
    attn_res_partial(
        blocks,
        num_blocks,
        norm_a,
        qk_a,
        1e-5,
        out_a,
        next_norm_weight=norm_b,
        next_qk_weight=qk_b,
        next_eps=2e-5,
        next_out=out_b,
    )
    single_a, _ = attn_res_partial_alloc(blocks, dual=False)
    single_b, _ = attn_res_partial_alloc(blocks, dual=False)
    attn_res_partial(blocks, num_blocks, norm_a, qk_a, 1e-5, single_a)
    attn_res_partial(blocks, num_blocks, norm_b, qk_b, 2e-5, single_b)
    for dual, single in ((out_a, single_a), (out_b, single_b)):
        for got, want in zip(dual, single):
            torch.testing.assert_close(got, want, atol=0, rtol=0)


class _StubLayer:
    """Just what KimiDecoderLayer.forward_attn_residual reads."""

    def __init__(self, layer_idx, block_size, hidden, device, split, max_tokens):
        from vllm.models.kimi_k3.amd import linear
        from vllm.utils.math_utils import cdiv

        def norm():
            return SimpleNamespace(
                weight=_weights(hidden, device)[0], variance_epsilon=1e-5
            )

        def proj():
            return SimpleNamespace(weight=_weights(hidden, device)[1].unsqueeze(0))

        self.self_attn = None
        self.mlp = self._mlp
        self.is_block_write_layer = layer_idx % block_size == 0
        self.block_write_idx = layer_idx // block_size
        self.prev_valid_blocks = cdiv(layer_idx, block_size)
        self.self_attention_res_norm, self.mlp_res_norm = norm(), norm()
        self.self_attention_res_proj, self.mlp_res_proj = proj(), proj()
        self.input_layernorm, self.post_attention_layernorm = norm(), norm()
        self.attn_scale = 1 + 0.5 * torch.rand(hidden, device=device)
        self.mlp_scale = 1 + 0.5 * torch.rand(hidden, device=device)
        self.attn_res_split = split
        self.attn_res_split_max_tokens = max_tokens
        self.attn_res_next = None
        self.attn_res_pending = None
        self.attn_res_split_step = False
        self._attn_res_events = None
        self.forward_attn_residual = (
            linear.KimiDecoderLayer.forward_attn_residual.__get__(self)
        )
        self._fork_attn_res_partials = (
            linear.KimiDecoderLayer._fork_attn_res_partials.__get__(self)
        )

    def _run_self_attn(self, positions, hidden_states):
        return torch.tanh(hidden_states.float() * self.attn_scale).bfloat16()

    def _mlp(self, hidden_states):
        return torch.sin(hidden_states.float() * self.mlp_scale).bfloat16()


def _run_chain(layers, embed, num_banks):
    from vllm.models.kimi_k3.amd import linear

    tokens, hidden = embed.shape
    block_residual = torch.zeros(
        tokens, num_banks, hidden, device=embed.device, dtype=embed.dtype
    )
    linear._begin_attn_res_split_step(
        [layer for layer in layers if layer.attn_res_split], tokens
    )
    hidden_states, residual, prefix_delta = embed.clone(), block_residual, None
    for layer in layers:
        hidden_states, residual, prefix_delta = layer.forward_attn_residual(
            None, hidden_states, residual, prefix_delta
        )
    last = layers[-1]
    out = linear._apply_attn_res(
        hidden_states,
        residual,
        last.self_attention_res_proj,
        last.self_attention_res_norm,
        num_banks,
        delta=prefix_delta,
    )
    return out, residual


def check_layer_chain(device: str) -> None:
    from vllm.models.kimi_k3.amd import linear

    num_layers, block_size, hidden, tokens = 7, 2, 64, 5
    num_banks = (num_layers + block_size - 1) // block_size

    def build(split, max_tokens=512):
        torch.manual_seed(2)
        layers = [
            _StubLayer(i, block_size, hidden, device, split, max_tokens)
            for i in range(num_layers)
        ]
        if split:
            linear._link_attn_res_split_layers(layers)
        return layers

    torch.manual_seed(3)
    embed = _rand(tokens, hidden, device=device)

    base_out, base_bank = _run_chain(build(False), embed, num_banks)

    calls = {"partial": 0, "combine": 0}
    orig_partial, orig_combine = linear.attn_res_partial, linear.attn_res_combine
    orig_allowed = linear._attn_res_split_allowed
    allowed_calls = []

    def counting(name, fn):
        def wrapped(*args, **kwargs):
            calls[name] += 1
            return fn(*args, **kwargs)

        return wrapped

    linear.attn_res_partial = counting("partial", orig_partial)
    linear.attn_res_combine = counting("combine", orig_combine)
    linear._attn_res_split_allowed = lambda *args: (
        allowed_calls.append(args) or orig_allowed(*args)
    )
    try:
        split_layers = build(True)
        assert all(layer.attn_res_next is not None for layer in split_layers[:-1])
        assert split_layers[-1].attn_res_next is None
        split_out, split_bank = _run_chain(split_layers, embed, num_banks)
        # One sweep per layer; every pre-MLP mix and every pre-attention mix
        # but layer 0's (no predecessor) combine.
        assert calls == {"partial": num_layers, "combine": 2 * num_layers - 1}
        assert all(layer.attn_res_pending is None for layer in split_layers)
        # Decided once per forward, not per layer.
        assert allowed_calls == [(tokens, 512)]
        torch.testing.assert_close(split_out, base_out, atol=3e-2, rtol=3e-2)
        torch.testing.assert_close(split_bank, base_bank, atol=3e-2, rtol=3e-2)

        # Above the token limit the split path must be the fused path exactly.
        calls.update(partial=0, combine=0)
        capped_layers = build(True, max_tokens=tokens - 1)
        capped_out, capped_bank = _run_chain(capped_layers, embed, num_banks)
        assert calls == {"partial": 0, "combine": 0}
        torch.testing.assert_close(capped_out, base_out, atol=0, rtol=0)
        torch.testing.assert_close(capped_bank, base_bank, atol=0, rtol=0)

        # A stale partial with a different token count is ignored.
        calls.update(partial=0, combine=0)
        stale_layers = build(True)
        stale_layers[0].attn_res_next = None
        linear._begin_attn_res_split_step(stale_layers, tokens)
        stale_layers[1].attn_res_pending = linear.attn_res_partial_alloc(
            torch.empty(tokens + 1, num_banks, hidden, device=device), dual=False
        )[0]
        hidden_states = embed.clone()
        bank = torch.zeros(tokens, num_banks, hidden, device=device).bfloat16()
        out = stale_layers[0].forward_attn_residual(None, hidden_states, bank, None)
        stale_layers[1].forward_attn_residual(None, *out)
        assert calls["combine"] == 2  # the two pre-MLP mixes only
        assert stale_layers[1].attn_res_pending is None

        # The per-forward reset drops a leftover partial.
        stale_layers[2].attn_res_pending = stale_layers[1].attn_res_pending = (
            linear.attn_res_partial_alloc(bank, dual=False)[0]
        )
        linear._begin_attn_res_split_step(stale_layers, tokens)
        assert all(layer.attn_res_pending is None for layer in stale_layers)
    finally:
        linear.attn_res_partial, linear.attn_res_combine = orig_partial, orig_combine
        linear._attn_res_split_allowed = orig_allowed


_CHECKS = {
    "combine": check_combine_matches_fused,
    "dual": check_dual_partial_matches_single,
    "chain": check_layer_chain,
}


@pytest.mark.parametrize("check", sorted(_CHECKS))
def test_attn_res_split(check: str) -> None:
    from vllm.platforms import current_platform

    if current_platform.is_rocm():
        _CHECKS[check]("cuda")
        return
    env = dict(os.environ, TRITON_INTERPRET="1", VLLM_TARGET_DEVICE="cpu")
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(_REPO_ROOT), env.get("PYTHONPATH", "")) if p
    )
    result = subprocess.run(
        [sys.executable, __file__, check],
        cwd=_REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]
    assert f"OK {check}" in result.stdout


def test_attn_res_split_default_off(monkeypatch: pytest.MonkeyPatch) -> None:
    from vllm.models.kimi_k3.amd import linear

    monkeypatch.delenv("VLLM_K3_ATTNRES_SPLIT", raising=False)
    assert not linear._attn_res_split_enabled()
    monkeypatch.setenv("VLLM_K3_ATTNRES_SPLIT", "1")
    assert linear._attn_res_split_enabled()
    # DBO ubatch threads share the layers' cross-layer split state: off.
    for use_ubatching in (False, True):
        config = SimpleNamespace(
            parallel_config=SimpleNamespace(use_ubatching=use_ubatching)
        )
        monkeypatch.setattr(
            linear, "get_current_vllm_config_or_none", lambda c=config: c
        )
        assert linear._attn_res_split_enabled() is not use_ubatching

    # Without a partial, _apply_attn_res is the fused kernel call, unchanged.
    seen = {}

    def fake_attn_res(*args, **kwargs):
        seen["args"], seen["kwargs"] = args, kwargs
        return torch.zeros(1)

    def fail(*args, **kwargs):
        raise AssertionError("combine must not run without a partial")

    monkeypatch.setattr(linear, "attn_res", fake_attn_res)
    monkeypatch.setattr(linear, "attn_res_combine", fail)
    prefix = torch.zeros(2, 4)
    blocks = torch.zeros(2, 3, 4)
    norm = SimpleNamespace(weight=torch.ones(4), variance_epsilon=1e-5)
    proj = SimpleNamespace(weight=torch.ones(1, 4))
    linear._apply_attn_res(prefix, blocks, proj, norm, 2, block_write_idx=-1)
    assert seen["args"][6:10] == (2, -1, 1e-5, 0.0)
    assert seen["kwargs"] == {"quant_dtype": None}


if __name__ == "__main__":
    _enable_interpreted_triton()
    _CHECKS[sys.argv[1]]("cpu")
    print(f"OK {sys.argv[1]}")
