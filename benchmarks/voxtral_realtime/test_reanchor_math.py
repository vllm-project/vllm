# SPDX-License-Identifier: Apache-2.0
"""
Unit test for the RoPE re-anchoring core math (config-free, matches vLLM's
rotary convention in vllm/model_executor/layers/rotary_embedding/common.py).

Property proved: a POST-rotation cached key that was rotated at absolute position
`p` and is then re-rotated by the constant R(-D) equals the key rotated at
position `p-D`. Hence after shifting the query clock by D and re-rotating every
live key by R(-D), every in-window relative attention score R(i-j) is preserved
exactly. Tested for BOTH styles used by Voxtral realtime:
  - decoder: NeoX,  rotary_dim=128, base=1e6
  - encoder: GPT-J, rotary_dim=64,  base=1e6

The CI-collected pytest equivalent (same property, asserted) lives at
tests/v1/worker/test_reanchor_rotary.py; this script stays as a manual harness
with printed per-case errors.

Run:  .venv/bin/python benchmarks/voxtral_realtime/test_reanchor_math.py
"""

import torch

# Import the PRODUCTION re-anchor op so this test guards the real function rather
# than a hand-copied duplicate. The helper lives in a dependency-light module so
# importing it does not pull the GPU model runner graph.
from vllm.v1.worker.reanchor_rotary import apply_inverse_rotary


def _inv_freq(rotary_dim: int, base: float) -> torch.Tensor:
    return 1.0 / (
        base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float64) / rotary_dim)
    )


def forward_rotary(
    x: torch.Tensor, pos: int, rotary_dim: int, base: float, is_neox: bool
) -> torch.Tensor:
    """Rotate x by +pos (vLLM convention)."""
    half = rotary_dim // 2
    ang = float(pos) * _inv_freq(rotary_dim, base)
    cos, sin = torch.cos(ang).to(x.dtype), torch.sin(ang).to(x.dtype)
    rot, pas = x[..., :rotary_dim], x[..., rotary_dim:]
    if is_neox:
        x1, x2 = rot[..., :half], rot[..., half:]
        out = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
    else:
        x1, x2 = rot[..., 0::2], rot[..., 1::2]
        o1, o2 = x1 * cos - x2 * sin, x2 * cos + x1 * sin
        out = torch.stack([o1, o2], dim=-1).flatten(-2)
    return torch.cat([out, pas], dim=-1) if pas.numel() else out


def main():
    torch.manual_seed(0)
    cases = [("decoder NeoX", 128, 1e6, True), ("encoder GPT-J", 64, 1e6, False)]
    ok = True
    for name, rdim, base, neox in cases:
        for p, D in [(5000, 4096), (9000, 7000), (3000, 2048), (120000, 100000)]:
            k = torch.randn(4, rdim, dtype=torch.float32)
            # (1) re-anchor identity: R(-D)·R(p)·k == R(p-D)·k
            cached = forward_rotary(k, p, rdim, base, neox)  # key in KV cache
            reanchored = apply_inverse_rotary(cached, D, rdim, base, neox)
            target = forward_rotary(k, p - D, rdim, base, neox)
            err1 = (reanchored - target).abs().max().item()
            # (2) relative-score preservation: q@i vs k@j  ==  q@(i-D) vs (reanchored)
            i, j = p + 60, p  # j within window below i
            q = torch.randn(4, rdim, dtype=torch.float32)
            s_orig = (
                forward_rotary(q, i, rdim, base, neox)
                * forward_rotary(k, j, rdim, base, neox)
            ).sum(-1)
            kj_re = apply_inverse_rotary(
                forward_rotary(k, j, rdim, base, neox), D, rdim, base, neox
            )
            s_re = (forward_rotary(q, i - D, rdim, base, neox) * kj_re).sum(-1)
            err2 = (s_orig - s_re).abs().max().item()
            status = "OK" if (err1 < 1e-3 and err2 < 1e-3) else "FAIL"
            if status == "FAIL":
                ok = False
            print(
                f"  [{status}] {name:14s} p={p:6d} D={D:6d}  "
                f"reanchor-identity err={err1:.1e}  rel-score err={err2:.1e}"
            )
    print(
        "\nRESULT:",
        "PASS: R(-D) re-rotation preserves in-window scores" if ok else "FAIL",
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
