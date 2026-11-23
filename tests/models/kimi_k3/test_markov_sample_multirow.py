# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""``VLLM_K31_MARKOV_ROWS``: the fused DSpark Markov launches
(``ops/markov_sample``) for more than one request. Unset (the default) they
take one request, as before; set to N they take up to N, 2 or more through
``_k31_markov_rows_kernel``.

No GPU here: each test runs in a fresh interpreter with Triton's CPU
interpreter (TRITON_INTERPRET=1) and checks the launches against a pure-torch
Markov loop (murmur3 Gumbel noise included), vLLM's own ``gumbel_sample`` loop
and the single-request launch, request by request. The Markov weights are
small multiples of 1/16, so every bias dot product is exact in FP32 whatever
its summation order and the comparisons are bitwise."""

import os
import subprocess
import sys

import pytest

ROWS_ENV = "VLLM_K31_MARKOV_ROWS"

# Triton's interpreter on a box without a GPU driver: an active (fake) driver
# whose target lookup fails (FLA's utils then take the CPU path), and vLLM's
# Triton check bypassed (it would install the placeholder).
_PRELUDE = r"""
import os
os.environ["TRITON_INTERPRET"] = "1"
import numpy as np
import torch
import triton
import triton.language as tl
from triton.backends import backends

backends["nvidia"].driver.is_active = staticmethod(lambda: True)


class _CPUDriver:
    def get_current_target(self):
        raise RuntimeError("no GPU")

    def get_current_device(self):
        return 0

    def get_active_torch_device(self):
        return torch.device("cpu")

    def get_device_interface(self):
        return torch.cuda

    def get_benchmarker(self):
        return lambda *a, **k: [0.0]


triton.runtime.driver.set_active(_CPUDriver())
# The interpreter keeps ``bool block & bool scalar`` (an int1 mask compiled) as
# a non-bool array, which its masked loads and stores refuse.
from triton.runtime import interpreter as _interp

_Builder = _interp.InterpreterBuilder
_load, _store = _Builder.create_masked_load, _Builder.create_masked_store


def _as_bool(mask):
    return _interp.TensorHandle(mask.data.astype(bool), mask.dtype)


_Builder.create_masked_load = lambda self, p, m, *a: _load(self, p, _as_bool(m), *a)
_Builder.create_masked_store = lambda self, p, v, m, *a: _store(
    self, p, v, _as_bool(m), *a
)
# The interpreter's FP32 -> bf16 cast truncates; the GPU's (and torch's)
# rounds to nearest even.
_cast = _Builder.cast_impl


def _cast_rtne(self, src, dst_type):
    if src.dtype.scalar == tl.float32 and dst_type.scalar == tl.bfloat16:
        x = torch.from_numpy(np.array(src.data, dtype=np.float32, copy=True))
        bits = x.to(torch.bfloat16).view(torch.int16).numpy()
        return _interp.TensorHandle(bits.view(np.uint16), tl.bfloat16)
    return _cast(self, src, dst_type)


_Builder.cast_impl = _cast_rtne
import vllm.platforms

vllm.platforms.vllm_version_matches_substr = lambda s: False
from vllm.triton_utils import HAS_TRITON

assert HAS_TRITON
from types import SimpleNamespace

from vllm.models.kimi_k3.amd.ops import markov_sample as ms

M32 = 0xFFFFFFFF
SALT = 1 << 30
N_SPEC = 5
VOCAB = 256
RANK = 512
MAX_REQS = 16


def _rotl(x, s):
    return ((x << s) | (x >> (32 - s))) & M32


def _mix(h, k):
    k = (k * 0xCC9E2D51) & M32
    k = _rotl(k, 15)
    k = (k * 0x1B873593) & M32
    h = h ^ k
    h = _rotl(h, 13)
    return (h * 5 + 0xE6546B64) & M32


def _fmix(h):
    h = h ^ (h >> 16)
    h = (h * 0x85EBCA6B) & M32
    h = h ^ (h >> 13)
    h = (h * 0xC2B2AE35) & M32
    return h ^ (h >> 16)


def _uniform(seed, pos, offset):
    # gumbel.murmur3_uniform32 in int64 arithmetic (low 32 bits kept).
    h = torch.zeros_like(offset)
    h = _mix(h, seed & M32)
    h = _mix(h, (seed >> 32) & M32)
    h = _mix(h, pos & M32)
    h = _mix(h, offset & M32)
    r = _fmix(h ^ 16)
    hi = (r >> 16).to(torch.float32)
    lo = (r & 0xFFFF).to(torch.float32)
    return hi * 1.52587890625e-05 + (lo + 0.5) * 2.3283064365386963e-10


def _log1p_neg(u):
    poly = torch.full_like(u, 1.0 / 8.0)
    for c in (7, 6, 5, 4, 3, 2):
        poly = 1.0 / c + u * poly
    poly = 1.0 + u * poly
    direct = torch.log(torch.clamp(1.0 - u, min=5.960464477539063e-08))
    return torch.where(u < 0.25, -u * poly, direct)


def torch_reference(case):
    # The speculator's Markov loop in torch: embedding, bias GEMV (bf16 out),
    # bf16 add, cache store before temperature, Gumbel-max with the step mixed
    # into the seed and the drafting salt on the position before the token.
    tokens = torch.zeros(MAX_REQS, N_SPEC, dtype=torch.int64)
    cache = case.cache.clone()
    w1, w2 = case.w1.float(), case.w2[:VOCAB].float()
    keys = torch.arange(VOCAB, dtype=torch.int64)
    for r in range(case.num_reqs):
        prev = int(case.input_ids[case.anchor[r]])
        for step in range(N_SPEC):
            bias = (w2 @ w1[prev]).to(torch.bfloat16)
            logits = case.base[r, step] + bias
            lf = logits.float()
            if not case.greedy:
                req = int(case.idx_map[r * N_SPEC + step])
                row = req
                if case.row_mapping is not None and req >= 0:
                    row = int(case.row_mapping[req])
                if req >= 0 and row >= 0:
                    cache[row, step] = logits
                temp = float(case.temperature[req]) if req >= 0 else 0.0
                if temp != 0.0:
                    seed = int(case.seeds[req]) ^ (step << 32)
                    pos = int(case.sample_pos[r * N_SPEC + step]) - 1 + SALT
                    seed_t = torch.full_like(keys, seed)
                    pos_t = torch.full_like(keys, pos)
                    u = _uniform(seed_t, pos_t, keys)
                    noise = -torch.log(-_log1p_neg(u))
                    lf = lf / torch.tensor(temp, dtype=torch.float32) + noise
            prev = int(torch.argmax(lf))
            tokens[r, step] = prev
    return tokens, cache


def vllm_loop(case):
    # The stock loop's sampler: vLLM's gumbel_sample (interpreted), Markov
    # step in torch.
    from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample

    n = case.num_reqs
    tokens = torch.zeros(MAX_REQS, N_SPEC, dtype=torch.int64)
    cache = case.cache.clone()
    idx_map = case.idx_map[: n * N_SPEC].view(n, N_SPEC)
    sample_pos = case.sample_pos[: n * N_SPEC].view(n, N_SPEC)
    prev = case.input_ids[case.anchor[:n]].long()
    for step in range(N_SPEC):
        bias = (case.w1[prev].float() @ case.w2[:VOCAB].float().T).to(torch.bfloat16)
        logits = case.base[:n, step] + bias
        if case.greedy:
            prev = logits.argmax(dim=-1)
        else:
            prev = gumbel_sample(
                logits,
                idx_map[:, step].contiguous(),
                case.temperature,
                case.seeds,
                sample_pos[:, step] - 1,
                apply_temperature=True,
                is_drafting=True,
                logits_cache=cache,
                logits_cache_col=torch.tensor(step),
                logits_cache_row_mapping=case.row_mapping,
            )
        tokens[:n, step] = prev
    return tokens, cache


def make_case(num_reqs, greedy, mapped, padded, seed=0):
    g = torch.Generator().manual_seed(seed * 1000 + num_reqs)

    def small(*shape):
        # Multiples of 1/16 in [-1/4, 1/4]: exact FP32 dot products.
        return (torch.randint(-4, 5, shape, generator=g) / 16).to(torch.bfloat16)

    case = SimpleNamespace(num_reqs=num_reqs, greedy=greedy)
    case.w1 = small(VOCAB, RANK)
    case.w2 = small(VOCAB, RANK)
    case.base = (torch.randn(num_reqs, N_SPEC, VOCAB, generator=g) * 2).to(
        torch.bfloat16
    )
    case.input_ids = torch.randint(0, VOCAB, (64,), generator=g, dtype=torch.int32)
    case.anchor = torch.randperm(64, generator=g)[:num_reqs].to(torch.int64)
    states = torch.randperm(MAX_REQS, generator=g)[:num_reqs].to(torch.int32)
    if padded:
        # Padded batch rows (a captured graph's spare requests): no request.
        states[-1] = -1
    case.idx_map = states.repeat_interleave(N_SPEC).contiguous()
    case.sample_pos = (
        torch.randint(10, 4000, (num_reqs,), generator=g)[:, None]
        + torch.arange(N_SPEC)[None, :]
    ).flatten()
    case.temperature = torch.tensor(
        [0.0, 0.7, 1.0, 1.3] * (MAX_REQS // 4), dtype=torch.float32
    )
    case.seeds = torch.randint(0, 1 << 62, (MAX_REQS,), generator=g)
    case.row_mapping = None
    rows = MAX_REQS
    if mapped:
        # A batch-sharded cache: half the requests have a row on this rank.
        case.row_mapping = torch.full((MAX_REQS,), -1, dtype=torch.int32)
        case.row_mapping[::2] = torch.arange(MAX_REQS // 2, dtype=torch.int32)
        rows = MAX_REQS // 2
    case.cache = torch.zeros(rows, N_SPEC, VOCAB, dtype=torch.bfloat16)
    return case


def make_spec(case, cache=None, rows=None):
    rows = slice(0, case.num_reqs) if rows is None else rows
    head = SimpleNamespace(
        markov_w1=SimpleNamespace(weight=case.w1),
        markov_w2=SimpleNamespace(weight=case.w2),
    )
    spec = SimpleNamespace(
        num_speculative_steps=N_SPEC,
        model=SimpleNamespace(model=SimpleNamespace(markov_head=head)),
        draft_logits=None,
        sample_idx_mapping=case.idx_map[rows.start * N_SPEC : rows.stop * N_SPEC],
        sample_pos=case.sample_pos[rows.start * N_SPEC : rows.stop * N_SPEC],
        draft_tokens=torch.zeros(MAX_REQS, N_SPEC, dtype=torch.int64),
        input_buffers=SimpleNamespace(input_ids=case.input_ids),
        _anchor_idx=case.anchor[rows],
        temperature=case.temperature,
        seeds=case.seeds,
    )
    if not case.greedy:
        spec.draft_logits = SimpleNamespace(
            values=case.cache.clone() if cache is None else cache,
            row_mapping=case.row_mapping,
        )
    return spec


def fused(case):
    spec = make_spec(case)
    assert ms.run(spec, case.num_reqs, case.base)
    slots = ms._SLOTS[(case.base.device, triton.next_power_of_2(N_SPEC + 1), ms.NSUB)]
    # The launches leave their slots zeroed (captured graphs reuse them).
    assert not slots.any()
    cache = None if case.greedy else spec.draft_logits.values
    return spec.draft_tokens, cache


CASES = [
    (n, greedy, mapped, padded)
    for n in (2, 3, 8, 10)
    for greedy in (True, False)
    for mapped, padded in ((False, False), (True, False), (False, True))
    if not (greedy and mapped)
]
"""


def _run(body: str, rows: str | None = "16") -> str:
    env = {k: v for k, v in os.environ.items() if k != ROWS_ENV}
    if rows is not None:
        env[ROWS_ENV] = rows
    out = subprocess.run(
        [sys.executable, "-c", _PRELUDE + body],
        env=env,
        capture_output=True,
        text=True,
    )
    assert out.returncode == 0, out.stderr[-4000:]
    # vLLM's logger writes to stdout too: the scripts' lines start with "@@ ".
    lines = out.stdout.splitlines()
    return "\n".join(line[3:] for line in lines if line.startswith("@@ "))


_WHY_NOT = r"""
class Draft:
    pass


ms.markov_drafts = lambda: (Draft,)
case = make_case(2, greedy=False, mapped=False, padded=False)
spec = make_spec(case)
spec.model = Draft()
spec.model.model = SimpleNamespace(markov_head=SimpleNamespace(
    markov_w1=SimpleNamespace(weight=case.w1),
    markov_w2=SimpleNamespace(weight=case.w2),
))
spec.draft_watermarker = spec._d2t_scatter_index = None
spec.acceptance_estimator = None
print("@@ MAX_ROWS", ms.MAX_ROWS)
for n in (1, 2, 16, 17):
    print("@@", n, ms.why_not(spec, n))
"""


@pytest.mark.parametrize(
    "rows, max_rows",
    [(None, 1), ("1", 1), ("0", 1), ("abc", 1), ("16", 16)],
)
def test_rows_knob(rows, max_rows):
    out = _run(_WHY_NOT, rows).splitlines()
    assert out[0] == f"MAX_ROWS {max_rows}"
    why = dict(line.split(" ", 1) for line in out[1:])
    assert why["1"] == "None"
    for n in ("2", "16", "17"):
        expect = "None" if int(n) <= max_rows else f"more than {max_rows} requests"
        assert why[n] == expect, (n, why[n])


_DISPATCH = r"""
calls = []
for name in ("_k31_markov_step_kernel", "_k31_markov_rows_kernel"):
    kernel = getattr(ms, name)

    class Rec:
        def __init__(self, name, kernel):
            self.name, self.kernel = name, kernel

        def __getitem__(self, grid):
            calls.append((self.name, grid))
            return self.kernel[grid]

    setattr(ms, name, Rec(name, kernel))
for n in (1, 3, 10):
    calls.clear()
    case = make_case(n, greedy=False, mapped=False, padded=False)
    if n == 1:
        assert ms.run(make_spec(case), n, case.base)
    else:
        fused(case)
    print("@@", n, sorted(set(calls)))
"""


def test_dispatch():
    """One request keeps the single-request launch; 2 or more take the
    multi-request one, ``ROW_TILE`` requests a program."""
    out = dict(line.split(" ", 1) for line in _run(_DISPATCH).splitlines())
    assert out["1"] == "[('_k31_markov_step_kernel', (1, 8))]"
    assert out["3"] == "[('_k31_markov_rows_kernel', (1, 8))]"
    assert out["10"] == "[('_k31_markov_rows_kernel', (2, 8))]"


def test_dispatch_without_the_knob():
    """``MAX_ROWS`` raised without the env knob (the GPU test
    tests/kernels/models/amd/test_k31_fused_markov.py does) keeps the
    single-request kernel's (requests, blocks) grid, as before the knob."""
    body = "ms.MAX_ROWS = 16\nprint('@@ rows_kernel', ms.ROWS_KERNEL)\n" + _DISPATCH
    out = dict(line.split(" ", 1) for line in _run(body, rows=None).splitlines())
    assert out["rows_kernel"] == "False"
    assert out["1"] == "[('_k31_markov_step_kernel', (1, 8))]"
    assert out["3"] == "[('_k31_markov_step_kernel', (3, 8))]"
    assert out["10"] == "[('_k31_markov_step_kernel', (10, 8))]"


def test_tp_split_reason_does_not_vary_with_requests():
    """The caller logs each reason once: one line for every batch size."""
    out = _run(
        "print('@@', {ms.tp_split_why_not(n, 163840, 8) for n in (2, 3, 16)})",
        rows=None,
    )
    assert out == "{'more than 1 request'}"


_TORCH = r"""
noisy = False
for n, greedy, mapped, padded in CASES:
    case = make_case(n, greedy, mapped, padded)
    tokens, cache = fused(case)
    ref_tokens, ref_cache = torch_reference(case)
    assert torch.equal(tokens, ref_tokens), (n, greedy, mapped, padded)
    if not greedy:
        assert torch.equal(cache, ref_cache), (n, greedy, mapped, padded)
        plain, _ = torch_reference(SimpleNamespace(**{**vars(case), "greedy": True}))
        noisy |= not torch.equal(ref_tokens, plain)
    print("@@ ok", n, greedy, mapped, padded)
"""


def test_multirow_matches_torch_reference():
    out = _run(_TORCH + 'assert noisy\nprint("@@ noisy")').splitlines()
    assert out.pop() == "noisy"
    assert len(out) == 20 and all(line.startswith("ok") for line in out)


_VLLM_LOOP = r"""
for n, greedy, mapped, padded in CASES:
    case = make_case(n, greedy, mapped, padded, seed=1)
    tokens, cache = fused(case)
    ref_tokens, ref_cache = vllm_loop(case)
    assert torch.equal(tokens, ref_tokens), (n, greedy, mapped, padded)
    if not greedy:
        assert torch.equal(cache, ref_cache), (n, greedy, mapped, padded)
    print("@@ ok", n, greedy, mapped, padded)
"""


def test_multirow_matches_vllm_loop():
    out = _run(_VLLM_LOOP).splitlines()
    assert len(out) == 20 and all(line.startswith("ok") for line in out)


_SINGLE = r"""
for n, greedy, mapped, padded in CASES:
    case = make_case(n, greedy, mapped, padded, seed=2)
    tokens, cache = fused(case)
    one_cache = None if greedy else case.cache.clone()
    for r in range(n):
        spec = make_spec(case, one_cache, slice(r, r + 1))
        assert ms.run(spec, 1, case.base[r : r + 1])
        assert torch.equal(spec.draft_tokens[0], tokens[r]), (n, r)
    if not greedy:
        assert torch.equal(cache, one_cache), (n, greedy, mapped, padded)
    # Twice: the same drafts from the cleared slots.
    again, _ = fused(case)
    assert torch.equal(again, tokens)
    print("@@ ok", n, greedy, mapped, padded)
"""


def test_multirow_matches_single_row_launch():
    out = _run(_SINGLE).splitlines()
    assert len(out) == 20 and all(line.startswith("ok") for line in out)
