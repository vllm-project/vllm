# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3-Flash KDA decode through FlashInfer's fused kernels.

The FlashInfer path runs conv + SiLU, the recurrence and the gated RMSNorm in
one launch for plain decode rows (also the decode rows of mixed steps) and for
draft-verify rows. It must keep the Triton path's cache contract: same conv
states, recurrent states within kernel rounding, null and padding slots
untouched, and normed layer output whatever the step type.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.models.glm5next.common import kda as glm_kda
from vllm.models.glm5next.common.kda import (
    Glm5NextLinearAttention,
    _is_flashinfer_kda_decode_supported,
    _resolve_kda_decode_backend,
)
from vllm.third_party.flash_linear_attention.ops.kda import FusedRMSNormGated
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
from vllm.v1.attention.backends.utils import compute_causal_conv1d_metadata

# FusedRMSNormGated is a CustomOp, which needs a current vLLM config.
pytestmark = pytest.mark.usefixtures("default_vllm_config")

D, W = 128, 4
LOWER_BOUND = -5.0
NORM_EPS = 1e-5
NUM_SLOTS = 24
PREFIX = "model.layers.0.self_attn"


def _require_flashinfer(num_heads, query_len, dim_first, state_dtype):
    if dim_first and query_len > 1:
        pytest.skip("FlashInfer's packed KDA decode supports only the SD layout")
    if not _is_flashinfer_kda_decode_supported(
        num_heads,
        D,
        W,
        query_len,
        torch.bfloat16,
        torch.bfloat16,
        state_dtype,
        LOWER_BOUND,
    ):
        pytest.skip("FlashInfer fused KDA decode does not support this config")


def _make_layer(
    num_heads: int,
    num_spec: int,
    state_dtype: torch.dtype,
    dim_first: bool,
    backend: str,
    seed: int = 0,
) -> Glm5NextLinearAttention:
    """A KDA layer with random weights and caches, built without the linear
    projections (only ``_forward`` runs)."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    proj = num_heads * D
    width = W - 1 + num_spec
    layer = Glm5NextLinearAttention.__new__(Glm5NextLinearAttention)
    nn.Module.__init__(layer)
    layer.prefix = PREFIX
    layer.head_dim, layer.conv_size, layer.num_spec = D, W, num_spec
    layer.local_num_heads, layer.local_projection_size = num_heads, proj
    layer.kda_safe_gate, layer.kda_lower_bound = True, LOWER_BOUND
    layer.kda_prefill_backend = "triton"
    layer._flashkda_buffer_specs = None
    layer._checkpoint_exporter = None
    layer._conv_state_dim_first = dim_first
    for name in ("q_conv1d", "k_conv1d", "v_conv1d"):
        weight = 0.3 * torch.randn(proj, 1, W, device="cuda", generator=g)
        setattr(layer, name, SimpleNamespace(weight=weight, bias=None))
    layer._merged_conv_weight = None
    layer.A_log = nn.Parameter(
        torch.log(torch.linspace(1, 16, num_heads, device="cuda")).view(
            1, 1, num_heads, 1
        )
    )
    layer.dt_bias = nn.Parameter(0.1 * torch.randn(proj, device="cuda", generator=g))
    layer.o_norm = FusedRMSNormGated(D, eps=NORM_EPS, activation="sigmoid").cuda()
    layer.o_norm.weight.data = (
        1 + 0.1 * torch.randn(D, device="cuda", generator=g)
    ).bfloat16()
    layer.kda_decode_backend = backend if num_spec == 0 else "triton"
    layer.kda_spec_decode_backend = backend if num_spec > 0 else "triton"
    layer._norm_in_forward = backend == "flashinfer"
    layer._fi_conv_weight = layer._fi_norm_weight = None
    # DS stores the conv state [slot, 3P, width], SD [slot, width, 3P].
    conv_shape = (
        (NUM_SLOTS, 3 * proj, width) if dim_first else (NUM_SLOTS, width, 3 * proj)
    )
    conv_state = torch.randn(conv_shape, device="cuda", generator=g).bfloat16()
    state = 0.2 * torch.randn(NUM_SLOTS, num_heads, D, D, device="cuda", generator=g)
    layer.kv_cache = (conv_state, state.to(state_dtype))
    return layer


def _make_inputs(num_heads: int, num_tokens: int, seed: int = 1) -> dict:
    """Inputs as ``forward`` passes them: q|k|v and beta are column slices of
    the fused projection output."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    proj = num_heads * D
    projected = torch.randn(
        num_tokens, 3 * proj + num_heads + 2 * D, device="cuda", generator=g
    ).bfloat16()
    return dict(
        qkv_proj_states=projected[:, : 3 * proj],
        beta=projected[:, 3 * proj : 3 * proj + num_heads].unsqueeze(0),
        g1=torch.randn(1, num_tokens, num_heads, D, device="cuda", generator=g)
        .bfloat16()
        .sub(2.0),
        g2=torch.randn(num_tokens, num_heads, D, device="cuda", generator=g).bfloat16(),
    )


def _non_spec_metadata(
    query_lens: list[int], num_decodes: int, slots: list[int], has_state: list[bool]
) -> GDNAttentionMetadata:
    """Non-spec metadata as the builder makes it: decodes first, then the
    prefill tail rebased for the peeled path."""
    qsl_cpu = torch.tensor([0, *torch.tensor(query_lens).cumsum(0).tolist()]).int()
    qsl = qsl_cpu.cuda()
    indices = torch.tensor(slots, dtype=torch.int32, device="cuda")
    has_initial_state = torch.tensor(has_state, device="cuda")
    num_decode_tokens = num_decodes
    num_prefills = len(query_lens) - num_decodes
    meta = GDNAttentionMetadata(
        num_prefills=num_prefills,
        num_prefill_tokens=int(qsl_cpu[-1]) - num_decode_tokens,
        num_decodes=num_decodes,
        num_decode_tokens=num_decode_tokens,
        num_spec_decodes=0,
        num_spec_decode_tokens=0,
        num_actual_tokens=int(qsl_cpu[-1]),
        non_spec_query_start_loc=qsl,
        non_spec_state_indices_tensor=indices,
    )
    if num_prefills:
        meta.has_initial_state = has_initial_state
        meta.nums_dict, meta.batch_ptr, meta.token_chunk_offset_ptr = (
            compute_causal_conv1d_metadata(qsl_cpu, device=qsl.device)
        )
        if num_decodes:
            meta.prefill_query_start_loc = qsl[num_decodes:] - num_decode_tokens
            meta.prefill_state_indices = indices[num_decodes:]
            meta.prefill_has_initial_state = has_initial_state[num_decodes:]
        else:
            meta.prefill_query_start_loc = qsl
            meta.prefill_state_indices = indices
            meta.prefill_has_initial_state = has_initial_state
    return meta


def _run(layer, meta, inputs, monkeypatch) -> torch.Tensor:
    """``Glm5NextLinearAttention.forward`` after the projections. The Triton
    decode conv writes its output over the input rows, so callers pass fresh
    inputs to every run."""
    monkeypatch.setattr(
        glm_kda,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata={PREFIX: meta}),
    )
    num_tokens, num_heads = inputs["g2"].shape[:2]
    out = torch.zeros(1, num_tokens, num_heads, D, dtype=torch.bfloat16, device="cuda")
    layer._forward(core_attn_out=out, **inputs)
    if not layer._norm_in_forward:
        out = layer.o_norm(out, inputs["g2"])
    return out


def _decode_reference(layer, inputs, slots, conv_state, state) -> torch.Tensor:
    """fp32 reference for single-token rows (conv + SiLU rounded to bf16 as
    the kernels do, q/k l2norm, bounded gate, delta rule, gated RMSNorm);
    updates the given caches (conv state in [slot, 3P, width] view)."""
    H = layer.local_num_heads
    weight = torch.cat(
        [m.weight.squeeze(1) for m in (layer.q_conv1d, layer.k_conv1d, layer.v_conv1d)]
    )
    a_log = layer.A_log.view(-1).float()
    out = torch.zeros(len(slots), H, D, device="cuda")
    for row, slot in enumerate(slots):
        if slot <= 0:
            continue
        x = inputs["qkv_proj_states"][row].float()
        window = torch.cat([conv_state[slot].float(), x[:, None]], dim=1)
        y = (window * weight).sum(-1)
        y = (y * torch.sigmoid(y)).bfloat16().float()
        conv_state[slot] = window[:, 1:].bfloat16()
        q, k, v = y.view(3, H, D).unbind(0)
        q = q / torch.sqrt(q.square().sum(-1, keepdim=True) + 1e-6) * D**-0.5
        k = k / torch.sqrt(k.square().sum(-1, keepdim=True) + 1e-6)
        raw_g = inputs["g1"][0, row].float() + layer.dt_bias.view(H, D)
        decay = (LOWER_BOUND * torch.sigmoid(a_log.exp()[:, None] * raw_g)).exp()
        beta = torch.sigmoid(inputs["beta"][0, row].float())
        s = state[slot].float() * decay[:, None, :]
        u = beta[:, None] * (v - torch.einsum("hvk,hk->hv", s, k))
        s = s + u[:, :, None] * k[:, None, :]
        state[slot] = s.to(state.dtype)
        o = torch.einsum("hvk,hk->hv", s, q).bfloat16().float()
        o = o * torch.rsqrt(o.square().mean(-1, keepdim=True) + NORM_EPS)
        out[row] = (
            o * layer.o_norm.weight.float() * torch.sigmoid(inputs["g2"][row].float())
        )
    return out


def _conv_view(layer, conv_state):
    return conv_state if layer._conv_state_dim_first else conv_state.transpose(-1, -2)


def test_resolve_kda_decode_backend(monkeypatch: pytest.MonkeyPatch):
    args = (64, 128, 4, 1, torch.bfloat16, torch.bfloat16, torch.float32, -5.0)
    assert _resolve_kda_decode_backend("triton", *args) == "triton"
    with pytest.raises(ValueError, match="Unsupported"):
        _resolve_kda_decode_backend("native", *args)

    monkeypatch.setattr(glm_kda.current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(
        glm_kda.current_platform, "is_device_capability_family", lambda c: c == 100
    )
    monkeypatch.setattr(glm_kda, "has_flashinfer_fused_kda_decode", lambda: True)
    monkeypatch.setattr(glm_kda, "has_flashinfer_packed_fused_kda_decode", lambda: True)
    monkeypatch.setattr(glm_kda, "is_conv_state_dim_first", lambda: False)
    assert _resolve_kda_decode_backend("auto", *args) == "flashinfer"
    assert _resolve_kda_decode_backend("flashinfer", *args) == "flashinfer"

    # Draft-verify rows: the packed kernel has no 8-head or DS variant.
    verify8 = (8, *args[1:3], 3, *args[4:])
    assert _resolve_kda_decode_backend("auto", *verify8) == "triton"
    with pytest.raises(RuntimeError, match="SM10x"):
        _resolve_kda_decode_backend("flashinfer", *verify8)
    verify64 = (*args[:3], 3, *args[4:])
    assert _resolve_kda_decode_backend("auto", *verify64) == "flashinfer"
    monkeypatch.setattr(glm_kda, "is_conv_state_dim_first", lambda: True)
    assert _resolve_kda_decode_backend("auto", *verify64) == "triton"
    assert _resolve_kda_decode_backend("auto", *args) == "flashinfer"

    # Unbounded gate or non-SM10x: Triton.
    unbounded = (*args[:7], None)
    assert _resolve_kda_decode_backend("auto", *unbounded) == "triton"
    monkeypatch.setattr(
        glm_kda.current_platform, "is_device_capability_family", lambda c: False
    )
    assert _resolve_kda_decode_backend("auto", *args) == "triton"
    with pytest.raises(RuntimeError, match="SM10x"):
        _resolve_kda_decode_backend("flashinfer", *args)


@pytest.mark.parametrize("dim_first", [False, True], ids=["SD", "DS"])
@pytest.mark.parametrize(
    "state_dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"]
)
@pytest.mark.parametrize("num_heads", [8, 16, 32, 64])
@torch.inference_mode()
def test_decode_step_matches_triton_and_reference(
    monkeypatch, num_heads, state_dtype, dim_first
):
    _require_flashinfer(num_heads, 1, dim_first, state_dtype)
    # Null rows (slot 0, also CUDA-graph padding) must not touch any cache.
    slots = [3, 0, 7, 11, 0, 19, 1, 22]
    meta = _non_spec_metadata([1] * len(slots), len(slots), slots, [True] * len(slots))
    results = {}
    for backend in ("triton", "flashinfer"):
        layer = _make_layer(num_heads, 0, state_dtype, dim_first, backend)
        inputs = _make_inputs(num_heads, len(slots))
        results[backend] = (_run(layer, meta, inputs, monkeypatch), layer.kv_cache)

    ref_layer = _make_layer(num_heads, 0, state_dtype, dim_first, "triton")
    conv_ref, state_ref = ref_layer.kv_cache
    ref = _decode_reference(
        ref_layer,
        _make_inputs(num_heads, len(slots)),
        slots,
        _conv_view(ref_layer, conv_ref),
        state_ref,
    )
    live = torch.tensor(slots) > 0
    (out_t, (conv_t, state_t)), (out_f, (conv_f, state_f)) = (
        results["triton"],
        results["flashinfer"],
    )
    torch.testing.assert_close(conv_f, conv_t, atol=0, rtol=0)
    torch.testing.assert_close(conv_f, conv_ref, atol=0, rtol=0)
    tol = 2e-2 if state_dtype == torch.bfloat16 else 2e-3
    torch.testing.assert_close(state_f, state_t, atol=tol, rtol=tol)
    torch.testing.assert_close(out_f[0, live], out_t[0, live], atol=3e-2, rtol=3e-2)
    # No worse than the Triton path against the fp32 reference.
    err_f = (out_f[0, live].float() - ref[live]).abs().max()
    err_t = (out_t[0, live].float() - ref[live]).abs().max()
    assert err_f <= 1.5 * err_t + 1e-2, (err_f, err_t)
    assert torch.count_nonzero(out_f[0, ~live]) == 0
    untouched = [s for s in range(NUM_SLOTS) if s not in slots or s == 0]
    fresh = _make_layer(num_heads, 0, state_dtype, dim_first, "triton").kv_cache
    torch.testing.assert_close(conv_f[untouched], fresh[0][untouched], atol=0, rtol=0)
    torch.testing.assert_close(state_f[untouched], fresh[1][untouched], atol=0, rtol=0)


@pytest.mark.parametrize("prefill_backend", ["triton", "flashkda"])
@pytest.mark.parametrize("num_heads", [16, 32, 64])
@torch.inference_mode()
def test_mixed_step_peels_decode_rows(
    monkeypatch, workspace_init, num_heads, prefill_backend
):
    """Decode rows of a mixed step give the same result as a decode-only step;
    prefill rows (fresh and continued) match the Triton decode path."""
    _require_flashinfer(num_heads, 1, False, torch.float32)
    flashkda = glm_kda._resolve_kda_prefill_backend(
        "auto", D, torch.bfloat16, LOWER_BOUND
    )
    if prefill_backend == "flashkda" and flashkda != "flashkda":
        pytest.skip("FlashKDA is not available")
    decode_slots = [5, 2, 20, 9]
    query_lens = [1] * len(decode_slots) + [37, 1, 130]
    slots = decode_slots + [12, 14, 17]
    has_state = [True] * len(decode_slots) + [False, True, True]
    meta = _non_spec_metadata(query_lens, len(decode_slots), slots, has_state)

    def make(backend):
        layer = _make_layer(num_heads, 0, torch.float32, False, backend)
        if prefill_backend == "flashkda":
            import vllm._flashkda_C  # noqa: F401

            layer.kda_prefill_backend = "flashkda"
            T, N = sum(query_lens), len(query_lens)
            ws = torch.ops._flashkda_C.get_workspace_size(T, num_heads, N)
            layer._flashkda_buffer_specs = (
                ((N, num_heads, D, D), torch.float32),
                ((N, num_heads, D, D), torch.float32),
                ((ws,), torch.uint8),
                ((1, T, num_heads, D), torch.bfloat16),
            )
        return layer

    out = {}
    for backend in ("triton", "flashinfer"):
        layer = make(backend)
        inputs = _make_inputs(num_heads, sum(query_lens))
        out[backend] = (_run(layer, meta, inputs, monkeypatch), layer.kv_cache)

    n = len(decode_slots)
    decode_only = make("flashinfer")
    inputs = _make_inputs(num_heads, sum(query_lens))
    decode_inputs = {
        "qkv_proj_states": inputs["qkv_proj_states"][:n],
        "beta": inputs["beta"][:, :n],
        "g1": inputs["g1"][:, :n],
        "g2": inputs["g2"][:n],
    }
    decode_meta = _non_spec_metadata([1] * n, n, decode_slots, [True] * n)
    decode_out = _run(decode_only, decode_meta, decode_inputs, monkeypatch)

    out_f, (conv_f, state_f) = out["flashinfer"]
    out_t, (conv_t, state_t) = out["triton"]
    torch.testing.assert_close(out_f[:, :n], decode_out, atol=0, rtol=0)
    live = [s for s in decode_slots if s > 0]
    torch.testing.assert_close(
        state_f[live], decode_only.kv_cache[1][live], atol=0, rtol=0
    )
    torch.testing.assert_close(conv_f, conv_t, atol=0, rtol=0)
    # Prefill rows and their states do not depend on which kernel ran the
    # decode rows (the Triton path runs them through the chunked kernel).
    prefill = slots[n:]
    torch.testing.assert_close(state_f[prefill], state_t[prefill], atol=0, rtol=0)
    torch.testing.assert_close(out_f[:, n:], out_t[:, n:], atol=0, rtol=0)
    torch.testing.assert_close(state_f[live], state_t[live], atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(out_f[:, :n], out_t[:, :n], atol=3e-2, rtol=3e-2)


def _spec_metadata(num_spec, accepted, prefill_lens, num_spec_rows, slots_per_seq):
    """Spec-verify rows first (each 1 + num_spec tokens), then prefills."""
    T = 1 + num_spec
    spec_lens = [T] * num_spec_rows
    lens = spec_lens + prefill_lens
    tokens = sum(lens)
    is_spec = torch.tensor([1] * num_spec_rows + [0] * len(prefill_lens)).bool()
    token_is_spec = torch.repeat_interleave(is_spec, torch.tensor(lens))
    spec_token_indx = token_is_spec.nonzero().flatten().cuda()
    non_spec_token_indx = (~token_is_spec).nonzero().flatten().cuda()
    spec_qsl = torch.arange(0, T * num_spec_rows + 1, T, dtype=torch.int32).cuda()
    meta = GDNAttentionMetadata(
        num_prefills=len(prefill_lens),
        num_prefill_tokens=sum(prefill_lens),
        num_decodes=0,
        num_decode_tokens=0,
        num_spec_decodes=num_spec_rows,
        num_spec_decode_tokens=T * num_spec_rows,
        num_actual_tokens=tokens,
        spec_query_start_loc=spec_qsl,
        spec_state_indices_tensor=torch.tensor(
            slots_per_seq[:num_spec_rows], dtype=torch.int32, device="cuda"
        ),
        spec_sequence_masks=is_spec.cuda(),
        spec_token_indx=spec_token_indx,
        non_spec_token_indx=non_spec_token_indx,
        num_accepted_tokens=torch.tensor(accepted, dtype=torch.int32, device="cuda"),
    )
    if prefill_lens:
        qsl_cpu = torch.tensor(
            [0, *torch.tensor(prefill_lens).cumsum(0).tolist()]
        ).int()
        prefill_slots = [s[0] for s in slots_per_seq[num_spec_rows:]]
        meta.non_spec_query_start_loc = qsl_cpu.cuda()
        meta.non_spec_state_indices_tensor = torch.tensor(
            prefill_slots, dtype=torch.int32, device="cuda"
        )
        meta.has_initial_state = torch.tensor([False, True], device="cuda")[
            : len(prefill_lens)
        ]
        meta.nums_dict, meta.batch_ptr, meta.token_chunk_offset_ptr = (
            compute_causal_conv1d_metadata(qsl_cpu, device=torch.device("cuda"))
        )
        meta.prefill_query_start_loc = meta.non_spec_query_start_loc
        meta.prefill_state_indices = meta.non_spec_state_indices_tensor
        meta.prefill_has_initial_state = meta.has_initial_state
    else:
        meta.non_spec_token_indx = torch.empty(0, dtype=torch.long, device="cuda")
        meta.spec_token_indx = torch.arange(tokens, device="cuda")
    return meta


@pytest.mark.parametrize("with_prefill", [False, True], ids=["verify", "mixed"])
@pytest.mark.parametrize(
    "state_dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"]
)
@pytest.mark.parametrize("num_heads", [16, 32, 64])
@torch.inference_mode()
def test_spec_verify_rows_match_triton(
    monkeypatch, num_heads, state_dtype, with_prefill
):
    num_spec = 2
    _require_flashinfer(num_heads, 1 + num_spec, False, state_dtype)
    # Rows: three verify rows (one a null row), then prefills (fresh, continued).
    slots = [[1, 2, 3], [4, 5, 6], [0, 0, 0], [9, 9, 9], [10, 10, 10]]
    accepted = [1, 3, 1]
    prefill_lens = [21, 6] if with_prefill else []
    meta = _spec_metadata(num_spec, accepted, prefill_lens, 3, slots)
    out = {}
    for backend in ("triton", "flashinfer"):
        layer = _make_layer(num_heads, num_spec, state_dtype, False, backend)
        inputs = _make_inputs(num_heads, meta.num_actual_tokens)
        out[backend] = (_run(layer, meta, inputs, monkeypatch), layer.kv_cache)
    out_f, (conv_f, state_f) = out["flashinfer"]
    out_t, (conv_t, state_t) = out["triton"]
    live_tokens = torch.ones(meta.num_actual_tokens, dtype=torch.bool)
    live_tokens[6:9] = False  # the null verify row
    torch.testing.assert_close(conv_f, conv_t, atol=0, rtol=0)
    tol = 2e-2 if state_dtype == torch.bfloat16 else 2e-3
    torch.testing.assert_close(state_f, state_t, atol=tol, rtol=tol)
    torch.testing.assert_close(
        out_f[0, live_tokens], out_t[0, live_tokens], atol=3e-2, rtol=3e-2
    )


@torch.inference_mode()
def test_decode_step_cuda_graph(monkeypatch):
    """The fused decode launch is graph-safe: a captured decode step replays
    with new inputs and slots like an eager one."""
    num_heads = 32
    _require_flashinfer(num_heads, 1, False, torch.float32)
    n = 6
    layer = _make_layer(num_heads, 0, torch.float32, False, "flashinfer")
    eager = _make_layer(num_heads, 0, torch.float32, False, "flashinfer")
    static_inputs = _make_inputs(num_heads, n, seed=1)
    meta = _non_spec_metadata([1] * n, n, [1, 2, 3, 4, 5, 6], [True] * n)
    _run(layer, meta, static_inputs, monkeypatch)  # builds the FlashInfer weights
    _run(eager, meta, _make_inputs(num_heads, n, seed=1), monkeypatch)
    out = torch.zeros(1, n, num_heads, D, dtype=torch.bfloat16, device="cuda")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        layer._forward(core_attn_out=out, **static_inputs)

    new_inputs = _make_inputs(num_heads, n, seed=2)
    for name, x in new_inputs.items():
        static_inputs[name].copy_(x)
    meta.non_spec_state_indices_tensor.copy_(
        torch.tensor([7, 0, 9, 10, 11, 12], dtype=torch.int32)
    )
    graph.replay()
    expected = _run(eager, meta, new_inputs, monkeypatch)
    torch.testing.assert_close(out, expected, atol=0, rtol=0)
    torch.testing.assert_close(layer.kv_cache[0], eager.kv_cache[0], atol=0, rtol=0)
    torch.testing.assert_close(layer.kv_cache[1], eager.kv_cache[1], atol=0, rtol=0)
