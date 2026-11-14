# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3-Flash KDA layer: the wiring of the planar prefill conv
(`kda_prefill_conv`, default `auto`).

test_kda_prefill_conv.py in this directory covers the kernel alone.
Here the layer's `_forward` runs one mixed chunked-prefill step (decode rows next
to a fresh and a continued prompt) with the stock conv and with the planar
conv on cloned caches, through the Triton chunk path and, where it is
available, FlashKDA, without and with a speculative sequence in the step (the
layer then gathers the non-spec rows into a dense tensor and the conv state
is wider), and once with fp32 conv states (`--mamba-cache-dtype float32`);
conv states must agree bit for bit, the layer output and the recurrent state
within 1e-2 (the conv outputs may differ by one bf16 ulp, see the kernel
test). The dispatch guard must send
a short step to the stock kernel, and the option resolver must resolve
`auto` by device family and reject what it should.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability

pytestmark = pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")

D = 128
HEADS = 2
LOWER_BOUND = -5.0
# Decode rows first, then one fresh and one continued prompt. 8.3K prefill
# tokens clear the dispatch guard's bar at 2 heads (8192 + 6 per decode row).
LONG_STEP = ([1, 1, 1, 5000, 3300], [True, True, True, False, True], [1, 2, 3, 4, 5])
SHORT_STEP = ([1, 1, 300, 200], [True, True, False, True], [1, 2, 3, 4])
NUM_SLOTS = 6  # the non-spec slots; a speculative sequence's slots follow


def _make_layer(kda, backend: str, dim_first: bool, device, monkeypatch):
    """A layer assembled without __init__, as test_kda_recurrent.py does."""
    from vllm.v1.worker.workspace import WorkspaceManager

    width = 3 * HEADS * D
    layer = object.__new__(kda.Glm5NextLinearAttention)
    torch.nn.Module.__init__(layer)
    layer.prefix = "layer.0"
    layer.local_num_heads = HEADS
    layer.head_dim = D
    layer.conv_size = 4
    layer.local_projection_size = HEADS * D
    layer.kda_safe_gate = True
    layer.kda_lower_bound = LOWER_BOUND
    layer.kda_prefill_backend = backend
    layer._conv_state_dim_first = dim_first
    layer.conv1d = SimpleNamespace(
        weight=torch.randn(width, 1, 4, device=device) * 0.1, bias=None
    )
    layer.A_log = torch.randn(1, 1, HEADS, 1, device=device) * 0.1
    layer.dt_bias = torch.randn(HEADS * D, device=device) * 0.1
    layer._flashkda_buffer_specs = None
    layer._checkpoint_exporter = None
    if backend == "flashkda":
        from vllm.model_executor.layers.mamba.kda_checkpoint import (
            FlashKDAPrefillCheckpointExporter,
        )

        layer._checkpoint_exporter = FlashKDAPrefillCheckpointExporter(state_len=3)
        monkeypatch.setattr(
            kda, "current_workspace_manager", lambda: WorkspaceManager(device)
        )
    return layer


def _flashkda_buffer_specs(step):
    """The layer's FlashKDA workspace buffers, sized for one step as __init__
    would size them for the whole batch."""
    num_tokens, num_seqs = sum(step[0]), len(step[0])
    state_shape = (num_seqs, HEADS, D, D)
    return (
        (state_shape, torch.float32),
        (state_shape, torch.float32),
        (
            (torch.ops._flashkda_C.get_workspace_size(num_tokens, HEADS, num_seqs),),
            torch.uint8,
        ),
        ((1, num_tokens, HEADS, D), torch.bfloat16),
    )


def _metadata(step, device):
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
    from vllm.v1.attention.backends.utils import compute_causal_conv1d_metadata

    lens, has_init, slots = step
    cu_cpu = torch.tensor([0, *torch.tensor(lens).cumsum(0).tolist()]).int()
    nums, batch_ptr, chunk_offsets = compute_causal_conv1d_metadata(
        cu_cpu, device=device
    )
    num_decodes = sum(n == 1 for n in lens)
    return GDNAttentionMetadata(
        num_prefills=len(lens) - num_decodes,
        num_prefill_tokens=sum(lens) - num_decodes,
        num_decodes=num_decodes,
        num_decode_tokens=num_decodes,
        num_spec_decodes=0,
        num_spec_decode_tokens=0,
        num_actual_tokens=sum(lens),
        has_initial_state=torch.tensor(has_init, device=device),
        non_spec_query_start_loc=cu_cpu.to(device),
        non_spec_state_indices_tensor=torch.tensor(
            slots, dtype=torch.int32, device=device
        ),
        nums_dict=nums,
        batch_ptr=batch_ptr,
        token_chunk_offset_ptr=chunk_offsets,
    )


def _add_spec_sequence(metadata, num_spec: int, num_seqs: int, device):
    """One speculative sequence of `num_spec + 1` rows ahead of the step's
    non-spec rows (as test_kda_recurrent.py sets it up), on the slots after
    the non-spec ones; returns its row count. The step's plain decode rows
    become prefills, as the metadata builder counts them once speculative
    rows are present."""
    spec_tokens = num_spec + 1
    metadata.num_spec_decodes = 1
    metadata.num_spec_decode_tokens = spec_tokens
    metadata.num_actual_tokens += spec_tokens
    metadata.num_prefills += metadata.num_decodes
    metadata.num_prefill_tokens += metadata.num_decode_tokens
    metadata.num_decodes = 0
    metadata.num_decode_tokens = 0
    metadata.spec_sequence_masks = torch.tensor(
        [True] + [False] * num_seqs, device=device
    )
    metadata.spec_query_start_loc = torch.tensor(
        [0, spec_tokens], dtype=torch.int32, device=device
    )
    metadata.spec_state_indices_tensor = torch.arange(
        NUM_SLOTS, NUM_SLOTS + spec_tokens, dtype=torch.int32, device=device
    ).unsqueeze(0)
    metadata.spec_token_indx = torch.arange(spec_tokens, device=device)
    metadata.non_spec_token_indx = torch.arange(
        spec_tokens, metadata.num_actual_tokens, device=device
    )
    metadata.num_accepted_tokens = torch.ones(1, dtype=torch.int32, device=device)
    return spec_tokens


def _run_step(
    kda, layer, step, conv, recurrent, conv_option, monkeypatch, device, num_spec=0
):
    """One _forward on clones of the caches; returns (out, conv, recurrent)."""
    lens = step[0]
    width = 3 * HEADS * D
    metadata = _metadata(step, device)
    spec_tokens = (
        _add_spec_sequence(metadata, num_spec, len(lens), device) if num_spec else 0
    )
    num_tokens = sum(lens) + spec_tokens
    torch.manual_seed(1)
    # q|k|v is the leading, row-strided slice of the fused projection.
    projected = torch.randn(
        num_tokens, width + HEADS + 2 * D, device=device, dtype=torch.bfloat16
    )
    qkv, beta = projected[:, :width], projected[:, width : width + HEADS].unsqueeze(0)
    gate = torch.randn(1, num_tokens, HEADS, D, device=device, dtype=torch.bfloat16)
    conv, recurrent = conv.clone(), recurrent.clone()
    layer.kv_cache = (conv, recurrent)
    layer.kda_prefill_conv = conv_option
    if layer.kda_prefill_backend == "flashkda":
        layer._flashkda_buffer_specs = _flashkda_buffer_specs(step)
    monkeypatch.setattr(
        kda,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata={layer.prefix: metadata}),
    )
    out = torch.empty(1, num_tokens, HEADS, D, device=device, dtype=torch.bfloat16)
    layer._forward(qkv, gate, beta, out)
    return out, conv, recurrent


def _check_planar_matches_stock_in_layer(
    monkeypatch, backend, dim_first, num_spec, conv_dtype
):
    from vllm.models.glm5next.common import kda

    if backend == "flashkda":
        if (
            kda._resolve_kda_prefill_backend("auto", D, torch.bfloat16, LOWER_BOUND)
            != "flashkda"
        ):
            pytest.skip("requires FlashKDA")
        import vllm._flashkda_C  # noqa: F401

    torch.manual_seed(42)
    device = torch.device("cuda")
    layer = _make_layer(kda, backend, dim_first, device, monkeypatch)
    width, state_len = 3 * HEADS * D, 3 + num_spec
    num_slots = NUM_SLOTS + (num_spec + 1 if num_spec else 0)
    conv_shape = (width, state_len) if dim_first else (state_len, width)
    conv = torch.randn(num_slots, *conv_shape, device=device, dtype=conv_dtype)
    recurrent = torch.randn(num_slots, HEADS, D, D, device=device) * 0.1

    planar_calls = []
    planar = kda.causal_conv1d_planar

    def counting_planar(*args, **kwargs):
        planar_calls.append(1)
        return planar(*args, **kwargs)

    monkeypatch.setattr(kda, "causal_conv1d_planar", counting_planar)

    args = (kda, layer, LONG_STEP, conv, recurrent)
    out_stock, conv_stock, rec_stock = _run_step(
        *args, "causal_conv1d", monkeypatch, device, num_spec
    )
    assert not planar_calls
    out_planar, conv_planar, rec_planar = _run_step(
        *args, "planar", monkeypatch, device, num_spec
    )
    assert len(planar_calls) == 1
    # Conv states match bit for bit; the conv outputs may differ from the
    # stock kernel's by one bf16 ulp on a few elements (see the kernel test),
    # which the layer output and the recurrent state carry as differences far
    # below these tolerances, where a wiring error shows as large ones.
    assert torch.equal(conv_planar, conv_stock)
    torch.testing.assert_close(out_planar, out_stock, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(rec_planar, rec_stock, atol=1e-2, rtol=1e-2)

    # Short step: the dispatch guard keeps the stock kernel.
    args = (kda, layer, SHORT_STEP, conv, recurrent)
    out_stock, conv_stock, _ = _run_step(
        *args, "causal_conv1d", monkeypatch, device, num_spec
    )
    out_guarded, conv_guarded, _ = _run_step(
        *args, "planar", monkeypatch, device, num_spec
    )
    assert len(planar_calls) == 1
    assert torch.equal(conv_guarded, conv_stock)
    assert torch.equal(out_guarded, out_stock)


@pytest.mark.parametrize("backend", ["triton", "flashkda"])
@pytest.mark.parametrize("dim_first", [False, True], ids=["SD", "DS"])
# 0: the projection's row-strided q|k|v reaches the conv; 3: a speculative
# sequence shares the step, the layer gathers the non-spec rows into a dense
# tensor and the conv state has num_spec more positions.
@pytest.mark.parametrize("num_spec", [0, 3])
@torch.inference_mode()
def test_planar_prefill_conv_matches_stock_in_layer(
    monkeypatch, backend, dim_first, num_spec
):
    _check_planar_matches_stock_in_layer(
        monkeypatch, backend, dim_first, num_spec, torch.bfloat16
    )


@pytest.mark.parametrize("dim_first", [False, True], ids=["SD", "DS"])
@torch.inference_mode()
def test_planar_prefill_conv_fp32_states_in_layer(monkeypatch, dim_first):
    # --mamba-cache-dtype float32: fp32 conv states under the layer's fp32 conv
    # weights, the other pairing the kernel matches (states bit for bit, outputs
    # within one ulp; the stock
    # kernel then runs on a copy of x widened to fp32), in a speculative step
    # with its wider state.
    _check_planar_matches_stock_in_layer(
        monkeypatch, "triton", dim_first, 3, torch.float32
    )


def test_resolve_kda_prefill_conv(monkeypatch):
    from vllm.models.glm5next.common import kda

    bf16 = torch.bfloat16
    assert kda._resolve_kda_prefill_conv("causal_conv1d", 4, bf16, bf16) == (
        "causal_conv1d"
    )
    assert kda._resolve_kda_prefill_conv("planar", 4, bf16, bf16) == "planar"
    with pytest.raises(ValueError, match="causal_conv1d"):
        kda._resolve_kda_prefill_conv("fused", 4, bf16, bf16)
    with pytest.raises(ValueError, match="width 4"):
        kda._resolve_kda_prefill_conv("planar", 3, bf16, bf16)
    # The stock kernel casts x to the conv-state dtype before multiplying; with
    # the layer's fp32 conv weights, bf16 activations match it (states exactly,
    # outputs within one ulp) with bf16 and with fp32 conv states, and the layer
    # accepts those two pairings.
    assert kda._resolve_kda_prefill_conv("planar", 4, bf16, torch.float32) == "planar"
    with pytest.raises(ValueError, match="bfloat16"):
        kda._resolve_kda_prefill_conv("planar", 4, bf16, torch.float16)
    with pytest.raises(ValueError, match="bfloat16"):
        kda._resolve_kda_prefill_conv("planar", 4, torch.float16, torch.float16)
    assert kda._resolve_kda_prefill_conv(
        "causal_conv1d", 4, torch.bfloat16, torch.float32
    )
    # The fp32-state pairing relies on fp32 conv weights (the layer's); with
    # bf16 weights it is refused like any other mixed pairing.
    with pytest.raises(ValueError, match="bfloat16"):
        kda._resolve_kda_prefill_conv("planar", 4, bf16, torch.float32, bf16)
    # "auto" (the default) takes the planar kernel on SM10x devices only, for
    # the two pairings above, and never raises; "planar" stays available on
    # other CUDA devices. The device capability is patched at class level so
    # the real family check runs.
    platform_cls = type(kda.current_platform)

    def set_capability(major, minor):
        monkeypatch.setattr(
            platform_cls,
            "get_device_capability",
            classmethod(lambda cls, device_id=0: DeviceCapability(major, minor)),
        )

    set_capability(10, 3)
    assert kda._resolve_kda_prefill_conv("auto", 4, bf16, bf16) == "planar"
    assert kda._resolve_kda_prefill_conv("auto", 4, bf16, torch.float32) == "planar"
    assert kda._resolve_kda_prefill_conv("auto", 4, bf16, torch.float16) == (
        "causal_conv1d"
    )
    assert kda._resolve_kda_prefill_conv("auto", 4, bf16, torch.float32, bf16) == (
        "causal_conv1d"
    )
    assert kda._resolve_kda_prefill_conv("auto", 3, bf16, bf16) == "causal_conv1d"
    for major, minor in ((9, 0), (12, 0)):
        set_capability(major, minor)
        assert kda._resolve_kda_prefill_conv("auto", 4, bf16, bf16) == ("causal_conv1d")
        assert kda._resolve_kda_prefill_conv("planar", 4, bf16, bf16) == "planar"
