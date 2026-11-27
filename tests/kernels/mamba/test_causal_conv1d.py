# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from collections.abc import Iterator

import pytest
import torch
import torch.nn.functional as F
from einops import rearrange

from tests.kernels.mamba.utils import record_triton_compiles
from tests.utils import create_new_process_for_each_test
from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
    causal_conv1d_fn,
    causal_conv1d_update,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

DEVICE = current_platform.device_type

pytestmark = pytest.mark.skipif(
    not (
        current_platform.is_cuda_alike()
        or current_platform.is_xpu()
        or current_platform.is_cpu()
    ),
    reason="causal_conv1d Triton kernels require CUDA-alike, XPU, or CPU",
)


def causal_conv1d_ref(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    initial_states: torch.Tensor | None = None,
    return_final_states: bool = False,
    final_states_out: torch.Tensor | None = None,
    activation: str | None = "silu",
):
    """x: (batch, dim, seqlen)
    weight: (dim, width)
    bias: (dim,)
    initial_states: (batch, dim, width - 1)
    final_states_out: (batch, dim, width - 1)

    out: (batch, dim, seqlen)
    """
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")
    dtype_in = x.dtype
    x = x.to(weight.dtype)
    seqlen = x.shape[-1]
    dim, width = weight.shape
    if initial_states is None:
        out = F.conv1d(x, weight.unsqueeze(1), bias, padding=width - 1, groups=dim)
    else:
        x = torch.cat([initial_states, x], dim=-1)
        out = F.conv1d(x, weight.unsqueeze(1), bias, padding=0, groups=dim)
    out = out[..., :seqlen]
    if return_final_states:
        final_states = F.pad(x, (width - 1 - x.shape[-1], 0)).to(
            dtype_in
        )  # (batch, dim, width - 1)
        if final_states_out is not None:
            final_states_out.copy_(final_states)
        else:
            final_states_out = final_states
    out = (out if activation is None else F.silu(out)).to(dtype=dtype_in)
    return (out, None) if not return_final_states else (out, final_states_out)


def causal_conv1d_update_ref(
    x, conv_state, weight, bias=None, activation=None, cache_seqlens=None
):
    """x: (batch, dim) or (batch, dim, seqlen)
    conv_state: (batch, dim, state_len), where state_len >= width - 1
    weight: (dim, width)
    bias: (dim,)
    cache_seqlens: (batch,), dtype int32.
        If not None, the conv_state is treated as a circular buffer.
        The conv_state will be updated by copying x to the
        conv_state starting at the index
        @cache_seqlens % state_len before performing the convolution.

    out: (batch, dim) or (batch, dim, seqlen)
    """
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")
    dtype_in = x.dtype
    unsqueeze = x.dim() == 2
    if unsqueeze:
        x = x.unsqueeze(-1)
    batch, dim, seqlen = x.shape
    width = weight.shape[1]
    state_len = conv_state.shape[-1]
    assert conv_state.shape == (batch, dim, state_len)
    assert weight.shape == (dim, width)
    if cache_seqlens is None:
        x_new = torch.cat([conv_state, x], dim=-1).to(
            weight.dtype
        )  # (batch, dim, state_len + seqlen)
        conv_state.copy_(x_new[:, :, -state_len:])
    else:
        width_idx = torch.arange(
            -(width - 1), 0, dtype=torch.long, device=x.device
        ).unsqueeze(0) + cache_seqlens.unsqueeze(1)
        width_idx = (
            torch.remainder(width_idx, state_len).unsqueeze(1).expand(-1, dim, -1)
        )
        x_new = torch.cat([conv_state.gather(2, width_idx), x], dim=-1).to(weight.dtype)
        copy_idx = torch.arange(seqlen, dtype=torch.long, device=x.device).unsqueeze(
            0
        ) + cache_seqlens.unsqueeze(1)
        copy_idx = torch.remainder(copy_idx, state_len).unsqueeze(1).expand(-1, dim, -1)
        conv_state.scatter_(2, copy_idx, x)
    out = F.conv1d(x_new, weight.unsqueeze(1), bias, padding=0, groups=dim)[
        :, :, -seqlen:
    ]
    if unsqueeze:
        out = out.squeeze(-1)
    return (out if activation is None else F.silu(out)).to(dtype=dtype_in)


@pytest.mark.parametrize("width", [1, 8])
@pytest.mark.skipif(
    current_platform.is_cpu(),
    reason="The width check lives in the Triton wrappers, which CPU bypasses.",
)
def test_causal_conv1d_rejects_unsupported_width(width):
    # The kernels branch on widths 2 to 7; any other width would compute
    # wrong values silently.
    dim = 8
    weight = torch.randn(dim, width, device=DEVICE)
    conv_state = torch.zeros(1, dim, 7, device=DEVICE)
    x = torch.randn(dim, 1, device=DEVICE)
    query_start_loc = torch.tensor([0, 1], dtype=torch.int32, device=DEVICE)
    with pytest.raises(ValueError, match="widths 2 to 7"):
        causal_conv1d_fn(x, weight, None, conv_state, query_start_loc)
    with pytest.raises(ValueError, match="widths 2 to 7"):
        causal_conv1d_update(x.T.contiguous(), conv_state, weight)


@pytest.mark.parametrize("itype", [torch.bfloat16])
@pytest.mark.parametrize("silu_activation", [False, True])
@pytest.mark.parametrize("has_bias", [False, True])
@pytest.mark.parametrize("seqlen", [1])
@pytest.mark.parametrize("width", [4, 7])
@pytest.mark.parametrize("dim", [2048, 2048 + 16, 4096])
def test_causal_conv1d_update(dim, width, seqlen, has_bias, silu_activation, itype):
    device = DEVICE
    rtol, atol = (3e-4, 1e-3) if itype == torch.float32 else (3e-3, 5e-3)
    if itype == torch.bfloat16:
        rtol, atol = 1e-2, 5e-2
    # set seed
    set_random_seed(0)
    batch = 2
    x = torch.randn(batch, dim, seqlen, device=device, dtype=itype)
    x_ref = x.clone()
    # +1 entry to reserve index 0 as null block
    conv_state = torch.randn(batch + 1, dim, width - 1, device=device, dtype=itype)

    weight = torch.randn(dim, width, device=device, dtype=itype)
    bias = torch.randn(dim, device=device, dtype=itype) if has_bias else None
    # Start indices from 1, skipping null block at index 0
    conv_state_indices = torch.arange(1, batch + 1, dtype=torch.int32, device=device)
    conv_state_ref = conv_state[conv_state_indices].detach().clone()
    activation = None if not silu_activation else "silu"

    out = causal_conv1d_update(
        x,
        conv_state,
        weight,
        bias,
        activation=activation,
        conv_state_indices=conv_state_indices,
    )
    out_ref = causal_conv1d_update_ref(
        x_ref, conv_state_ref, weight, bias, activation=activation
    )

    assert torch.equal(conv_state[conv_state_indices], conv_state_ref)
    assert torch.allclose(out, out_ref, rtol=rtol, atol=atol)


@pytest.mark.parametrize("itype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("silu_activation", [False, True])
@pytest.mark.parametrize("has_bias", [False, True])
@pytest.mark.parametrize("seqlen", [1, 3])
@pytest.mark.parametrize("width", [3, 4, 7])
@pytest.mark.parametrize("dim", [2048 + 16, 4096])
# tests correctness in case subset of the sequences are padded
@pytest.mark.parametrize("with_padding", [True, False])
@pytest.mark.parametrize("batch_size", [3])
def test_causal_conv1d_update_with_batch_gather(
    batch_size, with_padding, dim, width, seqlen, has_bias, silu_activation, itype
):
    device = DEVICE
    rtol, atol = (3e-4, 1e-3) if itype == torch.float32 else (3e-3, 5e-3)
    if itype == torch.bfloat16:
        rtol, atol = 1e-2, 5e-2

    # set seed
    set_random_seed(0)

    padding = 5 if with_padding else 0
    padded_batch_size = batch_size + padding
    # total_entries = number of cache line
    total_entries = 10 * batch_size

    # x will be (batch, dim, seqlen) with contiguous along dim-axis
    x = torch.randn(
        padded_batch_size, seqlen, dim, device=device, dtype=itype
    ).transpose(1, 2)

    x_ref = x.clone()

    # +1 to exclude index 0 (null block)
    conv_state_indices = (torch.randperm(total_entries - 1)[:batch_size] + 1).to(
        dtype=torch.int32, device=device
    )
    unused_states_bool = torch.ones(total_entries, dtype=torch.bool, device=device)
    unused_states_bool[conv_state_indices] = False
    padded_state_indices = torch.concat(
        [
            conv_state_indices,
            torch.as_tensor(
                [NULL_BLOCK_ID] * padding, dtype=torch.int32, device=device
            ),
        ],
        dim=0,
    )

    # conv_state will be (cache_lines, dim, state_len)
    # with contiguous along dim-axis
    conv_state = torch.randn(
        total_entries, width - 1, dim, device=device, dtype=itype
    ).transpose(1, 2)

    conv_state_for_padding_test = conv_state.clone()

    weight = torch.randn(dim, width, device=device, dtype=itype)
    bias = torch.randn(dim, device=device, dtype=itype) if has_bias else None
    conv_state_ref = conv_state[conv_state_indices, :].detach().clone()
    activation = None if not silu_activation else "silu"

    out = causal_conv1d_update(
        x,
        conv_state,
        weight,
        bias,
        activation=activation,
        conv_state_indices=padded_state_indices,
    )
    out_ref = causal_conv1d_update_ref(
        x_ref[:batch_size], conv_state_ref, weight, bias, activation=activation
    )

    assert torch.equal(conv_state[conv_state_indices, :], conv_state_ref)
    assert torch.equal(
        conv_state[unused_states_bool], conv_state_for_padding_test[unused_states_bool]
    )
    assert torch.allclose(out[:batch_size], out_ref, rtol=rtol, atol=atol)


@pytest.mark.parametrize("itype", [torch.bfloat16])
@pytest.mark.parametrize("silu_activation", [True])
@pytest.mark.parametrize("has_bias", [True])
@pytest.mark.parametrize("width", [4, 5, 6, 7])
@pytest.mark.parametrize("seqlen", [8, 249, 4096])
@pytest.mark.parametrize("dim", [64, 4096])
@pytest.mark.parametrize("with_padding", [True, False])
@pytest.mark.parametrize("batch", [4, 10])
def test_causal_conv1d_varlen(
    batch, with_padding, dim, seqlen, width, has_bias, silu_activation, itype
):
    device = DEVICE
    if not current_platform.is_cpu():
        torch.accelerator.empty_cache()
    rtol, atol = (3e-4, 1e-3) if itype == torch.float32 else (3e-3, 5e-3)
    if itype == torch.bfloat16:
        rtol, atol = 1e-2, 5e-2
    # set seed
    set_random_seed(0)
    seqlens = []
    batch_size = batch
    padding = 3 if with_padding else 0
    padded_batch_size = batch_size + padding
    nsplits = padded_batch_size - 1

    eos_pos = torch.randperm(seqlen - 1)[:nsplits].sort().values

    seqlens.append(
        torch.diff(
            torch.cat([torch.tensor([-1]), eos_pos, torch.tensor([seqlen - 1])])
        ).tolist()
    )
    assert sum(seqlens[-1]) == seqlen
    assert all(s > 0 for s in seqlens[-1])

    total_entries = batch_size * 10
    cumsum = torch.cumsum(torch.tensor(seqlens[0]), dim=0).to(torch.int32)
    cumsum = torch.concat([torch.tensor([0], dtype=torch.int32), cumsum], dim=0)
    x = rearrange(
        torch.randn(1, seqlen, 4096 + dim + 64, device=device, dtype=itype),
        "b s d -> b d s",
    )[:, 4096 : 4096 + dim, :]

    weight = torch.randn(dim, width, device=device, dtype=itype)

    bias = torch.randn(dim, device=device, dtype=itype) if has_bias else None
    x_ref = x.clone()
    weight_ref = weight.clone()
    bias_ref = bias.clone() if bias is not None else None
    activation = None if not silu_activation else "silu"
    final_states = torch.randn(
        total_entries, width - 1, dim, device=x.device, dtype=x.dtype
    ).transpose(1, 2)
    final_states_ref = final_states.clone()
    has_initial_states = torch.randint(
        0, 2, (cumsum.shape[0] - 1,), dtype=torch.bool, device=x.device
    )
    # +1 to exclude index 0 (null block)
    state_indices = (
        torch.randperm(total_entries - 1, dtype=torch.int32, device=x.device)[
            :batch_size
        ]
        + 1
    )
    padded_state_indices = torch.concat(
        [
            state_indices,
            torch.as_tensor(
                [NULL_BLOCK_ID] * padding, dtype=torch.int32, device=device
            ),
        ],
        dim=-1,
    )
    out = causal_conv1d_fn(
        x.squeeze(0),
        weight,
        bias=bias,
        conv_states=final_states,
        query_start_loc=cumsum.to(device),
        cache_indices=padded_state_indices,
        has_initial_state=has_initial_states,
        activation=activation,
    )

    out_ref = []
    out_ref_b = []

    splits = [torch.split(var, seqlens[0], dim=-1) for var in (x_ref)]
    for i in range(len(seqlens[0])):
        x_s = [v[i].unsqueeze(0) for v in splits][0]
        if padded_state_indices[i] == NULL_BLOCK_ID:
            continue
        out_ref_b.append(
            causal_conv1d_ref(
                x_s,
                weight_ref,
                bias_ref,
                activation=activation,
                return_final_states=True,
                final_states_out=final_states_ref[padded_state_indices[i]].unsqueeze(0),
                initial_states=final_states_ref[padded_state_indices[i]].unsqueeze(0)
                if has_initial_states[i]
                else None,
            )
        )
    out_ref.append(torch.cat([t[0] for t in out_ref_b], dim=2))
    out_ref_tensor = torch.cat(out_ref, dim=0)

    assert torch.allclose(
        final_states[state_indices],
        final_states_ref[state_indices],
        rtol=rtol,
        atol=atol,
    )
    unpadded_out = out[:, : out_ref_tensor.shape[-1]]
    assert torch.allclose(unpadded_out, out_ref_tensor, rtol=rtol, atol=atol)


@pytest.mark.parametrize("itype", [torch.bfloat16])
@pytest.mark.parametrize("silu_activation", [True])
@pytest.mark.parametrize("has_bias", [True])
@pytest.mark.parametrize("width", [4, 7])
@pytest.mark.parametrize("dim", [64, 2048])
@pytest.mark.parametrize("q_lens_cfg", [[3, 1, 0, 4, 2], [1, 5, 5, 1], [2, 0, 0, 3]])
@pytest.mark.skipif(
    current_platform.is_cpu(),
    reason="CPU kernel for causal_conv1d_update ignores num_accepted_tokens.",
)
def test_causal_conv1d_update_ragged_varlen(
    q_lens_cfg, dim, width, has_bias, silu_activation, itype
):
    mk_inputs = lambda: _create_ragged_varlen_inputs(
        q_lens_cfg, dim, width, has_bias, silu_activation, itype
    )
    inputs, state = mk_inputs()
    actual = causal_conv1d_update(conv_state=state, **inputs)

    ref_inputs, ref_state = mk_inputs()
    ref_outs = _create_ragged_varlen_ref_outs(conv_state=ref_state, **ref_inputs)

    for i, ref_out in enumerate(ref_outs):
        start = int(inputs["query_start_loc"][i])
        end = int(inputs["query_start_loc"][i + 1])
        entry = inputs["conv_state_indices"][i : i + 1]
        assert torch.allclose(actual[start:end], ref_out, rtol=1e-2, atol=5e-2)
        assert torch.equal(state[entry], ref_state[entry])


def _create_ragged_varlen_inputs(
    q_lens_cfg, dim, width, has_bias, silu_activation, itype
):
    device = DEVICE
    set_random_seed(0)

    batch_size = len(q_lens_cfg)
    max_query_len = 5
    assert max(q_lens_cfg) <= max_query_len
    state_len = width - 1 + max_query_len
    total_entries = 10 * batch_size

    q_lens = torch.tensor(q_lens_cfg, dtype=torch.int32, device=device)
    query_start_loc = torch.cat(
        [
            torch.zeros(1, dtype=torch.int32, device=device),
            torch.cumsum(q_lens, dim=0, dtype=torch.int32),
        ]
    )
    total_tokens = int(q_lens.sum())

    x = torch.randn(total_tokens, dim, device=device, dtype=itype)
    conv_state = torch.randn(
        total_entries, state_len, dim, device=device, dtype=itype
    ).transpose(1, 2)
    weight = torch.randn(dim, width, device=device, dtype=itype)
    bias = torch.randn(dim, device=device, dtype=itype) if has_bias else None
    activation = None if not silu_activation else "silu"

    conv_state_indices = (
        torch.randperm(total_entries - 1, device=device)[:batch_size] + 1
    ).to(torch.int32)
    num_accepted_tokens = torch.randint(
        1, max_query_len + 1, (batch_size,), dtype=torch.int32, device=device
    )
    inputs = dict(
        x=x,
        weight=weight,
        bias=bias,
        activation=activation,
        conv_state_indices=conv_state_indices,
        num_accepted_tokens=num_accepted_tokens,
        query_start_loc=query_start_loc,
        max_query_len=max_query_len,
    )
    return inputs, conv_state


def _create_ragged_varlen_ref_outs(
    x,
    conv_state,
    weight,
    bias,
    activation,
    conv_state_indices,
    num_accepted_tokens,
    query_start_loc,
    max_query_len,
) -> Iterator[torch.Tensor]:
    for seq_idx in range(len(conv_state_indices)):
        start = int(query_start_loc[seq_idx])
        end = int(query_start_loc[seq_idx + 1])
        entry = conv_state_indices[seq_idx : seq_idx + 1]
        if start == end:
            yield x[start:end]
            continue
        out_ref = causal_conv1d_update(
            x[start:end],
            conv_state,
            weight,
            bias,
            activation=activation,
            conv_state_indices=entry,
            num_accepted_tokens=num_accepted_tokens[seq_idx : seq_idx + 1],
            query_start_loc=query_start_loc.new_tensor([0, end - start]),
            max_query_len=max_query_len,
        )
        yield out_ref


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Triton JIT on CUDA")
@create_new_process_for_each_test("spawn")
def test_conv1d_update_compiles_once_for_graph_and_eager_state_indices():
    """CUDA-graph batches slice the builder's `(rows, 1 + num_spec)` state
    index buffer, eager batches a one-slot block-table gather. An eager batch
    must reuse the kernel a graph batch compiled, or the first one compiles
    during inference. The conv states are views into padded cache pages, as the
    mixer reads them. A fresh process keeps other tests' compiles out of the
    JIT caches."""
    set_random_seed(0)
    dim, width, num_spec = 256, 4, 3
    query_len, state_len = 1 + num_spec, width - 1 + num_spec
    dtype = torch.bfloat16
    pages = torch.randn(10, state_len * dim + 64, device=DEVICE, dtype=dtype)
    conv_state = pages[:, : state_len * dim].view(10, state_len, dim).transpose(1, 2)
    weight = torch.randn(dim, width, device=DEVICE, dtype=dtype)
    bias = torch.randn(dim, device=DEVICE, dtype=dtype)

    def decode(state_indices):
        rows = state_indices.size(0)
        causal_conv1d_update(
            torch.randn(rows * query_len, dim, device=DEVICE, dtype=dtype),
            conv_state,
            weight,
            bias,
            "silu",
            conv_state_indices=state_indices,
            num_accepted_tokens=torch.ones(rows, device=DEVICE).int(),
            query_start_loc=torch.arange(
                0, rows * query_len + 1, query_len, device=DEVICE
            ).int(),
            max_query_len=query_len,
        )

    # Two decodes padded to four rows of the graph buffer.
    graph_indices = torch.full(
        (8, query_len), NULL_BLOCK_ID, dtype=torch.int32, device=DEVICE
    )
    graph_indices[:2, 0] = torch.tensor([5, 9], device=DEVICE)
    decode(graph_indices[:4])
    torch.accelerator.synchronize()

    with record_triton_compiles() as compiled:
        decode(torch.tensor([[3], [6]], device=DEVICE).int())
        decode(torch.tensor([[3], [6], [7]], device=DEVICE).int())
        torch.accelerator.synchronize()

    assert not compiled, f"Triton compiled for an eager batch: {compiled}"


requires_flashinfer_conv1d = pytest.mark.skipif(
    not (
        current_platform.is_cuda()
        and current_platform.get_device_capability() in ((10, 0), (10, 3))
    ),
    reason="FlashInfer conv1d needs compute capability 10.0 or 10.3",
)

FI_DIM, FI_WIDTH, FI_SLOTS = 1280, 4, 64


def _fi_layer(state_len, dim_first):
    """Conv weights and a conv state pool laid out as a Mamba2 layer reads
    them: views into padded cache pages, dim-first or state-first."""
    dtype = torch.bfloat16
    weight = torch.randn(FI_DIM, FI_WIDTH, device=DEVICE, dtype=dtype)
    bias = torch.randn(FI_DIM, device=DEVICE, dtype=dtype)
    pages = torch.randn(FI_SLOTS, state_len * FI_DIM + 64, device=DEVICE, dtype=dtype)
    payload = pages[:, : state_len * FI_DIM]
    if dim_first:
        state = payload.view(FI_SLOTS, FI_DIM, state_len)
    else:
        state = payload.view(FI_SLOTS, state_len, FI_DIM).transpose(1, 2)
    return weight, bias, state


def _fi_rows(num_tokens):
    """Conv inputs as the mixer passes them: rows of a wider projection."""
    projected = torch.randn(
        num_tokens, FI_DIM + 96, device=DEVICE, dtype=torch.bfloat16
    )
    return projected[:, :FI_DIM]


def _prefill_metadata(lengths, slots, has_initial_states):
    from vllm.v1.attention.backends.mamba2_attn import Mamba2AttentionMetadata

    starts = torch.tensor((0, *lengths), dtype=torch.int32).cumsum(0).int()
    metadata = Mamba2AttentionMetadata(
        num_prefills=len(lengths),
        num_prefill_tokens=sum(lengths),
        num_decodes=0,
        num_decode_tokens=0,
        num_reqs=len(lengths),
        has_initial_states_p=torch.tensor(has_initial_states, device=DEVICE),
        query_start_loc_p=starts.to(DEVICE),
        # A block-table column, strided as with a speculative block.
        state_indices_tensor_p=torch.tensor(
            [[slot, 0] for slot in slots], dtype=torch.int32, device=DEVICE
        )[:, 0],
        state_indices_tensor_d=None,
        query_start_loc_d=None,
        num_accepted_tokens=None,
        seq_lens=torch.tensor(lengths, dtype=torch.int32, device=DEVICE),
    )
    return metadata, starts


@requires_flashinfer_conv1d
@pytest.mark.parametrize("dim_first", [False, True])
@pytest.mark.parametrize("num_spec", [0, 3])
@pytest.mark.parametrize(
    "lengths",
    [[37], [1, 2, 3, 70, 5, 6, 7, 300], [9] * 40],
    ids=["one", "eight", "forty"],
)
def test_flashinfer_conv1d_prefill_matches_triton(dim_first, num_spec, lengths):
    """Fed vLLM's prefill metadata, FlashInfer's prefill writes the same conv
    states as Triton's and outputs the same up to BF16 rounding, so either
    prefill kernel can precede either decode kernel. Forty prefills need
    FlashInfer's workspace."""
    from vllm.model_executor.layers.linear_attention.conv1d.prefill.flashinfer import (  # noqa: E501
        FlashInferConv1dPrefillKernel,
    )
    from vllm.model_executor.layers.linear_attention.conv1d.prefill.triton import (
        TritonConv1dPrefillKernel,
    )

    set_random_seed(0)
    weight, bias, state = _fi_layer(FI_WIDTH - 1 + num_spec, dim_first)
    slots = torch.randperm(FI_SLOTS - 1)[: len(lengths)] + 1
    has_initial_states = [i % 2 == 0 for i in range(len(lengths))]
    metadata, starts = _prefill_metadata(lengths, slots.tolist(), has_initial_states)
    x = _fi_rows(sum(lengths))

    outputs, states = [], []
    for kernel in (TritonConv1dPrefillKernel, FlashInferConv1dPrefillKernel):
        kernel_state = state.clone()
        out = kernel.forward(
            x.T,
            kernel_state,
            weight,
            bias,
            "silu",
            kernel.prepare_metadata(metadata, starts),
        )
        outputs.append(out.T)
        states.append(kernel_state)

    torch.testing.assert_close(outputs[1], outputs[0], atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(states[1], states[0], atol=0, rtol=0)


@requires_flashinfer_conv1d
@pytest.mark.parametrize("dim_first", [False, True])
@pytest.mark.parametrize("num_spec", [0, 3])
@pytest.mark.parametrize("padded", [False, True])
def test_flashinfer_conv1d_decode_matches_triton(dim_first, num_spec, padded):
    """Fed vLLM's decode metadata (a `(rows, 1 + num_spec)` state index
    buffer, padded with the null block in a full CUDA graph), FlashInfer's
    decode matches Triton's on the live rows and leaves the null block alone.
    With speculative decoding, each row starts after its accepted tokens."""
    from vllm.config import VllmConfig
    from vllm.model_executor.layers.linear_attention.conv1d.decode.flashinfer import (  # noqa: E501
        FlashInferConv1dDecodeKernel,
    )
    from vllm.model_executor.layers.linear_attention.conv1d.decode.triton import (
        TritonConv1dDecodeKernel,
    )
    from vllm.v1.attention.backends.mamba2_attn import Mamba2AttentionMetadata
    from vllm.v1.kv_cache_interface import MambaSpec

    set_random_seed(0)
    state_len, query_len = FI_WIDTH - 1 + num_spec, 1 + num_spec
    weight, bias, state = _fi_layer(state_len, dim_first)
    num_decodes = 5
    rows = 8 if padded else num_decodes
    indices = torch.full((rows, query_len), NULL_BLOCK_ID, dtype=torch.int32)
    indices[:num_decodes] = (
        torch.randperm(FI_SLOTS - 1)[: num_decodes * query_len] + 1
    ).view(num_decodes, query_len)
    accepted = torch.ones(rows, dtype=torch.int32)
    accepted[:num_decodes] = torch.randint(1, query_len + 1, (num_decodes,))
    metadata = Mamba2AttentionMetadata(
        num_prefills=0,
        num_prefill_tokens=0,
        num_decodes=num_decodes,
        num_decode_tokens=num_decodes * query_len,
        num_reqs=rows,
        has_initial_states_p=None,
        query_start_loc_p=None,
        state_indices_tensor_p=None,
        state_indices_tensor_d=indices.to(DEVICE),
        query_start_loc_d=torch.arange(
            0, rows * query_len + 1, query_len, dtype=torch.int32, device=DEVICE
        )
        if num_spec
        else None,
        num_accepted_tokens=accepted.to(DEVICE) if num_spec else None,
        seq_lens=torch.full((rows,), 100, dtype=torch.int32, device=DEVICE),
    )
    spec = MambaSpec(
        block_size=256,
        shapes=((state_len, FI_DIM), (8, 64, 128)),
        dtypes=(torch.bfloat16, torch.float32),
    )
    buffers = FlashInferConv1dDecodeKernel.allocate_buffers(
        VllmConfig(), spec, rows, torch.device(DEVICE)
    )
    fi_metadata = FlashInferConv1dDecodeKernel.prepare_metadata(
        buffers, metadata, rows, padded
    )
    x = _fi_rows(rows * query_len)

    triton_state, fi_state = state.clone(), state.clone()
    # Triton writes its output over its input.
    triton_out = TritonConv1dDecodeKernel.forward(
        x.clone(), triton_state, weight, bias, "silu", metadata, max_query_len=query_len
    )
    fi_out = FlashInferConv1dDecodeKernel.forward(
        x, fi_state, weight, bias, "silu", fi_metadata, max_query_len=query_len
    )

    live_rows = num_decodes * query_len
    torch.testing.assert_close(
        fi_out[:live_rows], triton_out[:live_rows], atol=2e-2, rtol=2e-2
    )
    live_slots = indices[:num_decodes, 0].long()
    torch.testing.assert_close(
        fi_state[live_slots], triton_state[live_slots], atol=0, rtol=0
    )
    assert torch.equal(fi_state[NULL_BLOCK_ID], state[NULL_BLOCK_ID])


@requires_flashinfer_conv1d
@create_new_process_for_each_test("spawn")
def test_registered_flashinfer_conv1d_prefill_warmup_leaves_nothing_to_compile():
    """FlashInfer compiles a conv prefill variant per batch shape class on
    first use. After the registry's warmup, prefills of one, a few and more
    than 32 sequences, short and long, with and without cached states, compile
    nothing: vLLM's JIT monitor, in error mode, raises on any CuTe DSL compile.
    FlashInfer's disk cache is off, so kernels compiled by earlier runs can't
    pass for warmed ones, and a fresh process keeps other tests' compiles out
    of FlashInfer's in-process cache."""
    import os

    from vllm.model_executor.layers.linear_attention.conv1d.prefill.flashinfer import (  # noqa: E501
        FlashInferConv1dPrefillKernel,
    )
    from vllm.model_executor.warmup.jit_warmup import JitWarmupRegistry
    from vllm.utils import jit_monitor

    os.environ["FLASHINFER_CUTE_DSL_DISABLE_CACHE"] = "1"
    set_random_seed(0)
    num_spec = 3
    weight, bias, state = _fi_layer(FI_WIDTH - 1 + num_spec, dim_first=False)
    registry = JitWarmupRegistry(None)
    with registry.activate():
        FlashInferConv1dPrefillKernel.register_warmup(
            dim=FI_DIM,
            state_len=FI_WIDTH - 1 + num_spec,
            dim_first=False,
            x_row_stride=_fi_rows(1).stride(0),
            weight_dtype=weight.dtype,
            bias_dtype=bias.dtype,
            max_num_seqs=64,
            max_num_batched_tokens=8192,
        )
    registry.warmup()

    jit_monitor.activate(mode="error")
    for lengths in ([1], [700], [3, 1, 40], [2000], [5] * 40, [128] * 40):
        slots = (torch.randperm(FI_SLOTS - 1)[: len(lengths)] + 1).tolist()
        has_initial_states = [i % 2 == 1 for i in range(len(lengths))]
        metadata, starts = _prefill_metadata(lengths, slots, has_initial_states)
        FlashInferConv1dPrefillKernel.forward(
            _fi_rows(sum(lengths)).T,
            state,
            weight,
            bias,
            "silu",
            FlashInferConv1dPrefillKernel.prepare_metadata(metadata, starts),
        )
    torch.accelerator.synchronize()
