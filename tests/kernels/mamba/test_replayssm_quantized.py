# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check quantized checkpoints against independent, exact arithmetic."""

import pytest
import torch


def _require_cuda():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    from vllm.platforms import current_platform

    if not current_platform.is_cuda():
        pytest.skip("requires the NVIDIA CUDA platform")


def _bytes(value):
    return value.contiguous().view(torch.uint8).cpu()


def _encode_rows(value, dtype):
    value = value.float().cpu()
    qmax = 127 if dtype == torch.int8 else 448
    amax = value.abs().amax(-1)
    multiplier = torch.where(amax == 0, 1.0, qmax / amax)
    scaled = value * multiplier[..., None]
    if dtype == torch.int8:
        scaled = scaled.round()
    return scaled.to(dtype), multiplier.reciprocal()


def _assert_rows(value, payload, scale):
    expected_payload, expected_scale = _encode_rows(value, payload.dtype)
    assert torch.equal(_bytes(payload), _bytes(expected_payload))
    torch.testing.assert_close(scale.cpu(), expected_scale, rtol=0, atol=0)


def _source(dtype):
    qmax = 127 if dtype == torch.int8 else 448
    value = torch.ones(3, 16, 64, 128)
    value[..., -1] = qmax
    factors = torch.exp2(torch.arange(16 * 64).reshape(16, 64) % 3 - 2)
    return value * factors[None, ..., None], factors


@pytest.mark.parametrize("dtype", [torch.int8, torch.float8_e4m3fn])
def test_shared_prefill_selected_slots_and_reuse(dtype):
    """Pack, gather, and replace prefill states without touching padding."""
    _require_cuda()
    from vllm.model_executor.layers.mamba.ops.replayssm_quantization import (
        dequantize_replayssm_state,
        quantize_replayssm_state,
    )

    source, _ = _source(dtype)
    source[0, 0, 0].zero_()
    source[2] *= 2
    source = source.cuda()
    # A larger outer stride reproduces padded cache pages.
    payload_pages = torch.full((6, 16, 64, 128), 7, dtype=dtype, device="cuda")
    scale_pages = torch.full((6, 16, 64), 7.0, device="cuda")
    payload, scale = payload_pages[::2], scale_pages[::2]
    ids = torch.tensor([2, 0, 1], dtype=torch.int32, device="cuda")
    quantize_replayssm_state(
        source,
        payload,
        scale,
        ids,
        source_indices=torch.tensor([2, -1, 0], device="cuda"),
    )
    _assert_rows(source[2], payload[2], scale[2])
    _assert_rows(source[0], payload[1], scale[1])
    assert torch.all(payload[0].float() == 7)
    assert torch.all(scale[0] == 7)
    decoded = dequantize_replayssm_state(
        payload,
        scale,
        torch.tensor([2, 1, 1, -1], dtype=torch.int32, device="cuda"),
        torch.tensor([True, True, False, True], device="cuda"),
    )
    assert decoded.dtype == torch.float32
    expected = torch.zeros_like(decoded)
    expected[0] = payload[2].float() * scale[2, ..., None]
    expected[1] = payload[1].float() * scale[1, ..., None]
    torch.testing.assert_close(decoded, expected, rtol=0, atol=0)
    torch.testing.assert_close(decoded[:2], source[[2, 0]], rtol=0, atol=0)
    preserved_payload, preserved_scale = _bytes(payload[1]), scale[1].clone()

    source.mul_(4)
    ids.copy_(torch.tensor([2, -1, -1], dtype=torch.int32, device="cuda"))
    quantize_replayssm_state(source, payload, scale, ids)
    _assert_rows(source[0], payload[2], scale[2])
    assert torch.equal(_bytes(payload[1]), preserved_payload)
    assert torch.equal(scale[1], preserved_scale)
    assert torch.all(payload_pages[1::2].float() == 7)
    assert torch.all(scale_pages[1::2] == 7)


@pytest.mark.parametrize("dtype", [torch.int8, torch.float8_e4m3fn])
def test_shared_graph_mixed_flush_checkpoint_bytes(dtype):
    """Two graph replays update only the flushing slot and preserve its scales."""
    _require_cuda()
    from flashinfer.mamba.checkpointing_ssu import checkpointing_ssu

    from vllm.model_executor.layers.mamba.ops.replayssm_quantization import (
        quantize_replayssm_state,
    )

    source, factors = _source(dtype)
    source[0].zero_()
    payload = torch.zeros_like(source, dtype=dtype, device="cuda")
    scale = torch.ones(3, 16, 64, device="cuda")
    quantize_replayssm_state(
        source.cuda(), payload, scale, torch.tensor([-1, 1, 2], device="cuda")
    )
    initial_payload, initial_scale = payload.clone(), scale.clone()
    x = (
        (factors / 8)
        .cuda()
        .to(torch.bfloat16)[None, None]
        .expand(3, 1, -1, -1)
        .contiguous()
    )
    b = torch.zeros(3, 1, 1, 128, device="cuda", dtype=torch.bfloat16)
    b[..., 0] = 1
    x_cache = x[:, 0, :, None].expand(-1, -1, 17, -1).contiguous()
    b_cache = b[:, 0, :, None].expand(-1, -1, 17, -1).contiguous()
    dt_cache = torch.ones(3, 16, 17, device="cuda")
    out = torch.empty_like(x)
    args = dict(
        state=payload,
        state_scale=scale,
        x_cache=x_cache,
        B_cache=b_cache,
        dt_cache=dt_cache,
        ring_start=torch.zeros(3, device="cuda", dtype=torch.int32),
        prev_num_accepted_tokens=torch.tensor(
            [0, 0, 16], device="cuda", dtype=torch.int32
        ),
        x=x,
        dt=torch.ones(3, 1, 16, 1, device="cuda", dtype=torch.bfloat16).expand(
            -1, -1, -1, 64
        ),
        A=torch.zeros(16, 1, 1, device="cuda").expand(-1, 64, 128),
        B=b,
        C=b,
        out=out,
        state_batch_indices=torch.tensor([2, 1, -1], device="cuda"),
        algorithm="monolith",
        d_split=1,
    )
    checkpointing_ssu(**args)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        checkpointing_ssu(**args)
    payload.copy_(initial_payload)
    scale.copy_(initial_scale)
    for iteration in range(1, 3):
        graph.replay()
        expected = source.clone()
        expected[2, ..., 0] += iteration * 2 * factors
        _assert_rows(expected, payload, scale)
        expected_out = torch.stack(
            ((1 + iteration * 2 + 0.125) * factors, 1.125 * factors)
        ).to(torch.bfloat16)
        assert torch.equal(out[:2, 0].cpu(), expected_out)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn, torch.int8])
def test_shared_bolt_t8_acceptance_and_ring_wrap(dtype):
    """Rejected candidates must not enter the next Bolt checkpoint or output."""
    _require_cuda()
    from flashinfer.mamba.checkpointing_ssu import checkpointing_ssu

    from vllm.model_executor.layers.mamba.ops.replayssm_quantization import (
        quantize_replayssm_state,
    )

    heads, groups, tokens, ring_len = 128, 8, 8, 24
    source = torch.zeros(4, heads, 64, 128)
    source[..., -1] = 127 if dtype == torch.int8 else 448
    source[0].zero_()
    state = source.to(device="cuda", dtype=dtype)
    scales = None
    if dtype != torch.bfloat16:
        scales = torch.ones(4, heads, 64, device="cuda")
        quantize_replayssm_state(
            source.cuda(), state, scales, torch.tensor([-1, 1, 2, 3], device="cuda")
        )
    x_ring = torch.full(
        (4, heads, ring_len, 64), 1 / 8, device="cuda", dtype=torch.bfloat16
    )
    b_ring = torch.zeros(4, groups, ring_len, 128, device="cuda", dtype=torch.bfloat16)
    b_ring[..., 0] = 1
    dt_ring = torch.ones(4, heads, ring_len, device="cuda")
    starts = torch.tensor([0, 21, 19, 23], device="cuda", dtype=torch.int32)
    history = torch.tensor([0, 8, 9, 16], device="cuda", dtype=torch.int32)
    slots = torch.tensor([3, 1, 2, -1], device="cuda", dtype=torch.int32)
    increments = torch.arange(1, tokens + 1).float() / 8
    x = (
        increments[None, :, None, None]
        .expand(4, tokens, heads, 64)
        .cuda()
        .bfloat16()
        .contiguous()
    )
    b = torch.zeros(4, tokens, groups, 128, device="cuda", dtype=torch.bfloat16)
    b[..., 0] = 1
    out = torch.empty_like(x)
    args = dict(
        state=state,
        state_scale=scales,
        x_cache=x_ring,
        B_cache=b_ring,
        dt_cache=dt_ring,
        ring_start=starts,
        prev_num_accepted_tokens=history,
        x=x,
        dt=torch.ones(4, tokens, heads, 1, device="cuda", dtype=torch.bfloat16).expand(
            -1, -1, -1, 64
        ),
        A=torch.zeros(heads, 1, 1, device="cuda").expand(-1, 64, 128),
        B=b,
        C=b,
        out=out,
        state_batch_indices=slots,
        algorithm="monolith",
        d_split=1,
    )
    saved = [value.clone() for value in (state, x_ring, b_ring, dt_ring)]
    saved_scale = None if scales is None else scales.clone()
    checkpointing_ssu(**args)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        checkpointing_ssu(**args)
    for value, initial in zip((state, x_ring, b_ring, dt_ring), saved, strict=True):
        value.copy_(initial)
    if scales is not None:
        scales.copy_(saved_scale)

    expected = source.clone()
    expected_ring = x_ring.cpu().clone()
    starts_cpu, history_cpu = starts.cpu(), history.cpu()
    for accepted in ([1, 7, 8], [8, 1, 7], [7, 8, 1]):
        expected_output = []
        for row, slot in enumerate((3, 1, 2)):
            start, count = int(starts_cpu[slot]), int(history_cpu[slot])
            positions = (start + torch.arange(count)) % ring_len
            old_sum = expected_ring[slot, 0, positions, 0].float().sum()
            current = expected[slot, 0, 0, 0] + old_sum
            expected_output.append((current + increments.cumsum(0)).bfloat16())
            if count + tokens > 16:
                expected[slot, ..., 0] += old_sum
                if scales is None:
                    expected[slot] = expected[slot].bfloat16().float()
                else:
                    packed, row_scale = _encode_rows(expected[slot], dtype)
                    expected[slot] = packed.float() * row_scale[..., None]
            written = (start + count + torch.arange(tokens)) % ring_len
            expected_ring[slot, :, written, :] = x[row].cpu().permute(1, 0, 2)
        graph.replay()
        for row, slot in enumerate((3, 1, 2)):
            if scales is None:
                torch.testing.assert_close(
                    state[slot].cpu(), expected[slot].bfloat16(), rtol=0, atol=0
                )
            else:
                _assert_rows(expected[slot], state[slot], scales[slot])
            torch.testing.assert_close(
                out[row].cpu(),
                expected_output[row][:, None, None].expand(-1, heads, 64),
                rtol=0,
                atol=0,
            )
            count = int(history_cpu[slot])
            flushed = count + tokens > 16
            if flushed:
                starts_cpu[slot] = (starts_cpu[slot] + count) % ring_len
            history_cpu[slot] = (0 if flushed else count) + accepted[row]
        torch.testing.assert_close(x_ring.cpu(), expected_ring, rtol=0, atol=0)
        assert torch.equal(_bytes(state[0]), _bytes(saved[0][0]))
        starts.copy_(starts_cpu)
        history.copy_(history_cpu)


def _separate_cache(fmt, groups, heads=16, max_seqlen=0):
    ring_len = 16 + max_seqlen
    event_shape = (2,) if max_seqlen else (1,)
    shapes = [
        (1,),
        (heads, 64, 128),
        (heads, ring_len, 64),
        (heads, ring_len),
        (groups, ring_len, 128),
    ]
    dtypes = [
        torch.bfloat16,
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.bfloat16,
    ]
    if fmt == "fp8":
        shapes += [(heads, 64, 128), (heads, 64), event_shape]
        dtypes += [torch.float8_e4m3fn, torch.float32, torch.int64]
    else:
        shapes += [
            (heads // 2, 128, 64),
            (heads // 2, 4, 256),
            (3, groups),
            (groups,),
            event_shape,
        ]
        dtypes += [torch.uint8, torch.uint8, torch.float32, torch.int32, torch.int64]
    # Keep outer padding in every tensor, as in the allocated cache pages.
    return tuple(
        torch.zeros(12, *shape, device="cuda", dtype=dtype)[::4]
        for shape, dtype in zip(shapes, dtypes, strict=True)
    )


def _encode_nvfp4(value):
    """Compute native packed bytes on the CPU, including the scale permutation."""
    value = value.float().cpu()
    global_scale = value.abs().amax() / 2688
    heads = value.shape[0]
    blocks = value.reshape(heads, 64, 8, 16)
    scales = (blocks.abs().amax(-1) / (6 * global_scale)).to(torch.float8_e4m3fn)
    normalized = blocks / (scales.float()[..., None] * global_scale)
    grid = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6])
    distances = (normalized.abs()[..., None] - grid).abs()
    codes = distances.argmin(-1)
    next_code = (codes + 1).clamp_max(7)
    tied = distances.gather(-1, codes[..., None]) == distances.gather(
        -1, next_code[..., None]
    )
    codes += (tied.squeeze(-1) & (codes % 2 == 1) & (codes < 7)).long()
    codes = codes.to(torch.uint8) | (torch.signbit(normalized).to(torch.uint8) << 3)
    codes = codes.reshape(heads, 64, 128)
    packed = (codes[..., ::2] | (codes[..., 1::2] << 4)).reshape(heads // 2, 128, 64)
    scales = scales.reshape(heads // 2, 4, 32, 2, 4).permute(0, 3, 2, 1, 4)
    return (
        packed,
        scales.contiguous().view(torch.uint8).reshape(heads // 2, 4, 256),
        global_scale,
    )


@pytest.mark.parametrize("fmt", ["fp8", "nvfp4"])
def test_separate_bolt_t8_commits_only_accepted_prefix(fmt):
    """A rejected crossing must preserve both checkpoints and the next output."""
    _require_cuda()
    if torch.cuda.get_device_capability() != (12, 1):
        pytest.skip("the pending-state integration requires SM121")
    from vllm.model_executor.layers.mamba.ops.replayssm_cutedsl import (
        initialize_replayssm_state,
        selective_state_update_replayssm,
    )

    heads, groups = 128, 8
    cache = _separate_cache(fmt, groups, heads=heads, max_seqlen=8)
    source = torch.full((3, heads, 64, 128), 0.6328125, device="cuda")
    if fmt == "fp8":
        source[..., -1] = 448
    else:
        source[..., 15::16] = 6
        source[:, -1, -1, -1] = 2688
    source[1] *= 2
    slots = torch.tensor([2, 1, 0], device="cuda", dtype=torch.int32)
    initialize_replayssm_state(cache, source, slots, state_dtype=fmt)
    increments = torch.tensor([1, 2, 3, 4, 5, 6, 7, 16]).float() / 32
    lengths = (8, 5, 3)
    values = torch.cat((increments, 2 * increments[:5], increments[:3]))
    x = (
        (values[:, None, None] / 32)
        .expand(-1, heads, 64)
        .cuda()
        .bfloat16()
        .contiguous()
    )
    dt = torch.full((16, heads), 32.0, device="cuda")
    b = torch.zeros(16, groups, 128, device="cuda", dtype=torch.bfloat16)
    b[..., 0] = 1
    out = torch.empty_like(x)
    accepted = torch.zeros(3, device="cuda", dtype=torch.int32)
    args = dict(
        state_dtype=fmt,
        num_accepted_tokens=accepted,
        cu_seqlens=torch.tensor([0, 8, 13, 16], device="cuda", dtype=torch.int32),
        max_seqlen=8,
    )
    a, d, bias = (torch.zeros(heads, device="cuda") for _ in range(3))

    def decode():
        selective_state_update_replayssm(
            cache, x, dt, a, b, b, d, bias, slots, out, **args
        )

    saved = [tensor.clone() for tensor in cache]
    decode()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        decode()
    for tensor, initial in zip(cache, saved, strict=True):
        tensor.copy_(initial)

    anchors = source[:2].cpu().clone()
    histories: list[list[float]] = [[], []]
    totals = [0, 0]
    pending: list[list[float]] = [[], []]
    for counts in ((0, 0), (8, 1), (7, 5), (1, 5), (7, 4), (8, 1), (1, 5)):
        accepted.copy_(torch.tensor([*counts, 0], device="cuda", dtype=torch.int32))
        graph.replay()
        offset = 0
        for row, slot in enumerate((2, 1)):
            additions = pending[row][: counts[row]]
            histories[row].extend(additions)
            totals[row] += len(additions)
            if len(histories[row]) >= 16:
                anchors[row, ..., 0] += sum(histories[row][:16])
                histories[row] = histories[row][16:]
            torch.testing.assert_close(
                cache[1][slot].cpu(), anchors[row].half(), rtol=0, atol=0
            )
            if fmt == "fp8":
                encoded, scale = _encode_rows(anchors[row], torch.float8_e4m3fn)
                assert torch.equal(_bytes(cache[5][slot]), _bytes(encoded))
                torch.testing.assert_close(cache[6][slot].cpu(), scale, rtol=0, atol=0)
                read_value = float(encoded[0, 0, 0]) * float(scale[0, 0])
            else:
                packed, scales, global_scale = _encode_nvfp4(anchors[row])
                assert torch.equal(_bytes(cache[5][slot]), _bytes(packed))
                assert torch.equal(_bytes(cache[6][slot]), _bytes(scales))
                grid = (0, 0.5, 1, 1.5, 2, 3, 4, 6)
                code = int(packed[0, 0, 0]) & 15
                read_value = grid[code & 7] * (-1 if code & 8 else 1)
                read_value *= float(scales.view(torch.float8_e4m3fn)[0, 0, 0]) * float(
                    global_scale
                )
                assert torch.all(cache[7][slot, 2].cpu() == global_scale)
                group_max = anchors[row].abs().reshape(groups, -1).amax(-1)
                torch.testing.assert_close(
                    cache[7][slot, :2].cpu(), group_max.expand(2, -1), rtol=0, atol=0
                )
                assert torch.all(cache[8][slot] == (totals[row] // 16) % 2)
            current = values[offset : offset + lengths[row]]
            expected = (read_value + sum(histories[row]) + current.cumsum(0)).bfloat16()
            torch.testing.assert_close(
                out[offset : offset + lengths[row]].cpu(),
                expected[:, None, None].expand(-1, heads, 64),
                rtol=0,
                atol=0,
            )
            assert cache[-1][slot].tolist() == [totals[row], lengths[row]]
            torch.testing.assert_close(
                cache[2][slot, :, 16 : 16 + lengths[row]],
                x[offset : offset + lengths[row]].permute(1, 0, 2),
                rtol=0,
                atol=0,
            )
            pending[row] = current.tolist()
            offset += lengths[row]
        assert torch.count_nonzero(out[13:]) == 0
        for tensor in cache:
            assert torch.count_nonzero(_bytes(tensor[0])) == 0

    # Reuse a slot while it still holds unconfirmed candidates.
    initialize_replayssm_state(cache, source[:1], slots[:1], state_dtype=fmt)
    assert cache[-1][2].tolist() == [0, 0]
    for ring in cache[2:5]:
        assert torch.count_nonzero(ring[2]) == 0
    accepted.fill_(8)
    graph.replay()
    torch.testing.assert_close(
        cache[1][2].cpu(), source[0].cpu().half(), rtol=0, atol=0
    )
    assert cache[-1][2].tolist() == [0, 8]


@pytest.mark.parametrize(
    "fmt,tokens,heads,groups",
    [
        ("fp8", 1, 16, 2),
        ("nvfp4", 1, 16, 2),
        ("fp8", 8, 128, 8),
        ("nvfp4", 8, 128, 8),
    ],
)
def test_separate_pending_general_arithmetic(fmt, tokens, heads, groups):
    """Check the MMA axes and decayed readout against a CPU FP32 recurrence."""
    _require_cuda()
    if torch.cuda.get_device_capability() != (12, 1):
        pytest.skip("the pending-state integration requires SM121")
    from vllm.model_executor.layers.mamba.ops.replayssm_cutedsl import (
        initialize_replayssm_state,
        selective_state_update_replayssm,
    )

    generator = torch.Generator().manual_seed(391)

    def values(*shape, scale=1.0):
        return (torch.randn(*shape, generator=generator) * scale).bfloat16()

    cache = _separate_cache(fmt, groups, heads=heads, max_seqlen=tokens)
    source = values(1, heads, 64, 128, scale=0.05).float()
    slots = torch.tensor([1], device="cuda", dtype=torch.int32)
    initialize_replayssm_state(cache, source.cuda(), slots, state_dtype=fmt)
    x = values(16 + tokens, heads, 64, scale=0.05)
    b = values(16 + tokens, groups, 128, scale=0.05)
    c = values(tokens, groups, 128, scale=0.1)
    raw_dt = values(16 + tokens, heads, scale=0.7)
    a = -torch.linspace(0.1, 0.4, heads)
    bias = torch.linspace(-0.2, 0.2, heads).bfloat16()
    d = torch.linspace(-0.3, 0.3, heads).bfloat16()
    dt = torch.nn.functional.softplus(raw_dt.float() + bias.float())
    # Feature splits preserve their parent token stride in the model.
    features = torch.cat((x.flatten(1), b.flatten(1)), dim=1).cuda()
    x_gpu = features[:, : heads * 64].view(-1, heads, 64)
    b_gpu = features[:, heads * 64 :].view(-1, groups, 128)
    c_gpu = torch.cat((c.flatten(1), c.flatten(1)), dim=1).cuda()
    c_gpu = c_gpu[:, : groups * 128].view(tokens, groups, 128)
    dt_gpu = torch.cat((raw_dt, raw_dt), dim=1).cuda()[:, :heads]
    dt_gpu = dt_gpu[..., None].expand(-1, -1, 64)
    accepted = torch.zeros(1, device="cuda", dtype=torch.int32)
    out = torch.empty(tokens, heads, 64, device="cuda", dtype=torch.bfloat16)
    cu = torch.tensor([0, tokens], device="cuda", dtype=torch.int32)
    for start in range(0, 17, tokens):
        selective_state_update_replayssm(
            cache,
            x_gpu[start : start + tokens],
            dt_gpu[start : start + tokens],
            a.cuda(),
            b_gpu[start : start + tokens],
            c_gpu,
            d.cuda(),
            bias.cuda(),
            slots,
            out,
            state_dtype=fmt,
            num_accepted_tokens=accepted,
            cu_seqlens=cu,
            max_seqlen=tokens,
        )
        accepted.fill_(tokens)

    # The flush rounds only the scaled B operand to BF16 before FP32 MMA.
    b_heads = b.float().repeat_interleave(heads // groups, dim=1)
    cumulative = (dt[:16] * a).cumsum(0)
    unrounded_weights = (
        b_heads[:16] * dt[:16, :, None] * (cumulative[-1] - cumulative).exp()[..., None]
    )
    weights = unrounded_weights.bfloat16()
    refreshed = source[0] * cumulative[-1].exp()[:, None, None]
    refreshed += torch.einsum("ijk,ijl->jkl", x[:16].float(), weights.float())
    anchor = cache[1][1].float().cpu()
    # Sixteen FP32 decay steps and fast transcendentals can cross a BF16 midpoint.
    # A 32-epsilon relative budget bounds that input error before BF16 rounding.
    # Propagate only possible operand changes, rather than widening every result.
    weight_error = unrounded_weights.abs() * (32 * torch.finfo(torch.float32).eps)
    weight_low = (unrounded_weights - weight_error).bfloat16().float()
    weight_high = (unrounded_weights + weight_error).bfloat16().float()
    weight_delta = torch.maximum(
        (weight_low - weights.float()).abs(), (weight_high - weights.float()).abs()
    )
    allowance = 2e-6 + refreshed.abs() * 2e-5
    allowance += torch.einsum("ijk,ijl->jkl", x[:16].float().abs(), weight_delta)
    # Apply the state uncertainty before the independent FP16 rounding step.
    low_state, high_state = refreshed - allowance, refreshed + allowance
    lower, upper = low_state.half(), high_state.half()
    lower = torch.where(
        lower.float() > low_state,
        torch.nextafter(lower, torch.full_like(lower, -torch.inf)),
        lower,
    ).float()
    upper = torch.where(
        upper.float() < high_state,
        torch.nextafter(upper, torch.full_like(upper, torch.inf)),
        upper,
    ).float()
    assert torch.all(anchor >= lower)
    assert torch.all(anchor <= upper)
    if fmt == "fp8":
        _, expected_scale = _encode_rows(refreshed, torch.float8_e4m3fn)
        assert torch.all(
            (cache[6][1].cpu() - expected_scale).abs()
            <= allowance.amax(-1) / 448 + 1e-8
        )
        scale = cache[6][1].cpu()[..., None]
        payload = cache[5][1].float().cpu()
        # E4M3 conversion is monotone across this independently bounded interval.
        payload_low = (low_state / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
        payload_high = (high_state / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
        assert torch.all(payload >= payload_low.float())
        assert torch.all(payload <= payload_high.float())
        decoded = payload * scale
    else:
        # The first flush uses the previous checkpoint's global maximum.
        global_scale = source.abs().amax() / 2688
        torch.testing.assert_close(
            cache[7][1, 2].cpu(), global_scale.expand(groups), rtol=1e-6, atol=0
        )
        row = torch.arange(heads)[:, None] % 2 * 64 + torch.arange(64)[None, :]
        block = torch.arange(8)
        offset = (
            block // 4 * 512
            + row[..., None] % 32 * 16
            + row[..., None] // 32 * 4
            + block % 4
        )
        scale_bytes = cache[6][1].cpu().view(torch.float8_e4m3fn)
        scales = scale_bytes.reshape(heads // 2, 1024).float()[
            torch.arange(heads)[:, None, None] // 2, offset
        ]
        expected_scales = (
            refreshed.reshape(heads, 64, 8, 16).abs().amax(-1) / (6 * global_scale)
        ).to(torch.float8_e4m3fn)
        torch.testing.assert_close(
            scales, expected_scales.float(), rtol=0.126, atol=0.002
        )
        packed = cache[5][1].cpu().reshape(heads, 64, 64)
        codes = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2).long()
        grid = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6])
        effective_scale = (scales * global_scale).repeat_interleave(16, dim=-1)
        decoded = grid[codes & 7] * torch.where(codes & 8 != 0, -1, 1)
        decoded *= effective_scale
        # Given each stored scale, the code must minimize the distance to FP32.
        candidates = grid[None, None, None, :] * effective_scale[..., None]
        nearest_error = (refreshed.abs()[..., None] - candidates).abs().amin(-1)
        assert torch.all((decoded - refreshed).abs() <= nearest_error + allowance)
    state = decoded.clone()
    c_heads = c.float().repeat_interleave(heads // groups, dim=1)
    expected = []
    for index in range(tokens):
        state *= (a * dt[16 + index]).exp()[:, None, None]
        state += (
            dt[16 + index, :, None, None]
            * x[16 + index].float()[..., None]
            * b_heads[16 + index, :, None, :]
        )
        expected.append(
            torch.einsum("hdn,hn->hd", state, c_heads[index])
            + d.float()[:, None] * x[16 + index].float()
        )
    # One BF16 output ulp allows the independent FP32 reduction order.
    torch.testing.assert_close(
        out.float().cpu(), torch.stack(expected), rtol=0.008, atol=2e-6
    )
    assert cache[-1][1].tolist() == [16, tokens]


@pytest.mark.parametrize("fmt", ["fp8", "nvfp4"])
def test_separate_intermediate_single_token_prefill(fmt, tmp_path):
    """A one-token prompt chunk must update the state used by later prefill."""
    _require_cuda()
    if torch.cuda.get_device_capability() != (12, 1):
        pytest.skip("the pending-state integration requires SM121")
    from transformers import NemotronHConfig

    from tests.v1.attention.utils import BatchSpec, create_common_attn_metadata
    from vllm.config import CacheConfig, ModelConfig, SchedulerConfig, VllmConfig
    from vllm.config.mamba import MambaBackendEnum, MambaConfig
    from vllm.model_executor.layers.mamba.ops.replayssm_cutedsl import (
        initialize_replayssm_state,
        selective_state_update_replayssm,
    )
    from vllm.model_executor.layers.mamba.ops.ssd_combined import (
        mamba_chunk_scan_combined_varlen,
    )
    from vllm.v1.attention.backends.mamba2_attn import Mamba2AttentionMetadataBuilder
    from vllm.v1.kv_cache_interface import MambaSpec

    heads, groups = 16, 2
    NemotronHConfig(
        architectures=["NemotronHForCausalLM"],
        hidden_size=512,
        expand=2,
        mamba_num_heads=heads,
        mamba_head_dim=64,
        n_groups=groups,
        ssm_state_size=128,
        chunk_size=16,
        layers_block_type=["mamba"],
        vocab_size=32,
    ).save_pretrained(tmp_path)
    config = VllmConfig(
        model_config=ModelConfig(
            model=str(tmp_path),
            skip_tokenizer_init=True,
            dtype="bfloat16",
            max_model_len=64,
        ),
        scheduler_config=SchedulerConfig(
            max_model_len=64,
            max_num_seqs=1,
            max_num_batched_tokens=64,
            is_encoder_decoder=False,
        ),
        cache_config=CacheConfig(
            use_replayssm=True,
            replayssm_checkpoint_mode="separate",
            replayssm_state_dtype=fmt,
            enable_prefix_caching=False,
            block_size=16,
        ),
        mamba_config=MambaConfig(backend=MambaBackendEnum.FLASHINFER),
    )
    cache = _separate_cache(fmt, groups, heads=heads, max_seqlen=8)
    spec = MambaSpec(
        block_size=16,
        shapes=tuple(tuple(t.shape[1:]) for t in cache),
        dtypes=tuple(t.dtype for t in cache),
        mamba_cache_mode="none",
    )
    device = torch.device("cuda")
    builder = Mamba2AttentionMetadataBuilder(spec, ["mamba"], config, device)
    source = torch.full((1, heads, 64, 128), 0.625, device=device)
    if fmt == "fp8":
        source[..., -1] = 448
    else:
        source[..., 15::16] = 6
        source[:, -1, -1, -1] = 2688
    slots = torch.tensor([1], device=device, dtype=torch.int32)
    initialize_replayssm_state(cache, source, slots, state_dtype=fmt)
    increments = torch.arange(1, 9, device=device).float() / 8
    x = (increments[:, None, None] / 32).expand(8, heads, 64).bfloat16().contiguous()
    dt = torch.full((8, heads), 32.0, device=device)
    b = torch.zeros(8, groups, 128, device=device, dtype=torch.bfloat16)
    b[..., 0] = 1
    a, d, bias = (torch.zeros(heads, device=device) for _ in range(3))
    output = torch.empty_like(x)
    reference = source[0].clone()
    start = 0
    classifications = []
    for count in (4, 1, 3):
        end = start + count
        # The initialized checkpoint represents an already computed prefix.
        common = create_common_attn_metadata(
            BatchSpec(seq_lens=[end + 2], query_lens=[count]), 16, device
        )
        common = common.replace(
            is_prefilling=torch.tensor([True]),
            block_table_tensor=torch.ones_like(common.block_table_tensor),
        )
        metadata = builder.build(0, common)
        classifications.append((metadata.num_prefills, metadata.num_decodes))
        if metadata.num_prefills:
            final_state = mamba_chunk_scan_combined_varlen(
                x[start:end],
                dt[start:end],
                a,
                b[start:end],
                b[start:end],
                chunk_size=metadata.chunk_size,
                cu_seqlens=metadata.query_start_loc_p,
                cu_chunk_seqlens=metadata.cu_chunk_seqlen_p,
                last_chunk_indices=metadata.last_chunk_indices_p,
                seq_idx=metadata.seq_idx_p,
                out=output[start:end],
                initial_states=cache[1][1:2].float(),
                dt_softplus=True,
                state_dtype=torch.float32,
            )
            initialize_replayssm_state(cache, final_state, slots, state_dtype=fmt)
        else:
            # The pre-fix classification takes this actual pending-decode path.
            selective_state_update_replayssm(
                cache,
                x[start:end],
                dt[start:end],
                a,
                b[start:end],
                b[start:end],
                d,
                bias,
                slots,
                output[start:end],
                state_dtype=fmt,
                num_accepted_tokens=torch.ones(1, device=device, dtype=torch.int32),
                cu_seqlens=torch.tensor([0, count], device=device, dtype=torch.int32),
                max_seqlen=8,
            )
        reference[..., 0] += increments[start:end].sum()
        torch.testing.assert_close(cache[1][1], reference.half(), rtol=0, atol=0)
        if fmt == "fp8":
            _assert_rows(reference, cache[5][1], cache[6][1])
        else:
            packed, scales, global_scale = _encode_nvfp4(reference)
            assert torch.equal(_bytes(cache[5][1]), _bytes(packed))
            assert torch.equal(_bytes(cache[6][1]), _bytes(scales))
            assert torch.all(cache[7][1, 2].cpu() == global_scale)
        assert cache[-1][1].tolist() == [0, 0]
        start = end
    expected = (0.625 + increments.cumsum(0))[:, None, None].expand_as(output)
    torch.testing.assert_close(output.float(), expected, rtol=0, atol=0)
    assert classifications == [(1, 0), (1, 0), (1, 0)]


def test_separate_sm121_cute_stochastic_rounding_thresholds():
    """Check the CuTe helper against probabilities from independent FP64 intervals."""
    _require_cuda()
    if torch.cuda.get_device_capability() != (12, 1):
        pytest.skip("the software stochastic-rounding path requires SM121")
    from cutlass import cute
    from cutlass.cute.runtime import from_dlpack

    from vllm.third_party.replayssm.mamba2.pending import _sr_half

    # Enumerate every finite, nonnegative FP16 value in increasing order.
    levels = torch.arange(0x7C00, dtype=torch.int32).to(torch.int16)
    levels = levels.view(torch.float16).double()
    midpoints = ((levels[:-1] + levels[1:]) / 2).float()
    positive = torch.cat(
        (
            levels.float(),
            midpoints,
            torch.nextafter(midpoints, torch.full_like(midpoints, -torch.inf)),
            torch.nextafter(midpoints, torch.full_like(midpoints, torch.inf)),
            torch.tensor([2.0**-149, 2.0**-126, 65520, 1e10, torch.inf]),
        )
    )
    values = torch.cat((positive, -positive))
    random = torch.tensor([0, 4095, 4096, 4097, 8191], dtype=torch.int32)
    values = values[:, None].expand(-1, random.numel()).contiguous().flatten()
    random = random.repeat(positive.numel() * 2)
    # High random bits must have no effect on the 13-bit decision.
    random |= 0x12340000
    magnitude = values.double().abs().clamp_max(65504)
    index = torch.searchsorted(levels, magnitude, right=True) - 1
    low = levels[index]
    high = levels[(index + 1).clamp_max(levels.numel() - 1)]
    probability = torch.where(high == low, 0.0, (magnitude - low) / (high - low))
    expected = torch.where(probability > (random & 8191).double() / 8192, high, low)
    expected = torch.copysign(expected, values.double()).half()

    @cute.kernel
    def kernel(x: cute.Tensor, r: cute.Tensor, y: cute.Tensor):
        index = cute.arch.block_idx()[0] * 128 + cute.arch.thread_idx()[0]
        if index < cute.size(x):
            y[index] = _sr_half(x[index], r[index])

    @cute.jit
    def launch(x: cute.Tensor, r: cute.Tensor, y: cute.Tensor):
        kernel(x, r, y).launch(
            grid=((cute.size(x) + 127) // 128, 1, 1), block=(128, 1, 1)
        )

    x_gpu, r_gpu = values.cuda(), random.cuda()
    out = torch.empty_like(x_gpu, dtype=torch.float16)
    arguments = tuple(from_dlpack(tensor) for tensor in (x_gpu, r_gpu, out))
    compiled = cute.compile(launch, *arguments)
    compiled(*arguments)
    # Byte equality also verifies signed zero and the strict midpoint decision.
    assert torch.equal(_bytes(out), _bytes(expected))


def test_separate_sm121_initializer_stochastic_rounding_intervals():
    """Software SR preserves exact values and chooses adjacent finite FP16 values."""
    _require_cuda()
    if torch.cuda.get_device_capability() != (12, 1):
        pytest.skip("the software FP16 stochastic rounding path requires SM121")
    from vllm.model_executor.layers.mamba.ops.replayssm_cutedsl import (
        initialize_replayssm_state,
    )

    values = torch.tensor(
        [1 + 2**-11, -1 - 2**-11, 0, 2**-25, 65520, -65520, 1, 2**-24, 65504, -65504]
    )
    indices = torch.arange(16 * 64 * 128) % len(values)
    source = values[indices].reshape(1, 16, 64, 128).cuda()
    cache = _separate_cache("fp8", groups=1, max_seqlen=1)
    initialize_replayssm_state(
        cache,
        source,
        torch.tensor([1], device="cuda", dtype=torch.int32),
        state_dtype="fp8",
        seed=42,
    )
    rounded = cache[1][1].flatten().float().cpu()
    for case, lower, upper in (
        (0, 1, 1 + 2**-10),
        (1, -1 - 2**-10, -1),
        (3, 0, 2**-24),
    ):
        samples = rounded[indices == case]
        assert torch.all((samples == lower) | (samples == upper))
        fraction_upper = (samples == upper).float().mean().item()
        assert 0.35 < fraction_upper < 0.65
    for case, expected in (
        (2, 0),
        (4, 65504),
        (5, -65504),
        (6, 1),
        (7, 2**-24),
        (8, 65504),
        (9, -65504),
    ):
        assert torch.all(rounded[indices == case] == expected)
    assert torch.isfinite(rounded).all()


@pytest.mark.parametrize("fmt", ["fp8", "nvfp4"])
@pytest.mark.parametrize("heads_per_group", [8, 16])
def test_separate_graph_prefill_reuse_and_mixed_flush(fmt, heads_per_group):
    """Check both copies after reuse and two flushes at different slot positions."""
    _require_cuda()
    if torch.cuda.get_device_capability() != (10, 3):
        pytest.skip("the native CuTe ReplaySSM kernels require SM103")
    from vllm.model_executor.layers.mamba.ops.replayssm_cutedsl import (
        initialize_replayssm_state,
        selective_state_update_replayssm,
    )

    groups = 16 // heads_per_group
    cache = _separate_cache(fmt, groups)
    source = torch.full((3, 16, 64, 128), 0.5, device="cuda")
    if fmt == "fp8":
        source[..., -1] = 448
    else:
        source[..., 15::16] = 6
        source[:, -1, -1, -1] = 2688
    source[1] *= 2
    ids = torch.tensor([2, 1, -1], dtype=torch.int32, device="cuda")
    initialize_replayssm_state(
        cache, source[:1].expand_as(source), ids, state_dtype=fmt
    )
    for tensor in cache[1:]:
        assert torch.equal(_bytes(tensor[1]), _bytes(tensor[2]))
    torch.testing.assert_close(cache[1][1], source[0].half(), rtol=0, atol=0)
    assert torch.count_nonzero(cache[-1]) == 0
    if fmt == "fp8":
        _assert_rows(source[0], cache[5][1], cache[6][1])
    else:
        packed, scales, global_scale = _encode_nvfp4(source[0])
        assert torch.equal(_bytes(cache[5][1]), _bytes(packed))
        assert torch.equal(_bytes(cache[6][1]), _bytes(scales))
        group_max = source[0].abs().reshape(groups, -1).amax(-1)
        torch.testing.assert_close(
            cache[7][1, :2], group_max.expand(2, -1), rtol=0, atol=0
        )
        assert torch.all(cache[7][1, 2].cpu() == global_scale)
        assert torch.count_nonzero(cache[8]) == 0
    initialize_replayssm_state(cache, source, ids, state_dtype=fmt)
    x = torch.full((3, 16, 64), 1 / 1024, dtype=torch.bfloat16, device="cuda")
    x[1] *= 2
    dt = torch.full((3, 16), 32.0, device="cuda")
    a = torch.zeros(16, device="cuda")
    b = torch.zeros(3, groups, 128, dtype=torch.bfloat16, device="cuda")
    b[..., 0] = 1
    d, bias = torch.zeros(16, device="cuda"), torch.zeros(16, device="cuda")
    out = torch.empty_like(x)

    def decode():
        selective_state_update_replayssm(
            cache, x, dt, a, b, b, d, bias, ids, out, state_dtype=fmt
        )

    # Advance both slots, then reuse slot 1 with a different prefill state.
    for _ in range(7):
        decode()
    source[1] *= 2
    initialize_replayssm_state(
        cache,
        source,
        torch.tensor([-1, 1, -1], dtype=torch.int32, device="cuda"),
        state_dtype=fmt,
    )
    x[1] *= 2
    for ring in cache[2:5]:
        assert torch.count_nonzero(ring[1]) == 0
    assert cache[-1][1].item() == 0
    assert cache[-1][2].item() == 7

    # Precompute the independent checkpoints. Every FP16 update is exact.
    checkpoints = {}
    for row, factor in ((0, 1), (1, 4)):
        for flushes in range(3):
            expected = source[row].cpu().clone()
            expected[..., 0] += flushes * factor / 2
            encoding = (
                _encode_rows(expected, torch.float8_e4m3fn)
                if fmt == "fp8"
                else _encode_nvfp4(expected)
            )
            checkpoints[row, flushes] = expected, encoding

    def check(total0, total1):
        for row, slot, total in ((0, 2, total0), (1, 1, total1)):
            expected, encoding = checkpoints[row, total // 16]
            torch.testing.assert_close(
                cache[1][slot].cpu(), expected.half(), rtol=0, atol=0
            )
            assert torch.equal(_bytes(cache[5][slot]), _bytes(encoding[0]))
            if fmt == "fp8":
                torch.testing.assert_close(
                    cache[6][slot].cpu(), encoding[1], rtol=0, atol=0
                )
            else:
                assert torch.equal(_bytes(cache[6][slot]), _bytes(encoding[1]))
                assert torch.all(cache[7][slot, 2].cpu() == encoding[2])
                group_max = expected.abs().reshape(groups, -1).amax(-1)
                torch.testing.assert_close(
                    cache[7][slot, :2].cpu(),
                    group_max.expand(2, -1),
                    rtol=0,
                    atol=0,
                )
                assert torch.all(cache[8][slot] == (total // 16) % 2)
            assert cache[-1][slot].item() == total
        for tensor in cache:
            assert torch.count_nonzero(_bytes(tensor[0])) == 0

    check(7, 0)
    saved = [tensor.clone() for tensor in cache]
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        decode()
    for tensor, original in zip(cache, saved, strict=True):
        tensor.copy_(original)
    for step in range(1, 33):
        graph.replay()
        check(7 + step, step)
        expected_out = torch.tensor(
            [0.5 + (7 + step) / 32, 4 * (0.5 + step / 32), 0],
            dtype=torch.bfloat16,
        )[:, None, None].expand(3, 16, 64)
        assert torch.equal(out.cpu(), expected_out)
