# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from tests.lora.utils import (
    PunicaTensors,
    assert_close,
    generate_data,
    generate_data_for_expand_nslices,
)
from vllm.lora.ops.xpu_ops import (
    bgmv_expand,
    bgmv_expand_slice,
    bgmv_shrink,
    lora_linear,
)
from vllm.platforms import current_platform


def torch_bgmv_expand(
    inputs: torch.Tensor,
    lora_b_weights: torch.Tensor,
    output_tensor: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    add_inputs: bool = True,
):
    selected_loras = lora_b_weights[lora_indices_tensor].to(dtype=output_tensor.dtype)
    if len(selected_loras.shape) == 4:
        selected_loras = selected_loras.squeeze(dim=1)
    inputs = inputs.to(dtype=output_tensor.dtype)
    outputs = torch.einsum("bi, boi -> bo", inputs, selected_loras)

    limit = output_tensor.shape[0]
    if outputs.shape[0] == 1 and output_tensor.shape[0] != 1:
        limit = 1

    # LoRA adapter and model may add different amounts of padding to output
    common_len = min(outputs.shape[1], output_tensor.shape[1])

    if add_inputs:
        output_tensor[:, :common_len] += outputs[:limit, :common_len]
    else:
        output_tensor[:, :common_len] = outputs[:limit, :common_len]


def torch_bgmv_shrink(
    inputs: torch.Tensor,
    lora_b_weights: torch.Tensor,
    output_tensor: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    scaling: float = 1.0,
):
    selected_loras = lora_b_weights[lora_indices_tensor].to(dtype=output_tensor.dtype)
    if len(selected_loras.shape) == 4:
        selected_loras = selected_loras.squeeze(dim=1)
    inputs = inputs.to(dtype=output_tensor.dtype)
    outputs = torch.einsum("bi, boi -> bo", inputs, selected_loras)

    output_tensor[:, : outputs.shape[1]] = scaling * outputs[:]


def torch_bgmv_expand_slice(
    inputs: torch.Tensor,
    lora_b_weights: torch.Tensor,
    output_tensor: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    slice_offset: int,
    slice_size: int,
    add_inputs: bool = True,
):
    selected_loras = lora_b_weights[lora_indices_tensor].to(dtype=output_tensor.dtype)
    inputs = inputs.to(dtype=output_tensor.dtype)
    if len(selected_loras.shape) == 4:
        selected_loras = selected_loras.squeeze(dim=1)
    outputs = torch.einsum("bi, boi -> bo", inputs, selected_loras)

    if add_inputs:
        output_tensor[:, slice_offset : slice_offset + slice_size] += outputs[:]
    else:
        output_tensor[:, slice_offset : slice_offset + slice_size] = outputs[:]


def check_bgmv_shrink(
    batches: int,
    num_loras: int,
    rank: int,
    hidden_size: int,
    dtype: torch.dtype,
    device: str,
    scaling: float,
):
    """Compare vllm.bgmv_shrink against a reference implementation."""
    seq_length = 1
    data: PunicaTensors = generate_data(
        batches,
        hidden_size,
        num_loras,
        rank,
        seq_length,
        dtype,
        "shrink",
        device,
    )

    bgmv_shrink(
        data.inputs_tensor,
        data.lora_weights,
        data.our_out_tensor,
        data.token_lora_mapping,
        scaling,
    )

    torch_bgmv_shrink(
        data.inputs_tensor,
        data.lora_weights,
        data.ref_out_tensor,
        data.token_lora_mapping,
        scaling,
    )

    data.ref_out_tensor = data.ref_out_tensor.to(torch.float32)
    assert_close(data.our_out_tensor, data.ref_out_tensor)


def check_bgmv_expand(
    batches: int,
    num_loras: int,
    rank: int,
    hidden_size: int,
    dtype: torch.dtype,
    device: str,
    add_inputs: bool,
):
    """Compare vllm.bgmv_expand against a reference implementation."""
    seq_length = 1
    data: PunicaTensors = generate_data(
        batches,
        hidden_size,
        num_loras,
        rank,
        seq_length,
        dtype,
        "expand",
        device,
    )

    bgmv_expand(
        data.inputs_tensor,
        data.lora_weights,
        data.our_out_tensor,
        data.token_lora_mapping,
        add_inputs=add_inputs,
    )
    torch_bgmv_expand(
        data.inputs_tensor,
        data.lora_weights,
        data.ref_out_tensor,
        data.token_lora_mapping,
        add_inputs=add_inputs,
    )
    assert_close(data.ref_out_tensor, data.our_out_tensor)


def check_bgmv_expand_slice(
    batches: int,
    num_loras: int,
    rank: int,
    hidden_size: int,
    nslices: int,
    dtype: torch.dtype,
    device: str,
    add_inputs: bool,
):
    """Compare vllm.bgmv_expand_slice against a reference implementation."""
    seq_length = 1
    data: PunicaTensors = generate_data_for_expand_nslices(
        batches,
        hidden_size,
        num_loras,
        rank,
        seq_length,
        dtype,
        nslices,
        device,
    )

    slice_offset = 0
    for index in range(nslices):
        bgmv_expand_slice(
            data.inputs_tensor,
            data.lora_weights[index],
            data.our_out_tensor,
            data.token_lora_mapping,
            slice_offset,
            slice_size=hidden_size,
            add_inputs=add_inputs,
        )
        torch_bgmv_expand_slice(
            data.inputs_tensor,
            data.lora_weights[index],
            data.ref_out_tensor,
            data.token_lora_mapping,
            slice_offset,
            slice_size=hidden_size,
            add_inputs=add_inputs,
        )

        slice_offset += hidden_size
    assert_close(data.ref_out_tensor, data.our_out_tensor)


# General tests params that tests for variations in all dimensions
# except hidden_size.
test_params = {
    "hidden_sizes": [2049],
    "batches": [4],
    "num_loras": [4],
    "max_ranks": [32],
}

DTYPES = [torch.float16, torch.bfloat16]
DEVICES = [f"xpu:{0}"]
SEED = [0]


@pytest.mark.parametrize("batches", test_params["batches"])
@pytest.mark.parametrize("num_loras", test_params["num_loras"])
@pytest.mark.parametrize("rank", test_params["max_ranks"])
@pytest.mark.parametrize("hidden_size", test_params["hidden_sizes"])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("seed", SEED)
@pytest.mark.parametrize("op_type", ["shrink", "expand"])
@pytest.mark.skipif(not current_platform.is_xpu(), reason="skip for non xpu platform")
def test_bgmv(
    batches: int,
    num_loras: int,
    rank: int,
    hidden_size: int,
    dtype: torch.dtype,
    device: str,
    seed: int,
    op_type: str,
):
    if op_type == "shrink":
        check_bgmv_shrink(
            batches=batches,
            num_loras=num_loras,
            rank=rank,
            hidden_size=hidden_size,
            dtype=dtype,
            device=device,
            scaling=0.5,
        )
    else:
        check_bgmv_expand(
            batches=batches,
            num_loras=num_loras,
            rank=rank,
            hidden_size=hidden_size,
            dtype=dtype,
            device=device,
            add_inputs=True,
        )


@pytest.mark.parametrize("batches", test_params["batches"])
@pytest.mark.parametrize("num_loras", test_params["num_loras"])
@pytest.mark.parametrize("rank", test_params["max_ranks"])
@pytest.mark.parametrize("hidden_size", test_params["hidden_sizes"])
@pytest.mark.parametrize("nslices", [2, 3])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("seed", SEED)
@pytest.mark.skipif(not current_platform.is_xpu(), reason="skip for non xpu platform")
def test_bgmv_expand_nslices(
    batches: int,
    num_loras: int,
    rank: int,
    hidden_size: int,
    nslices: int,
    dtype: torch.dtype,
    device: str,
    seed: int,
):
    check_bgmv_expand_slice(
        batches=batches,
        num_loras=num_loras,
        rank=rank,
        hidden_size=hidden_size,
        nslices=nslices,
        dtype=dtype,
        device=device,
        add_inputs=True,
    )


def _uses_gemm(fn) -> bool:
    """Whether `fn` took the dense GEMM path rather than BGMV."""
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU]
    ) as prof:
        fn()
    names = {event.name for event in prof.events()}
    return bool(names & {"aten::mm", "aten::addmm"})


def _lora_linear_reference(
    y, x, lora_a, lora_b, indices, output_slices, scaling, add_inputs
):
    expected = y.float().clone()
    offset = 0
    for a, b, n in zip(lora_a, lora_b, output_slices):
        for token, slot in enumerate(indices.tolist()):
            if slot < 0:
                # BGMV leaves rows without an adapter untouched.
                continue
            hidden = (x[token].float() @ a[slot, 0].float().t() * scaling).to(x.dtype)
            delta = hidden.float() @ b[slot, 0].float().t()
            if add_inputs:
                expected[token, offset : offset + n] += delta
            else:
                expected[token, offset : offset + n] = delta
        offset += n
    return expected.to(y.dtype)


def _make_lora_weights(output_slices, num_loras, rank, hidden_size, dtype, device):
    lora_a = [
        torch.randn(num_loras, 1, rank, hidden_size, dtype=dtype, device=device) * 0.1
        for _ in output_slices
    ]
    lora_b = [
        torch.randn(num_loras, 1, n, rank, dtype=dtype, device=device) * 0.1
        for n in output_slices
    ]
    return lora_a, lora_b


@pytest.mark.parametrize("mapping", ["single", "mixed", "with_base", "small"])
@pytest.mark.parametrize("add_inputs", [True, False])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.skipif(not current_platform.is_xpu(), reason="skip for non xpu platform")
def test_lora_linear(mapping: str, add_inputs: bool, dtype):
    """Both kernel paths of `lora_linear` must match the reference."""
    torch.manual_seed(0)
    device, rank, hidden_size, scaling = "xpu:0", 16, 512, 0.5
    output_slices = [384, 128, 128]
    num_tokens = 8 if mapping == "small" else 256
    lora_a, lora_b = _make_lora_weights(
        output_slices, 3, rank, hidden_size, dtype, device
    )
    indices = torch.full((num_tokens,), 1, dtype=torch.long)
    if mapping == "mixed":
        indices[num_tokens // 2 :] = 2
    elif mapping == "with_base":
        indices[num_tokens // 2 :] = -1
    x = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)
    y = torch.randn(num_tokens, sum(output_slices), dtype=dtype, device=device)
    expected = _lora_linear_reference(
        y, x, lora_a, lora_b, indices, output_slices, scaling, add_inputs
    )

    # The wrapper only reports a single slot when every token uses it.
    slot = 1 if mapping in ("single", "small") else -1
    single_lora_slot = torch.full((1,), slot, dtype=torch.int32)
    used_gemm = _uses_gemm(
        lambda: lora_linear(
            y,
            x,
            lora_a,
            lora_b,
            indices.to(device),
            single_lora_slot,
            output_slices,
            scaling,
            add_inputs,
            True,
            64,
        )
    )
    assert used_gemm == (mapping == "single")
    assert_close(y, expected)


@pytest.mark.parametrize(
    "cudagraph_mode, max_capture, mapping, num_tokens, expect_gemm",
    [
        # Work threshold for rank 16: 8192 / 16 = 512 tokens.
        ("NONE", 0, "single", 511, False),
        ("NONE", 0, "single", 512, True),
        # Batches that may replay a captured graph never take the GEMM path.
        ("FULL_AND_PIECEWISE", 1024, "single", 600, False),
        ("FULL_AND_PIECEWISE", 1024, "single", 1025, True),
        ("NONE", 0, "mixed", 1024, False),
        ("NONE", 0, "with_base", 1024, False),
        # Without a config the captured sizes are unknown: stay on BGMV.
        (None, 0, "single", 1024, False),
    ],
)
@pytest.mark.skipif(not current_platform.is_xpu(), reason="skip for non xpu platform")
def test_xpu_add_lora_linear_dispatch(
    cudagraph_mode, max_capture, mapping, num_tokens, expect_gemm
):
    from vllm.config import (
        CUDAGraphMode,
        VllmConfig,
        get_current_vllm_config_or_none,
        set_current_vllm_config,
    )
    from vllm.config.lora import LoRAConfig
    from vllm.lora.layers import LoRAMapping
    from vllm.lora.punica_wrapper.punica_xpu import PunicaWrapperXPU

    torch.manual_seed(0)
    dtype, device, rank, hidden_size = torch.float16, "xpu:0", 16, 256
    output_slices = (128, 64)
    lora_config = LoRAConfig(max_loras=2, max_lora_rank=rank)
    if cudagraph_mode is None:
        assert get_current_vllm_config_or_none() is None
        wrapper = PunicaWrapperXPU(2048, 4, device, lora_config=lora_config)
    else:
        vllm_config = VllmConfig()
        vllm_config.compilation_config.cudagraph_mode = CUDAGraphMode[cudagraph_mode]
        vllm_config.compilation_config.max_cudagraph_capture_size = max_capture
        with set_current_vllm_config(vllm_config):
            wrapper = PunicaWrapperXPU(2048, 4, device, lora_config=lora_config)

    # Adapter ids 2 and 3 occupy slots 0 and 1; id 0 means no adapter.
    token_ids = [3] * num_tokens
    if mapping == "mixed":
        token_ids[num_tokens // 2 :] = [2] * (num_tokens - num_tokens // 2)
    elif mapping == "with_base":
        token_ids[num_tokens // 2 :] = [0] * (num_tokens - num_tokens // 2)
    wrapper.update_metadata(
        LoRAMapping(tuple(token_ids), tuple(token_ids)), [2, 3], 2, 32000
    )
    slot_of = {0: -1, 2: 0, 3: 1}
    indices = torch.tensor([slot_of[i] for i in token_ids])

    lora_a, lora_b = _make_lora_weights(
        output_slices, 2, rank, hidden_size, dtype, device
    )
    x = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)
    y = torch.randn(num_tokens, sum(output_slices), dtype=dtype, device=device)
    expected = _lora_linear_reference(
        y, x, lora_a, lora_b, indices, output_slices, 1.0, True
    )
    used_gemm = _uses_gemm(
        lambda: wrapper.add_lora_linear(
            y, x, tuple(lora_a), tuple(lora_b), 1.0, output_slices
        )
    )
    assert used_gemm == expect_gemm
    assert_close(y, expected)
