# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Any, Union
import ctypes
import os
import time
import unittest.mock

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.multiprocess.native_completion import (
    DeviceHostFuncDispatcher,
    submit_callback_to_stream,
)
from lmcache.v1.platform import torch_ops as _py_ops

# ==========================================
# 0. utils functions.
# ==========================================


# Device utility functions
def device_sync(device: str) -> None:
    """Synchronize device operations.

    Args:
        device: Device string ("cuda", "xpu", or "cpu")

    This function synchronizes operations for GPU devices:
    - For CUDA devices, calls torch.cuda.synchronize()
    - For XPU devices, calls torch.xpu.synchronize()
    - For CPU, no synchronization is needed (returns immediately)
    """
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "xpu":
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            torch.xpu.synchronize()
    else:
        # TODO: add more device here
        pass
    # CPU requires no synchronization


# ==========================================
# 1. Backend Configuration
# ==========================================


def _build_backend_params() -> list:
    """Build pytest parameter list for the backend fixture.

    Returns one entry per available backend configuration:
    - cuda_c_ops: uses lmcache.c_ops (requires CUDA and the CUDA extension)
    - cuda_py_ops: uses lmcache.v1.platform.torch_ops with GPU visible
    - cpy_py_ops: uses lmcache.v1.platform.torch_ops with GPU mocked away
    - xpu_sycl_ops: uses lmcache.xpu_ops (requires XPU and the SYCL extension)
    - xpu_py_ops: uses lmcache.v1.platform.torch_ops with XPU visible
    """
    params = []
    cuda_available = torch.cuda.is_available()

    params.append(pytest.param(("cpu_py_ops", _py_ops, "cpu"), id="cpu_py_ops"))

    if cuda_available:
        try:
            # First Party
            import lmcache.c_ops as cuda_c_ops

            params.append(
                pytest.param(("cuda_c_ops", cuda_c_ops, "cuda"), id="cuda_c_ops")
            )
        except ImportError:
            pass

        if _py_ops._get_copy_lib() is not None:
            params.append(
                pytest.param(("cuda_py_ops", _py_ops, "cuda"), id="cuda_cuda_py_ops")
            )
        else:
            params.append(
                pytest.param(("cuda_py_ops", _py_ops, "cpu"), id="cuda_cpu_py_ops")
            )

    if hasattr(torch, "xpu") and torch.xpu.is_available():
        try:
            # First Party
            import lmcache.c_ops as xpu_sycl_ops

            params.append(
                pytest.param(("xpu_sycl_ops", xpu_sycl_ops, "xpu"), id="xpu_sycl_ops")
            )
        except ImportError:
            pass

    return params


_BACKEND_PARAMS = _build_backend_params()

# Module-level storage: (scenario_name, backend_id) -> {result_key: tensor}
_results: dict[tuple[str, str], dict[str, Any]] = {}


@pytest.fixture(scope="module", params=_BACKEND_PARAMS)
def backend(request: pytest.FixtureRequest) -> Any:
    """Yield (backend_id, ops_module, device) for each backend config.

    For the cpu_py_ops variant, torch.cuda.is_available is patched to
    return False so the scenario code behaves as if no GPU is present.
    """
    backend_id, ops, device = request.param
    if backend_id == "cpu_py_ops":
        with unittest.mock.patch("torch.cuda.is_available", return_value=False):
            yield backend_id, ops, device
    else:
        yield backend_id, ops, device


# ==========================================
# 2. Scenario functions
# ==========================================


def scenario_get_gpu_pci_bus_id(ops: Any, device: str) -> dict[str, torch.Tensor]:
    """Test get_gpu_pci_bus_id returns a valid string on CUDA backends."""
    res = ops.get_gpu_pci_bus_id(0)

    is_valid = isinstance(res, str) and len(res) > 0

    if torch.cuda.is_available() is False:
        # for non cuda device, not expecting a valid return
        # but crash should not happen
        # and we mock a valid return here
        is_valid = True

    # 1 = PASS (call succeeded without crash)
    # 0 = FAIL
    return {
        "get_gpu_pci_bus_id": torch.tensor([1 if is_valid else 0], dtype=torch.int32)
    }


def scenario_calculate_cdf(ops: Any, device: str) -> dict[str, torch.Tensor]:
    """Test calculate_cdf for multiple bin counts."""
    num_bins_list = [1, 2, 5, 11, 15, 31, 32, 63]
    results: dict[str, torch.Tensor] = {}

    # all bins should be smaller than 64 -> align with C ops
    assert all(n < 64 for n in num_bins_list), (
        f"All num_bins must be < 64, got {[n for n in num_bins_list if n >= 64]}"
    )

    for num_bins in num_bins_list:
        torch.manual_seed(42)

        # Create on CPU first for consistent RNG across backends
        input_tensor = torch.randint(0, num_bins, (1, 1000, 1), dtype=torch.int8).to(
            device
        )

        raw_output = ops.calculate_cdf(input_tensor, num_bins)

        # Both CUDA and non-CUDA return int16 with shape
        # [nlayers, nchannels, num_bins + 1]
        nlayers, _, nchannels = input_tensor.shape
        assert raw_output.shape == (nlayers, nchannels, num_bins + 1), (
            f"Expected shape ({nlayers}, {nchannels}, {num_bins + 1}), "
            f"got {raw_output.shape}"
        )

        out_cpu = raw_output.flatten().cpu()
        out_int32 = out_cpu.to(torch.int32)
        out_uint16 = torch.where(out_int32 < 0, out_int32 + 65536, out_int32)
        final_result = out_uint16.float() / 65536.0

        results[f"calculate_cdf_bins{num_bins}"] = final_result

    return results


def scenario_rotary_embedding_k_fused(ops: Any, device: str) -> dict[str, torch.Tensor]:
    """Test rotary_embedding_k_fused for both NeoX and GPT-J rotation styles."""
    torch.manual_seed(42)

    # 1. Setup Dimensions
    num_tokens = 128
    num_kv_heads = 32
    head_size = 128
    max_position = 2048
    rotary_dim = head_size

    # 2. Generate Inputs on CPU first for consistent RNG across backends
    old_positions = torch.randint(0, 1000, (num_tokens,), dtype=torch.long).to(device)
    new_positions = old_positions + 1

    cos_sin_cache = torch.randn(max_position, rotary_dim, dtype=torch.float32).to(
        device
    )

    # Test both is_neox=True (NeoX-style, contiguous halves) and
    # is_neox=False (GPT-J-style, interleaved)
    results: dict[str, torch.Tensor] = {}
    for is_neox in [True, False]:
        # Reset seed for consistent key tensor across both tests
        torch.manual_seed(42)

        key = torch.randn(num_tokens, num_kv_heads, head_size, dtype=torch.float32).to(
            device
        )

        # 3. Execute (in-place update on key)
        ops.rotary_embedding_k_fused(
            old_positions,
            new_positions,
            key,
            head_size,
            cos_sin_cache,
            is_neox,
        )

        # 4. Collect with is_neox suffix to distinguish the two test cases
        neox_suffix = "neox" if is_neox else "gptj"
        results[f"rotary_embedding_k_fused_{neox_suffix}"] = key.cpu()

    return results


def scenario_lmcache_memcpy_async(ops: Any, device: str) -> dict[str, torch.Tensor]:
    """Test lmcache_memcpy_async for H2D and D2H memory transfers.

    Tests both pointer mode (int pointers) and tensor mode (torch.Tensor objects).
    Uses pointer mode for CPU/CUDA devices and tensor mode for other devices.

    Exercises multiple boundary conditions to verify correct behaviour for
    both the CUDA c_ops backend (which chunks at alignment boundaries via
    cudaMemcpyAsync) and the Python fallback backend (which issues a single
    synchronous copy):
      - copy spanning exactly one aligned block
      - copy entirely within one aligned block (no boundary crossing)
      - copy crossing a single alignment boundary
      - copy crossing multiple alignment boundaries
      - copy starting exactly at an alignment boundary
      - copy with an unaligned start offset crossing many boundaries
    Both H2D and D2H directions are verified for each boundary condition.
    """
    torch.manual_seed(42)

    # Buffer large enough for all boundary test cases (max offset+nbytes = 16+200 = 216)
    buf_size = 256
    alignment = 64  # boundary interval in bytes (must be a power of two)

    src_host = torch.randint(0, 256, (buf_size,), dtype=torch.uint8)
    gpu_buffer = torch.zeros(buf_size, dtype=torch.uint8, device=device)

    dst_host = torch.zeros(buf_size, dtype=torch.uint8)
    if device in ("cuda", "xpu"):
        dst_host = dst_host.pin_memory()

    h2d_dir = ops.TransferDirection.H2D
    d2h_dir = ops.TransferDirection.D2H

    # Decide mode based on the running device.
    # The native CUDA/XPU backend only accepts a tensor of uint64 pointers;
    # only the Python fallback supports list[Tensor].
    use_tensor_mode = device not in ("cpu", "cuda", "xpu")

    # (host_buffer_offset, nbytes) boundary test cases:
    #   (0,  64): exactly one aligned block from the start
    #   (32, 32): entirely within one block, no boundary crossing
    #   (32, 64): crosses one boundary — [32..64) then [64..96)
    #   (0, 192): three full aligned blocks (boundaries at 64 and 128)
    #   (64, 128): starts at alignment boundary, spans two full blocks
    #   (16, 200): unaligned start, crosses multiple boundaries
    test_cases = [
        (0, 64),
        (32, 32),
        (32, 64),
        (0, 192),
        (64, 128),
        (16, 200),
    ]

    all_results: list[torch.Tensor] = []

    for offset, nbytes in test_cases:
        gpu_buffer.zero_()
        dst_host.zero_()
        expected = src_host[offset : offset + nbytes].clone()

        device_sync(device)

        if use_tensor_mode:
            ops.lmcache_memcpy_async(
                gpu_buffer[offset : offset + nbytes],
                src_host[offset : offset + nbytes],
                nbytes,
                h2d_dir,
                0,
                alignment,
            )
            device_sync(device)
            ops.lmcache_memcpy_async(
                dst_host[offset : offset + nbytes],
                gpu_buffer[offset : offset + nbytes],
                nbytes,
                d2h_dir,
                0,
                alignment,
            )
        else:
            ops.lmcache_memcpy_async(
                gpu_buffer.data_ptr() + offset,
                src_host.data_ptr() + offset,
                nbytes,
                h2d_dir,
                offset,
                alignment,
            )
            device_sync(device)
            ops.lmcache_memcpy_async(
                dst_host.data_ptr() + offset,
                gpu_buffer.data_ptr() + offset,
                nbytes,
                d2h_dir,
                offset,
                alignment,
            )

        device_sync(device)

        result = dst_host[offset : offset + nbytes].cpu()
        assert torch.equal(result, expected), (
            f"Boundary test (offset={offset}, nbytes={nbytes}, alignment={alignment}): "
            f"data corrupted, max diff = "
            f"{(result.float() - expected.float()).abs().max().item()}"
        )
        all_results.append(result)

    return {"lmcache_memcpy_async": torch.cat(all_results)}


def scenario_load_and_reshape_flash(ops: Any, device: str) -> dict[str, torch.Tensor]:
    """Test load_and_reshape_flash extracts KV cache tokens into contiguous buffer."""
    torch.manual_seed(42)

    # 1. Standard Params
    src_device = device
    dst_device = "cpu"

    num_blocks = 100
    block_size = 16
    num_heads = 8
    head_size = 128
    num_layers = 32
    num_tokens = 256
    chunk_size = 256
    dtype = torch.bfloat16

    # 2. Setup Data (Deterministic Pattern)
    total_elements = num_blocks * block_size * num_heads * head_size

    kv_cache_cpu = []
    for i in range(num_layers):
        base_tensor = torch.linspace(i, i + 1, total_elements, dtype=torch.float32)
        base_tensor = base_tensor.reshape(
            num_blocks, block_size, num_heads, head_size
        ).to(dtype)
        k = base_tensor
        v = base_tensor + 0.5
        kv_cache_cpu.append([k, v])

    kv_cache = [
        [layer[0].to(src_device), layer[1].to(src_device)] for layer in kv_cache_cpu
    ]

    # Slot mapping: deterministic strided selection
    step = (num_blocks * block_size) // num_tokens
    slot_indices = list(range(0, num_blocks * block_size, step))[:num_tokens]
    slot_mapping = torch.tensor(slot_indices, device=src_device, dtype=torch.int64)
    slot_mapping_chunked = torch.split(slot_mapping, chunk_size)

    # 3. Extract (to CPU pinned)
    extracted_chunks = []
    for chunk_id, slot_mapping_temp in enumerate(slot_mapping_chunked):
        mem_obj_shape = (2, num_layers, len(slot_mapping_temp), num_heads * head_size)
        mem_obj_tensor = torch.zeros(mem_obj_shape, dtype=dtype, device=dst_device)

        if device in ("cuda", "xpu"):
            mem_obj_tensor = mem_obj_tensor.pin_memory()

        for layer_id in range(num_layers):
            ops.load_and_reshape_flash(
                mem_obj_tensor,
                kv_cache[layer_id][0],
                kv_cache[layer_id][1],
                slot_mapping_temp,
                layer_id,
            )
        extracted_chunks.append(mem_obj_tensor)

    device_sync(device)

    # 4. Verify: compare extracted data against original kv_cache
    #    mem_obj_tensor layout:
    #       [2, num_layers, num_tokens_in_chunk, num_heads * head_size]
    #    dim 0: K=0, V=1
    #    Original kv_cache layout: [num_blocks, block_size, num_heads, head_size]
    #    slot_mapping tells us which (block, offset) each token comes from
    for chunk_id, slot_mapping_temp in enumerate(slot_mapping_chunked):
        slots = slot_mapping_temp.cpu()
        extracted = extracted_chunks[chunk_id].cpu()

        for layer_id in range(num_layers):
            orig_k = kv_cache_cpu[layer_id][
                0
            ]  # [num_blocks, block_size, num_heads, head_size]
            orig_v = kv_cache_cpu[layer_id][1]

            for tok_idx, slot in enumerate(slots):
                block_idx = slot.item() // block_size
                offset = slot.item() % block_size

                # Expected: flattened [num_heads * head_size]
                expected_k = orig_k[block_idx, offset].reshape(-1)
                expected_v = orig_v[block_idx, offset].reshape(-1)

                # Extracted
                got_k = extracted[0, layer_id, tok_idx]
                got_v = extracted[1, layer_id, tok_idx]

                k_diff = (got_k.float() - expected_k.float()).abs().max().item()
                assert torch.equal(got_k, expected_k), (
                    f"K mismatch layer={layer_id}, slot={slot.item()}, "
                    f"max diff={k_diff}"
                )

                v_diff = (got_v.float() - expected_v.float()).abs().max().item()
                assert torch.equal(got_v, expected_v), (
                    f"V mismatch layer={layer_id}, slot={slot.item()}, "
                    f"max diff={v_diff}"
                )

    # 5. Return ALL extracted chunks concatenated for cross-backend comparison
    return {
        "load_and_reshape_flash": torch.cat([c.cpu() for c in extracted_chunks], dim=2)
    }


def scenario_reshape_and_cache_back_flash(
    ops: Any, device: str
) -> dict[str, torch.Tensor]:
    """Test reshape_and_cache_back_flash writes tokens back into paged KV cache."""
    torch.manual_seed(42)

    # 1. Environment Setup
    src_device = "cpu"
    dst_device = device

    num_blocks = 100
    block_size = 16
    num_heads = 8
    head_size = 128
    num_layers = 32
    num_tokens = 256
    chunk_size = 256
    dtype = torch.bfloat16

    # 2. Prepare Source Data (CPU Buffer)
    # Shape: [2, num_layers, num_tokens, num_heads * head_size]
    mem_obj_shape = (2, num_layers, num_tokens, num_heads * head_size)
    src_buffer = torch.zeros(mem_obj_shape, dtype=dtype, device=src_device)

    # Data Pattern: Odd numbers (1.0, 3.0, 5.0, ...)
    for i in range(num_tokens):
        val = 1.0 + (i * 2.0)
        src_buffer[0, :, i, :] = val  # Key
        src_buffer[1, :, i, :] = val + 0.5  # Value

    if device in ("cuda", "xpu"):
        src_buffer = src_buffer.pin_memory()

    # 3. Prepare Destination (Empty Cache)
    kv_cache = [
        [
            torch.zeros(
                num_blocks,
                block_size,
                num_heads,
                head_size,
                device=dst_device,
                dtype=dtype,
            ),
            torch.zeros(
                num_blocks,
                block_size,
                num_heads,
                head_size,
                device=dst_device,
                dtype=dtype,
            ),
        ]
        for _ in range(num_layers)
    ]

    # 4. Slot Mapping (Continuous: Token 0 → Slot 0, Token 1 → Slot 1, ...)
    slot_indices = list(range(num_tokens))
    slot_mapping = torch.tensor(slot_indices, device=dst_device, dtype=torch.int64)
    slot_mapping_chunked = torch.split(slot_mapping, chunk_size)

    # 5. Execute Operator (Load Back)
    current_token_offset = 0
    for chunk_id, slot_chunk in enumerate(slot_mapping_chunked):
        chunk_len = len(slot_chunk)

        buffer_chunk = src_buffer[
            :, :, current_token_offset : current_token_offset + chunk_len, :
        ]
        if not buffer_chunk.is_contiguous():
            buffer_chunk = buffer_chunk.contiguous()

        for layer_id in range(num_layers):
            ops.reshape_and_cache_back_flash(
                buffer_chunk,
                kv_cache[layer_id][0],
                kv_cache[layer_id][1],
                slot_chunk,
                layer_id,
            )
        current_token_offset += chunk_len

    device_sync(device)

    # 6. Verify: check written values against source pattern
    for layer_id in range(num_layers):
        k_cache = kv_cache[layer_id][
            0
        ].cpu()  # [num_blocks, block_size, num_heads, head_size]
        v_cache = kv_cache[layer_id][1].cpu()

        for tok_idx, slot in enumerate(slot_indices):
            block_idx = slot // block_size
            offset = slot % block_size

            expected_k_val = 1.0 + (tok_idx * 2.0)
            expected_v_val = expected_k_val + 0.5

            got_k = k_cache[block_idx, offset]
            got_v = v_cache[block_idx, offset]

            expected_k = torch.full_like(got_k, expected_k_val)
            expected_v = torch.full_like(got_v, expected_v_val)

            assert torch.allclose(got_k.float(), expected_k.float(), atol=0.1), (
                f"K mismatch at layer={layer_id}, slot={slot}, "
                f"expected={expected_k_val}, got={got_k[0, 0].item()}"
            )
            assert torch.allclose(got_v.float(), expected_v.float(), atol=0.1), (
                f"V mismatch at layer={layer_id}, slot={slot}, "
                f"expected={expected_v_val}, got={got_v[0, 0].item()}"
            )

    # 7. Return first block of layer 0 key cache for cross-backend comparison
    return {"reshape_and_cache_back_flash": kv_cache[0][0][0].cpu()}


def scenario_encode_fast_new(ops: Any, device: str) -> dict[str, torch.Tensor]:
    """Test encode_fast_new produces valid, non-empty encoded output."""
    torch.manual_seed(42)

    # 1. Hyperparameters
    nlayers = 2
    nchannels = 4
    ntokens = 128
    alphabet_size = 16
    max_buf_len = ntokens * 2

    # 2. Construct Data on target device
    # A. CDF: uniform distribution, strictly increasing
    step = 100 // alphabet_size
    base_cdf = torch.arange(0, 100, step, dtype=torch.int32)
    base_cdf = base_cdf[:alphabet_size]

    cdf_cpu = (
        base_cdf.unsqueeze(0).unsqueeze(0).expand(nlayers, nchannels, -1).contiguous()
    )
    cdf = cdf_cpu.to(dtype=torch.int16, device=device)

    # B. Input symbols: cycling 0..14
    total_syms = nlayers * ntokens * nchannels
    input_cpu = torch.arange(total_syms, dtype=torch.float32)
    input_cpu = (input_cpu % (alphabet_size - 1)).to(torch.int8)
    input_cpu = input_cpu.reshape(nlayers, ntokens, nchannels)
    input_sym = input_cpu.to(device=device)

    # 3. Prepare Outputs
    output_buffer = torch.zeros(
        (nlayers, nchannels, max_buf_len),
        dtype=torch.uint8,
        device=device,
    )
    output_lengths = torch.zeros(
        (nlayers, nchannels),
        dtype=torch.int32,
        device=device,
    )

    # 4. Execute
    ops.encode_fast_new(
        cdf,
        input_sym,
        output_buffer,
        output_lengths,
    )

    device_sync(device)

    # 5. Verify
    lengths_cpu = output_lengths.cpu()

    assert (lengths_cpu > 0).all(), "Encoding produced zero-length output!"
    assert (lengths_cpu <= max_buf_len).all(), "Buffer overflow detected!"

    # 6. Return: first 200 bytes of layer 0, channel 0
    valid_len = int(lengths_cpu[0, 0].item())
    res = output_buffer[0, 0, : min(valid_len, 200)].cpu()
    return {"encode_fast_new": res}


def scenario_decode_fast_new(ops: Any, device: str) -> dict[str, torch.Tensor]:
    """Test decode_fast_new correctly decodes a round-tripped encoded stream."""
    torch.manual_seed(42)

    # 1. Config
    nlayers = 2
    nchannels = 4
    ntokens = 128
    alphabet_size = 16
    max_buf_len = ntokens * 2

    # 2. Data Generation on CPU first for consistent RNG across backends
    cdf = torch.randint(
        1,
        100,
        (nlayers, nchannels, alphabet_size),
        dtype=torch.int32,
    )
    cdf = torch.cumsum(cdf, dim=-1).to(torch.int16).to(device)

    input_sym = torch.randint(
        0,
        alphabet_size - 2,
        (nlayers, ntokens, nchannels),
        dtype=torch.int8,
    ).to(device)

    # 3. Encode first (need encoded data to test decode)
    encoded_buffer = torch.zeros(
        (nlayers, nchannels, max_buf_len),
        dtype=torch.uint8,
        device=device,
    )
    encoded_lengths = torch.zeros(
        (nlayers, nchannels),
        dtype=torch.int32,
        device=device,
    )

    ops.encode_fast_new(
        cdf,
        input_sym,
        encoded_buffer,
        encoded_lengths,
    )
    device_sync(device)

    # 4. Decode
    decoded_sym = torch.zeros_like(input_sym, dtype=torch.uint8)

    ops.decode_fast_new(
        cdf,
        encoded_buffer,
        encoded_lengths,
        decoded_sym,
    )
    device_sync(device)

    # 5. Verify: decoded must match original
    input_uint8 = input_sym.to(torch.uint8)
    mismatch = (input_uint8 != decoded_sym).sum().item()
    if mismatch > 0:
        mask = input_uint8 != decoded_sym
        ly, t, c = mask.nonzero()[0].tolist()
        pytest.fail(
            f"Decode mismatch: {mismatch} errors. "
            f"First diff at L{ly}T{t}C{c}: "
            f"orig={input_uint8[ly, t, c].item()} "
            f"decoded={decoded_sym[ly, t, c].item()}"
        )

    # 6. Return decoded slice for cross-backend comparison
    return {"decode_fast_new": decoded_sym[0, :20, 0].cpu()}


def scenario_decode_fast_prefsum(ops: Any, device: str) -> dict[str, torch.Tensor]:
    """Test decode_fast_prefsum correctly decodes with prefix-sum offsets."""
    torch.manual_seed(42)

    # 1. Configuration
    nlayers = 2
    nchannels = 4
    ntokens = 128
    alphabet_size = 16
    max_buf_len = ntokens * 2

    # 2. Data Generation (Normalized CDF) on CPU first for consistent RNG
    cdf = torch.randint(
        1,
        100,
        (nlayers, nchannels, alphabet_size),
        dtype=torch.int32,
    )
    cdf = torch.cumsum(cdf, dim=-1).float()
    cdf = (cdf / cdf[..., -1:] * 65536).to(torch.int32)
    cdf[..., -1] = 65536
    cdf = cdf.to(torch.int16).to(device).contiguous()

    input_sym = torch.randint(
        0,
        alphabet_size - 2,
        (nlayers, ntokens, nchannels),
        dtype=torch.int8,
    ).to(device)

    # 3. Encode to get variable lengths
    tmp_buf = torch.zeros(
        (nlayers, nchannels, max_buf_len),
        dtype=torch.uint8,
        device=device,
    )
    tmp_len = torch.zeros(
        (nlayers, nchannels),
        dtype=torch.int32,
        device=device,
    )
    ops.encode_fast_new(cdf, input_sym, tmp_buf, tmp_len)
    device_sync(device)

    # 4. Pack into 1D dense bytestream
    lens_flat = tmp_len.cpu().flatten().tolist()
    bufs_flat = tmp_buf.cpu().reshape(-1, max_buf_len).numpy()
    all_bytes = []
    for i, length in enumerate(lens_flat):
        all_bytes.extend(bufs_flat[i, :length].tolist())

    bytestream_1d = torch.tensor(
        all_bytes,
        dtype=torch.uint8,
        device=device,
    ).contiguous()

    # 5. Offsets (end-position via cumsum)
    lengths_prefsum = (
        tmp_len.flatten().cumsum(0).reshape(tmp_len.shape).to(torch.int64).to(device)
    ).contiguous()

    # 6. Decode
    decoded_sym = (
        torch.zeros_like(
            input_sym,
            dtype=torch.uint8,
        )
        .to(device)
        .contiguous()
    )

    ops.decode_fast_prefsum(
        cdf,
        bytestream_1d,
        lengths_prefsum,
        decoded_sym,
    )
    device_sync(device)

    # 7. Verify roundtrip
    input_ref = input_sym.to(torch.uint8)
    mismatch = (input_ref != decoded_sym).sum().item()
    if mismatch > 0:
        mask = input_ref != decoded_sym
        ly, t, c = mask.nonzero()[0].tolist()
        pytest.fail(
            f"Prefsum mismatch: {mismatch} errors. "
            f"First diff at L{ly}T{t}C{c}: "
            f"orig={input_ref[ly, t, c].item()} "
            f"decoded={decoded_sym[ly, t, c].item()}"
        )

    # 8. Return
    return {
        "decode_fast_prefsum": decoded_sym[0, :20, 0].cpu(),
    }


def scenario_single_layer_kv_transfer(ops: Any, device: str) -> dict[str, torch.Tensor]:
    """Test single_layer_kv_transfer for multiple KV formats and directions."""
    torch.manual_seed(42)

    num_tokens = 64
    num_blocks = 256
    block_size = 16
    num_heads = 12
    head_size = 64
    hidden_size = num_heads * head_size

    slot_mapping = torch.arange(
        0,
        num_tokens * 2,
        2,
        device=device,
    ).to(torch.int64)

    # (engine_kv_format, is_mla, token_major, direction)
    # direction: False = LMC→vLLM (H2D), True = vLLM→LMC (D2H)
    test_cases = [
        # flash attn: [2, NB, BS, NH, HS] — two_major
        (ops.EngineKVFormat.NL_X_TWO_NB_BS_NH_HS, False, True, False),
        (ops.EngineKVFormat.NL_X_TWO_NB_BS_NH_HS, False, False, False),
        (ops.EngineKVFormat.NL_X_TWO_NB_BS_NH_HS, False, True, True),
        # flash infer: [NB, 2, BS, NH, HS]
        (ops.EngineKVFormat.NL_X_NB_TWO_BS_NH_HS, False, True, False),
        (ops.EngineKVFormat.NL_X_NB_TWO_BS_NH_HS, False, False, False),
        (ops.EngineKVFormat.NL_X_NB_TWO_BS_NH_HS, False, True, True),
        # vLLM MLA: [NB, BS, HS]
        (ops.EngineKVFormat.NL_X_NB_BS_HS, True, True, False),
        (ops.EngineKVFormat.NL_X_NB_BS_HS, True, True, True),
    ]

    for engine_kv_format, is_mla, token_major, direction in test_cases:
        dir_tag = "v2l" if direction else "l2v"
        is_two_major = engine_kv_format == ops.EngineKVFormat.NL_X_TWO_NB_BS_NH_HS
        case_desc = (
            f"fmt={engine_kv_format}, MLA={is_mla}, TM={token_major}, Dir={dir_tag}"
        )

        # ── 1. Setup Shapes ──
        lmc_shape: tuple[int, ...] = ()
        vllm_shape: tuple[int, ...] = ()
        if is_mla:
            lmc_shape = (num_tokens, hidden_size)
            vllm_shape = (num_blocks, block_size, hidden_size)
        else:
            lmc_shape = (
                (num_tokens, 2, hidden_size)
                if token_major
                else (2, num_tokens, hidden_size)
            )
            if is_two_major:
                # flash attn: [2, num_blocks, block_size, num_heads, head_size]
                vllm_shape = (2, num_blocks, block_size, num_heads, head_size)
            else:
                # flash infer: [num_blocks, 2, block_size, num_heads, head_size]
                vllm_shape = (num_blocks, 2, block_size, num_heads, head_size)

        # ── 2. Deterministic Data ──
        lmc_size = 1
        for s in lmc_shape:
            lmc_size *= s
        vllm_size = 1
        for s in vllm_shape:
            vllm_size *= s

        lmc_tensor = (
            (torch.arange(lmc_size, device=device) % 1000)
            .to(torch.float16)
            .reshape(lmc_shape)
        )
        vllm_tensor = (
            (torch.arange(vllm_size, device=device) % 1000)
            .to(torch.float16)
            .reshape(vllm_shape)
        )

        # ── 3. Golden Reference ──
        lmc_ref = lmc_tensor.clone()
        vllm_ref = vllm_tensor.clone()
        block_indices = slot_mapping // block_size
        block_offsets = slot_mapping % block_size

        if not direction:  # LMC → vLLM
            if is_mla:
                vllm_ref[block_indices, block_offsets, :] = lmc_ref
            else:
                src = lmc_ref if token_major else lmc_ref.permute(1, 0, 2)
                src = src.view(num_tokens, 2, num_heads, head_size)
                if is_two_major:
                    # [2, NB, BS, NH, HS]
                    vllm_ref[0, block_indices, block_offsets] = src[:, 0, :, :]
                    vllm_ref[1, block_indices, block_offsets] = src[:, 1, :, :]
                else:
                    # [NB, 2, BS, NH, HS]
                    vllm_ref[block_indices, 0, block_offsets] = src[:, 0, :, :]
                    vllm_ref[block_indices, 1, block_offsets] = src[:, 1, :, :]
        else:  # vLLM → LMC
            if is_mla:
                lmc_ref = vllm_ref[block_indices, block_offsets, :]
            else:
                if is_two_major:
                    k = vllm_ref[0, block_indices, block_offsets]
                    v = vllm_ref[1, block_indices, block_offsets]
                else:
                    k = vllm_ref[block_indices, 0, block_offsets]
                    v = vllm_ref[block_indices, 1, block_offsets]
                combined = torch.stack(
                    [k, v],
                    dim=1,
                ).view(num_tokens, 2, hidden_size)
                lmc_ref = combined if token_major else combined.permute(1, 0, 2)

        # ── 4. Execute ──
        xfer_dir = ops.TransferDirection.D2H if direction else ops.TransferDirection.H2D
        ops.single_layer_kv_transfer(
            lmc_tensor,
            vllm_tensor,
            slot_mapping,
            xfer_dir,
            engine_kv_format,
            token_major,
        )
        device_sync(device)

        # ── 5. Verify ──
        if not direction:
            torch.testing.assert_close(
                vllm_tensor,
                vllm_ref,
                rtol=1e-3,
                atol=1e-3,
                msg=f"Mismatch in {case_desc}",
            )
        else:
            torch.testing.assert_close(
                lmc_tensor,
                lmc_ref,
                rtol=1e-3,
                atol=1e-3,
                msg=f"Mismatch in {case_desc}",
            )

    # ── 6. Collect canonical results for cross-backend comparison ──
    # Use flash attn (two_major) format to match original file names
    canonical_cases = [
        (False, True, False),  # l2v, non-MLA
        (False, True, True),  # v2l, non-MLA
        (True, True, False),  # l2v, MLA
        (True, True, True),  # v2l, MLA
    ]

    results: dict[str, torch.Tensor] = {}
    for is_mla, token_major, direction in canonical_cases:
        dir_tag = "v2l" if direction else "l2v"

        if is_mla:
            lmc_shape = (num_tokens, hidden_size)
            vllm_shape = (num_blocks, block_size, hidden_size)
            fmt = ops.EngineKVFormat.NL_X_NB_BS_HS
        else:
            lmc_shape = (num_tokens, 2, hidden_size)
            vllm_shape = (2, num_blocks, block_size, num_heads, head_size)
            fmt = ops.EngineKVFormat.NL_X_TWO_NB_BS_NH_HS

        lmc_size = 1
        for s in lmc_shape:
            lmc_size *= s
        vllm_size = 1
        for s in vllm_shape:
            vllm_size *= s

        lmc_tensor = (
            (torch.arange(lmc_size, device=device) % 1000)
            .to(torch.float16)
            .reshape(lmc_shape)
        )
        vllm_tensor = (
            (torch.arange(vllm_size, device=device) % 1000)
            .to(torch.float16)
            .reshape(vllm_shape)
        )

        xfer_dir = ops.TransferDirection.D2H if direction else ops.TransferDirection.H2D
        ops.single_layer_kv_transfer(
            lmc_tensor,
            vllm_tensor,
            slot_mapping,
            xfer_dir,
            fmt,
            token_major,
        )
        device_sync(device)

        result = lmc_tensor.cpu() if direction else vllm_tensor.cpu()
        results[f"single_layer_kv_transfer_{dir_tag}_mla_{is_mla}"] = result

    return results


def scenario_single_layer_kv_transfer_sgl(
    ops: Any, device: str
) -> dict[str, torch.Tensor]:
    """Test single_layer_kv_transfer_sgl for SGLang KV format."""
    torch.manual_seed(42)

    num_tokens = 32
    num_blocks = 128
    block_size = 16
    num_heads = 8
    head_size = 64
    hidden_size = num_heads * head_size

    slot_mapping = torch.arange(
        0,
        num_tokens * 3,
        3,
        device=device,
    ).to(torch.int64)

    # (token_major, direction)
    # direction: False = LMC→SGL, True = SGL→LMC
    test_cases = [
        (True, False),
        (False, False),
        (True, True),
        (False, True),
    ]

    results: dict[str, torch.Tensor] = {}
    for token_major, direction in test_cases:
        dir_tag = "s2l" if direction else "l2s"

        # 1. Setup Shapes
        lmc_shape = (
            (num_tokens, 2, hidden_size)
            if token_major
            else (2, num_tokens, hidden_size)
        )
        sgl_shape = (
            num_blocks,
            block_size,
            num_heads,
            head_size,
        )

        # 2. Deterministic Data
        lmc_size = 1
        for s in lmc_shape:
            lmc_size *= s
        sgl_size = 1
        for s in sgl_shape:
            sgl_size *= s

        lmc_tensor = (
            (torch.arange(lmc_size, device=device) % 500)
            .to(torch.float16)
            .reshape(lmc_shape)
        )
        sgl_k_tensor = (
            (torch.arange(sgl_size, device=device) % 500 + 500)
            .to(torch.float16)
            .reshape(sgl_shape)
        )
        sgl_v_tensor = (
            (torch.arange(sgl_size, device=device) % 500 + 1000)
            .to(torch.float16)
            .reshape(sgl_shape)
        )

        # 3. Golden Reference
        lmc_ref = lmc_tensor.clone()
        sgl_k_ref = sgl_k_tensor.clone()
        sgl_v_ref = sgl_v_tensor.clone()

        block_indices = slot_mapping // block_size
        block_offsets = slot_mapping % block_size

        if not direction:  # LMC → SGL
            src = lmc_ref if token_major else lmc_ref.permute(1, 0, 2)
            src_k = src[:, 0, :].view(
                num_tokens,
                num_heads,
                head_size,
            )
            src_v = src[:, 1, :].view(
                num_tokens,
                num_heads,
                head_size,
            )
            sgl_k_ref[block_indices, block_offsets] = src_k
            sgl_v_ref[block_indices, block_offsets] = src_v
        else:  # SGL → LMC
            k_data = sgl_k_ref[block_indices, block_offsets].reshape(
                num_tokens, hidden_size
            )
            v_data = sgl_v_ref[block_indices, block_offsets].reshape(
                num_tokens, hidden_size
            )

            combined = torch.stack(
                [k_data, v_data],
                dim=1,
            )  # [N, 2, H]
            lmc_ref = combined if token_major else combined.permute(1, 0, 2)

        # 4. Execute
        ops.single_layer_kv_transfer_sgl(
            lmc_tensor,
            sgl_k_tensor,
            sgl_v_tensor,
            slot_mapping,
            ops.TransferDirection.D2H if direction else ops.TransferDirection.H2D,
            token_major,
        )
        device_sync(device)

        # 5. Verify
        case_desc = f"TM={token_major}, Dir={dir_tag}"
        if not direction:
            torch.testing.assert_close(
                sgl_k_tensor,
                sgl_k_ref,
                rtol=1e-3,
                atol=1e-3,
                msg=f"K mismatch in {case_desc}",
            )
            torch.testing.assert_close(
                sgl_v_tensor,
                sgl_v_ref,
                rtol=1e-3,
                atol=1e-3,
                msg=f"V mismatch in {case_desc}",
            )
        else:
            torch.testing.assert_close(
                lmc_tensor,
                lmc_ref,
                rtol=1e-3,
                atol=1e-3,
                msg=f"Mismatch in {case_desc}",
            )

        # 6. Collect each case separately
        result = lmc_tensor.cpu() if direction else sgl_k_tensor.cpu()
        results[f"single_layer_kv_transfer_sgl_{dir_tag}_tm_{token_major}"] = result

    return results


def scenario_multi_layer_kv_transfer(ops: Any, device: str) -> dict[str, torch.Tensor]:
    """Test multi_layer_kv_transfer for multiple paged KV formats and directions.

    Tests both pointer mode (torch.Tensor of int64 pointers) and tensor list mode
    (list[torch.Tensor]) for key_value_ptrs.
    """
    torch.manual_seed(42)

    num_layers = 2
    num_tokens = 4
    head_size = 16
    page_buffer_size = 10
    block_size = 5
    dtype = torch.float32

    slot_mapping = torch.tensor(
        [0, 2, 5, 9],
        dtype=torch.int64,
        device=device,
    )

    # ── Format-specific test cases ──
    # Each: (engine_kv_format, is_mla, block_size_arg)
    format_cases = [
        (ops.EngineKVFormat.NB_NL_TWO_BS_NH_HS, False, 1),  # vLLM cross layer
        (ops.EngineKVFormat.NL_X_TWO_NB_BS_NH_HS, False, 1),  # flash attn
        (ops.EngineKVFormat.NL_X_NB_TWO_BS_NH_HS, False, block_size),  # flash infer
        (ops.EngineKVFormat.NL_X_NB_BS_HS, True, 1),  # vLLM MLA
        (ops.EngineKVFormat.NL_X_NBBS_ONE_HS, True, 1),  # SGLang MLA
    ]

    # Decide mode based on the running device.
    # The native CUDA/XPU backend only accepts a tensor of uint64 pointers;
    # only the Python fallback supports list[Tensor].
    use_tensor_list = device not in ("cpu", "cuda", "xpu")

    for engine_kv_format, is_mla, bs_arg in format_cases:
        k_or_v_size = 1 if is_mla else 2

        for direction in [True, False]:
            dir_tag = "paged2lmc" if direction else "lmc2paged"
            fmt_name = str(engine_kv_format).split(".")[-1]

            # ── 1. LMCache Tensor ──
            lmc_shape = (k_or_v_size, num_layers, num_tokens, head_size)
            key_value = torch.zeros(lmc_shape, dtype=dtype, device="cpu")
            if device in ("cuda", "xpu"):
                key_value = key_value.pin_memory()

            if not direction:  # LMC → Paged
                for kv in range(k_or_v_size):
                    for ly in range(num_layers):
                        for t in range(num_tokens):
                            val = (
                                kv * 5000
                                + ly * 1000
                                + t * 10
                                + torch.arange(head_size, device=device)
                            ).to(dtype)
                            key_value[kv, ly, t] = val

            # ── 2. Paged Buffers (one per layer) ──
            page_buffers = []
            for ly in range(num_layers):
                if engine_kv_format == ops.EngineKVFormat.NL_X_NB_TWO_BS_NH_HS:
                    num_blocks = page_buffer_size // bs_arg
                    pb = torch.zeros(
                        (num_blocks, 2, bs_arg, head_size),
                        dtype=dtype,
                        device=device,
                    )
                elif is_mla:
                    pb = torch.zeros(
                        (page_buffer_size, head_size),
                        dtype=dtype,
                        device=device,
                    )
                else:
                    # Handles NB_NL_TWO_BS_NH_HS and NL_X_TWO_NB_BS_NH_HS
                    pb = torch.zeros(
                        (2, page_buffer_size, head_size),
                        dtype=dtype,
                        device=device,
                    )

                if direction:  # Paged → LMC
                    for s in range(page_buffer_size):
                        for kv in range(k_or_v_size):
                            val = (
                                kv * 7000
                                + ly * 2000
                                + s * 10
                                + torch.arange(head_size, device=device)
                            ).to(dtype)
                            if (
                                engine_kv_format
                                == ops.EngineKVFormat.NL_X_NB_TWO_BS_NH_HS
                            ):
                                blk_idx = s // bs_arg
                                blk_off = s % bs_arg
                                pb[blk_idx, kv, blk_off] = val
                            elif is_mla:
                                pb[s] = val
                            else:
                                # Handles NB_NL_TWO_BS_NH_HS and NL_X_TWO_NB_BS_NH_HS
                                pb[kv, s] = val

                page_buffers.append(pb)

            # ── 3. Prepare key_value_ptrs (pointer mode or tensor list mode) ──
            key_value_ptrs: Union[list[torch.Tensor], torch.Tensor]
            if use_tensor_list:
                # Tensor list mode: pass the tensor objects directly
                key_value_ptrs = page_buffers
            else:
                # Pointer mode: create tensor of uint64 pointers
                key_value_ptrs = torch.tensor(
                    [pb.data_ptr() for pb in page_buffers],
                    dtype=torch.uint64,
                    device=device,
                )

            # ── 4. Execute ──
            xfer_dir = (
                ops.TransferDirection.D2H if direction else ops.TransferDirection.H2D
            )
            ops.multi_layer_kv_transfer(
                key_value,
                key_value_ptrs,
                slot_mapping,
                torch.device(device),
                page_buffer_size,
                xfer_dir,
                engine_kv_format,
                bs_arg,
            )
            device_sync(device)

            # ── 5. Verify (internal, per-format) ──
            for t_id in range(num_tokens):
                s_idx = int(slot_mapping[t_id].item())
                for ly in range(num_layers):
                    for kv in range(k_or_v_size):
                        lmc_val = key_value[kv, ly, t_id]

                        if engine_kv_format == ops.EngineKVFormat.NL_X_NB_TWO_BS_NH_HS:
                            blk_idx = s_idx // bs_arg
                            blk_off = s_idx % bs_arg
                            paged_val = page_buffers[ly][blk_idx, kv, blk_off]
                        elif is_mla:
                            paged_val = page_buffers[ly][s_idx]
                        else:
                            # Handles NB_NL_TWO_BS_NH_HS and NL_X_TWO_NB_BS_NH_HS
                            paged_val = page_buffers[ly][kv, s_idx]

                        torch.testing.assert_close(
                            lmc_val.to("cpu"),
                            paged_val.to("cpu"),
                            msg=(
                                f"Mismatch: {fmt_name} {dir_tag} "
                                f"(tensor_list={use_tensor_list}), "
                                f"kv={kv}, layer={ly}, token={t_id}"
                            ),
                        )

    # ── 6. Collect ONE canonical result for cross-backend comparison ──
    # Use flash attn format (NL_X_TWO_NB_BS_NH_HS), re-run canonical cases
    canonical_format = ops.EngineKVFormat.NL_X_TWO_NB_BS_NH_HS
    results: dict[str, torch.Tensor] = {}
    for direction in [True, False]:
        dir_tag = "paged2lmc" if direction else "lmc2paged"
        lmc_shape = (2, num_layers, num_tokens, head_size)
        key_value = torch.zeros(lmc_shape, dtype=dtype, device=device)

        if not direction:
            for ly in range(num_layers):
                for t in range(num_tokens):
                    val = (
                        ly * 1000 + t * 10 + torch.arange(head_size, device=device)
                    ).to(dtype)
                    key_value[0, ly, t] = val
                    key_value[1, ly, t] = val + 500

        page_buffers = []
        for ly in range(num_layers):
            pb = torch.zeros(
                (2, page_buffer_size, head_size),
                dtype=dtype,
                device=device,
            )
            if direction:
                for s in range(page_buffer_size):
                    val = (
                        ly * 2000 + s * 10 + torch.arange(head_size, device=device)
                    ).to(dtype)
                    pb[0, s] = val
                    pb[1, s] = val + 700
            page_buffers.append(pb)

        if use_tensor_list:
            key_value_ptrs = page_buffers
        else:
            key_value_ptrs = torch.tensor(
                [pb.data_ptr() for pb in page_buffers],
                dtype=torch.uint64,
                device=device,
            )

        xfer_dir = ops.TransferDirection.D2H if direction else ops.TransferDirection.H2D
        ops.multi_layer_kv_transfer(
            key_value,
            key_value_ptrs,
            slot_mapping,
            torch.device(device),
            page_buffer_size,
            xfer_dir,
            canonical_format,
            1,
        )
        device_sync(device)

        results[f"multi_layer_kv_transfer_{dir_tag}"] = key_value.cpu()

    return results


def scenario_multi_layer_kv_transfer_unilateral(
    ops: Any, device: str
) -> dict[str, torch.Tensor]:
    """Test multi_layer_kv_transfer_unilateral for non-interleaved K/V pointers.

    Tests both pointer mode (torch.Tensor of int64 pointers) and tensor list mode
    (list[torch.Tensor]) for key_value_ptrs.
    """
    torch.manual_seed(42)

    num_layers = 2
    num_tokens = 4
    head_size = 16
    page_buffer_size = 10
    dtype = torch.float32

    slot_mapping = torch.tensor(
        [1, 3, 4, 7],
        dtype=torch.int64,
        device=device,
    )

    # ── Test cases: (engine_kv_format, is_mla) ──
    format_cases = [
        (
            ops.EngineKVFormat.NL_X_TWO_NB_BS_NH_HS,
            False,
        ),  # SGLang MHA (unilateral path)
        (
            ops.EngineKVFormat.NL_X_NB_BS_HS,
            True,
        ),  # vLLM MLA (delegates to multi_layer_kv_transfer)
        (
            ops.EngineKVFormat.NL_X_NBBS_ONE_HS,
            True,
        ),  # SGLang MLA (delegates to multi_layer_kv_transfer)
    ]

    # Decide mode based on the running device.
    # The native CUDA/XPU backend only accepts a tensor of uint64 pointers;
    # only the Python fallback supports list[Tensor].
    use_tensor_list = device not in ("cpu", "cuda", "xpu")

    for engine_kv_format, is_mla in format_cases:
        k_or_v_size = 1 if is_mla else 2

        for direction in [True, False]:
            dir_tag = "p2l" if direction else "l2p"

            # ── 1. LMCache Tensor ──
            lmc_shape = (k_or_v_size, num_layers, num_tokens, head_size)
            lmc_tensor = torch.zeros(lmc_shape, dtype=dtype, device="cpu")
            if device in ("cuda", "xpu"):
                lmc_tensor = lmc_tensor.pin_memory()

            if not direction:  # LMC → Paged
                for kv in range(k_or_v_size):
                    for ly in range(num_layers):
                        for t in range(num_tokens):
                            val = (
                                kv * 5000
                                + ly * 1000
                                + t * 10
                                + torch.arange(head_size, device=device)
                            ).to(dtype)
                            lmc_tensor[kv, ly, t] = val

            # ── 2. Paged Buffers ──
            key_value_ptrs: Union[list[torch.Tensor], torch.Tensor]
            if is_mla:
                # MLA delegates to multi_layer_kv_transfer
                # ptrs: [layer0, layer1, ...], each -> [page_buffer_size, head_size]
                page_buffers = []
                for ly in range(num_layers):
                    pb = torch.zeros(
                        (page_buffer_size, head_size),
                        dtype=dtype,
                        device=device,
                    )
                    if direction:  # Paged → LMC
                        for s in range(page_buffer_size):
                            val = (
                                ly * 2000
                                + s * 10
                                + torch.arange(head_size, device=device)
                            ).to(dtype)
                            pb[s] = val
                    page_buffers.append(pb)

                if use_tensor_list:
                    key_value_ptrs = page_buffers
                else:
                    key_value_ptrs = torch.tensor(
                        [pb.data_ptr() for pb in page_buffers],
                        dtype=torch.uint64,
                        device=device,
                    )
            else:
                # Non-MLA unilateral: separate K/V buffers
                # ptrs: [K_l0, K_l1, ..., V_l0, V_l1, ...]
                # each -> [page_buffer_size, head_size]
                buffers = {}
                for kv in range(2):
                    for ly in range(num_layers):
                        pb = torch.zeros(
                            (page_buffer_size, head_size),
                            dtype=dtype,
                            device=device,
                        )
                        if direction:  # Paged → LMC
                            for s in range(page_buffer_size):
                                val = (
                                    kv * 7000
                                    + ly * 2000
                                    + s * 10
                                    + torch.arange(head_size, device=device)
                                ).to(dtype)
                                pb[s] = val
                        buffers[(kv, ly)] = pb

                if use_tensor_list:
                    # Tensor list mode: [K_l0, K_l1, ..., V_l0, V_l1, ...]
                    tensor_list = []
                    for ly in range(num_layers):
                        tensor_list.append(buffers[(0, ly)])
                    for ly in range(num_layers):
                        tensor_list.append(buffers[(1, ly)])
                    key_value_ptrs = tensor_list
                else:
                    # Pointer mode
                    ptr_list = []
                    for ly in range(num_layers):
                        ptr_list.append(buffers[(0, ly)].data_ptr())
                    for ly in range(num_layers):
                        ptr_list.append(buffers[(1, ly)].data_ptr())

                    key_value_ptrs = torch.tensor(
                        ptr_list,
                        dtype=torch.uint64,
                        device=device,
                    ).contiguous()

            # ── 3. Execute ──
            xfer_dir = (
                ops.TransferDirection.D2H if direction else ops.TransferDirection.H2D
            )
            ops.multi_layer_kv_transfer_unilateral(
                lmc_tensor,
                key_value_ptrs,
                slot_mapping,
                torch.device(device),
                page_buffer_size,
                xfer_dir,
                engine_kv_format,
            )
            device_sync(device)

            # ── 4. Verify ──
            for t_id in range(num_tokens):
                s_idx = int(slot_mapping[t_id].item())
                for ly in range(num_layers):
                    for kv in range(k_or_v_size):
                        lmc_val = lmc_tensor[kv, ly, t_id]

                        if is_mla:
                            paged_val = page_buffers[ly][s_idx]
                        else:
                            paged_val = buffers[(kv, ly)][s_idx]

                        torch.testing.assert_close(
                            lmc_val.to("cpu"),
                            paged_val.to("cpu"),
                            msg=(
                                f"Mismatch: {engine_kv_format} {dir_tag} "
                                f"(tensor_list={use_tensor_list}), "
                                f"KV={kv}, layer={ly}, "
                                f"token={t_id}, slot={s_idx}"
                            ),
                        )

    # ── 5. Collect canonical result for cross-backend comparison ──
    # Use non-MLA unilateral (the primary use case of this function)
    results: dict[str, torch.Tensor] = {}
    for direction in [True, False]:
        dir_tag = "p2l" if direction else "l2p"

        lmc_shape = (2, num_layers, num_tokens, head_size)
        lmc_tensor = torch.zeros(lmc_shape, dtype=dtype, device=device)

        if not direction:
            for kv in range(2):
                for ly in range(num_layers):
                    for t in range(num_tokens):
                        val = (
                            kv * 5000
                            + ly * 1000
                            + t * 10
                            + torch.arange(head_size, device=device)
                        ).to(dtype)
                        lmc_tensor[kv, ly, t] = val

        buffers = {}
        for kv in range(2):
            for ly in range(num_layers):
                pb = torch.zeros(
                    (page_buffer_size, head_size),
                    dtype=dtype,
                    device=device,
                )
                if direction:
                    for s in range(page_buffer_size):
                        val = (
                            kv * 7000
                            + ly * 2000
                            + s * 10
                            + torch.arange(head_size, device=device)
                        ).to(dtype)
                        pb[s] = val
                buffers[(kv, ly)] = pb

        key_value_ptrs: Union[list[torch.Tensor], torch.Tensor]  # type: ignore[no-redef]
        if use_tensor_list:
            # Tensor list mode: [K_l0, K_l1, ..., V_l0, V_l1, ...]
            tensor_list = []
            for ly in range(num_layers):
                tensor_list.append(buffers[(0, ly)])
            for ly in range(num_layers):
                tensor_list.append(buffers[(1, ly)])
            key_value_ptrs = tensor_list
        else:
            ptr_list = []
            for ly in range(num_layers):
                ptr_list.append(buffers[(0, ly)].data_ptr())
            for ly in range(num_layers):
                ptr_list.append(buffers[(1, ly)].data_ptr())
            key_value_ptrs = torch.tensor(
                ptr_list,
                dtype=torch.uint64,
                device=device,
            ).contiguous()

        xfer_dir = ops.TransferDirection.D2H if direction else ops.TransferDirection.H2D
        ops.multi_layer_kv_transfer_unilateral(
            lmc_tensor,
            key_value_ptrs,
            slot_mapping,
            torch.device(device),
            page_buffer_size,
            xfer_dir,
            ops.EngineKVFormat.NL_X_TWO_NB_BS_NH_HS,
        )
        device_sync(device)

        results[f"multi_layer_kv_transfer_unilateral_{dir_tag}"] = lmc_tensor.cpu()

    return results


def scenario_alloc_free_pinned_ptr(ops: Any, device: str) -> dict[str, torch.Tensor]:
    """Test alloc_pinned_ptr and free_pinned_ptr round-trip."""
    alloc_size = 4096
    flags = 0  # cudaHostAllocDefault

    # 1. Allocate
    ptr = ops.alloc_pinned_ptr(alloc_size, flags)
    assert isinstance(ptr, int), f"Expected int, got {type(ptr)}"
    assert ptr != 0, "alloc_pinned_ptr returned null"

    # 2. Free
    ops.free_pinned_ptr(ptr)

    return {"alloc_free_pinned_ptr": torch.tensor([1], dtype=torch.int32)}


def scenario_alloc_free_numa_ptr(ops: Any, device: str) -> dict[str, torch.Tensor]:
    """Test alloc_numa_ptr and free_numa_ptr round-trip."""
    alloc_size = 4096
    node = 0  # NUMA node 0 (always exists)

    # 1. Allocate
    ptr = ops.alloc_numa_ptr(alloc_size, node)
    assert isinstance(ptr, int), f"Expected int, got {type(ptr)}"
    assert ptr != 0, "alloc_numa_ptr returned null"

    # 2. Free (must pass same size as alloc)
    ops.free_numa_ptr(ptr, alloc_size)

    return {"alloc_free_numa_ptr": torch.tensor([1], dtype=torch.int32)}


def scenario_alloc_free_pinned_numa_ptr(
    ops: Any, device: str
) -> dict[str, torch.Tensor]:
    """Test alloc_pinned_numa_ptr and free_pinned_numa_ptr round-trip."""
    alloc_size = 4096
    node = 0  # NUMA node 0

    # 1. Allocate (NUMA + cudaHostRegister)
    ptr = ops.alloc_pinned_numa_ptr(alloc_size, node)
    assert isinstance(ptr, int), f"Expected int, got {type(ptr)}"
    assert ptr != 0, "alloc_pinned_numa_ptr returned null"

    # 2. Free (cudaHostUnregister + munmap)
    ops.free_pinned_numa_ptr(ptr, alloc_size)

    return {"alloc_free_pinned_numa_ptr": torch.tensor([1], dtype=torch.int32)}


def scenario_alloc_free_shm_pinned_ptr(
    ops: Any, device: str
) -> dict[str, torch.Tensor]:
    """Test alloc_shm_pinned_ptr and free_shm_pinned_ptr round-trip."""
    alloc_size = 4096
    shm_name = "/test_lmcache_shm"

    # 1. Allocate
    ptr = ops.alloc_shm_pinned_ptr(alloc_size, shm_name)
    assert isinstance(ptr, int), f"Expected int, got {type(ptr)}"
    assert ptr != 0, "alloc_shm_pinned_ptr returned null"

    # 2. Free
    ops.free_shm_pinned_ptr(ptr, alloc_size, shm_name)

    return {"alloc_free_shm_pinned_ptr": torch.tensor([1], dtype=torch.int32)}


def scenario_transfer_direction_enum(ops: Any, device: str) -> dict[str, torch.Tensor]:
    """Test TransferDirection enum has distinct H2D and D2H members."""
    # 1. Verify enum members exist
    td = ops.TransferDirection
    assert hasattr(td, "H2D"), "Missing TransferDirection.H2D"
    assert hasattr(td, "D2H"), "Missing TransferDirection.D2H"

    # 2. Verify values are distinct
    assert td.H2D != td.D2H, "H2D and D2H should be distinct"

    # 3. Extract int value (compatible with both
    #    pybind11 enum and Python enum)
    h2d = td.H2D
    d2h = td.D2H
    h2d_val = h2d.value if hasattr(h2d, "value") else int(h2d)
    d2h_val = d2h.value if hasattr(d2h, "value") else int(d2h)

    return {
        "transfer_direction_enum": torch.tensor(
            [h2d_val, d2h_val],
            dtype=torch.int32,
        )
    }


def scenario_record_drain_completion(ops: Any, device: str) -> dict[str, torch.Tensor]:
    """Test record_completion_on_stream / drain_recorded_completions contracts.

    Verified backend-agnostic: native c_ops uses cudaLaunchHostFunc on the
    default stream (ptr=0), which fires synchronously after device_sync; the
    fallback enqueues immediately. Both paths satisfy every assertion below.
    """
    ops.drain_recorded_completions()  # clear residual global state

    assert ops.drain_recorded_completions() == []

    ops.record_completion_on_stream(0, "kind-a", b"payload-a")
    ops.record_completion_on_stream(0, "kind-b", b"payload-b")
    device_sync(device)
    result = ops.drain_recorded_completions()
    assert result == [("kind-a", b"payload-a"), ("kind-b", b"payload-b")]
    assert all(isinstance(k, str) and isinstance(p, bytes) for k, p in result)

    assert ops.drain_recorded_completions() == []

    # Multiple records on the default stream (ptr=0) must all enqueue.
    # Note: ptr=0 is the only value safe on both backends — the fallback
    # ignores the field, but the native path casts it to cudaStream_t, so
    # arbitrary values like -1 / 2**32 would be invalid there.
    for _ in range(3):
        ops.record_completion_on_stream(0, "k", b"v")
    device_sync(device)
    assert len(ops.drain_recorded_completions()) == 3

    return {"record_drain_completion": torch.tensor([1], dtype=torch.int32)}


def scenario_dispatcher_integration(ops: Any, device: str) -> dict[str, torch.Tensor]:
    """Test submit_callback_to_stream -> DeviceHostFuncDispatcher -> handler.

    Works on all backends: submit_callback_to_stream only reads stream.ptr, so
    _FakeStream(ptr=0) is accepted; the native path routes through the CUDA
    default stream which fires before the dispatcher's drain loop polls.
    """
    # First Party
    import lmcache.v1.multiprocess.native_completion as nc

    original = nc._lmc_ops
    nc._lmc_ops = ops
    try:
        ops.drain_recorded_completions()

        class _FakeStream:
            ptr: int = 0

        dispatcher = DeviceHostFuncDispatcher(drain_interval_seconds=0.001)
        received: list[list[bytes]] = []
        dispatcher.register("finish_write", received.append, payload_type=list[bytes])
        dispatcher.start()
        try:
            submit_callback_to_stream(_FakeStream(), "finish_write", [b"k0", b"k1"])
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline and not received:
                time.sleep(0.01)
        finally:
            dispatcher.stop()

        assert received == [[b"k0", b"k1"]]
    finally:
        nc._lmc_ops = original

    return {"dispatcher_integration": torch.tensor([1], dtype=torch.int32)}


def scenario_multi_layer_block_kv_transfer(
    ops: Any, device: str
) -> dict[str, torch.Tensor]:
    """Test multi_layer_block_kv_transfer across all GPU KV formats.

    Exercises the block-based transfer path for:
    - NHD per-layer format (D2H and H2D round-trip)
    - FlashInfer NHD per-layer format (interleaved K/V)
    - HND per-layer format (with permute)
    - FlashInfer HND per-layer format (interleaved K/V)
    - MLA per-layer format (no K/V split)
    - SGLang MLA flat per-layer format
    - Cross-layer NHD (single tensor [NB, NL, 2, BS, NH, HS])
    - Cross-layer HND (single tensor [NB, NL, 2, NH, BS, HS])
    - SGLang MHA flat (2*NL tensors [NB*BS, NH, HS])
    - SGLang MHA block (2*NL tensors [NB, BS, NH, HS])
    - non-sequential block_ids and list[int] block_ids input
    - skip_prefix_n_blocks > 0
    - num_blocks > blocks_per_chunk (multi-chunk)
    """
    results = {}

    # C++ bindings (cuda_c_ops, xpu_sycl_ops) expect uint64 pointer tensors for
    # paged_buffer_ptrs_tensor and list[int] for lmcache_objects_ptrs.
    # The Python fallback also supports both modes on cpu/cuda (pointer inputs
    # are reconstructed internally via _tensor_from_ptr).
    use_tensor_list = device not in ("cpu", "cuda")

    def _alloc_chunks(shape: tuple[int, ...], count: int) -> list[torch.Tensor]:
        chunks = [torch.zeros(shape, dtype=dtype) for _ in range(count)]
        if device in ("cuda"):
            chunks = [chunk.pin_memory() for chunk in chunks]
        return chunks

    # --- NHD per-layer ---
    torch.manual_seed(123)
    num_layers, num_blocks, block_size = 2, 8, 4
    num_heads, head_size = 2, 8
    blocks_per_chunk = 4
    chunk_tokens = blocks_per_chunk * block_size
    hidden_dim = num_heads * head_size
    dtype = torch.float32

    paged_layers = [
        torch.randn(2, num_blocks, block_size, num_heads, head_size, dtype=dtype).to(
            device
        )
        for _ in range(num_layers)
    ]
    shape_desc = ops.PageBufferShapeDesc()
    shape_desc.nl = num_layers
    shape_desc.nb = num_blocks
    shape_desc.bs = block_size
    shape_desc.nh = num_heads
    shape_desc.hs = head_size
    shape_desc.element_size = dtype.itemsize
    shape_desc.kv_size = 2
    engine_kv_format = ops.EngineKVFormat.NL_X_TWO_NB_BS_NH_HS
    num_chunks = num_blocks // blocks_per_chunk
    d2h_chunks = _alloc_chunks((2, num_layers, chunk_tokens, hidden_dim), num_chunks)
    block_ids = list(range(num_blocks))

    ops.multi_layer_block_kv_transfer(
        paged_layers
        if use_tensor_list
        else torch.tensor(
            [layer.data_ptr() for layer in paged_layers],
            dtype=torch.uint64,
            device=device,
        ),
        d2h_chunks if use_tensor_list else [c.data_ptr() for c in d2h_chunks],
        torch.tensor(block_ids, dtype=torch.int64, device=device),
        torch.device(device),
        ops.TransferDirection.D2H,
        shape_desc,
        chunk_tokens,
        engine_kv_format,
        0,
    )
    paged_h2d = [torch.zeros_like(layer) for layer in paged_layers]
    ops.multi_layer_block_kv_transfer(
        paged_h2d
        if use_tensor_list
        else torch.tensor(
            [layer.data_ptr() for layer in paged_h2d], dtype=torch.uint64, device=device
        ),
        d2h_chunks if use_tensor_list else [c.data_ptr() for c in d2h_chunks],
        torch.tensor(block_ids, dtype=torch.int64, device=device),
        torch.device(device),
        ops.TransferDirection.H2D,
        shape_desc,
        chunk_tokens,
        engine_kv_format,
        0,
    )
    for i in range(num_layers):
        orig = paged_layers[i].cpu()
        recon = paged_h2d[i].cpu()
        results[f"nhd_layer{i}"] = torch.stack([orig, recon])
        assert torch.allclose(orig, recon, atol=1e-6), (
            f"NHD Layer {i} round-trip mismatch"
        )

    # --- FlashInfer NHD per-layer ---
    torch.manual_seed(234)
    paged_layers_fi_nhd = [
        torch.randn(num_blocks, 2, block_size, num_heads, head_size, dtype=dtype).to(
            device
        )
        for _ in range(num_layers)
    ]
    engine_kv_format_fi_nhd = ops.EngineKVFormat.NL_X_NB_TWO_BS_NH_HS
    d2h_chunks_fi_nhd = _alloc_chunks(
        (2, num_layers, chunk_tokens, hidden_dim), num_chunks
    )
    ops.multi_layer_block_kv_transfer(
        paged_layers_fi_nhd
        if use_tensor_list
        else torch.tensor(
            [layer.data_ptr() for layer in paged_layers_fi_nhd],
            dtype=torch.uint64,
            device=device,
        ),
        d2h_chunks_fi_nhd
        if use_tensor_list
        else [c.data_ptr() for c in d2h_chunks_fi_nhd],
        torch.tensor(block_ids, dtype=torch.int64, device=device),
        torch.device(device),
        ops.TransferDirection.D2H,
        shape_desc,
        chunk_tokens,
        engine_kv_format_fi_nhd,
        0,
    )
    paged_h2d_fi_nhd = [torch.zeros_like(layer) for layer in paged_layers_fi_nhd]
    ops.multi_layer_block_kv_transfer(
        paged_h2d_fi_nhd
        if use_tensor_list
        else torch.tensor(
            [layer.data_ptr() for layer in paged_h2d_fi_nhd],
            dtype=torch.uint64,
            device=device,
        ),
        d2h_chunks_fi_nhd
        if use_tensor_list
        else [c.data_ptr() for c in d2h_chunks_fi_nhd],
        torch.tensor(block_ids, dtype=torch.int64, device=device),
        torch.device(device),
        ops.TransferDirection.H2D,
        shape_desc,
        chunk_tokens,
        engine_kv_format_fi_nhd,
        0,
    )
    for i in range(num_layers):
        orig = paged_layers_fi_nhd[i].cpu()
        recon = paged_h2d_fi_nhd[i].cpu()
        results[f"flashinfer_nhd_layer{i}"] = torch.stack([orig, recon])
        assert torch.allclose(orig, recon, atol=1e-6), (
            f"FlashInfer NHD Layer {i} round-trip mismatch"
        )

    # --- HND per-layer ---
    torch.manual_seed(456)
    paged_layers_hnd = [
        torch.randn(2, num_blocks, num_heads, block_size, head_size, dtype=dtype).to(
            device
        )
        for _ in range(num_layers)
    ]
    engine_kv_format_hnd = ops.EngineKVFormat.NL_X_TWO_NB_NH_BS_HS
    d2h_chunks_hnd = _alloc_chunks(
        (2, num_layers, chunk_tokens, hidden_dim), num_chunks
    )
    ops.multi_layer_block_kv_transfer(
        paged_layers_hnd
        if use_tensor_list
        else torch.tensor(
            [layer.data_ptr() for layer in paged_layers_hnd],
            dtype=torch.uint64,
            device=device,
        ),
        d2h_chunks_hnd if use_tensor_list else [c.data_ptr() for c in d2h_chunks_hnd],
        torch.tensor(block_ids, dtype=torch.int64, device=device),
        torch.device(device),
        ops.TransferDirection.D2H,
        shape_desc,
        chunk_tokens,
        engine_kv_format_hnd,
        0,
    )
    paged_h2d_hnd = [torch.zeros_like(layer) for layer in paged_layers_hnd]
    ops.multi_layer_block_kv_transfer(
        paged_h2d_hnd
        if use_tensor_list
        else torch.tensor(
            [layer.data_ptr() for layer in paged_h2d_hnd],
            dtype=torch.uint64,
            device=device,
        ),
        d2h_chunks_hnd if use_tensor_list else [c.data_ptr() for c in d2h_chunks_hnd],
        torch.tensor(block_ids, dtype=torch.int64, device=device),
        torch.device(device),
        ops.TransferDirection.H2D,
        shape_desc,
        chunk_tokens,
        engine_kv_format_hnd,
        0,
    )
    for i in range(num_layers):
        orig = paged_layers_hnd[i].cpu()
        recon = paged_h2d_hnd[i].cpu()
        results[f"hnd_layer{i}"] = torch.stack([orig, recon])
        assert torch.allclose(orig, recon, atol=1e-6), (
            f"HND Layer {i} round-trip mismatch"
        )

    # --- FlashInfer HND per-layer ---
    torch.manual_seed(567)
    paged_layers_fi_hnd = [
        torch.randn(num_blocks, 2, num_heads, block_size, head_size, dtype=dtype).to(
            device
        )
        for _ in range(num_layers)
    ]
    engine_kv_format_fi_hnd = ops.EngineKVFormat.NL_X_NB_TWO_NH_BS_HS
    d2h_chunks_fi_hnd = _alloc_chunks(
        (2, num_layers, chunk_tokens, hidden_dim), num_chunks
    )
    ops.multi_layer_block_kv_transfer(
        paged_layers_fi_hnd
        if use_tensor_list
        else torch.tensor(
            [layer.data_ptr() for layer in paged_layers_fi_hnd],
            dtype=torch.uint64,
            device=device,
        ),
        d2h_chunks_fi_hnd
        if use_tensor_list
        else [c.data_ptr() for c in d2h_chunks_fi_hnd],
        torch.tensor(block_ids, dtype=torch.int64, device=device),
        torch.device(device),
        ops.TransferDirection.D2H,
        shape_desc,
        chunk_tokens,
        engine_kv_format_fi_hnd,
        0,
    )
    paged_h2d_fi_hnd = [torch.zeros_like(layer) for layer in paged_layers_fi_hnd]
    ops.multi_layer_block_kv_transfer(
        paged_h2d_fi_hnd
        if use_tensor_list
        else torch.tensor(
            [layer.data_ptr() for layer in paged_h2d_fi_hnd],
            dtype=torch.uint64,
            device=device,
        ),
        d2h_chunks_fi_hnd
        if use_tensor_list
        else [c.data_ptr() for c in d2h_chunks_fi_hnd],
        torch.tensor(block_ids, dtype=torch.int64, device=device),
        torch.device(device),
        ops.TransferDirection.H2D,
        shape_desc,
        chunk_tokens,
        engine_kv_format_fi_hnd,
        0,
    )
    for i in range(num_layers):
        orig = paged_layers_fi_hnd[i].cpu()
        recon = paged_h2d_fi_hnd[i].cpu()
        results[f"flashinfer_hnd_layer{i}"] = torch.stack([orig, recon])
        assert torch.allclose(orig, recon, atol=1e-6), (
            f"FlashInfer HND Layer {i} round-trip mismatch"
        )

    # --- MLA per-layer ---
    torch.manual_seed(789)
    mla_hidden = 16
    paged_layers_mla = [
        torch.randn(num_blocks, block_size, mla_hidden, dtype=dtype).to(device)
        for _ in range(num_layers)
    ]
    shape_desc_mla = ops.PageBufferShapeDesc()
    shape_desc_mla.nl = num_layers
    shape_desc_mla.nb = num_blocks
    shape_desc_mla.bs = block_size
    shape_desc_mla.nh = 1
    shape_desc_mla.hs = mla_hidden
    shape_desc_mla.element_size = dtype.itemsize
    shape_desc_mla.kv_size = 1
    engine_kv_format_mla = ops.EngineKVFormat.NL_X_NB_BS_HS
    d2h_chunks_mla = _alloc_chunks((num_layers, chunk_tokens, mla_hidden), num_chunks)
    ops.multi_layer_block_kv_transfer(
        paged_layers_mla
        if use_tensor_list
        else torch.tensor(
            [layer.data_ptr() for layer in paged_layers_mla],
            dtype=torch.uint64,
            device=device,
        ),
        d2h_chunks_mla if use_tensor_list else [c.data_ptr() for c in d2h_chunks_mla],
        torch.tensor(block_ids, dtype=torch.int64, device=device),
        torch.device(device),
        ops.TransferDirection.D2H,
        shape_desc_mla,
        chunk_tokens,
        engine_kv_format_mla,
        0,
    )
    paged_h2d_mla = [torch.zeros_like(layer) for layer in paged_layers_mla]
    ops.multi_layer_block_kv_transfer(
        paged_h2d_mla
        if use_tensor_list
        else torch.tensor(
            [layer.data_ptr() for layer in paged_h2d_mla],
            dtype=torch.uint64,
            device=device,
        ),
        d2h_chunks_mla if use_tensor_list else [c.data_ptr() for c in d2h_chunks_mla],
        torch.tensor(block_ids, dtype=torch.int64, device=device),
        torch.device(device),
        ops.TransferDirection.H2D,
        shape_desc_mla,
        chunk_tokens,
        engine_kv_format_mla,
        0,
    )
    for i in range(num_layers):
        orig = paged_layers_mla[i].cpu()
        recon = paged_h2d_mla[i].cpu()
        results[f"mla_layer{i}"] = torch.stack([orig, recon])
        assert torch.allclose(orig, recon, atol=1e-6), (
            f"MLA Layer {i} round-trip mismatch"
        )

    # --- SGLang MLA flat per-layer ---
    torch.manual_seed(890)
    paged_layers_sglang_mla = [
        torch.randn(num_blocks * block_size, 1, mla_hidden, dtype=dtype).to(device)
        for _ in range(num_layers)
    ]
    engine_kv_format_sglang_mla = ops.EngineKVFormat.NL_X_NBBS_ONE_HS
    d2h_chunks_sglang_mla = _alloc_chunks(
        (num_layers, chunk_tokens, mla_hidden), num_chunks
    )
    ops.multi_layer_block_kv_transfer(
        paged_layers_sglang_mla
        if use_tensor_list
        else torch.tensor(
            [layer.data_ptr() for layer in paged_layers_sglang_mla],
            dtype=torch.uint64,
            device=device,
        ),
        d2h_chunks_sglang_mla
        if use_tensor_list
        else [c.data_ptr() for c in d2h_chunks_sglang_mla],
        torch.tensor(block_ids, dtype=torch.int64, device=device),
        torch.device(device),
        ops.TransferDirection.D2H,
        shape_desc_mla,
        chunk_tokens,
        engine_kv_format_sglang_mla,
        0,
    )
    paged_h2d_sglang_mla = [
        torch.zeros_like(layer) for layer in paged_layers_sglang_mla
    ]
    ops.multi_layer_block_kv_transfer(
        paged_h2d_sglang_mla
        if use_tensor_list
        else torch.tensor(
            [layer.data_ptr() for layer in paged_h2d_sglang_mla],
            dtype=torch.uint64,
            device=device,
        ),
        d2h_chunks_sglang_mla
        if use_tensor_list
        else [c.data_ptr() for c in d2h_chunks_sglang_mla],
        torch.tensor(block_ids, dtype=torch.int64, device=device),
        torch.device(device),
        ops.TransferDirection.H2D,
        shape_desc_mla,
        chunk_tokens,
        engine_kv_format_sglang_mla,
        0,
    )
    for i in range(num_layers):
        orig = paged_layers_sglang_mla[i].cpu()
        recon = paged_h2d_sglang_mla[i].cpu()
        results[f"sglang_mla_layer{i}"] = torch.stack([orig, recon])
        assert torch.allclose(orig, recon, atol=1e-6), (
            f"SGLang MLA Layer {i} round-trip mismatch"
        )

    # --- Cross-layer NHD ---
    torch.manual_seed(101)
    paged_cross_nhd = torch.randn(
        num_blocks,
        num_layers,
        2,
        block_size,
        num_heads,
        head_size,
        dtype=dtype,
    ).to(device)
    engine_kv_format_cross_nhd = ops.EngineKVFormat.NB_NL_TWO_BS_NH_HS
    d2h_chunks_cross_nhd = _alloc_chunks(
        (2, num_layers, chunk_tokens, hidden_dim), num_chunks
    )
    ops.multi_layer_block_kv_transfer(
        paged_cross_nhd
        if use_tensor_list
        else torch.tensor(
            [paged_cross_nhd.data_ptr()], dtype=torch.uint64, device=device
        ),
        d2h_chunks_cross_nhd
        if use_tensor_list
        else [c.data_ptr() for c in d2h_chunks_cross_nhd],
        torch.tensor(block_ids, dtype=torch.int64, device=device),
        torch.device(device),
        ops.TransferDirection.D2H,
        shape_desc,
        chunk_tokens,
        engine_kv_format_cross_nhd,
        0,
    )
    paged_h2d_cross_nhd = torch.zeros_like(paged_cross_nhd)
    ops.multi_layer_block_kv_transfer(
        paged_h2d_cross_nhd
        if use_tensor_list
        else torch.tensor(
            [paged_h2d_cross_nhd.data_ptr()], dtype=torch.uint64, device=device
        ),
        d2h_chunks_cross_nhd
        if use_tensor_list
        else [c.data_ptr() for c in d2h_chunks_cross_nhd],
        torch.tensor(block_ids, dtype=torch.int64, device=device),
        torch.device(device),
        ops.TransferDirection.H2D,
        shape_desc,
        chunk_tokens,
        engine_kv_format_cross_nhd,
        0,
    )
    orig = paged_cross_nhd.cpu()
    recon = paged_h2d_cross_nhd.cpu()
    results["cross_nhd"] = torch.stack([orig, recon])
    assert torch.allclose(orig, recon, atol=1e-6), "Cross-layer NHD mismatch"

    # --- Cross-layer HND ---
    torch.manual_seed(202)
    paged_cross_hnd = torch.randn(
        num_blocks,
        num_layers,
        2,
        num_heads,
        block_size,
        head_size,
        dtype=dtype,
    ).to(device)
    engine_kv_format_cross_hnd = ops.EngineKVFormat.NB_NL_TWO_NH_BS_HS
    d2h_chunks_cross_hnd = _alloc_chunks(
        (2, num_layers, chunk_tokens, hidden_dim), num_chunks
    )
    ops.multi_layer_block_kv_transfer(
        paged_cross_hnd
        if use_tensor_list
        else torch.tensor(
            [paged_cross_hnd.data_ptr()], dtype=torch.uint64, device=device
        ),
        d2h_chunks_cross_hnd
        if use_tensor_list
        else [c.data_ptr() for c in d2h_chunks_cross_hnd],
        torch.tensor(block_ids, dtype=torch.int64, device=device),
        torch.device(device),
        ops.TransferDirection.D2H,
        shape_desc,
        chunk_tokens,
        engine_kv_format_cross_hnd,
        0,
    )
    paged_h2d_cross_hnd = torch.zeros_like(paged_cross_hnd)
    ops.multi_layer_block_kv_transfer(
        paged_h2d_cross_hnd
        if use_tensor_list
        else torch.tensor(
            [paged_h2d_cross_hnd.data_ptr()], dtype=torch.uint64, device=device
        ),
        d2h_chunks_cross_hnd
        if use_tensor_list
        else [c.data_ptr() for c in d2h_chunks_cross_hnd],
        torch.tensor(block_ids, dtype=torch.int64, device=device),
        torch.device(device),
        ops.TransferDirection.H2D,
        shape_desc,
        chunk_tokens,
        engine_kv_format_cross_hnd,
        0,
    )
    orig = paged_cross_hnd.cpu()
    recon = paged_h2d_cross_hnd.cpu()
    results["cross_hnd"] = torch.stack([orig, recon])
    assert torch.allclose(orig, recon, atol=1e-6), "Cross-layer HND mismatch"

    # --- SGLang MHA flat (NBBS) ---
    torch.manual_seed(303)
    pbs = num_blocks * block_size
    paged_sglang_nbbs = [
        [
            torch.randn(pbs, num_heads, head_size, dtype=dtype).to(device)
            for _ in range(num_layers)
        ],
        [
            torch.randn(pbs, num_heads, head_size, dtype=dtype).to(device)
            for _ in range(num_layers)
        ],
    ]
    engine_kv_format_sglang_nbbs = ops.EngineKVFormat.TWO_X_NL_X_NBBS_NH_HS
    d2h_chunks_sglang_nbbs = _alloc_chunks(
        (2, num_layers, chunk_tokens, hidden_dim), num_chunks
    )
    ops.multi_layer_block_kv_transfer(
        paged_sglang_nbbs
        if use_tensor_list
        else torch.tensor(
            [t.data_ptr() for t in paged_sglang_nbbs[0]]
            + [t.data_ptr() for t in paged_sglang_nbbs[1]],
            dtype=torch.uint64,
            device=device,
        ),
        d2h_chunks_sglang_nbbs
        if use_tensor_list
        else [c.data_ptr() for c in d2h_chunks_sglang_nbbs],
        torch.tensor(block_ids, dtype=torch.int64, device=device),
        torch.device(device),
        ops.TransferDirection.D2H,
        shape_desc,
        chunk_tokens,
        engine_kv_format_sglang_nbbs,
        0,
    )
    paged_h2d_sglang_nbbs = [
        [torch.zeros_like(t) for t in group] for group in paged_sglang_nbbs
    ]
    ops.multi_layer_block_kv_transfer(
        paged_h2d_sglang_nbbs
        if use_tensor_list
        else torch.tensor(
            [t.data_ptr() for t in paged_h2d_sglang_nbbs[0]]
            + [t.data_ptr() for t in paged_h2d_sglang_nbbs[1]],
            dtype=torch.uint64,
            device=device,
        ),
        d2h_chunks_sglang_nbbs
        if use_tensor_list
        else [c.data_ptr() for c in d2h_chunks_sglang_nbbs],
        torch.tensor(block_ids, dtype=torch.int64, device=device),
        torch.device(device),
        ops.TransferDirection.H2D,
        shape_desc,
        chunk_tokens,
        engine_kv_format_sglang_nbbs,
        0,
    )
    for kv in range(2):
        for i in range(num_layers):
            orig = paged_sglang_nbbs[kv][i].cpu()
            recon = paged_h2d_sglang_nbbs[kv][i].cpu()
            results[f"sglang_nbbs_kv{kv}_l{i}"] = torch.stack([orig, recon])
            assert torch.allclose(orig, recon, atol=1e-6), (
                f"SGLang NBBS kv={kv} layer={i} mismatch"
            )

    # --- SGLang MHA block (NB_BS) ---
    torch.manual_seed(404)
    paged_sglang_nb = [
        [
            torch.randn(num_blocks, block_size, num_heads, head_size, dtype=dtype).to(
                device
            )
            for _ in range(num_layers)
        ],
        [
            torch.randn(num_blocks, block_size, num_heads, head_size, dtype=dtype).to(
                device
            )
            for _ in range(num_layers)
        ],
    ]
    engine_kv_format_sglang_nb = ops.EngineKVFormat.TWO_X_NL_X_NB_BS_NH_HS
    d2h_chunks_sglang_nb = _alloc_chunks(
        (2, num_layers, chunk_tokens, hidden_dim), num_chunks
    )
    ops.multi_layer_block_kv_transfer(
        paged_sglang_nb
        if use_tensor_list
        else torch.tensor(
            [t.data_ptr() for t in paged_sglang_nb[0]]
            + [t.data_ptr() for t in paged_sglang_nb[1]],
            dtype=torch.uint64,
            device=device,
        ),
        d2h_chunks_sglang_nb
        if use_tensor_list
        else [c.data_ptr() for c in d2h_chunks_sglang_nb],
        torch.tensor(block_ids, dtype=torch.int64, device=device),
        torch.device(device),
        ops.TransferDirection.D2H,
        shape_desc,
        chunk_tokens,
        engine_kv_format_sglang_nb,
        0,
    )
    paged_h2d_sglang_nb = [
        [torch.zeros_like(t) for t in group] for group in paged_sglang_nb
    ]
    ops.multi_layer_block_kv_transfer(
        paged_h2d_sglang_nb
        if use_tensor_list
        else torch.tensor(
            [t.data_ptr() for t in paged_h2d_sglang_nb[0]]
            + [t.data_ptr() for t in paged_h2d_sglang_nb[1]],
            dtype=torch.uint64,
            device=device,
        ),
        d2h_chunks_sglang_nb
        if use_tensor_list
        else [c.data_ptr() for c in d2h_chunks_sglang_nb],
        torch.tensor(block_ids, dtype=torch.int64, device=device),
        torch.device(device),
        ops.TransferDirection.H2D,
        shape_desc,
        chunk_tokens,
        engine_kv_format_sglang_nb,
        0,
    )
    for kv in range(2):
        for i in range(num_layers):
            orig = paged_sglang_nb[kv][i].cpu()
            recon = paged_h2d_sglang_nb[kv][i].cpu()
            results[f"sglang_nb_kv{kv}_l{i}"] = torch.stack([orig, recon])
            assert torch.allclose(orig, recon, atol=1e-6), (
                f"SGLang NB kv={kv} layer={i} mismatch"
            )

    # --- skip_prefix_n_blocks > 0 ---
    torch.manual_seed(505)
    skip_n = 2
    paged_layers_skip = [
        torch.randn(2, num_blocks, block_size, num_heads, head_size, dtype=dtype).to(
            device
        )
        for _ in range(num_layers)
    ]
    engine_kv_format_nhd = ops.EngineKVFormat.NL_X_TWO_NB_BS_NH_HS
    # With skip=2, effective blocks start at index 2.
    # Object 0 occupies flat indices [0, blocks_per_chunk), skipping first 2.
    # Object 1 occupies flat indices [blocks_per_chunk, 2*blocks_per_chunk).
    d2h_chunks_skip = _alloc_chunks(
        (2, num_layers, chunk_tokens, hidden_dim), num_chunks
    )
    ops.multi_layer_block_kv_transfer(
        paged_layers_skip
        if use_tensor_list
        else torch.tensor(
            [layer.data_ptr() for layer in paged_layers_skip],
            dtype=torch.uint64,
            device=device,
        ),
        d2h_chunks_skip if use_tensor_list else [c.data_ptr() for c in d2h_chunks_skip],
        torch.tensor(block_ids, dtype=torch.int64, device=device),
        torch.device(device),
        ops.TransferDirection.D2H,
        shape_desc,
        chunk_tokens,
        engine_kv_format_nhd,
        skip_n,
    )
    paged_h2d_skip = [torch.zeros_like(layer) for layer in paged_layers_skip]
    ops.multi_layer_block_kv_transfer(
        paged_h2d_skip
        if use_tensor_list
        else torch.tensor(
            [layer.data_ptr() for layer in paged_h2d_skip],
            dtype=torch.uint64,
            device=device,
        ),
        d2h_chunks_skip if use_tensor_list else [c.data_ptr() for c in d2h_chunks_skip],
        torch.tensor(block_ids, dtype=torch.int64, device=device),
        torch.device(device),
        ops.TransferDirection.H2D,
        shape_desc,
        chunk_tokens,
        engine_kv_format_nhd,
        skip_n,
    )
    # Blocks [skip_n:num_blocks] should round-trip at their original positions
    for layer_idx in range(num_layers):
        for kv in range(2):
            orig_blocks = paged_layers_skip[layer_idx][kv, skip_n:num_blocks]
            recon_blocks = paged_h2d_skip[layer_idx][kv, skip_n:num_blocks]
            key = f"skip_l{layer_idx}_kv{kv}"
            results[key] = torch.stack([orig_blocks.cpu(), recon_blocks.cpu()])
            assert torch.allclose(orig_blocks, recon_blocks, atol=1e-6), (
                f"Skip mismatch at layer={layer_idx} kv={kv}"
            )
    # Skipped blocks [0:skip_n] should remain zero
    for layer_idx in range(num_layers):
        for kv in range(2):
            skipped = paged_h2d_skip[layer_idx][kv, :skip_n]
            assert torch.all(skipped == 0), (
                f"Skipped blocks not zero at layer={layer_idx} kv={kv}"
            )
    assert torch.all(d2h_chunks_skip[0][:, :, : skip_n * block_size] == 0), (
        "Skipped D2H chunk region should remain zero"
    )

    # --- Non-sequential block_ids ---
    torch.manual_seed(515)
    paged_layers_permuted = [
        torch.randn(2, num_blocks, block_size, num_heads, head_size, dtype=dtype).to(
            device
        )
        for _ in range(num_layers)
    ]
    generator = torch.Generator(device="cpu").manual_seed(616)
    permuted_block_ids = torch.randperm(num_blocks, generator=generator).tolist()
    d2h_chunks_permuted = _alloc_chunks(
        (2, num_layers, chunk_tokens, hidden_dim), num_chunks
    )
    ops.multi_layer_block_kv_transfer(
        paged_layers_permuted
        if use_tensor_list
        else torch.tensor(
            [layer.data_ptr() for layer in paged_layers_permuted],
            dtype=torch.uint64,
            device=device,
        ),
        d2h_chunks_permuted
        if use_tensor_list
        else [c.data_ptr() for c in d2h_chunks_permuted],
        torch.tensor(permuted_block_ids, dtype=torch.int64, device=device),
        torch.device(device),
        ops.TransferDirection.D2H,
        shape_desc,
        chunk_tokens,
        engine_kv_format_nhd,
        0,
    )
    paged_h2d_permuted = [torch.zeros_like(layer) for layer in paged_layers_permuted]
    ops.multi_layer_block_kv_transfer(
        paged_h2d_permuted
        if use_tensor_list
        else torch.tensor(
            [layer.data_ptr() for layer in paged_h2d_permuted],
            dtype=torch.uint64,
            device=device,
        ),
        d2h_chunks_permuted
        if use_tensor_list
        else [c.data_ptr() for c in d2h_chunks_permuted],
        torch.tensor(permuted_block_ids, dtype=torch.int64, device=device),
        torch.device(device),
        ops.TransferDirection.H2D,
        shape_desc,
        chunk_tokens,
        engine_kv_format_nhd,
        0,
    )
    for i in range(num_layers):
        orig = paged_layers_permuted[i].cpu()
        recon = paged_h2d_permuted[i].cpu()
        results[f"permuted_nhd_layer{i}"] = torch.stack([orig, recon])
        assert torch.allclose(orig, recon, atol=1e-6), (
            f"Permuted NHD Layer {i} round-trip mismatch"
        )

    # --- Multi-chunk (num_blocks > blocks_per_chunk) ---
    torch.manual_seed(606)
    num_blocks_mc = 12
    paged_layers_mc = [
        torch.randn(2, num_blocks_mc, block_size, num_heads, head_size, dtype=dtype).to(
            device
        )
        for _ in range(num_layers)
    ]
    shape_desc_mc = ops.PageBufferShapeDesc()
    shape_desc_mc.nl = num_layers
    shape_desc_mc.nb = num_blocks_mc
    shape_desc_mc.bs = block_size
    shape_desc_mc.nh = num_heads
    shape_desc_mc.hs = head_size
    shape_desc_mc.element_size = dtype.itemsize
    shape_desc_mc.kv_size = 2
    num_chunks_mc = num_blocks_mc // blocks_per_chunk  # 3 chunks
    d2h_chunks_mc = _alloc_chunks(
        (2, num_layers, chunk_tokens, hidden_dim), num_chunks_mc
    )
    block_ids_mc = list(range(num_blocks_mc))
    ops.multi_layer_block_kv_transfer(
        paged_layers_mc
        if use_tensor_list
        else torch.tensor(
            [layer.data_ptr() for layer in paged_layers_mc],
            dtype=torch.uint64,
            device=device,
        ),
        d2h_chunks_mc if use_tensor_list else [c.data_ptr() for c in d2h_chunks_mc],
        torch.tensor(block_ids_mc, dtype=torch.int64, device=device),
        torch.device(device),
        ops.TransferDirection.D2H,
        shape_desc_mc,
        chunk_tokens,
        engine_kv_format_nhd,
        0,
    )
    paged_h2d_mc = [torch.zeros_like(layer) for layer in paged_layers_mc]
    ops.multi_layer_block_kv_transfer(
        paged_h2d_mc
        if use_tensor_list
        else torch.tensor(
            [layer.data_ptr() for layer in paged_h2d_mc],
            dtype=torch.uint64,
            device=device,
        ),
        d2h_chunks_mc if use_tensor_list else [c.data_ptr() for c in d2h_chunks_mc],
        torch.tensor(block_ids_mc, dtype=torch.int64, device=device),
        torch.device(device),
        ops.TransferDirection.H2D,
        shape_desc_mc,
        chunk_tokens,
        engine_kv_format_nhd,
        0,
    )
    for i in range(num_layers):
        orig = paged_layers_mc[i].cpu()
        recon = paged_h2d_mc[i].cpu()
        results[f"multi_chunk_l{i}"] = torch.stack([orig, recon])
        assert torch.allclose(orig, recon, atol=1e-6), (
            f"Multi-chunk Layer {i} round-trip mismatch"
        )

    # --- Fused-K/V blocks-first (HND and NHD) ---
    # kv_size == 1 and hs == 2 * head_size: the K/V pair stays packed inside
    # each head copy, and the object is a single plane [NL, T, NH * 2 * HS].
    torch.manual_seed(707)
    fused_hs = 2 * head_size
    shape_desc_fused = ops.PageBufferShapeDesc()
    shape_desc_fused.nl = num_layers
    shape_desc_fused.nb = num_blocks
    shape_desc_fused.bs = block_size
    shape_desc_fused.nh = num_heads
    shape_desc_fused.hs = fused_hs
    shape_desc_fused.element_size = dtype.itemsize
    shape_desc_fused.kv_size = 1
    for fused_key, fused_fmt, fused_shape in (
        (
            "fused_hnd",
            ops.EngineKVFormat.NL_X_NB_NH_BS_TWO_HS,
            (num_blocks, num_heads, block_size, fused_hs),
        ),
        (
            "fused_nhd",
            ops.EngineKVFormat.NL_X_NB_BS_NH_TWO_HS,
            (num_blocks, block_size, num_heads, fused_hs),
        ),
        (
            "cs_hnd",
            ops.EngineKVFormat.NL_X_NB_NH_BS_CS,
            (num_blocks, num_heads, block_size, fused_hs),
        ),
        (
            "cs_nhd",
            ops.EngineKVFormat.NL_X_NB_BS_NH_CS,
            (num_blocks, block_size, num_heads, fused_hs),
        ),
    ):
        paged_fused = [
            torch.randn(*fused_shape, dtype=dtype).to(device) for _ in range(num_layers)
        ]
        d2h_chunks_fused = _alloc_chunks(
            (num_layers, chunk_tokens, num_heads * fused_hs), num_chunks
        )
        ops.multi_layer_block_kv_transfer(
            paged_fused
            if use_tensor_list
            else torch.tensor(
                [layer.data_ptr() for layer in paged_fused],
                dtype=torch.uint64,
                device=device,
            ),
            d2h_chunks_fused
            if use_tensor_list
            else [c.data_ptr() for c in d2h_chunks_fused],
            torch.tensor(block_ids, dtype=torch.int64, device=device),
            torch.device(device),
            ops.TransferDirection.D2H,
            shape_desc_fused,
            chunk_tokens,
            fused_fmt,
            0,
        )
        paged_fused_h2d = [torch.zeros_like(layer) for layer in paged_fused]
        ops.multi_layer_block_kv_transfer(
            paged_fused_h2d
            if use_tensor_list
            else torch.tensor(
                [layer.data_ptr() for layer in paged_fused_h2d],
                dtype=torch.uint64,
                device=device,
            ),
            d2h_chunks_fused
            if use_tensor_list
            else [c.data_ptr() for c in d2h_chunks_fused],
            torch.tensor(block_ids, dtype=torch.int64, device=device),
            torch.device(device),
            ops.TransferDirection.H2D,
            shape_desc_fused,
            chunk_tokens,
            fused_fmt,
            0,
        )
        for i in range(num_layers):
            orig = paged_fused[i].cpu()
            recon = paged_fused_h2d[i].cpu()
            results[f"{fused_key}_l{i}"] = torch.stack([orig, recon])
            assert torch.allclose(orig, recon, atol=1e-6), (
                f"{fused_key} layer {i} round-trip mismatch"
            )
        # Pin the object byte layout itself: the fallback and the device
        # kernel must write bit-identical chunks or caches would not be
        # portable across backends. (The .cpu() reads above synced the
        # stream, so the pinned chunks are safe to snapshot here.)
        for c_idx, c in enumerate(d2h_chunks_fused):
            results[f"{fused_key}_chunk{c_idx}"] = c.clone()

    return results


def scenario_record_drain_event(ops: Any, device: str) -> dict[str, torch.Tensor]:
    """Test record_event_on_stream / drain_recorded_events contracts.

    Verified backend-agnostic: native c_ops uses cudaLaunchHostFunc on the
    default stream (ptr=0), which fires synchronously after device_sync; the
    fallback enqueues immediately with time.time(). Both paths satisfy every
    assertion below.
    """
    ops.drain_recorded_events()  # clear residual global state

    assert ops.drain_recorded_events() == []

    ops.record_event_on_stream(
        0, "mp.store.start", "sess-1", {"device": "cuda:0"}, {"count": 10}
    )
    ops.record_event_on_stream(
        0, "mp.store.end", "sess-1", {"device": "cuda:0"}, {"count": 10}
    )
    device_sync(device)
    result = ops.drain_recorded_events()
    assert len(result) == 2

    # Each element is (event_type_name, session_id, timestamp, str_meta, int_meta)
    name0, sid0, ts0, str_meta0, int_meta0 = result[0]
    name1, sid1, ts1, str_meta1, int_meta1 = result[1]

    assert name0 == "mp.store.start"
    assert name1 == "mp.store.end"
    assert sid0 == "sess-1"
    assert sid1 == "sess-1"
    assert ts0 > 0.0
    assert ts1 >= ts0  # timestamps must be monotonically non-decreasing
    assert str_meta0 == {"device": "cuda:0"}
    assert int_meta0 == {"count": 10}
    assert str_meta1 == {"device": "cuda:0"}
    assert int_meta1 == {"count": 10}

    # Drain clears the buffer
    assert ops.drain_recorded_events() == []

    # Multiple records on the default stream (ptr=0) must all enqueue.
    for i in range(5):
        ops.record_event_on_stream(0, f"mp.test.{i}", f"sess-{i}", {}, {"idx": i})
    device_sync(device)
    events = ops.drain_recorded_events()
    assert len(events) == 5
    for i, (name, sid, ts, str_meta, int_meta) in enumerate(events):
        assert name == f"mp.test.{i}"
        assert sid == f"sess-{i}"
        assert int_meta == {"idx": i}
        assert ts > 0.0

    return {"record_drain_event": torch.tensor([1], dtype=torch.int32)}


# ==========================================
# 3. Registry
# ==========================================

# cover pybind list in csrc/pybind.cpp
SCENARIO_REGISTRY = {
    "transfer_direction_enum": scenario_transfer_direction_enum,
    "multi_layer_kv_transfer": scenario_multi_layer_kv_transfer,
    "multi_layer_kv_transfer_unilateral": scenario_multi_layer_kv_transfer_unilateral,
    "multi_layer_block_kv_transfer": scenario_multi_layer_block_kv_transfer,
    "single_layer_kv_transfer": scenario_single_layer_kv_transfer,
    "single_layer_kv_transfer_sgl": scenario_single_layer_kv_transfer_sgl,
    "load_and_reshape_flash": scenario_load_and_reshape_flash,
    "reshape_and_cache_back_flash": scenario_reshape_and_cache_back_flash,
    "lmcache_memcpy_async": scenario_lmcache_memcpy_async,
    "encode_fast_new": scenario_encode_fast_new,
    "decode_fast_new": scenario_decode_fast_new,
    "decode_fast_prefsum": scenario_decode_fast_prefsum,
    "calculate_cdf": scenario_calculate_cdf,
    "rotary_embedding_k_fused": scenario_rotary_embedding_k_fused,
    "alloc_free_pinned_ptr": scenario_alloc_free_pinned_ptr,
    "alloc_free_pinned_numa_ptr": scenario_alloc_free_pinned_numa_ptr,
    "alloc_free_numa_ptr": scenario_alloc_free_numa_ptr,
    "alloc_free_shm_pinned_ptr": scenario_alloc_free_shm_pinned_ptr,
    "get_gpu_pci_bus_id": scenario_get_gpu_pci_bus_id,
    "record_drain_completion": scenario_record_drain_completion,
    "dispatcher_integration": scenario_dispatcher_integration,
    "record_drain_event": scenario_record_drain_event,
}


# ==========================================
# 4. Test functions pytest sees
# ==========================================


class TestScenarios:
    """Test class to ensure test_scenario runs before test_compare.

    **Execution Order Guarantee:**
    By grouping tests in a class and using alphabetically ordered method names
    (test_1_scenario, test_2_compare), pytest will execute all scenario tests
    before any compare tests, ensuring _results dict is fully populated.

    """

    @pytest.mark.parametrize("name,fn", list(SCENARIO_REGISTRY.items()))
    def test_1_scenario(
        self,
        backend: tuple,
        name: str,
        fn: Any,
    ) -> None:
        """Run a single scenario with a specific backend configuration.

        Each (scenario, backend) pair is a separate pytest test case, giving
        per-scenario per-backend failure reporting without subprocess indirection.

        Results are stored in the module-level _results dict for later comparison
        by test_2_compare.
        """
        backend_id, ops, device = backend
        result = fn(ops, device)
        if result is not None:
            _results[(name, backend_id)] = result

    @pytest.mark.parametrize("name", list(SCENARIO_REGISTRY.keys()))
    def test_2_compare(self, name: str) -> None:
        """Compare results across backends for a single scenario.

        When multiple backends ran (CUDA available), asserts that python
        fallback equivalents produce numerically identical results to cuda_ops.
        When only one backend ran (no CUDA), simply verifies results were stored.

        This test runs after test_1_scenario due to alphabetical ordering of
        method names within the TestScenarios class.
        """
        available = [p.values[0][0] for p in _BACKEND_PARAMS]  # list of backend_ids
        backend_results = {
            bid: _results[(name, bid)] for bid in available if (name, bid) in _results
        }

        assert len(backend_results) >= 1, (
            f"{name}: no results collected — were all test_1_scenario tests skipped?"
        )

        if len(backend_results) < 2:
            # Only one backend available (no CUDA); just verify results exist
            return

        # Collect all result keys across all backends for this scenario
        all_keys: set[str] = set()
        for res in backend_results.values():
            all_keys.update(res.keys())

        base_bid = next(iter(backend_results))  # first available backend as reference

        for key in sorted(all_keys):
            base_val = backend_results[base_bid].get(key)
            if base_val is None:
                continue

            for bid, res in backend_results.items():
                if bid == base_bid:
                    continue
                val = res.get(key)
                if val is None:
                    pytest.fail(
                        f"{name}/{key}: backend '{bid}' has no result "
                        f"(reference backend '{base_bid}' does)"
                    )
                    continue

                if isinstance(val, torch.Tensor):
                    v_current = val.detach().cpu().float()
                    v_base = base_val.detach().cpu().float()
                    if not torch.allclose(v_current, v_base, rtol=1e-4, atol=1e-4):
                        max_diff = (v_current - v_base).abs().max().item()
                        pytest.fail(
                            f"{name}/{key}: '{bid}' vs '{base_bid}' mismatch, "
                            f"max diff = {max_diff:.2e}"
                        )
                else:
                    if val != base_val:
                        pytest.fail(
                            f"{name}/{key}: '{bid}'={val} != '{base_bid}'={base_val}"
                        )


# ==========================================
# Allocation page alignment
# ==========================================
#
# Rust raw-block backend with O_DIRECT requires page-aligned buffer
# pointers; CUDA path gets this for free via cudaHostAlloc, and the
# non-CUDA fallback in lmcache.non_cuda_equivalents shall mirror the same
# guarantee.

_PINNED_ALLOC_SIZES = [1, 4095, 4096, 8192, 1024 * 1024]


@pytest.mark.parametrize("size", _PINNED_ALLOC_SIZES)
def test_alloc_pinned_ptr_is_page_aligned(size: int) -> None:
    page_size = os.sysconf("SC_PAGESIZE")
    ptr = _py_ops.alloc_pinned_ptr(size)
    try:
        assert ptr != 0
        if ptr % page_size != 0:
            raise AssertionError(
                f"alloc_pinned_ptr({size}) returned non-page-aligned ptr "
                f"{hex(ptr)} (page size {page_size})"
            )
        # Touch every byte in the requested region through the raw pointer
        # to confirm the registered view covers the full requested size
        # (an undersized view would corrupt adjacent memory or segfault).
        buf = (ctypes.c_uint8 * size).from_address(ptr)
        for i in range(size):
            buf[i] = (i & 0xFF) ^ 0xA5
        for i in range(size):
            assert buf[i] == ((i & 0xFF) ^ 0xA5)
    finally:
        _py_ops.free_pinned_ptr(ptr)
