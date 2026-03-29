"""Debug script to isolate fused kernel issues."""
import math
import torch

# Test the encode kernel's packing vs reference packing
NUM_TOKENS = 4
NUM_KV_HEADS = 4
HEAD_SIZE = 128
OUTLIER_FRACTION = 0.15
BIT_WIDTH = 4
BLOCK_SIZE = 16
NUM_BLOCKS = 32

def _next_power_of_2(n):
    if n <= 0:
        return 1
    return 1 << (n - 1).bit_length()

def setup():
    device = torch.device("cuda")
    n_outliers = max(1, int(HEAD_SIZE * OUTLIER_FRACTION))
    normal_size = HEAD_SIZE - n_outliers
    padded_d = _next_power_of_2(normal_size)

    gen = torch.Generator(device="cpu").manual_seed(42)
    sign_flips = torch.where(
        torch.rand(padded_d, generator=gen) > 0.5,
        torch.ones(padded_d), -torch.ones(padded_d),
    ).to(device=device, dtype=torch.float32)

    num_centroids = 2**BIT_WIDTH
    codebook = torch.linspace(-1.5, 1.5, num_centroids, device=device)
    boundaries = (codebook[:-1] + codebook[1:]) / 2

    outlier_idx = torch.arange(n_outliers, dtype=torch.long, device=device)
    normal_idx = torch.arange(n_outliers, HEAD_SIZE, dtype=torch.long, device=device)
    return {
        "sign_flips": sign_flips, "codebook": codebook,
        "boundaries": boundaries, "outlier_idx": outlier_idx,
        "normal_idx": normal_idx, "normal_size": normal_size,
        "n_outliers": n_outliers, "padded_d": padded_d,
    }

def test_encode_packing():
    """Test: does the fused kernel produce the same packed bytes?"""
    device = torch.device("cuda")
    state = setup()
    normal_size = state["normal_size"]
    packed_bytes = math.ceil(normal_size * BIT_WIDTH / 8)
    n_outliers = state["n_outliers"]
    outlier_u8_count = n_outliers * 2
    slot_bytes = outlier_u8_count + packed_bytes + 2

    x = torch.randn(NUM_TOKENS, NUM_KV_HEADS, HEAD_SIZE, device=device, dtype=torch.bfloat16)
    normal_x = x[..., state["normal_idx"]].contiguous()
    outlier_x = x[..., state["outlier_idx"]]

    # Reference encode
    from vllm.v1.attention.ops.triton_hadamard_turboquant import hadamard_turboquant_encode
    ref_indices, ref_norms = hadamard_turboquant_encode(
        normal_x, state["sign_flips"], state["codebook"], state["boundaries"],
    )

    # Reference pack
    flat_idx = ref_indices.reshape(-1, normal_size)
    N = flat_idx.shape[0]
    if normal_size % 2 != 0:
        flat_idx_padded = torch.nn.functional.pad(flat_idx, (0, 1), value=0)
    else:
        flat_idx_padded = flat_idx
    ref_packed = flat_idx_padded[:, 0::2] | (flat_idx_padded[:, 1::2] << 4)
    ref_packed = ref_packed[:, :packed_bytes]

    # Fused encode
    from vllm.v1.attention.ops.triton_fused_turboquant import fused_hadamard_encode_and_store
    cache = torch.zeros(NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, slot_bytes, dtype=torch.uint8, device=device)
    slot_mapping = torch.arange(NUM_TOKENS, device=device)
    block_indices = slot_mapping // BLOCK_SIZE
    block_offsets = slot_mapping % BLOCK_SIZE

    fused_hadamard_encode_and_store(
        normal_x=normal_x, outlier_x=outlier_x,
        sign_flips=state["sign_flips"], boundaries=state["boundaries"],
        cache=cache, block_indices=block_indices, block_offsets=block_offsets,
        bit_width=BIT_WIDTH,
    )

    # Extract packed bytes from cache
    fused_packed_list = []
    for t in range(NUM_TOKENS):
        bi = block_indices[t].item()
        bo = block_offsets[t].item()
        for h in range(NUM_KV_HEADS):
            slot = cache[bi, bo, h]
            fused_packed_list.append(slot[outlier_u8_count:outlier_u8_count+packed_bytes])
    fused_packed = torch.stack(fused_packed_list)

    # Compare packed bytes
    diff = (ref_packed.to(torch.uint8) != fused_packed)
    if diff.any():
        bad_rows = diff.any(dim=1).nonzero().flatten()
        print(f"PACKED MISMATCH in rows: {bad_rows.tolist()}")
        for r in bad_rows[:3]:
            bad_cols = diff[r].nonzero().flatten()[:5]
            print(f"  Row {r.item()}, cols {bad_cols.tolist()}:")
            print(f"    ref:   {ref_packed[r, bad_cols]}")
            print(f"    fused: {fused_packed[r, bad_cols]}")
            # Show the source indices
            print(f"    ref indices[{bad_cols[0]*2}:{bad_cols[0]*2+2}]: "
                  f"{flat_idx[r, bad_cols[0]*2:bad_cols[0]*2+2]}")
    else:
        print("PACKED BYTES MATCH!")

    # Also compare norms
    fused_norm_list = []
    for t in range(NUM_TOKENS):
        bi = block_indices[t].item()
        bo = block_offsets[t].item()
        for h in range(NUM_KV_HEADS):
            slot = cache[bi, bo, h]
            norm_bytes = slot[outlier_u8_count+packed_bytes:outlier_u8_count+packed_bytes+2]
            fused_norm_list.append(norm_bytes)
    fused_norm_bytes = torch.stack(fused_norm_list)
    ref_norm_bytes = ref_norms.reshape(N).to(torch.float16).view(torch.uint8).reshape(N, 2)

    norm_diff = (ref_norm_bytes != fused_norm_bytes)
    if norm_diff.any():
        bad_rows = norm_diff.any(dim=1).nonzero().flatten()
        print(f"NORM MISMATCH in rows: {bad_rows.tolist()}")
        for r in bad_rows[:3]:
            print(f"  Row {r.item()}: ref={ref_norm_bytes[r]}, fused={fused_norm_bytes[r]}")
    else:
        print("NORM BYTES MATCH!")

    # Compare outlier bytes
    fused_outlier_list = []
    for t in range(NUM_TOKENS):
        bi = block_indices[t].item()
        bo = block_offsets[t].item()
        for h in range(NUM_KV_HEADS):
            slot = cache[bi, bo, h]
            fused_outlier_list.append(slot[:outlier_u8_count])
    fused_outlier = torch.stack(fused_outlier_list)
    ref_outlier = outlier_x.reshape(N, n_outliers).to(torch.bfloat16).view(torch.uint8).reshape(N, outlier_u8_count)

    outlier_diff = (ref_outlier != fused_outlier)
    if outlier_diff.any():
        bad_rows = outlier_diff.any(dim=1).nonzero().flatten()
        print(f"OUTLIER MISMATCH in rows: {bad_rows.tolist()}")
    else:
        print("OUTLIER BYTES MATCH!")

def test_decode_simple():
    """Test: does the fused decode kernel match the reference Hadamard decode?"""
    device = torch.device("cuda")
    state = setup()
    normal_size = state["normal_size"]
    packed_bytes = math.ceil(normal_size * BIT_WIDTH / 8)

    # Create known indices and norms
    N = 4
    indices = torch.randint(0, 16, (N, normal_size), dtype=torch.uint8, device=device)
    norms = torch.rand(N, dtype=torch.float16, device=device) + 0.5

    # Reference: Triton Hadamard decode
    from vllm.v1.attention.ops.triton_hadamard_turboquant import hadamard_turboquant_decode
    indices_3d = indices.reshape(N, 1, normal_size)
    norms_2d = norms.reshape(N, 1)
    ref = hadamard_turboquant_decode(
        indices_3d, norms_2d, state["sign_flips"], state["codebook"],
        output_dtype=torch.bfloat16,
    ).reshape(N, normal_size)

    # Pack indices into bytes
    flat = indices.clone()
    if normal_size % 2 != 0:
        flat = torch.nn.functional.pad(flat, (0, 1), value=0)
    packed = flat[:, 0::2] | (flat[:, 1::2] << 4)
    packed = packed[:, :packed_bytes].to(torch.uint8)

    # Fused decode with packed + norms directly
    from vllm.v1.attention.ops.triton_fused_turboquant import _fused_decode_kernel, _next_power_of_2
    BLOCK_D = state["sign_flips"].shape[0]
    LOG2_D = int(math.log2(BLOCK_D))
    BLOCK_PACKED = _next_power_of_2(packed_bytes)

    fused_out = torch.empty(N, normal_size, dtype=torch.bfloat16, device=device)
    scratch = torch.empty(N, BLOCK_D, dtype=torch.float32, device=device)

    _fused_decode_kernel[(N,)](
        packed_ptr=packed, signs_ptr=state["sign_flips"],
        codebook_ptr=state["codebook"], scratch_ptr=scratch,
        norms_ptr=norms, out_ptr=fused_out,
        normal_size=normal_size, packed_bytes=packed_bytes,
        LOG2_D=LOG2_D, BLOCK_D=BLOCK_D, BLOCK_PACKED=BLOCK_PACKED,
        num_warps=4, num_stages=1,
    )

    diff = (ref - fused_out).abs()
    print(f"Decode max diff: {diff.max().item():.6f}")
    print(f"Decode mean diff: {diff.mean().item():.6f}")
    if diff.max() > 1e-2:
        # Find where the max diff is
        bad_row = diff.max(dim=1).values.argmax().item()
        bad_col = diff[bad_row].argmax().item()
        print(f"  Worst at row={bad_row}, col={bad_col}")
        print(f"  ref[{bad_row},{bad_col}] = {ref[bad_row, bad_col]}")
        print(f"  fused[{bad_row},{bad_col}] = {fused_out[bad_row, bad_col]}")
        # Check if indices were unpacked correctly
        # Verify packed data
        byte_idx = bad_col // 2
        is_high = bad_col % 2
        packed_byte = packed[bad_row, byte_idx].item()
        if is_high:
            expected_idx = (packed_byte >> 4) & 0xF
        else:
            expected_idx = packed_byte & 0xF
        original_idx = indices[bad_row, bad_col].item()
        print(f"  original index: {original_idx}, from packed: {expected_idx}")
    else:
        print("DECODE MATCHES!")

if __name__ == "__main__":
    print("=== ENCODE PACKING TEST ===")
    test_encode_packing()
    print()
    print("=== DECODE TEST ===")
    test_decode_simple()
