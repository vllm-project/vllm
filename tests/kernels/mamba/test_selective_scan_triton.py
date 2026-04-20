# SPDX-License-Identifier: Apache-2.0
"""Test script for the Triton selective scan implementation on XPU."""

import torch
import torch.nn.functional as F

from vllm.model_executor.layers.mamba.ops.selective_scan_triton import (
    selective_scan_fwd_triton,
)


def selective_scan_ref(
    u, delta, A, B, C, D=None, z=None, delta_bias=None,
    delta_softplus=False, prev_state=None,
):
    """Reference implementation (pure PyTorch, sequential scan)."""
    dtype_in = u.dtype
    u = u.float()
    delta = delta.float()
    if delta_bias is not None:
        delta = delta + delta_bias[..., None].float()
    if delta_softplus:
        delta = F.softplus(delta)
    batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]
    B = B.float()
    C = C.float()
    x = A.new_zeros((batch, dim, dstate)) if prev_state is None else prev_state.float()
    ys = []
    deltaA = torch.exp(torch.einsum("bdl,dn->bdln", delta, A))
    if B.dim() == 3:
        deltaB_u = torch.einsum("bdl,bnl,bdl->bdln", delta, B, u)
    else:
        from einops import repeat
        B = repeat(B, "B G N L -> B (G H) N L", H=dim // B.shape[1])
        deltaB_u = torch.einsum("bdl,bdnl,bdl->bdln", delta, B, u)
    if C.dim() == 4:
        from einops import repeat
        C = repeat(C, "B G N L -> B (G H) N L", H=dim // C.shape[1])
    last_state = None
    for i in range(u.shape[2]):
        x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
        if C.dim() == 3:
            y = torch.einsum("bdn,bn->bd", x, C[:, :, i])
        else:
            y = torch.einsum("bdn,bdn->bd", x, C[:, :, :, i])
        if i == u.shape[2] - 1:
            last_state = x
        ys.append(y)
    y = torch.stack(ys, dim=2)
    out = y if D is None else y + u * D[None, :, None]
    if z is not None:
        out = out * F.silu(z.float())
    out = out.to(dtype=dtype_in)
    return out, last_state


def test_basic(device, itype, seqlen, has_z, has_D, has_delta_bias,
               delta_softplus, varBC_groups, batch_size=1, dim=4, dstate=8):
    """Test basic selective scan correctness."""
    torch.manual_seed(42)
    wtype = torch.float32

    rtol, atol = (6e-4, 2e-3) if itype == torch.float32 else (3e-3, 5e-3)
    if itype == torch.bfloat16:
        rtol, atol = 3e-2, 5e-2

    A = -0.5 * torch.rand(dim, dstate, device=device, dtype=wtype)
    B_shape = [batch_size, varBC_groups, dstate, seqlen]
    B = torch.randn(B_shape, device=device, dtype=itype)
    C_shape = [batch_size, varBC_groups, dstate, seqlen]
    C = torch.randn(C_shape, device=device, dtype=itype)
    D = torch.randn(dim, device=device, dtype=torch.float32) if has_D else None
    z = (torch.randn(batch_size, dim, seqlen, device=device, dtype=itype)
         if has_z else None)
    delta_bias = (0.5 * torch.rand(dim, device=device, dtype=torch.float32)
                  if has_delta_bias else None)
    u = torch.randn(batch_size, dim, seqlen, device=device, dtype=itype)
    delta = 0.5 * torch.rand(batch_size, dim, seqlen, device=device, dtype=itype)
    ssm_states = torch.zeros(batch_size, dim, dstate, device=device, dtype=itype)

    # Reference
    u_ref = u.clone()
    delta_ref = delta.clone()
    z_ref = z.clone() if z is not None else None
    out_ref, state_ref = selective_scan_ref(
        u_ref, delta_ref, A.clone(), B.clone(), C.clone(),
        D=D.clone() if D is not None else None,
        z=z_ref,
        delta_bias=delta_bias,
        delta_softplus=delta_softplus,
    )

    # Prepare inputs for our kernel (need contiguous, matching expected shapes)
    u_test = u.clone()
    delta_test = delta.clone()
    z_test = z.clone() if z is not None else None
    B_test = B.clone()
    C_test = C.clone()
    ssm_states_test = ssm_states.clone()

    selective_scan_fwd_triton(
        u_test, delta_test, A, B_test, C_test,
        D, z_test, delta_bias,
        delta_softplus,
        query_start_loc=None,
        cache_indices=None,
        has_initial_state=None,
        ssm_states=ssm_states_test,
        null_block_id=-1,
        block_size=2048,
    )

    # Get output (z if has_z, else delta)
    out_test = z_test if has_z else delta_test

    # Compare
    if not torch.allclose(out_test, out_ref, rtol=rtol, atol=atol):
        max_diff = (out_test - out_ref).abs().max().item()
        print(f"  FAIL: max_diff={max_diff:.6e} (rtol={rtol}, atol={atol})")
        return False

    if not torch.allclose(ssm_states_test, state_ref.to(itype), rtol=rtol, atol=atol):
        max_diff = (ssm_states_test - state_ref.to(itype)).abs().max().item()
        print(f"  FAIL (states): max_diff={max_diff:.6e}")
        return False

    return True


def test_with_initial_state(device, itype, seqlen):
    """Test with initial state (chunked scan simulation)."""
    torch.manual_seed(42)
    dim, dstate, batch_size = 4, 8, 1
    varBC_groups = 1

    rtol, atol = (6e-4, 2e-3) if itype == torch.float32 else (3e-3, 5e-3)
    if itype == torch.bfloat16:
        rtol, atol = 3e-2, 5e-2

    A = -0.5 * torch.rand(dim, dstate, device=device, dtype=torch.float32)
    B = torch.randn(batch_size, varBC_groups, dstate, seqlen, device=device, dtype=itype)
    C = torch.randn(batch_size, varBC_groups, dstate, seqlen, device=device, dtype=itype)
    D = torch.randn(dim, device=device, dtype=torch.float32)
    z = torch.randn(batch_size, dim, seqlen, device=device, dtype=itype)
    delta_bias = 0.5 * torch.rand(dim, device=device, dtype=torch.float32)
    u = torch.randn(batch_size, dim, seqlen, device=device, dtype=itype)
    delta = 0.5 * torch.rand(batch_size, dim, seqlen, device=device, dtype=itype)

    # Reference: full scan
    out_ref, state_ref = selective_scan_ref(
        u.clone(), delta.clone(), A.clone(), B.clone(), C.clone(),
        D=D.clone(), z=z.clone(), delta_bias=delta_bias, delta_softplus=True,
    )

    # Triton: two-chunk scan
    mid = seqlen // 2
    # First chunk
    ssm_states = torch.zeros(batch_size, dim, dstate, device=device, dtype=itype)
    delta1 = delta[..., :mid].clone()
    z1 = z[..., :mid].clone()
    selective_scan_fwd_triton(
        u[..., :mid].contiguous(), delta1, A, B[..., :mid].contiguous(),
        C[..., :mid].contiguous(), D, z1, delta_bias, True,
        None, None, None, ssm_states, -1, 2048,
    )
    out1 = z1

    # Second chunk with initial state
    delta2 = delta[..., mid:].clone()
    z2 = z[..., mid:].clone()
    selective_scan_fwd_triton(
        u[..., mid:].contiguous(), delta2, A, B[..., mid:].contiguous(),
        C[..., mid:].contiguous(), D, z2, delta_bias, True,
        None, None,
        torch.ones(batch_size, device=device, dtype=torch.bool),
        ssm_states, -1, 2048,
    )
    out2 = z2

    out_test = torch.cat([out1, out2], dim=-1)

    if not torch.allclose(out_test, out_ref, rtol=rtol, atol=atol):
        max_diff = (out_test - out_ref).abs().max().item()
        print(f"  FAIL: max_diff={max_diff:.6e}")
        return False

    if not torch.allclose(ssm_states, state_ref.to(itype), rtol=rtol, atol=atol):
        max_diff = (ssm_states - state_ref.to(itype)).abs().max().item()
        print(f"  FAIL (states): max_diff={max_diff:.6e}")
        return False

    return True


def test_varlen(device, itype, seqlens):
    """Test varlen mode with query_start_loc."""
    torch.manual_seed(42)
    dim, dstate = 4, 8
    n_groups = 1
    batch_size = len(seqlens)
    total_len = sum(seqlens)

    rtol, atol = (6e-4, 2e-3) if itype == torch.float32 else (3e-3, 5e-3)
    if itype == torch.bfloat16:
        rtol, atol = 3e-2, 5e-2

    A = -0.5 * torch.rand(dim, dstate, device=device, dtype=torch.float32)
    D = torch.randn(dim, device=device, dtype=torch.float32)
    delta_bias = 0.5 * torch.rand(dim, device=device, dtype=torch.float32)

    # Varlen layout: u=(dim, total_len), B=(n_groups, dstate, total_len)
    u = torch.randn(dim, total_len, device=device, dtype=itype)
    delta = 0.5 * torch.rand(dim, total_len, device=device, dtype=itype)
    z = torch.randn(dim, total_len, device=device, dtype=itype)
    B = torch.randn(n_groups, dstate, total_len, device=device, dtype=itype)
    C = torch.randn(n_groups, dstate, total_len, device=device, dtype=itype)
    ssm_states = torch.zeros(batch_size, dim, dstate, device=device, dtype=itype)

    query_start_loc = torch.zeros(batch_size + 1, device=device, dtype=torch.int32)
    for i, sl in enumerate(seqlens):
        query_start_loc[i + 1] = query_start_loc[i] + sl

    # Reference: process each sequence separately in batch mode
    out_ref_parts = []
    state_refs = []
    offset = 0
    for i, sl in enumerate(seqlens):
        u_i = u[:, offset:offset + sl].unsqueeze(0)   # (1, dim, sl)
        delta_i = delta[:, offset:offset + sl].unsqueeze(0)
        z_i = z[:, offset:offset + sl].unsqueeze(0)
        B_i = B[:, :, offset:offset + sl].unsqueeze(0)  # (1, n_groups, dstate, sl)
        C_i = C[:, :, offset:offset + sl].unsqueeze(0)
        out_i, state_i = selective_scan_ref(
            u_i, delta_i, A.clone(), B_i, C_i,
            D=D.clone(), z=z_i, delta_bias=delta_bias, delta_softplus=True,
        )
        out_ref_parts.append(out_i.squeeze(0))  # (dim, sl)
        state_refs.append(state_i.squeeze(0))
        offset += sl
    out_ref = torch.cat(out_ref_parts, dim=-1)  # (dim, total_len)
    state_ref = torch.stack(state_refs, dim=0)   # (batch, dim, dstate)

    # Triton varlen
    delta_test = delta.clone()
    z_test = z.clone()
    ssm_states_test = ssm_states.clone()
    selective_scan_fwd_triton(
        u.clone(), delta_test, A, B.clone(), C.clone(),
        D, z_test, delta_bias, True,
        query_start_loc, None, None, ssm_states_test, -1, 2048,
    )
    out_test = z_test  # z is present

    if not torch.allclose(out_test, out_ref, rtol=rtol, atol=atol):
        max_diff = (out_test - out_ref).abs().max().item()
        print(f"  FAIL: max_diff={max_diff:.6e}")
        return False

    if not torch.allclose(ssm_states_test, state_ref.to(itype), rtol=rtol, atol=atol):
        max_diff = (ssm_states_test - state_ref.to(itype)).abs().max().item()
        print(f"  FAIL (states): max_diff={max_diff:.6e}")
        return False

    return True


def main():
    device = "xpu:0"
    print(f"Testing on {device}")
    print(f"Device: {torch.xpu.get_device_name(0)}")
    print()

    total = 0
    passed = 0

    # Test 1: Basic configurations
    print("=== Basic tests ===")
    for itype in [torch.float32, torch.bfloat16]:
        for seqlen in [16, 128, 512]:
            for has_z in [True, False]:
                for has_D in [True, False]:
                    for varBC_groups in [1, 2]:
                        total += 1
                        name = (f"basic itype={itype}, seqlen={seqlen}, "
                                f"z={has_z}, D={has_D}, groups={varBC_groups}")
                        try:
                            ok = test_basic(
                                device, itype, seqlen,
                                has_z=has_z, has_D=has_D,
                                has_delta_bias=True,
                                delta_softplus=True,
                                varBC_groups=varBC_groups,
                            )
                            if ok:
                                passed += 1
                                print(f"  PASS: {name}")
                            else:
                                print(f"  FAIL: {name}")
                        except Exception as e:
                            print(f"  ERROR: {name}: {e}")
                            import traceback
                            traceback.print_exc()

    # Test 2: Chunked scan with initial state
    print("\n=== Chunked scan (initial state) tests ===")
    for itype in [torch.float32, torch.bfloat16]:
        for seqlen in [64, 256]:
            total += 1
            name = f"chunked itype={itype}, seqlen={seqlen}"
            try:
                ok = test_with_initial_state(device, itype, seqlen)
                if ok:
                    passed += 1
                    print(f"  PASS: {name}")
                else:
                    print(f"  FAIL: {name}")
            except Exception as e:
                print(f"  ERROR: {name}: {e}")
                import traceback
                traceback.print_exc()

    # Test 3: Varlen mode
    print("\n=== Varlen tests ===")
    for itype in [torch.float32, torch.bfloat16]:
        for seqlens in [[32, 16], [64, 32, 16]]:
            total += 1
            name = f"varlen itype={itype}, seqlens={seqlens}"
            try:
                ok = test_varlen(device, itype, seqlens)
                if ok:
                    passed += 1
                    print(f"  PASS: {name}")
                else:
                    print(f"  FAIL: {name}")
            except Exception as e:
                print(f"  ERROR: {name}: {e}")
                import traceback
                traceback.print_exc()

    # Test 4: Larger dimensions (closer to real model)
    print("\n=== Large dimension tests ===")
    for itype in [torch.bfloat16]:
        for dim in [256, 1024]:
            for seqlen in [128, 512]:
                total += 1
                name = f"large dim={dim}, seqlen={seqlen}, itype={itype}"
                try:
                    ok = test_basic(
                        device, itype, seqlen,
                        has_z=True, has_D=True,
                        has_delta_bias=True,
                        delta_softplus=True,
                        varBC_groups=1,
                        dim=dim, dstate=16,
                    )
                    if ok:
                        passed += 1
                        print(f"  PASS: {name}")
                    else:
                        print(f"  FAIL: {name}")
                except Exception as e:
                    print(f"  ERROR: {name}: {e}")
                    import traceback
                    traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"Results: {passed}/{total} tests passed")
    if passed == total:
        print("ALL TESTS PASSED!")
    else:
        print(f"FAILURES: {total - passed}")

    return passed == total


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
