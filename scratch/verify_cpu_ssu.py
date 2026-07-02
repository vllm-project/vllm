import torch
import torch.nn.functional as F
from vllm.model_executor.layers.mamba.ops.mamba_ssm import selective_state_update
import time

def selective_state_update_ref(
    state, x, dt, A, B, C, D=None, z=None, dt_bias=None, dt_softplus=False
):
    has_heads = state.dim() > 3
    if state.dim() == 3:
        state = state.unsqueeze(1)
    if x.dim() == 2:
        x = x.unsqueeze(1)
    if dt.dim() == 2:
        dt = dt.unsqueeze(1)
    if A.dim() == 2:
        A = A.unsqueeze(0)
    if B.dim() == 2:
        B = B.unsqueeze(1)
    if C.dim() == 2:
        C = C.unsqueeze(1)
    if D is not None and D.dim() == 1:
        D = D.unsqueeze(0)
    if z is not None and z.dim() == 2:
        z = z.unsqueeze(1)
    if dt_bias is not None and dt_bias.dim() == 1:
        dt_bias = dt_bias.unsqueeze(0)
    batch, nheads, dim, dstate = state.shape
    
    dt = dt.float()
    if dt_bias is not None:
        dt = dt + dt_bias.float()
    dt = F.softplus(dt) if dt_softplus else dt
    
    dA = torch.exp(dt.unsqueeze(-1) * A.float())  # (batch, nheads, dim, dstate)
    ngroups = B.shape[1]
    B = B.repeat_interleave(nheads // ngroups, dim=1)
    C = C.repeat_interleave(nheads // ngroups, dim=1)
    
    dBx = (dt.unsqueeze(-1) * B.float().unsqueeze(2)) * x.float().unsqueeze(-1)
    
    state.copy_(state.float() * dA + dBx)
    
    out = torch.einsum("bhdn,bhn->bhd", state.float(), C.float())
    if D is not None:
        out += (x.float() * D.float())
    
    if z is not None:
        out = out * F.silu(z.float())
        
    if not has_heads:
        out = out.squeeze(1)
    return out.to(x.dtype)

def test_cpu_ssu():
    device = "cpu"
    torch.manual_seed(42)
    
    # Mamba-2 style dimensions
    batch_size = 2
    nheads = 128
    dim = 64
    dstate = 128
    itype = torch.bfloat16
    
    print(f"Testing CPU selective_state_update with {itype}...")
    
    state = torch.randn(batch_size, nheads, dim, dstate, dtype=itype, device=device)
    x = torch.randn(batch_size, nheads, dim, device=device, dtype=itype)
    dt = torch.randn(batch_size, nheads, dim, device=device, dtype=itype)
    dt_bias = torch.rand(nheads, dim, device=device) - 4.0
    A = -torch.rand(nheads, dim, dstate, device=device) - 1.0
    B = torch.randn(batch_size, 1, dstate, device=device, dtype=itype)
    C = torch.randn(batch_size, 1, dstate, device=device, dtype=itype)
    D = torch.randn(nheads, dim, device=device)
    z = torch.randn_like(x)
    
    out = torch.empty_like(x)
    state_ref = state.detach().clone()
    
    start = time.time()
    selective_state_update(
        state, x, dt, A, B, C, D=D, z=z, dt_bias=dt_bias, dt_softplus=True, out=out
    )
    end = time.time()
    print(f"Kernel time: {end - start:.4f}s")
    
    start = time.time()
    out_ref = selective_state_update_ref(
        state_ref, x, dt, A, B, C, D=D, z=z, dt_bias=dt_bias, dt_softplus=True
    )
    end = time.time()
    print(f"Ref time: {end - start:.4f}s")
    
    state_diff = (state.float() - state_ref.float()).abs().max().item()
    out_diff = (out.float() - out_ref.float()).abs().max().item()
    
    print(f"State max diff: {state_diff}")
    print(f"Out max diff: {out_diff}")
    
    if state_diff < 1e-2 and out_diff < 1e-2:
        print("SUCCESS")
    else:
        print("FAILURE")

if __name__ == "__main__":
    test_cpu_ssu()
