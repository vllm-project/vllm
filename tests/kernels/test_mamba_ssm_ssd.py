import pytest
import torch
import torch.nn.functional as F
from einops import rearrange, repeat

from vllm.model_executor.layers.mamba.ops.ssd_combined import (
    mamba_chunk_scan_combined)
from vllm.platforms import current_platform

# Added by the IBM Team, 2024

# Adapted from https://github.com/state-spaces/mamba/tree/main/mamba_ssm/ops/triton


def segsum(x):
    """More stable segment sum calculation."""
    T = x.size(-1)
    x = repeat(x, "... d -> ... d e", e=T)
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=bool),
                      diagonal=-1)
    x = x.masked_fill(~mask, 0)
    x_segsum = torch.cumsum(x, dim=-2)
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=bool),
                      diagonal=0)
    x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
    return x_segsum


def ssd_minimal_discrete(X, A, B, C, block_len, initial_states=None):
    """
    Arguments:
        X: (batch, length, n_heads, d_head)
        A: (batch, length, n_heads)
        B: (batch, length, n_heads, d_state)
        C: (batch, length, n_heads, d_state)
    Return:
        Y: (batch, length, n_heads, d_head)
    """
    assert X.dtype == A.dtype == B.dtype == C.dtype
    assert X.shape[1] % block_len == 0

    # Rearrange into blocks/chunks
    X, A, B, C = (rearrange(x, "b (c l) ... -> b c l ...", l=block_len)
                  for x in (X, A, B, C))

    A = rearrange(A, "b c l h -> b h c l")
    A_cumsum = torch.cumsum(A, dim=-1)

    # 1. Compute the output for each intra-chunk (diagonal blocks)
    L = torch.exp(segsum(A))
    Y_diag = torch.einsum("bclhn,bcshn,bhcls,bcshp->bclhp", C, B, L, X)

    # 2. Compute the state for each intra-chunk
    # (right term of low-rank factorization of off-diagonal blocks; B terms)
    decay_states = torch.exp(A_cumsum[:, :, :, -1:] - A_cumsum)
    states = torch.einsum("bclhn,bhcl,bclhp->bchpn", B, decay_states, X)

    # 3. Compute the inter-chunk SSM recurrence; produces correct SSM states at
    #    chunk boundaries
    # (middle term of factorization of off-diag blocks; A terms)
    if initial_states is None:
        initial_states = torch.zeros_like(states[:, :1])
    states = torch.cat([initial_states, states], dim=1)
    decay_chunk = torch.exp(segsum(F.pad(A_cumsum[:, :, :, -1], (1, 0))))
    new_states = torch.einsum("bhzc,bchpn->bzhpn", decay_chunk, states)
    states, final_state = new_states[:, :-1], new_states[:, -1]

    # 4. Compute state -> output conversion per chunk
    # (left term of low-rank factorization of off-diagonal blocks; C terms)
    state_decay_out = torch.exp(A_cumsum)
    Y_off = torch.einsum('bclhn,bchpn,bhcl->bclhp', C, states, state_decay_out)

    # Add output of intra-chunk and inter-chunk terms
    # (diagonal and off-diagonal blocks)
    Y = rearrange(Y_diag + Y_off, "b c l h p -> b (c l) h p")
    return Y, final_state


@pytest.mark.parametrize("itype",
                         [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("n_heads", [4, 16, 32])
@pytest.mark.parametrize("dim", [128, 512])
def test_mamba_chunk_scan(dim, n_heads, itype):
    device = "cuda"
    # set seed
    current_platform.seed_everything(0)
    batch = 1  # batch_size
    seqlen = 128
    chunk_size = 32
    d_head = dim // n_heads

    A = (-torch.exp(torch.rand(n_heads, dtype=itype, device=device)))
    dt = F.softplus(
        torch.randn(batch, seqlen, n_heads, dtype=itype, device=device) - 4)
    X = torch.randn((batch, seqlen, n_heads, d_head),
                    dtype=itype,
                    device=device)
    B = torch.randn((batch, seqlen, n_heads, d_head),
                    dtype=itype,
                    device=device)
    C = torch.randn((batch, seqlen, n_heads, d_head),
                    dtype=itype,
                    device=device)

    Y_min, final_state_min = ssd_minimal_discrete(X * dt.unsqueeze(-1), A * dt,
                                                  B, C, chunk_size)

    Y, final_state = mamba_chunk_scan_combined(X,
                                               dt,
                                               A,
                                               B,
                                               C,
                                               chunk_size,
                                               D=None,
                                               return_final_states=True)

    # just test the last in sequence
    torch.testing.assert_close(Y[:, -1], Y_min[:, -1], atol=1e-2, rtol=1e1)

    # just test the last head
    # NOTE, in the kernel we always cast states to fp32
    torch.testing.assert_close(final_state[:, -1],
                               final_state_min[:, -1].to(torch.float32),
                               atol=1e-2,
                               rtol=1e1)
