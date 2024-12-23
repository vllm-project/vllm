import pytest
import torch
import torch.nn.functional as F
from einops import rearrange, repeat

from vllm.model_executor.layers.mamba.ops.ssd_combined import (
    mamba_chunk_scan_combined)
from vllm.platforms import current_platform

import numpy as np

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


def generate_random_inputs(batch_size,
                           seqlen,
                           n_heads,
                           d_head,
                           itype,
                           device='cuda'):

    current_platform.seed_everything(0)
    A = (-torch.exp(torch.rand(n_heads, dtype=itype, device=device)))
    dt = F.softplus(
        torch.randn(batch_size, seqlen, n_heads, dtype=itype, device=device) -
        4)
    X = torch.randn((batch_size, seqlen, n_heads, d_head),
                    dtype=itype,
                    device=device)
    B = torch.randn((batch_size, seqlen, n_heads, d_head),
                    dtype=itype,
                    device=device)
    C = torch.randn((batch_size, seqlen, n_heads, d_head),
                    dtype=itype,
                    device=device)

    return A, dt, X, B, C


def generate_continous_batched_examples(example_lens_by_batch,
                                        num_examples,
                                        full_length,
                                        last_taken,
                                        exhausted,
                                        n_heads,
                                        d_head,
                                        itype,
                                        device='cuda'):

    # this function generates a random examples of certain length
    # and then cut according to "example_lens_by_batch" and feed
    # them in continuous batches to the kernels

    # generate the full-length example
    A, dt, X, B, C = generate_random_inputs(num_examples, full_length, n_heads,
                                            d_head, itype)

    Y_min, final_state_min = ssd_minimal_discrete(X * dt.unsqueeze(-1),
                                                  A * dt,
                                                  B,
                                                  C,
                                                  block_len=full_length // 4)

    # internal function to
    def take(example_lens):

        indices = []
        for i, l in enumerate(example_lens):
            c = last_taken.get(i, 0)
            indices.append((c, c + l))
            last_taken[i] = (c + l) % full_length
            exhausted[i] = last_taken[i] == 0

        return (torch.concat([x[i, s:e] for i, (s, e) in enumerate(indices)
                              ]).unsqueeze(0) for x in (dt, X, B, C))

    def end_boundary(n):
        return n - ((n - 1) // full_length) * full_length

    IND_E = None
    for i, spec in enumerate(example_lens_by_batch):

        # get the (maybe partial) example seen in this cont batch
        dt2, X2, B2, C2 = take(spec)

        # get the metadata
        cu_seqlens = torch.tensor((0, ) + spec, device=device).cumsum(dim=0)
        sed_idx = torch.zeros(cu_seqlens[-1],
                              dtype=torch.int32,
                              device=cu_seqlens.device)
        for i, (srt, end) in enumerate(zip(
                cu_seqlens,
                cu_seqlens[1:],
        )):
            sed_idx[srt:end] = i

        # for cont batch
        # IND = np.insert(np.cumsum(spec), [0], [0]) # torch.cumsum
        if IND_E is None:
            IND_S = [0 for _ in range(len(spec))]
        else:
            IND_S = [x % full_length for x in IND_E]
        IND_E = [end_boundary(x + y) for x, y in zip(IND_S, spec)]

        yield ([Y_min[s, IND_S[s]:IND_E[s]] for s in range(num_examples)],
               cu_seqlens, sed_idx.unsqueeze(0), (A, dt2, X2, B2, C2))


@pytest.mark.parametrize("itype",
                         [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("n_heads", [4, 16, 32])
@pytest.mark.parametrize("dim", [128, 512])
@pytest.mark.parametrize("seq_len_chunk_size", [(32, 128)])
def test_mamba_chunk_scan_single_example(dim, n_heads, seq_len_chunk_size,
                                         itype):

    # this tests the kernels on a single example (no batching)

    # set seed
    batch_size = 1  # batch_size
    seqlen, chunk_size = seq_len_chunk_size
    d_head = dim // n_heads

    A, dt, X, B, C = generate_random_inputs(batch_size, seqlen, n_heads,
                                            d_head, itype)

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


@pytest.mark.parametrize("itype", [torch.float16])
@pytest.mark.parametrize("n_heads", [4])
@pytest.mark.parametrize("dim", [64])
@pytest.mark.parametrize("seq_len_chunk_size_cases", [
    64,
    8,
    2,
    [(32, 32), (32, 32)],
])
def test_mamba_chunk_scan_batch(dim, n_heads, seq_len_chunk_size_cases, itype):

    # this test with multiple examples in a continuous batch

    seqlen, chunk_size, num_examples, cases = seq_len_chunk_size_cases
    d_head = dim // n_heads

    # hold state during the cutting process so we know if an
    # example has been exhausted and needs to cycle
    last_taken = {}  # map: eg -> pointer to last taken sample
    exhausted = {}  # map: eg -> boolean indicating example is exhausted

    states = None
    for Y_min, cu_seqlens, sed_idx, (A, dt, X, B,
                                     C) in generate_continous_batched_examples(
                                         cases, num_examples, seqlen,
                                         last_taken, exhausted, n_heads,
                                         d_head, itype):

        Y, new_states = mamba_chunk_scan_combined(
            X,
            dt,
            A,
            B,
            C,
            chunk_size,
            D=None,
            cu_seqlens=cu_seqlens,
            seq_idx=sed_idx,
            return_varlen_states=True,
            initial_states=states,
        )

        # just test the last in sequence
        for i in range(num_examples):

            # just test one dim and dstate
            Y_eg = Y[0, cu_seqlens[i]:cu_seqlens[i + 1], 0, 0]
            Y_min_eg = Y_min[i][:, 0, 0]
            torch.testing.assert_close(Y_eg, Y_min_eg, atol=1e-2, rtol=1e1)

        # update states
        states = new_states
        for i in [i for i, clear in exhausted.items() if clear]:
            states[i].fill_(0.)
        exhausted = {}
