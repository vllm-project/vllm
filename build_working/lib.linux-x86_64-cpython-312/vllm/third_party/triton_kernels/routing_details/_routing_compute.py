import triton
import triton.language as tl

from ._expt_data import _expt_data_compute, _expt_data_memset


@triton.jit
def _routing_compute_expt_offs(ExpertHist, FinalExpertOffs, hist_size,  # histogram
                               BLOCK_N: tl.constexpr):
    loop_iterations = (hist_size + BLOCK_N - 1) // BLOCK_N
    x = tl.zeros([BLOCK_N], ExpertHist.dtype.element_ty)
    for i in range(loop_iterations):
        offs_n = i * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < hist_size
        hist2 = tl.load(ExpertHist + offs_n, mask=mask_n)
        tok_starts = tl.cumsum(hist2, 0) - hist2 + x
        x += tl.sum(hist2, 0)
        tl.store(FinalExpertOffs + offs_n, tok_starts, mask=mask_n)
        offs_n += BLOCK_N


@triton.jit
def _routing_compute_indx_offs(PartialHist, shape_pm, stride_pm, stride_pn, BLOCK_M: tl.constexpr, expt_id):
    offs_m = tl.arange(0, BLOCK_M)
    # iterate over input data
    curr_sum = 0
    for _ in range(0, shape_pm, BLOCK_M):
        offs = offs_m * stride_pm + expt_id * stride_pn
        curr = tl.load(PartialHist + offs, mask=offs_m < shape_pm)
        out = tl.cumsum(curr, 0) + curr_sum
        curr_sum += tl.sum(curr, 0)
        tl.store(PartialHist + offs, out - curr, mask=offs_m < shape_pm)
        offs_m += BLOCK_M


@triton.jit
def _keyed_add(x, y):

    # we keep the key in the upper 16 bits of a uint32:
    key_mask: tl.constexpr = 0xffff0000

    kx = x & key_mask
    ky = y & key_mask
    z = tl.where(kx == ky, x + y - kx, y)
    return z


@triton.jit
def _routing_compute_indx(pid_m, GatherIndx, ScatterIndx, GateScal, ExptScal, ExptIndx, PartialOffs, stride_pm,
                          stride_pn, TokensStart, n_tokens, BLOCK_M: tl.constexpr, N_EXPTS_ACT: tl.constexpr):

    if isinstance(n_tokens, tl.tensor) and n_tokens.dtype.is_ptr():
        n_tokens = tl.load(n_tokens)
    n_gates = n_tokens * N_EXPTS_ACT

    tl.static_assert(N_EXPTS_ACT * BLOCK_M <= 32768)

    local_offs = tl.arange(0, N_EXPTS_ACT * BLOCK_M)
    offs = pid_m * BLOCK_M * N_EXPTS_ACT + local_offs
    expert = tl.load(ExptIndx + offs, mask=(offs < n_gates), other=-1).to(tl.uint32)

    # stable-sort by expert ID:
    kv_pairs = ((expert << 16) | local_offs).to(tl.uint32)
    kv_pairs = tl.sort(kv_pairs, 0)
    expert = kv_pairs >> 16
    offs = pid_m * BLOCK_M * N_EXPTS_ACT + (kv_pairs & 0xffff)
    mask = expert != 0xffff
    gate_scal = tl.load(ExptScal + offs, mask=mask)

    # compute run lengths in expert-sorted order:
    x = (kv_pairs & 0xffff0000 | 0x00000001)
    expts_and_inclusive_run_lengths = tl.associative_scan(x, 0, _keyed_add)
    exclusive_run_lengths = (expts_and_inclusive_run_lengths - 1) & 0xffff

    gates = tl.load(PartialOffs + pid_m * stride_pm + expert * stride_pn, mask=mask)
    gates += tl.load(TokensStart + expert, mask=mask)
    gates += exclusive_run_lengths

    tl.store(ScatterIndx + offs, gates, mask=mask)
    tl.store(GatherIndx + gates, offs, mask=mask)
    tl.store(GateScal + gates, gate_scal, mask=mask)


@triton.jit
def _combined_routing_compute(GatherIndx, ScatterIndx, GateScal, ExptScal, ExptIndx, PartialOffs, stride_pm, stride_pn,
                              TokensStart, n_tokens, BLOCK_M: tl.constexpr, N_EXPTS_ACT: tl.constexpr, Hist,
                              MDTileStarts, tile_starts_stridem, MDTileInfo, tile_info_stridem, first_tile_dim_log2,
                              SIZES: tl.constexpr, BLOCK: tl.constexpr, blocks2a):

    pid = tl.program_id(0)
    if pid < blocks2a:
        _expt_data_compute(Hist, MDTileStarts, tile_starts_stridem, MDTileInfo, tile_info_stridem, first_tile_dim_log2,
                           SIZES, BLOCK)
    else:
        pid -= blocks2a
        _routing_compute_indx(pid, GatherIndx, ScatterIndx, GateScal, ExptScal, ExptIndx, PartialOffs, stride_pm,
                              stride_pn, TokensStart, n_tokens, BLOCK_M, N_EXPTS_ACT)


@triton.jit
def _routing_clear_bitmatrix(Bitmatrix, stride_bm, stride_bn, shape_bn, cutoff, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    cutoff_word = cutoff // 32
    cutoff_bit = cutoff % 32
    cutoff_mask = (1 << (cutoff_bit)) - 1
    for start_n in range(0, shape_bn, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        values = tl.load(Bitmatrix + pid_m * stride_bm + offs_n * stride_bn, mask=offs_n < shape_bn)
        values = tl.where(offs_n == cutoff_word, values & cutoff_mask, values)
        values = tl.where(offs_n > cutoff_word, 0, values)
        tl.store(Bitmatrix + pid_m * stride_bm + offs_n * stride_bn, values, mask=offs_n < shape_bn)


@triton.jit
def _combined_routing_memset(Indx, size, sentinel, BLOCK: tl.constexpr, ExpertHist, FinalExpertOffs, hist_size,
                             n_expts_tot, PartialHist, shape_pm, stride_pm, stride_pn, MDStarts, tile_starts_stridem,
                             blocks1a, MDTileInfo, first_tile_dim_log2, SIZES: tl.constexpr, BLOCK_A: tl.constexpr,
                             BLOCK_N: tl.constexpr, BLOCK_M: tl.constexpr):
    """
    This kernel essentially combines 6 different pieces of functionality,
    statically branching on the value of tl.program_id(0) to decide which
    codepath to take.

        pid == 0:                                  create the token cumsum
        1 <= pid <= SIZES:                         create a tile cumsum
        SIZES < pid < blocks1a:                    initialise MDTileInfo to 0xffffffff
        blocks1a <= pid < blocks1a + n_expts_tot:  compute_indx_offs
        pid == blocks1a + n_expts_tot:             compute_expt_offs
        pid > blocks1a + n_expts_tot:              initialise Indx to sentinel

    As each of these is a relatively trivial workload, launching them from
    this single trampoline is beneficial as they can execute on different
    streaming multiprocesses in parallel.
    """

    pid = tl.program_id(0)

    if pid < blocks1a:
        _expt_data_memset(ExpertHist, n_expts_tot, MDStarts, tile_starts_stridem, MDTileInfo, first_tile_dim_log2,
                          SIZES, BLOCK_A)
    elif pid == n_expts_tot + blocks1a:
        _routing_compute_expt_offs(ExpertHist, FinalExpertOffs, hist_size, BLOCK_N)
    elif pid < n_expts_tot + blocks1a:
        _routing_compute_indx_offs(PartialHist, shape_pm, stride_pm, stride_pn, BLOCK_M, pid - blocks1a)
    else:
        offs = (pid - n_expts_tot - blocks1a - 1) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < size
        tl.store(Indx + offs, sentinel, mask=mask)
