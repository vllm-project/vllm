import torch
import triton
from dataclasses import dataclass, field
from .routing_details._routing_compute import _combined_routing_compute
from .routing_details._routing_compute import _combined_routing_memset
from .routing_details._routing_compute import _routing_clear_bitmatrix
from .routing_details._expt_data import _expt_data_memset
from .routing_details._expt_data import _expt_data_compute
from .target_info import is_hip


@dataclass
class GatherIndx:
    """
    Indices for an operation that performs:
    Y = X[src_idx, :]
    """
    # array such that `dst_idx[src_idx] = arange(0, N)`
    src_indx: torch.Tensor
    dst_indx: torch.Tensor


@dataclass
class ScatterIndx:
    """
    Indices for an operation that performs:
    Y[dst_idx, :] = X
    """
    # array such that `dst_idx[src_idx] = arange(0, N)`
    src_indx: torch.Tensor
    dst_indx: torch.Tensor


@dataclass
class ExptData:
    # hist[i] is the number of tokens routed to expert i
    hist: torch.Tensor
    # token_offs_raw[i] is the offset of the first token routed
    # to expert i in an expert-sorted array
    token_offs_raw: torch.Tensor
    # token_offs_pad[block][i] is the offset of the first token routed
    # to expert i in an expert-sorted array, assuming histogram
    # rounded to the next multiple of `block`
    token_offs_pad: dict[int, torch.Tensor]
    # block_id_map[block] contain one value for each `pid`` launched by
    # the matrix multiplication kernel launched with BLOCK_M=block:
    # - the value is -1 if the `pid` has no work to do
    # - otherwise, the value is two int16 (packed as an int32) that
    #   correspond respectively to (1) the expert assigned to
    #   the tokens processed by this pid; (2) the block assigned to the
    #   tokens processed by this pid (think `pid_m` in a regular matmul)
    # see `test_routing.py` for a reference implementation and more details
    block_pid_map: dict[int, torch.Tensor]

    def __post_init__(self):
        if self.hist is not None:
            assert self.hist.dtype == torch.int32
        if self.token_offs_raw is not None:
            assert self.token_offs_raw.dtype == torch.int32
        if self.token_offs_pad is not None:
            for v in self.token_offs_pad.values():
                assert v.dtype == torch.int32
        if self.block_pid_map is not None:
            for v in self.block_pid_map.values():
                assert v.dtype == torch.int32


@dataclass
class RoutingData:
    gate_scal: torch.Tensor = field()
    expt_hist: torch.Tensor = field()
    n_expts_tot: int = field()
    n_expts_act: int = field()
    expt_data: ExptData = None

    # Used to make perf annotation cleaner: when we use expert sharding, we can
    # use this to tell the "expected" number of local tokens per expert, because
    # the actual number can vary per each input.
    expected_tokens_per_expt: int = field(default=None)

    def n_blocks(self, n_rows, block_m):
        if n_rows <= self.n_expts_tot:
            return n_rows
        else:
            return triton.cdiv(max(n_rows - self.n_expts_tot + 1, 0), block_m) + self.n_expts_tot - 1


# --------------------------
# sort tokens by expert
# --------------------------


class SortTokens(torch.autograd.Function):

    @staticmethod
    def forward(ctx, expt_scal, expt_indx, n_expts_tot, bitmatrix):
        HIST_BLOCK_M = 32
        INDX_OFFS_BLOCK_M = 512
        MEMSET_BLOCK = 1024
        cdiv = triton.cdiv

        device = expt_scal.device
        dtype = expt_scal.dtype
        n_tokens_raw, _ = bitmatrix.shape
        n_tokens_pad, n_expts_act = expt_scal.shape
        n_gates_pad = n_tokens_pad * n_expts_act

        hist, partial_hist = bitmatrix.sum(partials_block_size=HIST_BLOCK_M)
        hist = hist[:n_expts_tot]
        assert hist.dtype == torch.int32
        # scratchpad
        expt_offs = torch.empty(n_expts_tot, dtype=torch.int32, device=device)
        combined_indx = torch.empty(n_gates_pad * 2, dtype=torch.int32, device=device)
        # output
        topk_indx = combined_indx[:n_gates_pad]
        gate_indx = combined_indx[n_gates_pad:]
        gate_scal = torch.empty(n_gates_pad, dtype=dtype, device=device)

        token_offs_combined, token_offs_raw, token_offs_pad, block_pid_map, blocks1a, blocks2a, MEMSET_BLOCK_A, HIST2_BLOCK_M, block_m_log2_start, block_m_num = _compute_expt_data_internal(
            hist, n_expts_tot, n_gates_pad)

        blocks1b = cdiv(n_gates_pad * 2, MEMSET_BLOCK) + n_expts_tot + 1
        blocks2b = cdiv(n_tokens_pad, HIST_BLOCK_M)

        _combined_routing_memset[(blocks1a + blocks1b, )](
            combined_indx, n_gates_pad * 2, -1, MEMSET_BLOCK, hist,  #
            expt_offs, hist.shape[0], n_expts_tot, partial_hist,  # inputs
            partial_hist.shape[0], partial_hist.stride(0), partial_hist.stride(1),  # outputs
            token_offs_combined, token_offs_combined.stride(0),  #
            blocks1a, block_pid_map,  #
            block_m_log2_start, SIZES=block_m_num, BLOCK_A=MEMSET_BLOCK_A,  # optimization parameters
            BLOCK_N=512, BLOCK_M=INDX_OFFS_BLOCK_M,  # tunable parameters
        )

        indx_offs = partial_hist

        _combined_routing_compute[(blocks2a + blocks2b, )](
            topk_indx, gate_indx, gate_scal,  # outputs
            expt_scal, expt_indx, indx_offs, indx_offs.stride(0), indx_offs.stride(1),  # inputs
            expt_offs, n_tokens_raw,  # input shape
            HIST_BLOCK_M, n_expts_act,  # constants
            hist, token_offs_pad, token_offs_pad.stride(0), block_pid_map, block_pid_map.stride(0),  # outputs
            block_m_log2_start, block_m_num, HIST2_BLOCK_M, blocks2a,  # etc.
        )

        ctx.n_tokens_raw = n_tokens_raw
        ctx.n_tokens_pad = n_tokens_pad
        ctx.n_expts_act = n_expts_act
        ctx.save_for_backward(gate_indx)
        return hist, topk_indx, gate_indx, gate_scal, token_offs_raw, token_offs_pad, block_pid_map

    @staticmethod
    def backward(ctx, _0, _1, _2, dgate_scal, _3, _4, _5):
        (gate_indx, ) = ctx.saved_tensors
        dgate_scal = dgate_scal[gate_indx]
        dgate_scal = dgate_scal.reshape(ctx.n_tokens_pad, ctx.n_expts_act)
        return dgate_scal, None, None, None


def sort_tokens(expt_scal, expt_indx, n_expts_tot, bitmatrix):
    return SortTokens.apply(expt_scal, expt_indx, n_expts_tot, bitmatrix)


# --------------------------
# prune routing
# --------------------------


class PruneRouting(torch.autograd.Function):

    @staticmethod
    def forward(ctx, expt_scal, expt_indx, bitmatrix, n_expts_tot, simulated_ep):
        from .compaction import compaction
        n_tokens_pad = expt_scal.shape[0]
        assert n_expts_tot % simulated_ep == 0
        _routing_clear_bitmatrix[(n_tokens_pad, )](
            bitmatrix.storage.data,
            bitmatrix.storage.data.stride(0),
            bitmatrix.storage.data.stride(1),
            bitmatrix.storage.data.shape[1],
            n_expts_tot // simulated_ep,
            BLOCK_N=512,
        )
        # perform compaction to update expt_scal / expt_indx
        expt_scal, expt_indx = compaction(expt_scal, expt_indx, bitmatrix)
        n_expts_tot = n_expts_tot // simulated_ep
        bitmatrix.shape[-1] = n_expts_tot
        return expt_scal, expt_indx, bitmatrix


def prune_routing(expt_scal, expt_indx, bitmatrix, n_expts_tot, simulated_ep):
    return PruneRouting.apply(expt_scal, expt_indx, bitmatrix, n_expts_tot, simulated_ep)


# --------------------------
# expt_data
# --------------------------


def log2_power_of_two(x):
    assert x > 0 and (x & (x - 1)) == 0, "x must be a power of two"
    return x.bit_length() - 1


block_m_log2_start = 4


def _compute_expt_data_internal(expt_hist, n_expts_tot, n_gates):

    MEMSET_BLOCK = 512
    HIST2_BLOCK_M = 512
    device = expt_hist.device
    n_expts_tot = n_expts_tot
    cdiv = triton.cdiv
    # block_ms are all powers-of-two between 16 and 128 (inclusive)
    block_m_log2_end = 9 if is_hip() else 8
    block_m_num = block_m_log2_end - block_m_log2_start
    if n_gates <= n_expts_tot:
        max_n_tiles = n_gates
    else:
        max_n_tiles = n_expts_tot - 1 - ((n_expts_tot - n_gates - 1) // 2**block_m_log2_start)
    # allocate memory
    pad = lambda x: cdiv(x, MEMSET_BLOCK) * MEMSET_BLOCK
    dtype = torch.int32

    token_offs_combined = torch.empty((block_m_num + 1, pad(n_expts_tot + 1)), dtype=dtype, device=device)

    token_offs_raw = token_offs_combined[0][:n_expts_tot + 1]
    token_offs_pad = token_offs_combined[1:]

    block_pid_map = torch.empty((block_m_num, pad(max_n_tiles)), dtype=dtype, device=device)
    memset_grid = torch.numel(block_pid_map) // MEMSET_BLOCK  # exact division
    # compute outputs
    token_offs_pad = token_offs_pad[:, :n_expts_tot + 1]
    block_pid_map = block_pid_map[:, :max_n_tiles]

    blocks1 = memset_grid + block_m_num + 1
    blocks2 = n_expts_tot * block_m_num
    return token_offs_combined, token_offs_raw, token_offs_pad, block_pid_map, blocks1, blocks2, MEMSET_BLOCK, HIST2_BLOCK_M, block_m_log2_start, block_m_num


def _unpack_into_dict(x):

    block_m_log2_end = block_m_log2_start + x.shape[0]
    x = {2**j: x[i, :] for i, j in enumerate(range(block_m_log2_start, block_m_log2_end))}
    return x


def compute_expt_data(expt_hist, n_expts_tot, n_gates):

    if expt_hist is None:
        return ExptData(None, None, None, None)

    # this just computes the kernel arguments:
    token_offs_combined, token_offs_raw, token_offs_pad, block_pid_map, blocks1, blocks2, MEMSET_BLOCK, HIST2_BLOCK_M, block_m_log2_start, block_m_num = _compute_expt_data_internal(
        expt_hist, n_expts_tot, n_gates)

    _expt_data_memset[(blocks1, )](
        expt_hist, n_expts_tot,  #
        token_offs_combined, token_offs_combined.stride(0),  #
        block_pid_map,  #
        block_m_log2_start, SIZES=block_m_num, BLOCK=MEMSET_BLOCK,  # optimization parameters
        num_warps=4)
    _expt_data_compute[(blocks2, )](
        expt_hist, token_offs_pad, token_offs_pad.stride(0), block_pid_map, block_pid_map.stride(0),  # outputs
        block_m_log2_start, SIZES=block_m_num, BLOCK=HIST2_BLOCK_M,  # optimization parameters
        num_warps=4)

    token_offs_pad = _unpack_into_dict(token_offs_pad)
    block_pid_map = _unpack_into_dict(block_pid_map)
    return ExptData(expt_hist, token_offs_raw, token_offs_pad, block_pid_map)


# --------------------------
# routing
# --------------------------


def routing_from_bitmatrix(bitmatrix, expt_scal, expt_indx, n_expts_tot, n_expts_act):
    hist, topk_indx, gate_indx, gate_scal, token_offs_raw, token_offs_pad, block_pid_map = sort_tokens(
        expt_scal, expt_indx, n_expts_tot, bitmatrix)
    token_offs_pad = _unpack_into_dict(token_offs_pad)
    block_pid_map = _unpack_into_dict(block_pid_map)
    expt_data = ExptData(hist, token_offs_raw, token_offs_pad, block_pid_map)

    # pack the matmul data structure
    gather_indx = GatherIndx(src_indx=topk_indx, dst_indx=gate_indx)
    scatter_indx = ScatterIndx(src_indx=gate_indx, dst_indx=topk_indx)
    return RoutingData(gate_scal, hist, n_expts_tot, n_expts_act, expt_data), gather_indx, scatter_indx


def routing(logits, n_expts_act, sm_first=False, expt_indx=None, simulated_ep=1, n_rows=None):
    from .topk import topk
    if sm_first:
        logits = torch.softmax(logits, dim=-1)
    expt_scal, expt_indx, bitmatrix = topk(logits, n_expts_act,  #
                                           apply_softmax=not sm_first, y_indx=expt_indx, n_rows=n_rows)
    n_expts_tot = logits.shape[-1] // simulated_ep
    # mutate bitmatrix
    if simulated_ep > 1:
        expt_scal, expt_indx, bitmatrix = prune_routing(expt_scal, expt_indx, bitmatrix, logits.shape[-1], simulated_ep)

    return routing_from_bitmatrix(bitmatrix, expt_scal, expt_indx, n_expts_tot, n_expts_act)


# --------------------------
# torch reference
# --------------------------


def compute_expt_data_torch(hist, n_expts_tot, n_gates):
    # offset for each experts
    device = hist.device
    token_offs_raw = torch.cumsum(hist, dim=0)
    token_offs_raw = torch.cat((torch.zeros(1, device=device), token_offs_raw))
    token_offs_raw = token_offs_raw.int()
    # maximum number of tiles for all values of `block_m` considered
    block_ms = [16, 32, 64, 128]
    if is_hip():
        block_ms.append(256)
    if n_gates <= n_expts_tot:
        max_n_tiles = n_gates
    else:
        # ceil_div(n_gates - n_experts + 1, d_tile) + n_experts - 1
        # ceil_div(x, y): -(-x // y)
        max_n_tiles = n_expts_tot - 1 - ((n_expts_tot - n_gates - 1) // min(block_ms))
    # fill up tile offset/infos for each block
    token_offs_pad = dict()
    block_pid_map = dict()
    for block_m in block_ms:
        n_tiles = (hist + block_m - 1) // block_m  # matmul blocks needed
        token_offs_pad[block_m] = torch.cumsum(n_tiles, dim=0)
        token_offs_pad[block_m] = torch.cat((torch.zeros(1, device=device), token_offs_pad[block_m]))
        token_offs_pad[block_m] = token_offs_pad[block_m].int()
        # compute data required to drive ragged batch matmul
        block_pid_map[block_m] = -torch.ones(max_n_tiles, dtype=torch.int32, device=device)

        # for e in range(n_expts_tot):
        #     offset = token_offs_pad[block_m][e]
        #     for b in range(n_tiles[e]):
        #         block_pid_map[block_m][offset + b] = (b << 16) + e

        col = torch.arange(max_n_tiles, device=device)
        map_vals = torch.arange(n_expts_tot, device=device)[:, None] + (col << 16)[None, :]
        map_idxs = token_offs_pad[block_m][:-1, None] + col[None, :]
        mask = col[None, :] < n_tiles[:, None]
        block_pid_map[block_m].index_put_((map_idxs[mask], ), map_vals.int()[mask])
    return ExptData(hist, token_offs_raw, token_offs_pad, block_pid_map)


def topk_torch(vals, k, expt_indx, has_user_provided_indx=False):
    # topk of experts
    if has_user_provided_indx:
        tk_indx = expt_indx
    else:
        tk_indx = torch.argsort(-vals, dim=1, stable=True)[:, :k]
    tk_indx = tk_indx.long()
    tk_val = torch.take_along_dim(vals, tk_indx, dim=1)
    tk_indx = tk_indx.int()
    return tk_val, tk_indx


def routing_torch(logits, n_expts_act, sm_first=False, expt_indx=None, n_rows=None):
    has_user_provided_indx = expt_indx is not None
    n_gates_pad = logits.shape[0] * n_expts_act

    if n_rows is not None:
        logits = logits[:n_rows, :]
    _, n_expts_tot = logits.shape
    if sm_first:
        logits = torch.softmax(logits, dim=-1)
    expt_scal, expt_indx = topk_torch(logits, n_expts_act, expt_indx, has_user_provided_indx=has_user_provided_indx)
    if not sm_first:
        expt_scal = torch.softmax(expt_scal, dim=-1)
    # sort each token's selections by expert
    if not has_user_provided_indx:
        expt_indx, sort_indices = torch.sort(expt_indx, dim=1)
        expt_scal = torch.gather(expt_scal, 1, sort_indices)
    # flatten topk data
    expt_scal = expt_scal.reshape(-1)
    expt_indx = expt_indx.reshape(-1).to(torch.int32)
    # sort by expert_id so experts are contiguous for the matmul
    topk_indx = torch.argsort(expt_indx, stable=True)
    gate_indx = torch.argsort(topk_indx, stable=True)
    gate_scal = expt_scal[topk_indx]
    hist = torch.histc(expt_indx, bins=n_expts_tot, max=n_expts_tot - 1).int()  # histogram of tokens over experts
    # pack the matmul data structure
    gather_indx = GatherIndx(src_indx=topk_indx.int(), dst_indx=gate_indx.int())
    scatter_indx = ScatterIndx(src_indx=gate_indx.int(), dst_indx=topk_indx.int())
    # compute expt_data
    expt_data = compute_expt_data_torch(hist, n_expts_tot, n_gates_pad)
    return RoutingData(gate_scal, hist, n_expts_tot, n_expts_act, expt_data), gather_indx, scatter_indx
