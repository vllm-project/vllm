import torch
from .compaction_details._masked_compaction import _masked_compaction
from .tensor import Bitmatrix


def compaction(yv, yi, bitmask, sentinel=-1):
    """
    Return compacted copies of *yv* and *yi* based on a per-row bitmask.

    Only the elements whose index appears among the active bits of *bitmask*
    are kept; the rest are replaced by *sentinel*.  Kept elements preserve
    their original left-to-right order.

    Parameters
    ----------
    yv : torch.Tensor, shape (B, K)
        Values tensor.
    yi : torch.Tensor, shape (B, K), dtype torch.long
        Integer indices (0 ≤ index < 32) associated with *yv*.
    bitmask : torch.Tensor, shape (B,) **or** (B, 32)
        Per-row mask of active indices.  See the in-place version for details.
    sentinel : int, default -1
        Value written into dropped positions of the returned tensors.

    Returns
    -------
    (yv_out, yi_out) : Tuple[torch.Tensor, torch.Tensor], each shape (B, K)
        New tensors with the same dtype/device as the inputs.

    """

    n_rows, n_cols = yi.shape
    ret_yv = torch.empty_like(yv)
    ret_yi = torch.empty_like(yi)
    if isinstance(bitmask, Bitmatrix):
        bitmask = bitmask.storage.data

    _masked_compaction[(n_rows, )](
        yv, yi, bitmask, bitmask.stride(0), bitmask.stride(1),  # inputs
        ret_yv, ret_yi,  # outputs
        sentinel,  # sentinel
        K=n_cols  # constants
    )
    return ret_yv, ret_yi


def compaction_torch(yv: torch.Tensor, yi: torch.Tensor, bitmask: torch.Tensor, sentinel=-1):
    """
    reference implementation of `masked_compact`
    """
    B, K = yi.shape
    device = yi.device
    # Expand bitmask to a boolean matrix of active bits  (B, 32)
    w = (1 << torch.arange(32, device=device, dtype=bitmask.dtype))
    bits = (bitmask.unsqueeze(-1) & w) != 0
    mask = bits.flatten(start_dim=-2)  # or bits.reshape(B, -1)
    # For every yi element decide whether it should be kept
    keep = mask.gather(1, yi.long())
    # Build a stable permutation that brings all "keep" items forward
    #    False→0, True→1  ==> invert so kept==0, dropped==1, then argsort
    order = (~keep).to(torch.int).argsort(dim=1, stable=True)
    # Re‑order tensors according to above permutation
    yi_sorted = yi.gather(1, order)
    yv_sorted = yv.gather(1, order)
    # fill relevant positions with sentinel
    keep_sorted = keep.gather(1, order)
    yi_sorted[~keep_sorted] = sentinel
    yv_sorted[~keep_sorted] = sentinel
    return yv_sorted, yi_sorted
