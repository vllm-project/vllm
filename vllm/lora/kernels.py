# torch implementation of LoRA kernels.

from typing import Optional

import torch


def dispatch_bgmv(
    y: torch.Tensor,
    x: torch.Tensor,
    w_t_all: torch.Tensor,
    indicies: torch.LongTensor,
    layer_idx: int,
    scale: float,
):
    """
    Semantics:
      y[i] += (
          x[i].unsqueeze(0)
          @ w_t_all[indices[i], layer_idx, :, :].transpose(-1, -2)
          * scale
        ).squeeze(0)

    Args:
      y: Shape: `[B, H2]`. Output vectors. Will be changed in-place.
      x: Shape: `[B, H1]`. Input vectors.
      w_t_all: Shape: `[None, L, H2, H1]`. All of the transposed weight
        matrices.
      indicies: Shape: `[B]`. Indices of the weight matrices.
      layer_idx: Layer index of the weight matrices.
      scale: Scaling factor.
    """
    y += (
        x.unsqueeze(1)
        @ w_t_all[indicies, layer_idx, :, :].transpose(-1, -2).to(x.dtype)
        * scale
    ).squeeze(1)


def dispatch_bgmv_low_level(y: torch.Tensor, x: torch.Tensor,
                            w_t_all: torch.Tensor, indicies: torch.LongTensor,
                            layer_idx: int, scale: float, h_in: int, h_out :int,
                            y_offset: int):
    """
    Same as `bgmv` but you can operate on slices of y.
    Pass whole y, define y_offset and y_slice_size.

    Semantics:
      y[i] += (
          x[i].unsqueeze(0)
          @ w_t_all[indices[i], layer_idx, :, :].transpose(-1, -2)
          * scale
        ).squeeze(0)

    Args:
      y: Shape: `[B, H2]`. Output vectors. Will be changed in-place.
      x: Shape: `[B, H1]`. Input vectors.
      w_t_all: Shape: `[None, L, y_slice_size, H1]`. Column partition of
        all of the transposed LoRA matrices.
      indicies: Shape: `[B]`. Indices of the LoRA weights.
      layer_idx: Layer index of LoRA weights.
      scale: Scaling factor.
      y_offset: Offset to apply to the starting column of y.
      y_slice_size: Size of the y column slice.
    """
    y[:, y_offset:y_offset+h_out] += (x[:, :h_in].unsqueeze(1) @ w_t_all[indicies, layer_idx, :, :].transpose(-1, -2).to(x.dtype) * scale).squeeze(1)
