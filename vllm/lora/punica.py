# Based on code from https://github.com/punica-ai/punica

from typing import Optional

import torch

from vllm import _custom_ops as ops
from vllm.platforms import current_platform


def _check_punica_support():
    if ops.is_custom_op_supported("_punica_C::dispatch_bgmv"):
        return

    if current_platform.get_device_capability() < (8, 0):
        raise ImportError(
            "punica LoRA kernels require compute capability >= 8.0")
    else:
        raise ImportError(
            "punica LoRA kernels could not be imported. If you built vLLM "
            "from source, make sure VLLM_INSTALL_PUNICA_KERNELS=1 env var "
            "was set.")


def bgmv(
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
    _check_punica_support()

    ops.dispatch_bgmv(y, x, w_t_all, indicies, layer_idx, scale)


def dispatch_bgmv_low_level(y: torch.Tensor, x: torch.Tensor,
                            w_t_all: torch.Tensor, indicies: torch.LongTensor,
                            layer_idx: int, scale: float, y_offset: int,
                            y_slice_size: int):
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
    _check_punica_support()

    ops.dispatch_bgmv_low_level(
        y,
        x,
        w_t_all,
        indicies,
        layer_idx,
        scale,
        x.size(1),
        y_slice_size,
        y_offset,
    )


def add_lora(y: torch.Tensor,
             x: torch.Tensor,
             wa_t_all: torch.Tensor,
             wb_t_all: torch.Tensor,
             indicies: torch.LongTensor,
             layer_idx: int,
             scale: float,
             *,
             buffer: Optional[torch.Tensor] = None):
    """
    Semantics:
      y[i] += (
          x[i].unsqueeze(0)
          @ wa_t_all[indices[i], layer_idx, :, :].transpose(-1, -2)
          @ wb_t_all[indices[i], layer_idx, :, :].transpose(-1, -2)
          * scale
        ).squeeze(0)

    Args:
      y: Shape: `[B, H2]`. Output vectors. Will be changed in-place.
      x: Shape: `[B, H1]`. Input vectors.
      wa_t_all: Shape: `[None, L, R, H1]`. All of the transposed
        LoRA A matrices.
      wb_t_all: Shape: `[None, L, H2, R]`. All of the transposed
        LoRA B matrices.
      indicies: Shape: `[B]`. Indices of the LoRA weights.
      layer_idx: Layer index of LoRA weights.
      scale: Scaling factor.
      buffer: Optional. Shape: `[B, R]`. Temporary buffer.
    """
    _check_punica_support()

    r = wb_t_all.size(-1)
    if buffer is None:
        # We set the buffer to be float32 by default to avoid
        # numerical inaccuracies that would otherwise happen
        # due to downcasting.
        buffer = torch.zeros((x.size(0), r),
                             dtype=torch.float32,
                             device=x.device)
    ops.dispatch_bgmv(buffer, x, wa_t_all, indicies, layer_idx, 1.0)
    ops.dispatch_bgmv(y, buffer, wb_t_all, indicies, layer_idx, scale)


def add_lora_slice(y: torch.Tensor,
                   x: torch.Tensor,
                   wa_t_all: torch.Tensor,
                   wb_t_all: torch.Tensor,
                   indicies: torch.LongTensor,
                   layer_idx: int,
                   scale: float,
                   y_offset: int,
                   y_slice_size: int,
                   *,
                   buffer: Optional[torch.Tensor] = None):
    """
    Same as `add_lora` but you can operate on slices of y.
    Pass whole y, define y_offset and y_slice_size.

    Semantics:
      y[i] += (
          x[i].unsqueeze(0)
          @ wa_t_all[indices[i], layer_idx, :, :].transpose(-1, -2)
          @ wb_t_all[indices[i], layer_idx, :, :].transpose(-1, -2)
          * scale
        ).squeeze(0)

    Args:
      y: Shape: `[B, H2]`. Output vectors. Will be changed in-place.
      x: Shape: `[B, H1]`. Input vectors.
      wa_t_all: Shape: `[None, L, R, H1]`. All of the transposed
        LoRA A matrices.
      wb_t_all: Shape: `[None, L, H2, R]`. All of the transposed
        LoRA B matrices.
      indicies: Shape: `[B]`. Indices of the LoRA weights.
      layer_idx: Layer index of LoRA weights.
      scale: Scaling factor.
      y_offset: Offset to apply to the starting column of y.
      y_slice_size: Size of the y column slice.
    """
    _check_punica_support()

    r = wb_t_all.size(-1)
    if buffer is None:
        # We set the buffer to be float32 by default to avoid
        # numerical inaccuracies that would otherwise happen
        # due to downcasting.
        buffer = torch.zeros((x.size(0), r),
                             dtype=torch.float32,
                             device=x.device)
    ops.dispatch_bgmv_low_level(
        buffer,
        x,
        wa_t_all,
        indicies,
        layer_idx,
        1.0,
        x.size(1),
        buffer.size(1),
        0,
    )
    ops.dispatch_bgmv_low_level(
        y,
        buffer,
        wb_t_all,
        indicies,
        layer_idx,
        scale,
        buffer.size(1),
        y_slice_size,
        y_offset,
    )
