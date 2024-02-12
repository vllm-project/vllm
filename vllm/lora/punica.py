# Based on code from https://github.com/punica-ai/punica

from typing import Optional

import torch

import_exc = None

try:
    import vllm._punica_C as punica_kernels
except ImportError as e:
    import_exc = e

if import_exc is None:

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
        punica_kernels.dispatch_bgmv(y, x, w_t_all, indicies, layer_idx, scale)

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
        r = wb_t_all.size(-1)
        if buffer is None:
            # We set the buffer to be float32 by default to avoid
            # numerical innacuracies that would otherwise happen
            # due to downcasting.
            buffer = torch.zeros((x.size(0), r),
                                 dtype=torch.float32,
                                 device=x.device)
        punica_kernels.dispatch_bgmv(buffer, x, wa_t_all, indicies, layer_idx,
                                     1.0)
        punica_kernels.dispatch_bgmv(y, buffer, wb_t_all, indicies, layer_idx,
                                     scale)

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
        r = wb_t_all.size(-1)
        if buffer is None:
            # We set the buffer to be float32 by default to avoid
            # numerical inaccuracies that would otherwise happen
            # due to downcasting.
            buffer = torch.zeros((x.size(0), r),
                                 dtype=torch.float32,
                                 device=x.device)
        punica_kernels.dispatch_bgmv_low_level(
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
        punica_kernels.dispatch_bgmv_low_level(
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

else:

    def _raise_exc(
        *args,  # pylint: disable=unused-argument
        **kwargs  # pylint: disable=unused-argument
    ):
        if torch.cuda.get_device_capability() < (8, 0):
            raise ImportError("punica LoRA kernels require compute "
                              "capability>=8.0") from import_exc
        else:
            raise ImportError(
                "punica LoRA kernels could not be imported. If you built vLLM "
                "from source, make sure VLLM_INSTALL_PUNICA_KERNELS=1 env var "
                "was set.") from import_exc

    bgmv = _raise_exc
    add_lora = _raise_exc
    add_lora_slice = _raise_exc

__all__ = [
    "bgmv",
    "add_lora",
    "add_lora_slice",
]
