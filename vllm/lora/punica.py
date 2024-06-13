# Based on code from https://github.com/punica-ai/punica

from typing import Optional

import torch

from vllm.lora.ops.bgmv_expand import bgmv_expand
from vllm.lora.ops.bgmv_expand_slice import bgmv_expand_slice
from vllm.lora.ops.bgmv_shrink import bgmv_shrink
from vllm.lora.ops.sgmv_expand import sgmv_expand
from vllm.lora.ops.sgmv_expand_slice import sgmv_expand_slice
from vllm.lora.ops.sgmv_shrink import sgmv_shrink


def _raise_import_error(e):
    if torch.cuda.get_device_capability() < (8, 0):
        raise ImportError(
            "punica LoRA kernels require compute capability >= 8.0") from e
    else:
        raise ImportError(
            "punica LoRA kernels could not be imported. If you built vLLM "
            "from source, make sure VLLM_INSTALL_PUNICA_KERNELS=1 env var "
            "was set.") from e


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
    try:
        import vllm._punica_C as punica_kernels
    except ImportError as e:
        _raise_import_error(e)

    punica_kernels.dispatch_bgmv(y, x, w_t_all, indicies, layer_idx, scale)


def dispatch_bgmv_low_level(
    y: torch.Tensor,
    x: torch.Tensor,
    w_t_all: torch.Tensor,
    indicies: torch.LongTensor,
    layer_idx: int,
    scale: float,
    y_offset: int,
    y_slice_size: int,
):
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
    try:
        import vllm._punica_C as punica_kernels
    except ImportError as e:
        _raise_import_error(e)
    punica_kernels.dispatch_bgmv_low_level(
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


def add_lora(
    y: torch.Tensor,
    x: torch.Tensor,
    wa_t_all: torch.Tensor,
    wb_t_all: torch.Tensor,
    indicies: torch.LongTensor,
    layer_idx: int,
    scale: float,
    *,
    buffer: Optional[torch.Tensor] = None,
):
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
    try:
        import vllm._punica_C as punica_kernels
    except ImportError as e:
        _raise_import_error(e)

    r = wb_t_all.size(-1)
    if buffer is None:
        # We set the buffer to be float32 by default to avoid
        # numerical inaccuracies that would otherwise happen
        # due to downcasting.
        buffer = torch.zeros((x.size(0), r),
                             dtype=torch.float32,
                             device=x.device)
    punica_kernels.dispatch_bgmv(buffer, x, wa_t_all, indicies, layer_idx, 1.0)
    punica_kernels.dispatch_bgmv(y, buffer, wb_t_all, indicies, layer_idx,
                                 scale)


def add_lora_slice(
    y: torch.Tensor,
    x: torch.Tensor,
    wa_t_all: torch.Tensor,
    wb_t_all: torch.Tensor,
    indicies: torch.LongTensor,
    layer_idx: int,
    scale: float,
    y_offset: int,
    y_slice_size: int,
    *,
    buffer: Optional[torch.Tensor] = None,
):
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
    try:
        import vllm._punica_C as punica_kernels
    except ImportError as e:
        _raise_import_error(e)

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


def add_lora_triton(
    y: torch.Tensor,
    x: torch.Tensor,
    wa_t_all: torch.Tensor,
    wb_t_all: torch.Tensor,
    b_seq_start_tensor: torch.Tensor,
    seq_length_tensor: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    batch_size: int,
    max_length: int,
    layer_idx: int,
    scale: float,
    is_prefilling: bool,
    *,
    buffer: Optional[torch.Tensor] = None,
):
    """
    Semantics:
      y[i] += (
          x[i].unsqueeze(0)
          @ wa_t_all[lora_index_tensor[i], layer_idx, :, :].transpose(-1, -2)
          @ wb_t_all[lora_index_tensor[i], layer_idx, :, :].transpose(-1, -2)
          * scale
        ).squeeze(0)
    Args:
        y (torch.Tensor):  (batch_size, output_dim).Will be changed in-place.
        x (torch.Tensor):  (batch_size, hidden_dim)
        wa_t_all (torch.Tensor):  (num_loras, lora_rank, hidden_dim)
        wb_t_all (torch.Tensor): (num_loras, output_dim, lora_rank)
        b_seq_start_tensor (torch.Tensor): (batch_size,). The cumulative
            sequence lengths of the sequences in the batch, used to index
            into sequence. E.g.,if the sequence length is [4, 6], it is
            [0, 4]. Used only during the prefilling stage.
        seq_length_tensor (torch.Tensor): batch_size,). record the sequence
            length of the sequences in the batch. Used only during the
            prefilling stage.
        lora_index_tensor (torch.Tensor): (batch_size,). The LoRA index
            corresponding to each batch
        batch_size (int): batch size. Used only during the prefilling stage.
        max_length (int):  maximum seq length in the batch.Used only during the
            prefilling stage.
        layer_idx (int): Layer index of LoRA weights.
        scale (float):  Scaling factor.
        is_prefilling (bool): True indicates the prefilling stage, while False
        indicates the decoding stage."
        buffer (Optional[torch.Tensor], optional): (batch_size,rank)
    """
    r = wb_t_all.size(-1)
    if buffer is None:
        # We set the buffer to be float32 by default ,refer to:
        # https://github.com/triton-lang/triton/issues/1387

        buffer = torch.zeros((x.size(0), r),
                             dtype=torch.float32,
                             device=x.device)
    if is_prefilling:
        _lora_sgmv(
            y,
            x,
            wa_t_all,
            wb_t_all,
            b_seq_start_tensor,
            seq_length_tensor,
            lora_indices_tensor,
            batch_size,
            max_length,
            layer_idx,
            scale,
            buffer=buffer,
        )
    else:
        _lora_bgmv(
            y,
            x,
            wa_t_all,
            wb_t_all,
            lora_indices_tensor,
            layer_idx,
            scale,
            buffer=buffer,
        )


def _lora_sgmv(
    y: torch.Tensor,
    x: torch.Tensor,
    wa_t_all: torch.Tensor,
    wb_t_all: torch.Tensor,
    b_seq_start_tensor: torch.Tensor,
    seq_length_tensor: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    batch_size: int,
    max_length: int,
    layer_idx: int,
    scale: float,
    buffer: torch.Tensor,
):
    sgmv_shrink(
        x,
        wa_t_all,
        buffer,
        b_seq_start_tensor,
        seq_length_tensor,
        lora_indices_tensor,
        batch_size,
        max_length,
        scale,
    )
    sgmv_expand(
        buffer,
        wb_t_all,
        y,
        b_seq_start_tensor,
        seq_length_tensor,
        lora_indices_tensor,
        batch_size,
        max_length,
        add_inputs=True,
    )


def _lora_bgmv(
    y: torch.Tensor,
    x: torch.Tensor,
    wa_t_all: torch.Tensor,
    wb_t_all: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    layer_idx: int,
    scale: float,
    buffer: torch.Tensor,
):
    bgmv_shrink(x, wa_t_all, buffer, lora_indices_tensor, scale)
    bgmv_expand(buffer, wb_t_all, y, lora_indices_tensor, add_inputs=True)


def add_lora_triton_slice(
    y: torch.Tensor,
    x: torch.Tensor,
    wa_t_all: torch.Tensor,
    wb_t_all: torch.Tensor,
    b_seq_start_tensor: torch.Tensor,
    seq_length_tensor: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    batch_size: int,
    max_length: int,
    layer_idx: int,
    scale: float,
    y_offset: int,
    y_slice_size: int,
    is_prefilling: bool,
    *,
    buffer: Optional[torch.Tensor] = None,
):
    """
    Same as `add_lora_triton` but you can operate on slices of y.
    Pass whole y, define y_offset and y_slice_size.
    """
    # try:
    #     import vllm._punica_C as punica_kernels
    # except ImportError as e:
    #     _raise_import_error(e)

    r = wb_t_all.size(-1)
    if buffer is None:
        # We set the buffer to be float32 by default to avoid
        # numerical inaccuracies that would otherwise happen
        # due to downcasting.
        buffer = torch.zeros((x.size(0), r),
                             dtype=torch.float32,
                             device=x.device)
    if is_prefilling:
        _lora_sgmv_nslice(
            y,
            x,
            wa_t_all,
            wb_t_all,
            b_seq_start_tensor,
            seq_length_tensor,
            lora_indices_tensor,
            batch_size,
            max_length,
            layer_idx,
            scale,
            y_offset,
            y_slice_size,
            buffer,
        )
    else:
        _lora_bgmv_nslice(
            y,
            x,
            wa_t_all,
            wb_t_all,
            lora_indices_tensor,
            layer_idx,
            scale,
            y_offset,
            y_slice_size,
            buffer,
        )


def _lora_sgmv_nslice(
    y: torch.Tensor,
    x: torch.Tensor,
    wa_t_all: torch.Tensor,
    wb_t_all: torch.Tensor,
    b_seq_start_tensor: torch.Tensor,
    seq_length_tensor: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    batch_size: int,
    max_length: int,
    layer_idx: int,
    scale: float,
    y_offset: int,
    y_slice_size: int,
    buffer,
):
    sgmv_shrink(
        x,
        wa_t_all,
        buffer,
        b_seq_start_tensor,
        seq_length_tensor,
        lora_indices_tensor,
        batch_size,
        max_length,
        scale,
    )
    sgmv_expand_slice(
        buffer,
        wb_t_all,
        y,
        b_seq_start_tensor,
        seq_length_tensor,
        lora_indices_tensor,
        batch_size,
        max_length,
        y_offset,
        y_slice_size,
        add_inputs=True,
    )


def _lora_bgmv_nslice(
    y: torch.Tensor,
    x: torch.Tensor,
    wa_t_all: torch.Tensor,
    wb_t_all: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    layer_idx: int,
    scale: float,
    y_offset: int,
    y_slice_size: int,
    buffer,
):
    bgmv_shrink(x, wa_t_all, buffer, lora_indices_tensor, scale)
    bgmv_expand_slice(
        buffer,
        wb_t_all,
        y,
        lora_indices_tensor,
        y_offset,
        y_slice_size,
        add_inputs=True,
    )
