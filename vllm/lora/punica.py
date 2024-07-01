# Based on code from https://github.com/punica-ai/punica

from typing import Dict, Optional, Tuple

import torch

from vllm.lora.ops.bgmv_expand import bgmv_expand
from vllm.lora.ops.bgmv_expand_slice import bgmv_expand_slice
from vllm.lora.ops.bgmv_shrink import bgmv_shrink
from vllm.lora.ops.sgmv_expand import sgmv_expand
from vllm.lora.ops.sgmv_expand_slice import sgmv_expand_slice
from vllm.lora.ops.sgmv_shrink import sgmv_shrink

_PARAMS_CACHE: Dict[int, Tuple] = {}

def _compute_params(
    token_lora_tensor: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int, ]:
    """
    Get the information required for the sgmv kernel. With the  features:
    1. If consecutive requests in the batch use the same LoRA, this function 
    will combine them into a single request, improving sgmv kernel inference 
    performance.
    2. At the beginning of each prefilling stage inference, recalculations are 
    needed based on the input, but only once. 
    """
    pointer = token_lora_tensor.data_ptr()
    if pointer not in _PARAMS_CACHE:
        lora_indices_tensor, seq_length_tensor = torch.unique_consecutive(
            token_lora_tensor, return_counts=True)
        cum_result = torch.cumsum(seq_length_tensor, dim=0)
        b_seq_start_tensor = torch.zeros_like(seq_length_tensor)
        b_seq_start_tensor[1:].copy_(cum_result[:-1])
        max_length = seq_length_tensor.max().item()
        batch_size = lora_indices_tensor.size(0)
        _PARAMS_CACHE[pointer] = (
            b_seq_start_tensor,
            seq_length_tensor,
            lora_indices_tensor,
            batch_size,
            max_length,
        )
    return _PARAMS_CACHE[pointer]


def reset_params_cache():
    """At the beginning of the prefilling stage, we need  clear the
    cache explicitly
    """
    _PARAMS_CACHE.clear()
    torch.cuda.empty_cache()


def _get_prefilling_params(token_lora_tensor: torch.Tensor,
                           cache_clear: bool = False):
    if cache_clear:
        reset_params_cache()
    return _compute_params(token_lora_tensor)


def add_shrink(
    y: torch.Tensor,
    x: torch.Tensor,
    w_t_all: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    layer_idx: int,
    scale: float,
    is_prefilling: bool,
    cache_clear: bool = False,
):
    """
    y=x@w_t_all
    When `is_prefilling` is True, will launch `sgmv_shrink`
    """
    if is_prefilling:
        (
            b_seq_start_tensor,
            seq_length_tensor,
            last_lora_indices_tensor,
            batch_size,
            max_length,
        ) = _get_prefilling_params(lora_indices_tensor, cache_clear)
        sgmv_shrink(
            x,
            w_t_all,
            y,
            b_seq_start_tensor,
            seq_length_tensor,
            last_lora_indices_tensor,
            batch_size,
            max_length,
            scale,
        )
    else:
        bgmv_shrink(x, w_t_all, y, lora_indices_tensor, scale)


def add_expand(
    y: torch.Tensor,
    x: torch.Tensor,
    w_t_all: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    layer_idx: int,
    is_prefilling: bool,
    add_input: bool = True,
    cache_clear: bool = False,
):
    """
    y+=x@w_t_all
    When `is_prefilling` is True, will launch `sgmv_expand`, 
    """
    if is_prefilling:
        (
            b_seq_start_tensor,
            seq_length_tensor,
            last_lora_indices_tensor,
            batch_size,
            max_length,
        ) = _get_prefilling_params(lora_indices_tensor, cache_clear)
        sgmv_expand(
            x,
            w_t_all,
            y,
            b_seq_start_tensor,
            seq_length_tensor,
            last_lora_indices_tensor,
            batch_size,
            max_length,
            add_input,
        )
    else:
        bgmv_expand(x, w_t_all, y, lora_indices_tensor, add_inputs=add_input)


def add_expand_slice(
    y: torch.Tensor,
    x: torch.Tensor,
    w_t_all: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    layer_idx: int,
    is_prefilling: bool,
    y_offset: Optional[int],
    y_slice_size: Optional[int],
    add_input: bool = True,
    cache_clear: bool = False,
):
    """
    y+=x@w_t_all
    """
    if is_prefilling:
        (
            b_seq_start_tensor,
            seq_length_tensor,
            last_lora_indices_tensor,
            batch_size,
            max_length,
        ) = _get_prefilling_params(lora_indices_tensor, cache_clear)
        sgmv_expand_slice(
            x,
            w_t_all,
            y,
            b_seq_start_tensor,
            seq_length_tensor,
            last_lora_indices_tensor,
            batch_size,
            max_length,
            y_offset,
            y_slice_size,
            add_input,
        )
    else:
        bgmv_expand_slice(
            x,
            w_t_all,
            y,
            lora_indices_tensor,
            y_offset,
            y_slice_size,
            add_inputs=add_input,
        )


def add_lora(
    y: torch.Tensor,
    x: torch.Tensor,
    wa_t_all: torch.Tensor,
    wb_t_all: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    layer_idx: int,
    scale: float,
    is_prefilling: bool,
    y_offset: Optional[int] = None,
    y_slice_size: Optional[int] = None,
    *,
    buffer: Optional[torch.Tensor] = None,
    cache_clear: bool = False,
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
        y (torch.Tensor):  Output tensor. Will be changed in-place.
        x (torch.Tensor): Input tensor
        wa_t_all (torch.Tensor): lora_a's weight
        wb_t_all (torch.Tensor): lora_b's weight
        lora_indices_tensor (torch.Tensor): _description_
        layer_idx (int): Layer index of LoRA weights.
        scale (float): Scaling factor.
        is_prefilling (bool): prefiling stage
        y_offset (Optional[int], optional): Offset to apply to the starting 
            column of y.
        y_slice_size (Optional[int], optional): Size of the y column slice..
        buffer (Optional[torch.Tensor], optional): Defaults to None.
        cache_clear (bool, optional):  Defaults to False.
    """

    r = wb_t_all.size(-1)
    if buffer is None:
        # We set the buffer to be float32 by default ,refer to:
        # https://github.com/triton-lang/triton/issues/1387
        buffer = torch.zeros((x.size(0), r),
                             dtype=torch.float32,
                             device=x.device)

    add_shrink(
        buffer,
        x,
        wa_t_all,
        lora_indices_tensor,
        0,
        scale,
        is_prefilling,
        cache_clear=cache_clear,
    )
    if y_offset is None and y_slice_size is None:
        add_expand(
            y,
            buffer,
            wb_t_all,
            lora_indices_tensor,
            0,
            is_prefilling,
            add_input=True,
            cache_clear=cache_clear,
        )
    else:
        add_expand_slice(
            y,
            buffer,
            wb_t_all,
            lora_indices_tensor,
            0,
            is_prefilling,
            y_offset,
            y_slice_size,
            add_input=True,
            cache_clear=cache_clear,
        )
