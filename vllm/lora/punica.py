"""
Based on:
Chen, L., Ye, Z., Wu, Y., Zhuo, D., Ceze, L., & Krishnamurthy, A. (2023). 
Punica: Multi-Tenant LoRA Serving. 
https://arxiv.org/abs/2310.18547
"""

from typing import Optional, Tuple

import torch

from vllm.lora.ops.bgmv_expand import bgmv_expand
from vllm.lora.ops.bgmv_expand_slice import bgmv_expand_slice
from vllm.lora.ops.bgmv_shrink import bgmv_shrink
from vllm.lora.ops.sgmv_expand import sgmv_expand
from vllm.lora.ops.sgmv_expand_slice import sgmv_expand_slice
from vllm.lora.ops.sgmv_shrink import sgmv_shrink


def _compute_meta(
    token_lora_tensor: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int, ]:
    """
    Get the information required for the sgmv kernel. With the  features:
    1. If consecutive requests in the batch use the same LoRA, this function 
    will combine them into a single request, improving sgmv kernel inference 
    performance.
    2. At the beginning of each prefill stage inference, recalculations are 
    needed based on the input, but only once. 
    """

    lora_indices_tensor, seq_length_tensor = torch.unique_consecutive(
        token_lora_tensor, return_counts=True)
    cum_result = torch.cumsum(seq_length_tensor, dim=0)
    b_seq_start_tensor = torch.zeros_like(seq_length_tensor)
    b_seq_start_tensor[1:].copy_(cum_result[:-1])
    max_length = seq_length_tensor.max().item()
    batch_size = lora_indices_tensor.size(0)
    return (
        b_seq_start_tensor,
        seq_length_tensor,
        lora_indices_tensor,
        batch_size,
        max_length,
    )


class PrefillHelper:
    """PrefillHelper is designed to manage and provide metadata for the sgmv 
    kernel during  prefill stage, utilizing the singleton pattern to guarantee 
    the existence of only one instance of the class.
    """
    _instance: Optional["PrefillHelper"] = None
    initialized: bool

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.initialized = False
        return cls._instance

    def __init__(self, max_batches: int = 256, device: str = "cuda"):
        """
        Args:
            max_batches (int, optional):  the maximum batch to pre-allocate.
                Defaults to 256.
            device (str, optional): Defaults to "cuda".
        """
        if not self.initialized:
            self.initialized = True
            # these attributes are the information required for sgmv kernel
            self.b_seq_start_tensor = torch.zeros(max_batches,
                                                  dtype=torch.long,
                                                  device=device)
            self.seq_length_tensor = torch.empty(max_batches,
                                                 dtype=torch.long,
                                                 device=device)
            self.lora_indices_tensor = torch.empty(max_batches,
                                                   dtype=torch.long,
                                                   device=device)
            self.max_length: int = 0
            self.batch_size: int = -1

    def _update_metada(self, token_lora_tensor: torch.Tensor) -> None:

        (b_seq_start_tensor, seq_length_tensor, lora_indices_tensor,
         batch_size, max_length) = _compute_meta(token_lora_tensor)

        self.b_seq_start_tensor[:b_seq_start_tensor.shape[0]].copy_(
            b_seq_start_tensor)
        self.seq_length_tensor[:seq_length_tensor.shape[0]].copy_(
            seq_length_tensor)
        self.lora_indices_tensor[:lora_indices_tensor.shape[0]].copy_(
            lora_indices_tensor)
        self.batch_size = batch_size
        self.max_length = max_length

    def get_metadata(
        self,
        token_lora_tensor: torch.Tensor,
        need_update: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int, ]:

        #Need to recalculate and fill metadata.
        if need_update:
            self._update_metada(token_lora_tensor)

        return (self.b_seq_start_tensor[:self.batch_size],
                self.seq_length_tensor[:self.batch_size],
                self.lora_indices_tensor[:self.batch_size], self.batch_size,
                self.max_length)


def get_prefill_meta(token_lora_tensor: torch.Tensor,
                     need_update: bool = False):
    prefill_helper = PrefillHelper(max_batches=256,
                                   device=str(token_lora_tensor.device))
    return prefill_helper.get_metadata(token_lora_tensor, need_update)


def shrink_prefill(
    y: torch.Tensor,
    x: torch.Tensor,
    w_t_all: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    layer_idx: int,
    scale: float,
    need_update: bool = False,
):
    (
        b_seq_start_tensor,
        seq_length_tensor,
        last_lora_indices_tensor,
        batch_size,
        max_length,
    ) = get_prefill_meta(lora_indices_tensor, need_update)
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


def shrink_decode(
    y: torch.Tensor,
    x: torch.Tensor,
    w_t_all: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    layer_idx: int,
    scale: float,
):
    bgmv_shrink(x, w_t_all, y, lora_indices_tensor, scale)


def expand_prefill(
    y: torch.Tensor,
    x: torch.Tensor,
    w_t_all: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    layer_idx: int,
    add_input: bool,
    need_update: bool = False,
):
    (
        b_seq_start_tensor,
        seq_length_tensor,
        last_lora_indices_tensor,
        batch_size,
        max_length,
    ) = get_prefill_meta(lora_indices_tensor, need_update)
    sgmv_expand(x, w_t_all, y, b_seq_start_tensor, seq_length_tensor,
                last_lora_indices_tensor, batch_size, max_length, add_input)


def expand_decode(
    y: torch.Tensor,
    x: torch.Tensor,
    w_t_all: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    layer_idx: int,
    add_input: bool,
):
    bgmv_expand(x, w_t_all, y, lora_indices_tensor, add_input)


def expand_slice_prefill(
    y: torch.Tensor,
    x: torch.Tensor,
    w_t_all: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    layer_idx: int,
    y_offset: Optional[int],
    y_slice_size: Optional[int],
    add_input: bool,
    need_update: bool = False,
):
    (
        b_seq_start_tensor,
        seq_length_tensor,
        last_lora_indices_tensor,
        batch_size,
        max_length,
    ) = get_prefill_meta(lora_indices_tensor, need_update)
    sgmv_expand_slice(x, w_t_all, y, b_seq_start_tensor, seq_length_tensor,
                      last_lora_indices_tensor, batch_size, max_length,
                      y_offset, y_slice_size, add_input)


def expand_slice_decode(y: torch.Tensor, x: torch.Tensor,
                        w_t_all: torch.Tensor,
                        lora_indices_tensor: torch.Tensor, layer_idx: int,
                        y_offset: Optional[int], y_slice_size: Optional[int],
                        add_input: bool):
    bgmv_expand_slice(x, w_t_all, y, lora_indices_tensor, y_offset,
                      y_slice_size, add_input)


def add_shrink(
    y: torch.Tensor,
    x: torch.Tensor,
    w_t_all: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    layer_idx: int,
    scale: float,
    is_prefill: bool,
    need_update: bool = False,
):
    """
    Perform the ` y+=x@w_t_all` computation, which is suitable for the 
    GEMM of lora'a.
    When `is_prefill is` true, it indicates that it is currently the 
    prefill stage, and the `shrink_prefill` function should be called. 
    Otherwise, it is the decode stage, and the shrink_decode function 
    should be called.
    """
    if is_prefill:
        shrink_prefill(y, x, w_t_all, lora_indices_tensor, layer_idx, scale,
                       need_update)
    else:
        shrink_decode(y, x, w_t_all, lora_indices_tensor, layer_idx, scale)


def add_expand(
    y: torch.Tensor,
    x: torch.Tensor,
    w_t_all: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    layer_idx: int,
    is_prefill: bool,
    add_input: bool = True,
    need_update: bool = False,
):
    """
    Perform the ` y+=x@w_t_all` computation, which is suitable for the 
    GEMM of lora'b.
    When `is_prefill` is true, it indicates that it is currently the 
    prefill stage, and the `expand_prefill` function should be called. 
    Otherwise, it is the decode stage, and the expand_decode function 
    should be called.
    """
    if is_prefill:
        expand_prefill(y, x, w_t_all, lora_indices_tensor, layer_idx,
                       add_input, need_update)
    else:
        expand_decode(y, x, w_t_all, lora_indices_tensor, layer_idx, add_input)


def add_expand_slice(
    y: torch.Tensor,
    x: torch.Tensor,
    w_t_all: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    layer_idx: int,
    is_prefill: bool,
    y_offset: Optional[int],
    y_slice_size: Optional[int],
    add_input: bool = True,
    need_update: bool = False,
):
    """
    Similar to `add_expand`
    """
    if is_prefill:
        expand_slice_prefill(y, x, w_t_all, lora_indices_tensor, layer_idx,
                             y_offset, y_slice_size, add_input, need_update)
    else:
        expand_slice_decode(y, x, w_t_all, lora_indices_tensor, layer_idx,
                            y_offset, y_slice_size, add_input)


def add_lora(
    y: torch.Tensor,
    x: torch.Tensor,
    wa_t_all: torch.Tensor,
    wb_t_all: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    layer_idx: int,
    scale: float,
    is_prefill: bool,
    y_offset: Optional[int] = None,
    y_slice_size: Optional[int] = None,
    *,
    buffer: Optional[torch.Tensor] = None,
    need_update: bool = False,
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
        is_prefill (bool): prefiling stage
        y_offset (Optional[int], optional): Offset to apply to the starting 
            column of y.
        y_slice_size (Optional[int], optional): Size of the y column slice..
        buffer (Optional[torch.Tensor], optional): Defaults to None.
        need_update (bool, optional): Indicates whether updating sgmv metadata 
            is needed. Defaults to False.
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
        is_prefill,
        need_update=need_update,
    )
    if y_offset is None and y_slice_size is None:
        add_expand(y,
                   buffer,
                   wb_t_all,
                   lora_indices_tensor,
                   0,
                   is_prefill,
                   add_input=True,
                   need_update=need_update)
    else:
        add_expand_slice(y,
                         buffer,
                         wb_t_all,
                         lora_indices_tensor,
                         0,
                         is_prefill,
                         y_offset,
                         y_slice_size,
                         add_input=True,
                         need_update=need_update)
