"""
Based on:
Chen, L., Ye, Z., Wu, Y., Zhuo, D., Ceze, L., & Krishnamurthy, A. (2023).
Punica: Multi-Tenant LoRA Serving.
https://arxiv.org/abs/2310.18547
"""

from typing import Callable, Optional, Tuple, Union, final

import torch

from vllm.lora.punica_base import PunicaWrapperBase
from vllm.triton_utils import HAS_TRITON

if HAS_TRITON:
    from vllm.lora.ops.bgmv_expand import bgmv_expand
    from vllm.lora.ops.bgmv_expand_slice import bgmv_expand_slice
    from vllm.lora.ops.bgmv_shrink import bgmv_shrink
    from vllm.lora.ops.sgmv_expand import sgmv_expand
    from vllm.lora.ops.sgmv_expand_slice import sgmv_expand_slice
    from vllm.lora.ops.sgmv_shrink import sgmv_shrink


@final
class PunicaWrapperGPU(PunicaWrapperBase):
    """
    PunicaWrapper is designed to manage and provide metadata for the punica
    kernel. The main function is to maintain the state information for
    Multi-LoRA, and to provide the interface for the punica kernel.
    """

    def __init__(self, max_num_batched_tokens: int, max_batches: int,
                 device: Union[torch.device, str], **kwargs):
        PunicaWrapperBase.__init__(self, max_num_batched_tokens, max_batches,
                                   device)

    def shrink_prefill(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        w_t_all: torch.Tensor,
        scale: float,
    ):
        #No LoRA request, so return directly
        if self.no_lora:
            return
        sgmv_shrink(
            x,
            w_t_all,
            y,
            *self.prefill_metadata,
            scale,
        )

    def shrink_decode(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        w_t_all: torch.Tensor,
        scale: float,
    ):
        bgmv_shrink(x, w_t_all, y, self.token_lora_indices, scale)

    def expand_prefill(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        w_t_all: torch.Tensor,
        add_input: bool,
    ):
        # No LoRA request, so return directly
        if self.no_lora:
            return
        sgmv_expand(
            x,
            w_t_all,
            y,
            *self.prefill_metadata,
            add_input,
        )

    def expand_decode(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        w_t_all: torch.Tensor,
        add_input: bool,
    ):
        bgmv_expand(x, w_t_all, y, self.token_lora_indices, add_input)

    def expand_slice_prefill(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        w_t_all: torch.Tensor,
        y_offset: Optional[int],
        y_slice_size: Optional[int],
        add_input: bool,
    ):
        # No LoRA request, so return directly
        if self.no_lora:
            return
        sgmv_expand_slice(
            x,
            w_t_all,
            y,
            *self.prefill_metadata,
            y_offset,
            y_slice_size,
            add_input,
        )

    def expand_slice_decode(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        w_t_all: torch.Tensor,
        y_offset: Optional[int],
        y_slice_size: Optional[int],
        add_input: bool,
    ):
        bgmv_expand_slice(
            x,
            w_t_all,
            y,
            self.token_lora_indices,
            y_offset,
            y_slice_size,
            add_input,
        )

    def add_shrink(self, y: torch.Tensor, x: torch.Tensor,
                   w_t_all: torch.Tensor, scale: float, **kwarg):
        """
        Perform the ` y+=x@w_t_all` computation, which is suitable for the
        GEMM of lora'a.
        When `is_prefill is` true, it indicates that it is currently the
        prefill stage, and the `shrink_prefill` function should be called.
        Otherwise, it is the decode stage, and the shrink_decode function
        should be called.
        """
        shrink_fun: Callable = (self.shrink_prefill
                                if self.is_prefill else self.shrink_decode)
        shrink_fun(y, x, w_t_all, scale)

    def add_expand(self,
                   y: torch.Tensor,
                   x: torch.Tensor,
                   w_t_all: torch.Tensor,
                   add_input: bool = True,
                   **kwarg):
        """
        Perform the ` y+=x@w_t_all` computation, which is suitable for the
        GEMM of lora'b.
        When `is_prefill` is true, it indicates that it is currently the
        prefill stage, and the `expand_prefill` function should be called.
        Otherwise, it is the decode stage, and the expand_decode function
        should be called.
        """

        expand_fun: Callable = (self.expand_prefill
                                if self.is_prefill else self.expand_decode)
        expand_fun(y, x, w_t_all, add_input)

    def add_expand_slice(self,
                         y: torch.Tensor,
                         x: torch.Tensor,
                         w_t_all: torch.Tensor,
                         y_offset: Optional[int],
                         y_slice_size: Optional[int],
                         add_input: bool = True,
                         **kwarg):
        """
        Similar to `add_expand`
        """

        expand_slice_fun: Callable = (self.expand_slice_prefill
                                      if self.is_prefill else
                                      self.expand_slice_decode)
        expand_slice_fun(y, x, w_t_all, y_offset, y_slice_size, add_input)

    def add_lora(self,
                 y: torch.Tensor,
                 x: torch.Tensor,
                 wa_t_all: torch.Tensor,
                 wb_t_all: torch.Tensor,
                 scale: float,
                 y_offset: Optional[int] = None,
                 y_slice_size: Optional[int] = None,
                 *,
                 buffer: Optional[torch.Tensor] = None,
                 **kwarg) -> None:
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
            scale (float): Scaling factor.
            y_offset (Optional[int], optional): Offset to apply to the starting
                column of y.
            y_slice_size (Optional[int], optional): Size of the y column slice.
            buffer (Optional[torch.Tensor], optional): Defaults to None.
        """
        y_org = y
        y = y.view(-1, y.shape[-1])
        x = x.view(-1, x.shape[-1])
        r = wb_t_all.size(-1)
        if buffer is None:
            # We set the buffer to be float32 by default ,refer to:
            # https://github.com/triton-lang/triton/issues/1387
            buffer = torch.zeros((x.size(0), r),
                                 dtype=torch.float32,
                                 device=x.device)

        self.add_shrink(buffer, x, wa_t_all, scale)
        if y_offset is None and y_slice_size is None:
            self.add_expand(y, buffer, wb_t_all, add_input=True)
        else:
            self.add_expand_slice(y,
                                  buffer,
                                  wb_t_all,
                                  y_offset,
                                  y_slice_size,
                                  add_input=True)
        y = y.view_as(y_org)

    def add_lora_packed_nslice(self, y: torch.Tensor, x: torch.Tensor,
                               lora_a_stacked: Tuple[torch.Tensor,
                                                     torch.Tensor,
                                                     torch.Tensor],
                               lora_b_stacked: Tuple[torch.Tensor,
                                                     torch.Tensor,
                                                     torch.Tensor],
                               scale: float, output_slices: Tuple[int, ...],
                               **kwarg) -> None:
        """
        Applies lora to each input. Similar to add_lora, This method is
        used for layers that are composed of multiple sublayers
        (slices) packed together.
        """
        y_org = y
        x = x.view(-1, x.shape[-1])
        y = y.view(-1, y.shape[-1])
        offset_left = 0
        # TODO fuse these kernels
        for slice_idx in range(len(output_slices)):
            self.add_lora(
                y,
                x,
                lora_a_stacked[slice_idx],
                lora_b_stacked[slice_idx],
                scale,
                offset_left,
                output_slices[slice_idx],
            )
            offset_left += output_slices[slice_idx]

        y = y.view_as(y_org)

    def add_lora_embedding(self,
                           y: torch.Tensor,
                           x: torch.Tensor,
                           w_t_all: torch.Tensor,
                           add_input: bool = True,
                           **kwarg):
        """
        VocabParallelEmbeddingWithLoRA only need expand op
        """

        expand_fun: Callable = (self.expand_prefill
                                if self.is_prefill else self.expand_decode)
        expand_fun(y, x, w_t_all, add_input)

    def add_lora_logits(self,
                        y: torch.Tensor,
                        x: torch.Tensor,
                        wa_t_all: torch.Tensor,
                        wb_t_all: torch.Tensor,
                        scale,
                        *,
                        buffer: Optional[torch.Tensor] = None,
                        **kwarg) -> None:
        """
        LogitsProcessorWithLoRA always using bgmv
        """
        y_org = y
        y = y.view(-1, y.shape[-1])
        x = x.view(-1, x.shape[-1])
        r = wb_t_all.size(-1)
        if buffer is None:
            # We set the buffer to be float32 by default ,refer to:
            # https://github.com/triton-lang/triton/issues/1387
            buffer = torch.zeros((x.size(0), r),
                                 dtype=torch.float32,
                                 device=x.device)

        bgmv_shrink(x, wa_t_all, buffer, self.sampler_indices, scale)
        bgmv_expand(buffer, wb_t_all, y, self.sampler_indices, add_inputs=True)
        y = y.view_as(y_org)
