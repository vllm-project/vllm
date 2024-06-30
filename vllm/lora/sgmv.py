# Based on code from https://github.com/punica-ai/punica

from typing import Optional

import torch

from vllm.model_executor.layers.lora import sgmv_triton


def sgmv(
    y: torch.Tensor,
    x: torch.Tensor,
    w_t_all: torch.Tensor,
    indices: torch.LongTensor,
    ranks: torch.LongTensor,
    repeats: torch.LongTensor,
    max_repeats: int,
    out_col_offset: int = 0,
    scale: float = 1.0,
):
    """
	Semantics:
		y[i, out_col_offset : out_col_offset + w_t_all.shape[2]] += (
			x[i].unsqueeze(0)
			@ w_t_all[indices[i], 0, :, :].transpose(-1, -2)
			* scale
		).squeeze(0)
	Args:
		y: Shape: `[B, H2]`. Output vectors. Will be changed in-place.
		x: Shape: `[B, H1]`. Input vectors.
		w_t_all: Shape: `[None, L, H2, H1]`. All of the transposed weight
			matrices.
		indices: Shape: `[B]`. Indices of the weight matrices.
		ranks: rank of the LoRA for each group of 32 tokens + remainder
			the LoRA applies to for each LoRA in the batch
		repeats: similar to ranks, but number of tokens for the LoRA group
		max_repeats: repeats.max(), just cached so it isn't recomputed
		out_col_offset: for sgmv_expand/LoRA B, offset output along hidden out
		scale: Scaling factor.
	"""
    h_out, h_in = w_t_all.shape[-2:]
    if h_out <= h_in:
        sgmv_triton.sgmv_shrink(x, w_t_all, y, ranks, indices, repeats,
                                max_repeats)
    else:
        sgmv_triton.sgmv_expand(x, w_t_all, y, ranks, indices, repeats,
                                max_repeats, out_col_offset, scale)


def add_lora(y: torch.Tensor,
             x: torch.Tensor,
             wa_t_all: torch.Tensor,
             wb_t_all: torch.Tensor,
             indices: torch.LongTensor,
             ranks: torch.LongTensor,
             repeats: torch.LongTensor,
             max_repeats: int,
             out_col_offset: int = 0,
             scale: float = 1.0,
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
		indices: Shape: `[B]`. Indices of the LoRA weights.
		ranks: rank of the LoRA for each group of 32 tokens + remainder
			the LoRA applies to for each LoRA in the batch
		repeats: similar to ranks, but number of tokens for the LoRA group
		max_repeats: repeats.max(), just cached so it isn't recomputed
		out_col_offset: for sgmv_expand/LoRA B, offset output along hidden out
		scale: Scaling factor.
		buffer: Optional. Shape: `[B, R]`. Temporary buffer.
	"""
    r = wb_t_all.size(-1)
    if buffer is None:
        # We set the buffer to be float32 by default to avoid
        # numerical inaccuracies that would otherwise happen
        # due to downcasting.
        buffer = torch.zeros((x.size(0), r),
                             dtype=torch.float32,
                             device=x.device)
    sgmv(  # LoRA A shrink
        buffer, x, wa_t_all, indices, ranks, repeats, max_repeats)
    torch.cuda.synchronize()
    sgmv(  # LoRA B expand
        y, buffer, wb_t_all, indices, ranks, repeats, max_repeats,
        out_col_offset, scale)
