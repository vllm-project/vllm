"""
FP8 quantization helpers for LoRA weights and activations.

- Weight quantization: block-wise, runs once at load time.
- Activation quantization: per-token, runs on every forward pass.

For LoRA, the rank dimension is typically small (8–64), so we use
``max_rank`` as the block size for that dimension and a fixed large
block (default 128) for the hidden/output dimension.
"""

import torch

HIDDEN_BLOCK = 128


def _cdiv(a: int, b: int) -> int:
    return -(a // -b)


class FP8LoRAQuantizer:
    """FP8 quantizer tailored for LoRA weight and activation tensors.

    Block sizes are rank-aware:
      - lora_a (rank, hidden_size) → block = (max_rank, HIDDEN_BLOCK)
      - lora_b (output_size, rank) → block = (HIDDEN_BLOCK, max_rank)

    The shrink / expand Triton kernels receive matching ``group_k``
    and ``group_n`` so that scale indexing is consistent.
    """

    def __init__(
        self,
        max_rank: int,
        hidden_block: int = HIDDEN_BLOCK,
        fp8_dtype: torch.dtype = torch.float8_e4m3fn,
    ):
        self.max_rank = max_rank
        self.hidden_block = hidden_block
        self.fp8_dtype = fp8_dtype
        self.fp8_max = torch.finfo(fp8_dtype).max

    @property
    def lora_a_block_size(self) -> tuple[int, int]:
        """lora_a shape (rank, hidden_size)."""
        return (self.max_rank, self.hidden_block)

    @property
    def lora_b_block_size(self) -> tuple[int, int]:
        """lora_b shape (output_size, rank)."""
        return (self.hidden_block, self.max_rank)

    @property
    def shrink_group_k(self) -> int:
        return self.hidden_block

    @property
    def shrink_group_n(self) -> int:
        return self.max_rank

    @property
    def expand_group_k(self) -> int:
        return self.max_rank

    @property
    def expand_group_n(self) -> int:
        return self.hidden_block

    @staticmethod
    def _make_a_scale(scale_inv: torch.Tensor, K: int, group_k: int) -> torch.Tensor:
        """Build 2-D a_scale with stride-0 on the K-block axis."""
        N = scale_inv.size(0)
        num_k_blocks = _cdiv(K, group_k) if group_k > 0 else 1
        return scale_inv.as_strided((N, num_k_blocks), (1, 0))

    def per_token_quant(
        self, x: torch.Tensor, group_k: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize a 2-D activation tensor per-token.

        Args:
            x: (num_tokens, K) in any float dtype.
            group_k: block size along the K dimension (must match the
                     weight's corresponding block size).
        """
        assert x.ndim == 2
        amax = x.float().abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
        scale_inv = (amax / self.fp8_max).to(torch.float32)
        x_fp8 = (x.float() / scale_inv).to(self.fp8_dtype)
        a_scale = self._make_a_scale(scale_inv.squeeze(1), x.size(1), group_k)
        return x_fp8, a_scale

    def per_token_quant_3d(
        self, x: torch.Tensor, group_k: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize a 3-D tensor (num_slices, tokens, K) per-token.

        A single per-token scale is computed across all slices (the
        expand kernel indexes a_scale by token, not by slice).
        """
        assert x.ndim == 3
        S, N, K = x.shape
        amax = x.float().abs().amax(dim=(0, 2), keepdim=False).clamp(min=1e-12)
        scale_inv = (amax / self.fp8_max).to(torch.float32)
        x_fp8 = (x.float() / scale_inv.unsqueeze(0).unsqueeze(2)).to(self.fp8_dtype)
        a_scale = self._make_a_scale(scale_inv, K, group_k)
        return x_fp8, a_scale

    def _per_block_quantize(
        self, weight: torch.Tensor, block_size: tuple[int, int]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Core block-wise quantization with explicit block_size."""
        assert weight.ndim == 2, f"Expected 2-D tensor, got {weight.ndim}-D"
        M, N = weight.shape
        block_m, block_n = block_size

        pad_m = (_cdiv(M, block_m) * block_m) - M
        pad_n = (_cdiv(N, block_n) * block_n) - N

        if pad_m or pad_n:
            x = torch.nn.functional.pad(weight.float(), (0, pad_n, 0, pad_m))
        else:
            x = weight.float()

        Mp, Np = x.shape
        x_blocked = x.reshape(Mp // block_m, block_m, Np // block_n, block_n)
        x_blocked = x_blocked.permute(0, 2, 1, 3)

        amax = x_blocked.abs().amax(dim=(2, 3)).clamp(min=1e-12)
        scale = amax / self.fp8_max
        scale_for_quant = self.fp8_max / amax

        x_scaled = x_blocked * scale_for_quant.unsqueeze(-1).unsqueeze(-1)
        x_fp8 = x_scaled.to(self.fp8_dtype)

        x_fp8 = x_fp8.permute(0, 2, 1, 3).reshape(Mp, Np)[:M, :N].contiguous()
        return x_fp8, scale

    def quantize_lora_a(
        self, weight: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize a lora_a weight (rank, hidden_size)."""
        return self._per_block_quantize(weight, self.lora_a_block_size)

    def quantize_lora_b(
        self, weight: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize a lora_b weight (output_size, rank)."""
        return self._per_block_quantize(weight, self.lora_b_block_size)

    @staticmethod
    def _quantize_and_set(
        dest_weight: torch.Tensor,
        dest_scale: torch.Tensor,
        index: int,
        fp8_w: torch.Tensor,
        scale_inv: torch.Tensor,
    ) -> None:
        dest_weight[index, 0, : fp8_w.shape[0], : fp8_w.shape[1]].copy_(
            fp8_w, non_blocking=True
        )
        dest_scale[index, 0, : scale_inv.shape[0], : scale_inv.shape[1]].copy_(
            scale_inv, non_blocking=True
        )

    def quantize_and_set_lora_a(
        self,
        dest_weight: torch.Tensor,
        dest_scale: torch.Tensor,
        index: int,
        weight_2d: torch.Tensor,
    ) -> None:
        fp8_w, scale_inv = self.quantize_lora_a(weight_2d)
        self._quantize_and_set(dest_weight, dest_scale, index, fp8_w, scale_inv)

    def quantize_and_set_lora_b(
        self,
        dest_weight: torch.Tensor,
        dest_scale: torch.Tensor,
        index: int,
        weight_2d: torch.Tensor,
    ) -> None:
        fp8_w, scale_inv = self.quantize_lora_b(weight_2d)
        self._quantize_and_set(dest_weight, dest_scale, index, fp8_w, scale_inv)
