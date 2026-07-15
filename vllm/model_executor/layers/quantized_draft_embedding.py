# SPDX-License-Identifier: Apache-2.0
"""Low-bit vocab embedding for speculative-decode draft models (A1c).

The MTP draft loads its own full-precision copy of a huge vocab embedding on the
last PP rank, which is what pushes 27B+draft over 2x16 GiB. Because the draft is
only a speculative proposer (rejection sampling guarantees the final output equals
target greedy regardless of draft quality), a *lossy* low-bit draft embedding is
correctness-safe: it only lowers acceptance, never changes the output.

Critically (RUN-A on the real 27B): the int storage is allocated **at
construction** and the fp16 checkpoint weight is quantized **as it loads**, so the
full fp16 [vocab, hidden] table never materializes on the GPU (a post-load swap
OOMs at the load peak before it can free anything). The checkpoint's
``embed_tokens.weight`` is routed through ``weight`` — an empty placeholder
Parameter carrying a ``weight_loader`` that quantizes the incoming fp16 tensor
(which lives on CPU during loading) into the pre-allocated GPU int storage.

Targets TP=1 (the draft runs single-rank), so the lookup is a plain gather with
no all-reduce. Bit-widths: 8 (int8, 1 byte/weight) and 4 (int4 packed 2
nibbles/byte, 0.5 byte/weight; requires an even hidden dimension).
"""

import torch
from torch import nn


def _quantize_per_row(
    weight: torch.Tensor, bits: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-row symmetric quantization. Returns (qweight, scale).

    qweight is int8 [V, H] for bits=8, or uint8 [V, H//2] (two packed signed
    nibbles per byte) for bits=4. scale is fp32 [V].
    """
    qmax = 2 ** (bits - 1) - 1  # 127 (int8) / 7 (int4)
    row_absmax = weight.abs().amax(dim=1).clamp_min(1e-12)
    scale = (row_absmax / qmax).to(torch.float32)
    q = (weight.to(torch.float32) / scale.unsqueeze(1)).round().clamp_(-qmax, qmax)
    if bits == 8:
        return q.to(torch.int8), scale
    # bits == 4: pack two signed nibbles per uint8 byte.
    u = (q + 8).to(torch.uint8)  # [-7, 7] -> [1, 15], fits 4 bits
    qweight = (u[:, 0::2] | (u[:, 1::2] << 4)).contiguous()
    return qweight, scale


class QuantizedVocabEmbedding(nn.Module):
    """Per-row symmetric low-bit vocab embedding (lookup + dequant).

    Args:
        num_embeddings: vocabulary size (V).
        embedding_dim: hidden size (H).
        bits: quantization bit-width (8 or 4).
        params_dtype: dtype the lookup returns (matches the model's dtype).
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *,
        bits: int = 8,
        params_dtype: torch.dtype = torch.float16,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        if bits not in (4, 8):
            raise NotImplementedError(f"bits={bits} not yet supported")
        if bits == 4 and embedding_dim % 2 != 0:
            raise ValueError("int4 packing requires an even hidden size")
        self.bits = bits
        self.out_dtype = params_dtype
        self.num_embeddings = num_embeddings
        self.hidden_size = embedding_dim

        packed_cols = embedding_dim if bits == 8 else embedding_dim // 2
        qdtype = torch.int8 if bits == 8 else torch.uint8
        self.register_buffer(
            "qweight",
            torch.zeros((num_embeddings, packed_cols), dtype=qdtype, device=device),
            persistent=False,
        )
        self.register_buffer(
            "scale",
            torch.zeros(num_embeddings, dtype=torch.float32, device=device),
            persistent=False,
        )
        # Entry point for the checkpoint loader: an EMPTY placeholder (never a
        # full fp16 [V, H] table — that is the OOM peak we are avoiding). The
        # AutoWeightsLoader matches the checkpoint's ``embed_tokens.weight`` to
        # this param and calls its ``weight_loader``, which quantizes on the fly.
        self.weight = nn.Parameter(
            torch.empty(0, dtype=params_dtype, device=device), requires_grad=False
        )
        self.weight.weight_loader = self._weight_loader

    @classmethod
    def from_weight(
        cls, weight: torch.Tensor, *, bits: int = 8
    ) -> "QuantizedVocabEmbedding":
        """Build and quantize from a full fp16 weight (tests / post-load)."""
        m = cls(
            weight.shape[0],
            weight.shape[1],
            bits=bits,
            params_dtype=weight.dtype,
            device=weight.device,
        )
        m.load_full_weight(weight)
        return m

    def load_full_weight(self, weight: torch.Tensor) -> None:
        """Quantize a full fp16 [V, H] table into the pre-allocated int storage."""
        qweight, scale = _quantize_per_row(weight, self.bits)
        self.qweight.copy_(qweight)
        self.scale.copy_(scale)

    def _weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
        # ``param`` is the empty placeholder; quantize the incoming fp16 tensor
        # (on CPU during loading) straight into the GPU int storage.
        self.load_full_weight(loaded_weight)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        rows = self.qweight[input_ids]
        if self.bits == 8:
            deq = rows.to(torch.float32)
        else:  # unpack nibbles back to signed int4 values
            low = (rows & 0x0F).to(torch.int16) - 8
            high = ((rows >> 4) & 0x0F).to(torch.int16) - 8
            deq = torch.stack([low, high], dim=-1)
            deq = deq.reshape(*rows.shape[:-1], self.hidden_size).to(torch.float32)
        out = deq * self.scale[input_ids].unsqueeze(-1)
        return out.to(self.out_dtype)
