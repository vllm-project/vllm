# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

_FLASH_NAMES = (
    "flash_attention_2",
    "flash_attention_3",
    "flash_attention_4",
    "kernels-community/flash-attn2",
    "kernels-community/vllm-flash-attn3",
    "kernels-community/aiter-flash-attn",
    "kernels-community/flash-attn4",
)


def broadcast_audio_key_padding_mask(
    keep: torch.Tensor,
    attn_implementation: str | None,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Key-padding mask with a singleton query axis.

    ``keep`` is bool ``[B, S]`` with True on visible keys. ``None`` and
    ``sdpa`` return bool ``[B, 1, 1, S]``. Flash names return long ``[B, S]``
    with 1 on visible keys. Eager returns float ``[B, 1, 1, S]`` with 0 on
    visible keys and ``-inf`` on pad keys.
    """
    if keep.ndim != 2 or keep.dtype != torch.bool:
        raise ValueError("keep must be a bool tensor of shape [batch, time]")

    name = "sdpa" if attn_implementation is None else attn_implementation
    if name.startswith("paged|"):
        name = name.removeprefix("paged|")

    batch, seq_len = keep.shape
    if name == "sdpa":
        return keep.to(device=device).view(batch, 1, 1, seq_len)
    if name in _FLASH_NAMES or "flash" in name:
        if (keep.sum(dim=-1) == 0).any():
            raise ValueError("flash attention rejects an all-pad audio row")
        return keep.to(dtype=torch.long, device=device)
    if name == "eager":
        keep_on_device = keep.to(device=device)
        mask = torch.zeros((batch, 1, 1, seq_len), dtype=dtype, device=device)
        mask.masked_fill_(~keep_on_device.view(batch, 1, 1, seq_len), float("-inf"))
        return mask
    raise ValueError(
        f"unsupported audio attention implementation {attn_implementation!r}"
    )
