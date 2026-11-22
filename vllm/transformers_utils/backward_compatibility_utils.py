# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Backward compatibility for remote code written against older Transformers.

Transformers >= 5.24 removed `transformers.modeling_attn_mask_utils`, but remote
code (e.g. MiniCPM-V and Nanbeige) still imports it. Its API is rebuilt here on
`transformers.masking_utils`, returning the same float masks as before.
"""

import importlib.util
import sys
from collections.abc import Callable

import torch
from transformers.masking_utils import (
    bidirectional_mask_function,
    causal_mask_function,
    eager_mask,
    sliding_window_causal_mask_function,
)
from transformers.utils.import_utils import is_tracing

ATTN_MASK_UTILS = "transformers.modeling_attn_mask_utils"


def _causal_mask_function(sliding_window: int | None) -> Callable:
    if sliding_window is None:
        return causal_mask_function
    # The legacy window also attends to the key `sliding_window` tokens back
    return sliding_window_causal_mask_function(sliding_window + 1)


class AttentionMaskConverter:
    def __init__(self, is_causal: bool, sliding_window: int | None = None):
        if sliding_window is not None and sliding_window <= 0:
            raise ValueError(
                "Make sure that when passing `sliding_window` that its value is a "
                f"strictly positive integer, not `{sliding_window}`"
            )
        self.is_causal = is_causal
        self.sliding_window = sliding_window

    def to_causal_4d(
        self,
        batch_size: int,
        query_length: int,
        key_value_length: int,
        dtype: torch.dtype,
        device: torch.device | str = "cpu",
    ) -> torch.Tensor | None:
        if not self.is_causal:
            raise ValueError("`to_causal_4d` requires `is_causal=True`.")
        if query_length == 1 and self.sliding_window is None:
            return None
        return self._make_causal_mask(
            (batch_size, query_length),
            dtype,
            device=device,
            past_key_values_length=key_value_length - query_length,
            sliding_window=self.sliding_window,
        )

    def to_4d(
        self,
        attention_mask_2d: torch.Tensor,
        query_length: int,
        dtype: torch.dtype,
        key_value_length: int | None = None,
    ) -> torch.Tensor:
        batch_size, kv_length = attention_mask_2d.shape
        q_offset = 0
        mask_function = bidirectional_mask_function
        if self.is_causal and (query_length > 1 or self.sliding_window is not None):
            if key_value_length is None:
                raise ValueError(
                    "This attention mask converter is causal. Make sure to pass "
                    "`key_value_length` to correctly create a causal mask."
                )
            q_offset = key_value_length - query_length
            kv_length = key_value_length
            mask_function = _causal_mask_function(self.sliding_window)
        elif self.sliding_window is not None:
            raise NotImplementedError(
                "Sliding window is currently only implemented for causal masking"
            )
        return eager_mask(
            batch_size,
            query_length,
            kv_length,
            q_offset=q_offset,
            mask_function=mask_function,
            attention_mask=attention_mask_2d == 1,
            dtype=dtype,
            device=attention_mask_2d.device,
        )

    @staticmethod
    def _make_causal_mask(
        input_ids_shape: torch.Size | tuple[int, int],
        dtype: torch.dtype,
        device: torch.device | str,
        past_key_values_length: int = 0,
        sliding_window: int | None = None,
    ) -> torch.Tensor:
        batch_size, query_length = input_ids_shape
        return eager_mask(
            batch_size,
            query_length,
            query_length + past_key_values_length,
            q_offset=past_key_values_length,
            mask_function=_causal_mask_function(sliding_window),
            dtype=dtype,
            device=device,
        )

    @staticmethod
    def _expand_mask(
        mask: torch.Tensor, dtype: torch.dtype, tgt_len: int | None = None
    ) -> torch.Tensor:
        batch_size, src_len = mask.shape
        return eager_mask(
            batch_size,
            tgt_len if tgt_len is not None else src_len,
            src_len,
            mask_function=bidirectional_mask_function,
            attention_mask=mask == 1,
            dtype=dtype,
            device=mask.device,
        )

    @staticmethod
    def _unmask_unattended(
        expanded_mask: torch.Tensor, min_dtype: float
    ) -> torch.Tensor:
        if expanded_mask.dtype == torch.bool:
            raise ValueError(
                "AttentionMaskConverter._unmask_unattended expects a float "
                "`expanded_mask`, got a BoolTensor."
            )
        fully_masked = torch.all(expanded_mask == min_dtype, dim=-1, keepdim=True)
        return expanded_mask.mul(~fully_masked)

    @staticmethod
    def _ignore_causal_mask_sdpa(
        attention_mask: torch.Tensor | None,
        inputs_embeds: torch.Tensor,
        past_key_values_length: int,
        sliding_window: int | None = None,
        is_training: bool = False,
    ) -> bool:
        query_length = inputs_embeds.shape[1]
        key_value_length = query_length + past_key_values_length
        is_square = query_length == 1 or key_value_length == query_length
        in_window = sliding_window is None or key_value_length < sliding_window
        if attention_mask is None:
            tracing = is_tracing(inputs_embeds)
            return (is_training or not tracing) and is_square and in_window
        if not in_window or attention_mask.dim() == 4:
            return False
        all_attended = not is_tracing(inputs_embeds) and torch.all(attention_mask == 1)
        return bool(all_attended) and is_square


def _prepare_4d_causal_attention_mask(
    attention_mask: torch.Tensor | None,
    input_shape: torch.Size | tuple | list,
    inputs_embeds: torch.Tensor,
    past_key_values_length: int,
    sliding_window: int | None = None,
) -> torch.Tensor | None:
    converter = AttentionMaskConverter(is_causal=True, sliding_window=sliding_window)
    key_value_length = input_shape[-1] + past_key_values_length
    dtype = inputs_embeds.dtype
    if attention_mask is not None and attention_mask.dim() == 2:
        return converter.to_4d(
            attention_mask, input_shape[-1], dtype, key_value_length=key_value_length
        )
    if attention_mask is not None and attention_mask.dim() == 4:
        expected_shape = (input_shape[0], 1, input_shape[1], key_value_length)
        if tuple(attention_mask.shape) != expected_shape:
            raise ValueError(
                f"Incorrect 4D attention_mask shape: {tuple(attention_mask.shape)}; "
                f"expected: {expected_shape}."
            )
        inverted_mask = 1.0 - attention_mask
        return inverted_mask.masked_fill(
            inverted_mask.to(torch.bool), torch.finfo(dtype).min
        )
    return converter.to_causal_4d(
        input_shape[0],
        input_shape[-1],
        key_value_length,
        dtype,
        device=inputs_embeds.device,
    )


def _prepare_4d_causal_attention_mask_for_sdpa(
    attention_mask: torch.Tensor | None,
    input_shape: torch.Size | tuple | list,
    inputs_embeds: torch.Tensor,
    past_key_values_length: int,
    sliding_window: int | None = None,
) -> torch.Tensor | None:
    if AttentionMaskConverter._ignore_causal_mask_sdpa(
        attention_mask, inputs_embeds, past_key_values_length, sliding_window
    ):
        return None
    converter = AttentionMaskConverter(is_causal=True, sliding_window=sliding_window)
    key_value_length = input_shape[-1] + past_key_values_length
    dtype = inputs_embeds.dtype
    if attention_mask is None:
        return converter.to_causal_4d(
            input_shape[0],
            input_shape[-1],
            key_value_length,
            dtype,
            device=inputs_embeds.device,
        )
    if attention_mask.dim() == 4:
        mask = attention_mask
    else:
        mask = converter.to_4d(
            attention_mask, input_shape[-1], dtype, key_value_length=key_value_length
        )
    # Fully masked rows (e.g. left padding) break SDPA's memory-efficient kernel
    if not is_tracing(inputs_embeds) and mask.device.type in ("cuda", "xpu"):
        mask = AttentionMaskConverter._unmask_unattended(mask, torch.finfo(dtype).min)
    return mask


def _prepare_4d_attention_mask(
    mask: torch.Tensor, dtype: torch.dtype, tgt_len: int | None = None
) -> torch.Tensor:
    return AttentionMaskConverter._expand_mask(mask, dtype, tgt_len=tgt_len)


def _prepare_4d_attention_mask_for_sdpa(
    mask: torch.Tensor, dtype: torch.dtype, tgt_len: int | None = None
) -> torch.Tensor | None:
    if not is_tracing(mask) and torch.all(mask == 1):
        return None
    return AttentionMaskConverter._expand_mask(mask, dtype, tgt_len=tgt_len)


def _create_4d_causal_attention_mask(
    input_shape: torch.Size | tuple | list,
    dtype: torch.dtype,
    device: torch.device,
    past_key_values_length: int = 0,
    sliding_window: int | None = None,
) -> torch.Tensor | None:
    converter = AttentionMaskConverter(is_causal=True, sliding_window=sliding_window)
    return converter.to_causal_4d(
        input_shape[0],
        input_shape[-1],
        past_key_values_length + input_shape[-1],
        dtype,
        device=device,
    )


def install() -> None:
    """Serve this module as `transformers.modeling_attn_mask_utils` if missing."""
    # Transformers >= 5.24
    missing = importlib.util.find_spec(ATTN_MASK_UTILS) is None
    if ATTN_MASK_UTILS not in sys.modules and missing:
        sys.modules[ATTN_MASK_UTILS] = sys.modules[__name__]
