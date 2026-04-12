# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

_MIN_HEAD_SIZE_FOR_LL4MI = 64


def should_use_unified_decode_fallback(
    head_size: int,
    sliding_window: tuple[int, int],
) -> bool:
    return head_size < _MIN_HEAD_SIZE_FOR_LL4MI or sliding_window[0] != -1
