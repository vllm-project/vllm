# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Mapping

KV_CHECKPOINT_RESTORE_ID_ARG = "vllm_kv_checkpoint_restore_id"
KV_CHECKPOINT_SAVE_ID_ARG = "vllm_kv_checkpoint_save_id"


def get_optional_str_arg(
    extra_args: Mapping[str, object] | None,
    key: str,
) -> str | None:
    if extra_args is None:
        return None
    value = extra_args.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"`{key}` must be a string when provided")
    if value == "":
        raise ValueError(f"`{key}` must not be empty")
    return value
