# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

from vllm.exceptions import VLLMValidationError
from vllm.renderers.inputs.preprocess import prompt_to_seq


def completion_engine_input_count(prompt: Any, prompt_embeds: Any) -> int:
    """Count inputs as `OnlineRenderer.preprocess_completion` expands them."""
    return sum(
        len(prompt_to_seq(value))
        for value in (prompt, prompt_embeds)
        if value is not None
    )


def validate_remote_kv_source_fanout(
    kv_transfer_params: dict[str, Any] | None,
    *,
    input_count: int,
    n: int | None,
) -> None:
    remote_kv_source = (
        kv_transfer_params.get("remote_kv_source")
        if isinstance(kv_transfer_params, dict)
        else None
    )
    if remote_kv_source is None or (input_count <= 1 and (n is None or n <= 1)):
        return

    raise VLLMValidationError(
        "`kv_transfer_params['remote_kv_source']` cannot be shared across "
        "multiple engine requests. Submit one engine request at a time "
        "(`n=1`) with a distinct `kv_request_id`.",
        parameter="kv_transfer_params.remote_kv_source",
    )
