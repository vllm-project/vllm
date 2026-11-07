# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from collections.abc import Mapping
from typing import Any


def load_tool_call_arguments(tool_call: Mapping[str, Any]) -> dict[str, Any]:
    raw_arguments = tool_call.get("arguments")
    if isinstance(raw_arguments, str):
        try:
            arguments = json.loads(raw_arguments)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "DeepSeek tool call function.arguments must be a valid JSON object."
            ) from exc
    else:
        arguments = raw_arguments

    if not isinstance(arguments, dict):
        raise ValueError("DeepSeek tool call function.arguments must be a JSON object.")

    return arguments
