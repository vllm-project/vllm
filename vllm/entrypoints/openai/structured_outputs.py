# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import dataclasses
from typing import Any, cast

from vllm.sampling_params import StructuredOutputsParams


def merge_structured_outputs(
    base: StructuredOutputsParams | None,
    updates: dict[str, Any],
) -> StructuredOutputsParams:
    if base is None:
        return StructuredOutputsParams(**updates)
    data = {
        data_field.name: getattr(base, data_field.name)
        for data_field in dataclasses.fields(cast(Any, base))
        if data_field.init
    }
    data.update(updates)
    return StructuredOutputsParams(**data)
