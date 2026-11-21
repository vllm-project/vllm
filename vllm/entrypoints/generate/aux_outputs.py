# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared request fields and inline auxiliary-output serialization."""

from typing import Any, Literal

import torch
from pydantic import BaseModel, Field, model_validator

from vllm.outputs import CompletionOutput, WeightVersionSpan
from vllm.utils.serial_utils import numpy2base64


class HiddenStatesRequest(BaseModel):
    aux_outputs: tuple[Literal["hidden_states", "prompt_hidden_states"], ...] | None = (
        None
    )
    hidden_states_start: int = Field(default=0, ge=0, strict=True)
    hidden_states_end: int | None = Field(default=None, gt=0, strict=True)

    @model_validator(mode="after")
    def validate_hidden_states_request(self):
        if self.aux_outputs:
            if getattr(self, "use_beam_search", False):
                raise ValueError("Hidden states do not support beam search")
            if (
                self.hidden_states_end is not None
                and self.hidden_states_end <= self.hidden_states_start
            ):
                raise ValueError("hidden_states_end must exceed hidden_states_start")
        return self


class HiddenStatesResponse(BaseModel):
    prompt_hidden_states: str | None = Field(
        default=None, exclude_if=lambda v: v is None
    )
    prompt_hidden_states_dtype: Literal["bfloat16", "float16", "float32"] | None = (
        Field(default=None, exclude_if=lambda v: v is None)
    )
    prompt_hidden_states_start: int | None = Field(
        default=None, exclude_if=lambda v: v is None
    )
    prompt_hidden_states_weight_versions: list[WeightVersionSpan] | None = Field(
        default=None, exclude_if=lambda v: v is None
    )
    hidden_states: str | None = Field(default=None, exclude_if=lambda v: v is None)
    hidden_states_dtype: Literal["bfloat16", "float16", "float32"] | None = Field(
        default=None,
        exclude_if=lambda v: v is None,
    )
    hidden_states_start: int | None = Field(
        default=None, exclude_if=lambda v: v is None
    )


def hidden_states_response_fields(output: CompletionOutput) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    for name in ("hidden_states", "prompt_hidden_states"):
        rows = getattr(output, name)
        if rows is None:
            continue
        array = (
            rows.view(torch.uint16).numpy()
            if rows.dtype == torch.bfloat16
            else rows.numpy()
        )
        fields[name] = numpy2base64(array)
        fields[name + "_dtype"] = str(rows.dtype).removeprefix("torch.")
        fields[name + "_start"] = getattr(output, name + "_start")
    if output.prompt_hidden_states is not None:
        fields["prompt_hidden_states_weight_versions"] = [
            {"version": span.version, "start": span.start, "end": span.end}
            for span in (output.prompt_hidden_states_weight_versions or [])
        ]
    return fields
