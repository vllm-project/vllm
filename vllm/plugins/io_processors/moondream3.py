# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, cast

from vllm.model_executor.models.moondream3_io import (
    MOONDREAM3_MAX_OBJECTS_DEFAULT,
    MOONDREAM3_TASK_DETECT,
    MOONDREAM3_TASK_POINT,
    build_moondream3_detect_point_prompt,
    decode_moondream3_detect_point_output_json,
)
from vllm.outputs import RequestOutput
from vllm.plugins.io_processors.interface import IOProcessor
from vllm.sampling_params import SamplingParams


@dataclass
class Moondream3DetectPointRequest:
    task: Literal["detect", "point"]
    target: str
    image: object
    max_objects: int = MOONDREAM3_MAX_OBJECTS_DEFAULT


class Moondream3DetectPointIOProcessor(
    IOProcessor[Moondream3DetectPointRequest, object]
):
    """IO processor for Moondream3 detect/point generation requests."""

    def parse_data(self, data: object) -> Moondream3DetectPointRequest:
        if not isinstance(data, Mapping):
            raise TypeError(
                "Moondream3 IO processor expects `data` to be a mapping, "
                f"got {type(data).__name__}."
            )

        task = data.get("task")
        if task not in (MOONDREAM3_TASK_DETECT, MOONDREAM3_TASK_POINT):
            raise ValueError(
                "Moondream3 IO processor requires task to be either "
                f"{MOONDREAM3_TASK_DETECT!r} or {MOONDREAM3_TASK_POINT!r}."
            )

        target = data.get("object")
        if not isinstance(target, str) or not target.strip():
            raise ValueError(
                "Moondream3 IO processor requires a non-empty string in the "
                "`object` field."
            )

        if "image" not in data:
            raise ValueError(
                "Moondream3 IO processor requires an `image` field in `data`."
            )
        image = data["image"]

        raw_max_objects = data.get("max_objects", MOONDREAM3_MAX_OBJECTS_DEFAULT)
        try:
            max_objects = int(raw_max_objects)
        except (TypeError, ValueError):
            raise ValueError(
                "`max_objects` must be an integer for the Moondream3 IO processor."
            ) from None
        if max_objects < 1:
            raise ValueError("`max_objects` must be >= 1 for Moondream3.")

        return Moondream3DetectPointRequest(
            task=cast(Literal["detect", "point"], task),
            target=target.strip(),
            image=image,
            max_objects=max_objects,
        )

    def merge_sampling_params(
        self,
        params: SamplingParams | None = None,
    ) -> SamplingParams:
        return params or SamplingParams()

    def merge_sampling_params_for_prompt(
        self,
        prompt: Moondream3DetectPointRequest,
        params: SamplingParams | None = None,
    ) -> SamplingParams:
        params = self.merge_sampling_params(params)
        extra_args = dict(params.extra_args or {})
        extra_args["moondream3_task"] = prompt.task
        extra_args["moondream3_max_objects"] = prompt.max_objects
        params.extra_args = extra_args
        return params

    def pre_process(
        self,
        prompt: Moondream3DetectPointRequest,
        request_id: str | None = None,
        **kwargs: Any,
    ) -> dict[str, object]:
        return {
            "prompt": build_moondream3_detect_point_prompt(
                prompt.task,
                prompt.target,
            ),
            "multi_modal_data": {"image": prompt.image},
        }

    def post_process(
        self,
        model_output,
        request_id: str | None = None,
        **kwargs: Any,
    ):
        return model_output

    def post_process_generate(
        self,
        model_output: RequestOutput,
        request_id: str | None = None,
        **kwargs: Any,
    ) -> RequestOutput:
        for completion in model_output.outputs:
            if completion.model_extra_output is None:
                continue

            decoded = decode_moondream3_detect_point_output_json(
                completion.model_extra_output
            )
            if decoded is not None:
                completion.text = decoded

        return model_output
