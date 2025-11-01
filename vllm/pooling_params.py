# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from copy import deepcopy
from typing import Annotated, Any, Optional

import msgspec

from vllm.config import ModelConfig, PoolerConfig
from vllm.config.pooler import get_use_activation
from vllm.sampling_params import RequestOutputKind
from vllm.tasks import PoolingTask


class PoolingParams(
    msgspec.Struct,
    omit_defaults=True,  # type: ignore[call-arg]
    array_like=True,
):  # type: ignore[call-arg]
    """API parameters for pooling models.

    Attributes:
        truncate_prompt_tokens: Controls prompt truncation.
            Set to -1 to use the model's default truncation size.
            Set to k to keep only the last k tokens (left truncation).
            Set to None to disable truncation.
        dimensions: Reduce the dimensions of embeddings
            if model support matryoshka representation.
        normalize: Whether to normalize the embeddings outputs.
        softmax: softmax will be deprecated, please use use_activation instead.
        activation: activation will be deprecated, please use use_activation instead.
        use_activation: Whether to apply activation function to
            the classification outputs.
    """

    # --8<-- [start:common-pooling-params]
    truncate_prompt_tokens: Annotated[int, msgspec.Meta(ge=-1)] | None = None
    # --8<-- [end:common-pooling-params]

    ## for embeddings models
    # --8<-- [start:embedding-pooling-params]
    dimensions: int | None = None
    normalize: bool | None = None
    # --8<-- [end:embedding-pooling-params]

    ## for classification, scoring and rerank
    # --8<-- [start:classification-pooling-params]
    softmax: bool | None = None
    activation: bool | None = None
    use_activation: bool | None = None
    # --8<-- [end:classification-pooling-params]

    ## for step pooling models
    step_tag_id: int | None = None
    returned_token_ids: list[int] | None = None

    ## Internal use only
    task: PoolingTask | None = None
    requires_token_ids: bool = False
    extra_kwargs: dict[str, Any] | None = None
    output_kind: RequestOutputKind = RequestOutputKind.FINAL_ONLY

    @property
    def all_parameters(self) -> list[str]:
        return ["dimensions", "normalize", "use_activation"]

    @property
    def valid_parameters(self):
        return {
            "embed": ["dimensions", "normalize"],
            "classify": ["use_activation"],
            "score": ["use_activation"],
            "token_embed": ["dimensions", "normalize"],
            "token_classify": ["use_activation"],
        }

    def clone(self) -> "PoolingParams":
        """Returns a deep copy of the PoolingParams instance."""
        return deepcopy(self)

    def verify(
        self, task: PoolingTask, model_config: Optional["ModelConfig"] = None
    ) -> None:
        if self.task is None:
            self.task = task
        elif self.task != task:
            msg = f"You cannot overwrite {self.task=!r} with {task=!r}!"
            raise ValueError(msg)

        # raise deprecated warning for softmax and activation
        self.use_activation = get_use_activation(self)

        # plugin task uses io_processor.parse_request to verify inputs,
        # skipping PoolingParams verify
        if self.task == "plugin":
            return

        # NOTE: Task validation needs to done against the model instance,
        # which is not available in model config. So, it's not included
        # in this method
        self._merge_default_parameters(model_config)
        self._set_default_parameters(model_config)
        self._verify_valid_parameters()

    def _merge_default_parameters(
        self, model_config: Optional["ModelConfig"] = None
    ) -> None:
        if model_config is None:
            return

        pooler_config = model_config.pooler_config
        if pooler_config is None:
            return

        assert self.task is not None, "task must be set"
        valid_parameters = self.valid_parameters[self.task]

        for k in valid_parameters:
            if getattr(pooler_config, k, None) is None:
                continue

            if getattr(self, k, None) is None:
                setattr(self, k, getattr(pooler_config, k))

        self._verify_step_pooling(pooler_config, valid_parameters)

    def _verify_step_pooling(
        self, pooler_config: "PoolerConfig", valid_parameters: list[str]
    ):
        step_pooling_parameters = ["step_tag_id", "returned_token_ids"]
        if pooler_config.pooling_type != "STEP":
            invalid_parameters = []
            for k in step_pooling_parameters:
                if getattr(self, k, None) is not None:
                    invalid_parameters.append(k)

            if invalid_parameters:
                raise ValueError(
                    f"Task {self.task} only supports {valid_parameters} "
                    f"parameters, does not support "
                    f"{invalid_parameters} parameters"
                )
        else:
            for k in step_pooling_parameters:
                if getattr(pooler_config, k, None) is None:
                    continue

                if getattr(self, k, None) is None:
                    setattr(self, k, getattr(pooler_config, k))

    def _set_default_parameters(self, model_config: Optional["ModelConfig"]):
        if self.task in ["embed", "token_embed"]:
            if self.normalize is None:
                self.normalize = True

            if self.dimensions is not None and model_config is not None:
                if not model_config.is_matryoshka:
                    raise ValueError(
                        f'Model "{model_config.served_model_name}" does not '
                        f"support matryoshka representation, "
                        f"changing output dimensions will lead to poor results."
                    )

                mds = model_config.matryoshka_dimensions
                if mds is not None:
                    if self.dimensions not in mds:
                        raise ValueError(
                            f'Model "{model_config.served_model_name}" '
                            f"only supports {str(mds)} matryoshka dimensions, "
                            f"use other output dimensions will "
                            f"lead to poor results."
                        )
                elif self.dimensions < 1:
                    raise ValueError("Dimensions must be greater than 0")

        elif self.task in ["classify", "score", "token_classify"]:
            if self.use_activation is None:
                self.use_activation = True
        else:
            raise ValueError(f"Unknown pooling task: {self.task}")

    def _verify_valid_parameters(self):
        assert self.task is not None, "task must be set"
        valid_parameters = self.valid_parameters[self.task]
        invalid_parameters = []
        for k in self.all_parameters:
            if k in valid_parameters:
                continue

            if getattr(self, k, None) is not None:
                invalid_parameters.append(k)

        if invalid_parameters:
            raise ValueError(
                f"Task {self.task} only supports {valid_parameters} "
                f"parameters, does not support "
                f"{invalid_parameters} parameters"
            )

    def __repr__(self) -> str:
        return (
            f"PoolingParams("
            f"task={self.task}, "
            f"normalize={self.normalize}, "
            f"dimensions={self.dimensions}, "
            f"use_activation={self.use_activation}, "
            f"step_tag_id={self.step_tag_id}, "
            f"returned_token_ids={self.returned_token_ids}, "
            f"requires_token_ids={self.requires_token_ids}, "
            f"extra_kwargs={self.extra_kwargs})"
        )

    def __post_init__(self) -> None:
        assert self.output_kind == RequestOutputKind.FINAL_ONLY, (
            "For pooling output_kind has to be FINAL_ONLY"
        )
