# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from copy import deepcopy
from typing import TYPE_CHECKING, Any, Optional

import msgspec

from vllm.sampling_params import RequestOutputKind
from vllm.tasks import PoolingTask

if TYPE_CHECKING:
    from vllm.config import ModelConfig


class PoolingParams(
        msgspec.Struct,
        omit_defaults=True,  # type: ignore[call-arg]
        array_like=True):  # type: ignore[call-arg]
    """API parameters for pooling models.

    Attributes:
        normalize: Whether to normalize the embeddings outputs.
        dimensions: Reduce the dimensions of embeddings
                    if model support matryoshka representation.
        activation: Whether to apply activation function to
                    the classification outputs.
        softmax: Whether to apply softmax to the reward outputs.
    """

    ## for embeddings models
    dimensions: Optional[int] = None
    normalize: Optional[bool] = None

    ## for classification models
    activation: Optional[bool] = None

    ## for reward models
    softmax: Optional[bool] = None
    step_tag_id: Optional[int] = None
    returned_token_ids: Optional[list[int]] = None

    task: Optional[PoolingTask] = None
    """Internal use only."""

    requires_token_ids: bool = False
    """Internal use only."""

    extra_kwargs: Optional[dict[str, Any]] = None
    """Internal use only."""

    output_kind: RequestOutputKind = RequestOutputKind.FINAL_ONLY

    @property
    def all_parameters(self) -> list[str]:
        return [
            "dimensions", "normalize", "activation", "softmax", "step_tag_id",
            "returned_token_ids"
        ]

    @property
    def valid_parameters(self):
        return {
            "embed": ["dimensions", "normalize"],
            "classify": ["activation"],
            "score": ["activation"],
            "encode": ["softmax", "step_tag_id", "returned_token_ids"],
        }

    def clone(self) -> "PoolingParams":
        """Returns a deep copy of the PoolingParams instance."""
        return deepcopy(self)

    def verify(self,
               task: PoolingTask,
               model_config: Optional["ModelConfig"] = None) -> None:

        if self.task is None:
            self.task = task
        elif self.task != task:
            msg = f"You cannot overwrite {self.task=!r} with {task=!r}!"
            raise ValueError(msg)

        # NOTE: Task validation needs to done against the model instance,
        # which is not available in model config. So, it's not included
        # in this method

        self._merge_default_parameters(model_config)
        self._set_default_parameters(model_config)
        self._verify_valid_parameters()

    def _merge_default_parameters(self,
                                  model_config: Optional["ModelConfig"] = None
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

    def _set_default_parameters(self, model_config: Optional["ModelConfig"]):
        if self.task == "embed":
            if self.normalize is None:
                self.normalize = True

            if self.dimensions is not None and model_config is not None:
                if not model_config.is_matryoshka:
                    raise ValueError(
                        f'Model "{model_config.served_model_name}" does not '
                        f'support matryoshka representation, '
                        f'changing output dimensions will lead to poor results.'
                    )

                mds = model_config.matryoshka_dimensions
                if mds is not None:
                    if self.dimensions not in mds:
                        raise ValueError(
                            f'Model "{model_config.served_model_name}" '
                            f'only supports {str(mds)} matryoshka dimensions, '
                            f'use other output dimensions will '
                            f'lead to poor results.')
                elif self.dimensions < 1:
                    raise ValueError("Dimensions must be greater than 0")

        elif self.task in ["classify", "score"]:
            if self.activation is None:
                self.activation = True

        elif self.task == "encode":
            if self.softmax is None:
                self.softmax = True
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
                f"{invalid_parameters} parameters")

    def __repr__(self) -> str:
        return (f"PoolingParams("
                f"task={self.task}, "
                f"normalize={self.normalize}, "
                f"dimensions={self.dimensions}, "
                f"activation={self.activation}, "
                f"softmax={self.softmax}, "
                f"step_tag_id={self.step_tag_id}, "
                f"returned_token_ids={self.returned_token_ids}, "
                f"requires_token_ids={self.requires_token_ids}, "
                f"extra_kwargs={self.extra_kwargs})")

    def __post_init__(self) -> None:
        assert self.output_kind == RequestOutputKind.FINAL_ONLY,\
            "For pooling output_kind has to be FINAL_ONLY"
