# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING, Optional, assert_never

import msgspec

from vllm.sampling_params import RequestOutputKind
from vllm.tasks import PoolingTask

if TYPE_CHECKING:
    from vllm.config import ModelConfig, PoolerConfig


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

    output_kind: RequestOutputKind = RequestOutputKind.FINAL_ONLY

    task: Optional[PoolingTask] = None
    """Internal use only."""

    requires_token_ids: bool = False
    """Internal use only."""

    @property
    def all_parameters(self) -> list[str]:
        return ["dimensions", "normalize", "activation", "softmax"]

    def clone(self) -> "PoolingParams":
        """Returns a deep copy of the PoolingParams instance."""
        return PoolingParams(
            dimensions=self.dimensions,
            normalize=self.normalize,
            activation=self.activation,
            softmax=self.softmax,
            task=self.task,
            requires_token_ids=self.requires_token_ids,
        )

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

        if self.task == "embed":
            legal_parameters = ["dimensions", "normalize"]

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

            if self.normalize is None:
                self.normalize = True

        elif self.task in ["classify", "score"]:
            legal_parameters = ["activation"]
            if self.activation is None:
                self.activation = True

        elif self.task == "encode":
            legal_parameters = ["softmax"]
            if self.softmax is None:
                self.softmax = True
        else:
            assert_never(self.task)

        invalid_parameters = []
        for k in self.all_parameters:
            if k in legal_parameters:
                continue

            if getattr(self, k, None) is not None:
                invalid_parameters.append(k)

        if invalid_parameters:
            raise ValueError(
                f"{self.task} only supports {legal_parameters} parameters, "
                f"does not support {invalid_parameters} parameters")

    def merge_default_parameters(self, pooler_config: "PoolerConfig") -> None:
        for k in self.all_parameters:
            if getattr(self, k, None) is None:
                setattr(self, k, getattr(pooler_config, k))

    def __repr__(self) -> str:
        return (f"PoolingParams("
                f"dimensions={self.dimensions}, "
                f"task={self.task}, "
                f"softmax={self.softmax}, "
                f"normalize={self.normalize}, "
                f"activation={self.activation}, "
                f"requires_token_ids={self.requires_token_ids})")

    def __post_init__(self) -> None:
        assert self.output_kind == RequestOutputKind.FINAL_ONLY,\
            "For pooling output_kind has to be FINAL_ONLY"
