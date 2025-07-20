# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING, Literal, Optional

import msgspec

from vllm.sampling_params import RequestOutputKind

if TYPE_CHECKING:
    from vllm.config import ModelConfig

PoolingTask = Literal["encode", "embed", "classify", "score"]


class PoolingParams(
        msgspec.Struct,
        omit_defaults=True,  # type: ignore[call-arg]
        array_like=True):  # type: ignore[call-arg]
    """API parameters for pooling models.

    Attributes:
        dimensions: Reduce the dimensions of embeddings
                    if model support matryoshka representation.
    """

    dimensions: Optional[int] = None

    output_kind: RequestOutputKind = RequestOutputKind.FINAL_ONLY

    task: Optional[PoolingTask] = None
    """Internal use only."""

    requires_token_ids: bool = False
    """Internal use only."""

    def clone(self) -> "PoolingParams":
        """Returns a deep copy of the PoolingParams instance."""
        return PoolingParams(
            dimensions=self.dimensions,
            task=self.task,
            requires_token_ids=self.requires_token_ids,
        )

    def verify(self, task: PoolingTask, model_config: "ModelConfig") -> None:
        if self.task is None:
            self.task = task
        elif self.task != task:
            msg = f"You cannot overwrite {self.task=!r} with {task=!r}!"
            raise ValueError(msg)

        # NOTE: Task validation needs to done against the model instance,
        # which is not available in model config. So, it's not included
        # in this method

        if self.dimensions is not None:
            if not model_config.is_matryoshka:
                raise ValueError(
                    f'Model "{model_config.served_model_name}" does not '
                    f'support matryoshka representation, '
                    f'changing output dimensions will lead to poor results.')

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

    def __repr__(self) -> str:
        return (f"PoolingParams("
                f"dimensions={self.dimensions}, "
                f"task={self.task}, "
                f"requires_token_ids={self.requires_token_ids})")

    def __post_init__(self) -> None:
        assert self.output_kind == RequestOutputKind.FINAL_ONLY,\
            "For pooling output_kind has to be FINAL_ONLY"
