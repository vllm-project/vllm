# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from copy import deepcopy
from typing import Any

import msgspec

from vllm.config import ModelConfig, PoolerConfig
from vllm.sampling_params import RequestOutputKind
from vllm.tasks import PoolingTask


class PoolingParams(
    msgspec.Struct,
    omit_defaults=True,  # type: ignore[call-arg]
    array_like=True,
):  # type: ignore[call-arg]
    """API parameters for pooling models.

    Attributes:
        use_activation: Whether to apply activation function to the pooler outputs.
            `None` uses the pooler's default, which is `True` in most cases.
        dimensions: Reduce the dimensions of embeddings
            if model support matryoshka representation.
    """

    # --8<-- [start:common-pooling-params]
    use_activation: bool | None = None
    # --8<-- [end:common-pooling-params]

    ## for embeddings models
    # --8<-- [start:embed-pooling-params]
    dimensions: int | None = None
    # --8<-- [end:embed-pooling-params]

    ## for classification, scoring and rerank
    # --8<-- [start:classify-pooling-params]
    # --8<-- [end:classify-pooling-params]

    ## for step pooling models
    step_tag_id: int | None = None
    returned_token_ids: list[int] | None = None

    ## Internal use only
    task: PoolingTask | None = None
    requires_token_ids: bool = False
    skip_reading_prefix_cache: bool | None = None
    extra_kwargs: dict[str, Any] | None = None
    output_kind: RequestOutputKind = RequestOutputKind.FINAL_ONLY

    @property
    def all_parameters(self) -> list[str]:
        return ["dimensions", "use_activation"]

    @property
    def valid_parameters(self):
        return {
            "embed": ["dimensions", "use_activation"],
            "classify": ["use_activation"],
            "score": ["use_activation"],
            "token_embed": ["dimensions", "use_activation"],
            "token_classify": ["use_activation"],
        }

    def clone(self) -> "PoolingParams":
        """Returns a deep copy of the PoolingParams instance."""
        return deepcopy(self)

    def verify(self, model_config: ModelConfig) -> None:
        # plugin task uses io_processor.parse_request to verify inputs,
        # skipping PoolingParams verify
        tasks = self.get_tasks()
        if "plugin" in tasks:
            if self.skip_reading_prefix_cache is None:
                self.skip_reading_prefix_cache = True
            return

        # NOTE: Task validation needs to done against the model instance,
        # which is not available in model config. So, it's not included
        # in this method
        self._merge_default_parameters(model_config)
        self._set_default_parameters(model_config)
        self._verify_valid_parameters()

    def _merge_default_parameters(self, model_config: ModelConfig) -> None:
        pooler_config = model_config.pooler_config
        if pooler_config is None:
            return
        tasks = self.get_tasks()
        assert tasks, "task must be set"

        # For multi-task, we use parameters from the first task
        # In the future, we could support per-task parameters
        task = tasks[0]
        valid_parameters = self.valid_parameters[task]

        for k in valid_parameters:
            if getattr(pooler_config, k, None) is None:
                continue

            if getattr(self, k, None) is None:
                setattr(self, k, getattr(pooler_config, k))

        if self.skip_reading_prefix_cache is None:
            # If prefix caching is enabled,
            # the output of all pooling may less than n_prompt_tokens,
            # we need to skip reading cache at this request.
            if any(t in ["token_embed", "token_classify"] for t in tasks):
                self.skip_reading_prefix_cache = True
            else:
                self.skip_reading_prefix_cache = False

        self._verify_step_pooling(pooler_config, valid_parameters, task)

    def _verify_step_pooling(
        self,
        pooler_config: PoolerConfig,
        valid_parameters: list[str],
        task: PoolingTask,
    ):
        step_pooling_parameters = ["step_tag_id", "returned_token_ids"]
        if pooler_config.tok_pooling_type != "STEP":
            invalid_parameters = []
            for k in step_pooling_parameters:
                if getattr(self, k, None) is not None:
                    invalid_parameters.append(k)

            if invalid_parameters:
                raise ValueError(
                    f"Task {task} only supports {valid_parameters} "
                    f"parameters, does not support "
                    f"{invalid_parameters} parameters"
                )
        else:
            for k in step_pooling_parameters:
                if getattr(pooler_config, k, None) is None:
                    continue

                if getattr(self, k, None) is None:
                    setattr(self, k, getattr(pooler_config, k))

    def _set_default_parameters(self, model_config: ModelConfig):
        tasks = self.get_tasks()
        if not tasks:
            raise ValueError("At least one task must be set")

        # For multi-task, we check all tasks to set appropriate defaults
        # Using first task for backward compatibility with single-task behavior
        task = tasks[0]
        if task in ["embed", "token_embed"]:
            if self.use_activation is None:
                self.use_activation = True

            if self.dimensions is not None:
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
                            f"Model {model_config.served_model_name!r} "
                            f"only supports {str(mds)} matryoshka dimensions, "
                            f"use other output dimensions will "
                            f"lead to poor results."
                        )
                elif self.dimensions < 1:
                    raise ValueError("Dimensions must be greater than 0")

        elif task in ["classify", "score", "token_classify"]:
            if self.use_activation is None:
                self.use_activation = True
        else:
            raise ValueError(f"Unknown pooling task: {task!r}")

    def _verify_valid_parameters(self):
        tasks = self.get_tasks()
        if not tasks:
            raise ValueError("At least one task must be set")

        # For multi-task, we need to ensure parameters are valid for all tasks
        # We collect all valid parameters across all tasks
        all_valid_parameters = set()
        invalid_parameters_for_task = {}

        for task in tasks:
            valid_parameters = set(self.valid_parameters[task])
            all_valid_parameters.update(valid_parameters)

            # Check if any parameters are invalid for this specific task
            task_invalid = []
            for k in self.all_parameters:
                if k in valid_parameters:
                    continue
                if getattr(self, k, None) is not None:
                    task_invalid.append(k)

            if task_invalid:
                invalid_parameters_for_task[task] = task_invalid
        # multi-tasks may have different valid parameters, skip check,
        # e.g. token_classify&embed
        if len(tasks) == 1 and invalid_parameters_for_task:
            # Build an informative error message showing which parameters
            # are invalid for which tasks
            errors = []
            for task, invalid in invalid_parameters_for_task.items():
                valid_params = self.valid_parameters[task]
                errors.append(
                    f"Task {task!r} only supports {valid_params} "
                    f"parameters, does not support {invalid} parameters"
                )
            raise ValueError("\n".join(errors))

    def __repr__(self) -> str:
        return (
            f"PoolingParams("
            f"task={self.task}, "
            f"dimensions={self.dimensions}, "
            f"use_activation={self.use_activation}, "
            f"step_tag_id={self.step_tag_id}, "
            f"returned_token_ids={self.returned_token_ids}, "
            f"requires_token_ids={self.requires_token_ids}, "
            f"skip_reading_prefix_cache={self.skip_reading_prefix_cache}, "
            f"extra_kwargs={self.extra_kwargs})"
        )

    def get_tasks(self) -> list[PoolingTask]:
        """Get the list of pooling tasks for this request.

        Returns multiple tasks if `tasks` is set, otherwise returns
        a single task from `task` attribute for backward compatibility.
        """
        if self.task == "token_classify+embed":
            return ["token_classify", "embed"]
        if self.task is not None:
            return [self.task]
        return []

    def __post_init__(self) -> None:
        assert self.output_kind == RequestOutputKind.FINAL_ONLY, (
            "For pooling output_kind has to be FINAL_ONLY"
        )
