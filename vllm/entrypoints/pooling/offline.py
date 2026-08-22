# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable, Sequence
from typing import Any

from tqdm.auto import tqdm

from vllm.entrypoints.chat_utils import ChatTemplateConfig
from vllm.entrypoints.offline_utils import OfflineInferenceMixin
from vllm.inputs import DataPrompt, PromptType
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.outputs import (
    ClassificationRequestOutput,
    EmbeddingRequestOutput,
    PoolingRequestOutput,
    ScoringRequestOutput,
)
from vllm.pooling_params import PoolingParams
from vllm.tasks import SCORE_TYPE_MAP, PoolingTask, SupportedTask

from .base.io_processor import PoolingIOProcessor
from .factories import init_pooling_io_processors
from .scoring.io_processor import ScoringIOProcessor
from .scoring.typing import ScoreInput
from .typing import (
    AnyOfflineInputsContext,
    OfflineEncodeInputsContext,
    OfflineOutputsContext,
    OfflinePluginInputsContext,
    OfflineScoringInputsContext,
    RequestFactory,
)

logger = init_logger(__name__)


class PoolingOfflineMixin(OfflineInferenceMixin):
    """Offline inference for pooling models"""

    runner_type: str
    chat_template: str | None
    supported_tasks: tuple[SupportedTask, ...]

    def __init__(self):
        self.pooling_task = self.model_config.get_pooling_task(self.supported_tasks)
        if self.pooling_task is not None:
            logger.info("Supported pooling task: %s", self.pooling_task)

        self.chat_template_config = ChatTemplateConfig(chat_template=self.chat_template)
        self.pooling_io_processors = init_pooling_io_processors(
            supported_tasks=self.supported_tasks,
            vllm_config=self.llm_engine.vllm_config,
            renderer=self.renderer,
            chat_template_config=self.chat_template_config,
        )

        # Use thread pool executor to accelerate preprocessing.
        self._executor = self.renderer._executor

    def encode(
        self,
        prompts: PromptType | Sequence[PromptType] | DataPrompt,
        pooling_params: PoolingParams | Sequence[PoolingParams] | None = None,
        *,
        use_tqdm: bool | Callable[..., tqdm] = True,
        lora_request: list[LoRARequest] | LoRARequest | None = None,
        pooling_task: PoolingTask | None = None,
        tokenization_kwargs: dict[str, Any] | None = None,
    ) -> list[PoolingRequestOutput]:
        """Apply pooling to the hidden states corresponding to the input
        prompts.

        This class automatically batches the given prompts, considering
        the memory constraint. For the best performance, put all of your prompts
        into a single list and pass it to this method.

        Args:
            prompts: The prompts to the LLM. You may pass a sequence of prompts
                for batch inference. See [PromptType][vllm.inputs.PromptType]
                for more details about the format of each prompt.
            pooling_params: The pooling parameters for pooling. If None, we
                use the default pooling parameters.
            use_tqdm: If `True`, shows a tqdm progress bar.
                If a callable (e.g., `functools.partial(tqdm, leave=False)`),
                it is used to create the progress bar.
                If `False`, no progress bar is created.
            lora_request: LoRA request to use for generation, if any.
            pooling_task: Override the pooling task to use.
            tokenization_kwargs: Overrides for `tokenizer.encode`.

        Returns:
            A list of `PoolingRequestOutput` objects containing the
            pooled hidden states in the same order as the input prompts.
        """

        if isinstance(prompts, dict) and "data" in prompts and pooling_task != "plugin":
            raise ValueError(
                "The 'data' field is only supported for the 'plugin' pooling task."
            )
        self._verify_pooling_task(pooling_task)
        assert pooling_task is not None and pooling_task in self.pooling_io_processors

        io_processor = self.pooling_io_processors[pooling_task]

        ctx: AnyOfflineInputsContext
        if isinstance(prompts, dict) and "data" in prompts:
            ctx = OfflinePluginInputsContext(
                pooling_task=pooling_task,
                prompts=prompts,  # type: ignore[arg-type]
                tokenization_kwargs=tokenization_kwargs,
                pooling_params=pooling_params,
                lora_request=lora_request,
                priorities=None,
            )
        else:
            ctx = OfflineEncodeInputsContext(
                pooling_task=pooling_task,
                prompts=prompts,
                tokenization_kwargs=tokenization_kwargs,
                pooling_params=pooling_params,
                lora_request=lora_request,
                priorities=None,
            )

        request_factory, num_requests = io_processor.get_request_factory_offline(ctx)
        outputs = self._run_tiling_engine(
            io_processor, request_factory, num_requests, use_tqdm=use_tqdm
        )
        outputs = io_processor.post_process_offline(
            ctx=OfflineOutputsContext(outputs=outputs)
        )
        return outputs

    def _verify_pooling_task(self, pooling_task: PoolingTask | None):
        if self.runner_type != "pooling":
            raise ValueError(
                "LLM.encode() is only supported for pooling models. "
                "Try passing `--runner pooling` to use the model as a "
                "pooling model."
            )

        if pooling_task is None:
            raise ValueError(
                """
                pooling_task required for `LLM.encode`.
                Please use one of the more specific methods or set the pooling_task when using `LLM.encode`:
                  - For embeddings, use `LLM.embed(...)` or `pooling_task="embed"`.
                  - For classification logits, use `LLM.classify(...)` or `pooling_task="classify"`.
                  - For similarity scores, use `LLM.score(...)`.
                  - For rewards, `pooling_task="classify"` or `pooling_task="token_classify"`.
                  - For token classification, use `pooling_task="token_classify"`.
                  - For multi-vector retrieval, use `pooling_task="token_embed"`.
                """  # noqa: E501
            )

        if (
            pooling_task in ("embed", "token_embed")
            and pooling_task not in self.supported_tasks
        ):
            raise ValueError(
                "Embedding API is not supported by this model. "
                "Try converting the model using `--convert embed`."
            )

        if (
            pooling_task in ("classify", "token_classify")
            and pooling_task not in self.supported_tasks
        ):
            raise ValueError(
                "Classification API is not supported by this model. "
                "Try converting the model using `--convert classify`."
            )

        # plugin task uses io_processor.parse_request to verify inputs
        if pooling_task != "plugin" and pooling_task != self.pooling_task:
            if pooling_task not in self.supported_tasks:
                raise ValueError(
                    f"Unsupported task: {pooling_task!r} "
                    f"Supported tasks: {self.supported_tasks}"
                )
            else:
                raise ValueError(
                    f"Try switching the model's pooling_task "
                    f'via `PoolerConfig(task="{pooling_task}")`'
                )

        if pooling_task == "plugin" and "plugin" not in self.pooling_io_processors:
            raise ValueError(
                "No IOProcessor plugin installed. Please refer "
                "to the documentation and to the "
                "'prithvi_geospatial_mae_io_processor' "
                "offline inference example for more details."
            )

    def embed(
        self,
        prompts: PromptType | Sequence[PromptType],
        *,
        use_tqdm: bool | Callable[..., tqdm] = True,
        pooling_params: PoolingParams | Sequence[PoolingParams] | None = None,
        lora_request: list[LoRARequest] | LoRARequest | None = None,
        tokenization_kwargs: dict[str, Any] | None = None,
    ) -> list[EmbeddingRequestOutput]:
        """
        Generate an embedding vector for each prompt.

        This class automatically batches the given prompts, considering
        the memory constraint. For the best performance, put all of your prompts
        into a single list and pass it to this method.

        Args:
            prompts: The prompts to the LLM. You may pass a sequence of prompts
                for batch inference. See [PromptType][vllm.inputs.PromptType]
                for more details about the format of each prompt.
            pooling_params: The pooling parameters for pooling. If None, we
                use the default pooling parameters.
            use_tqdm: If `True`, shows a tqdm progress bar.
                If a callable (e.g., `functools.partial(tqdm, leave=False)`),
                it is used to create the progress bar.
                If `False`, no progress bar is created.
            lora_request: LoRA request to use for generation, if any.
            tokenization_kwargs: Overrides for `tokenizer.encode`.

        Returns:
            A list of `EmbeddingRequestOutput` objects containing the
            embedding vectors in the same order as the input prompts.
        """

        items = self.encode(
            prompts,
            use_tqdm=use_tqdm,
            pooling_params=pooling_params,
            lora_request=lora_request,
            pooling_task="embed",
            tokenization_kwargs=tokenization_kwargs,
        )

        return [EmbeddingRequestOutput.from_base(item) for item in items]

    def classify(
        self,
        prompts: PromptType | Sequence[PromptType],
        *,
        pooling_params: PoolingParams | Sequence[PoolingParams] | None = None,
        use_tqdm: bool | Callable[..., tqdm] = True,
        lora_request: list[LoRARequest] | LoRARequest | None = None,
        tokenization_kwargs: dict[str, Any] | None = None,
    ) -> list[ClassificationRequestOutput]:
        """
        Generate class logits for each prompt.

        This class automatically batches the given prompts, considering
        the memory constraint. For the best performance, put all of your prompts
        into a single list and pass it to this method.

        Args:
            prompts: The prompts to the LLM. You may pass a sequence of prompts
                for batch inference. See [PromptType][vllm.inputs.PromptType]
                for more details about the format of each prompt.
            pooling_params: The pooling parameters for pooling. If None, we
                use the default pooling parameters.
            use_tqdm: If `True`, shows a tqdm progress bar.
                If a callable (e.g., `functools.partial(tqdm, leave=False)`),
                it is used to create the progress bar.
                If `False`, no progress bar is created.
            lora_request: LoRA request to use for generation, if any.
            tokenization_kwargs: Overrides for `tokenizer.encode`.

        Returns:
            A list of `ClassificationRequestOutput` objects containing the
            embedding vectors in the same order as the input prompts.
        """

        items = self.encode(
            prompts,
            use_tqdm=use_tqdm,
            pooling_params=pooling_params,
            lora_request=lora_request,
            pooling_task="classify",
            tokenization_kwargs=tokenization_kwargs,
        )

        return [ClassificationRequestOutput.from_base(item) for item in items]

    def score(
        self,
        data_1: ScoreInput | list[ScoreInput],
        data_2: ScoreInput | list[ScoreInput],
        /,
        *,
        use_tqdm: bool | Callable[..., tqdm] = True,
        pooling_params: PoolingParams | None = None,
        lora_request: list[LoRARequest] | LoRARequest | None = None,
        tokenization_kwargs: dict[str, Any] | None = None,
        chat_template: str | None = None,
    ) -> list[ScoringRequestOutput]:
        """Generate similarity scores for all pairs `<text,text_pair>` or
          `<multi-modal data, multi-modal data pair>`.

        The inputs can be `1 -> 1`, `1 -> N` or `N -> N`.
        In the `1 - N` case the `data_1` input will be replicated `N`
        times to pair with the `data_2` inputs.
        The input pairs are used to build a list of prompts for the
        cross encoder model. This class automatically batches the prompts,
        considering the memory constraint. For the best performance, put all
        of your inputs into a single list and pass it to this method.

        Supports both text and multi-modal data (images, etc.) when used with
        appropriate multi-modal models. For multi-modal inputs, ensure the
        prompt structure matches the model's expected input format.

        Args:
            data_1: Can be a single prompt, a list of prompts or
                `ScoreMultiModalParam`, which can contain either text or
                multi-modal data. When a list, it must have the same length as
                the `data_2` list.
            data_2: The data to pair with the query to form the input to
                the LLM. Can be text or multi-modal data. See [PromptType]
                [vllm.inputs.PromptType] for more details about the format of
                each prompt.
            pooling_params: The pooling parameters for pooling. If None, we
                use the default pooling parameters.
            use_tqdm: If `True`, shows a tqdm progress bar.
                If a callable (e.g., `functools.partial(tqdm, leave=False)`),
                it is used to create the progress bar.
                If `False`, no progress bar is created.
            lora_request: LoRA request to use for generation, if any.
            chat_template: The chat template to use for the scoring. If None, we
                use the model's default chat template.
            tokenization_kwargs: Overrides for `tokenizer.encode`.
        Returns:
            A list of `ScoringRequestOutput` objects containing the
            generated scores in the same order as the input prompts.
        """

        if self.runner_type != "pooling":
            raise ValueError(
                "LLM.score() is only supported for pooling models. "
                "Try passing `--runner pooling` to use the model as a "
                "pooling model."
            )

        score_type: str | None = SCORE_TYPE_MAP.get(self.pooling_task, None)  # type: ignore[arg-type]
        if (
            score_type == "cross-encoder"
            and getattr(self.model_config.hf_config, "num_labels", 0) != 1
        ):
            raise ValueError("Scoring API is only enabled for num_labels == 1.")

        if score_type is None or score_type not in self.pooling_io_processors:
            raise ValueError("This model does not support the Scoring API.")

        io_processor = self.pooling_io_processors[score_type]
        assert isinstance(io_processor, ScoringIOProcessor)

        scoring_data = io_processor.valid_inputs(data_1, data_2)
        n_queries = len(scoring_data.data_1)

        if pooling_params is None:
            pooling_params = PoolingParams()

        assert isinstance(pooling_params, PoolingParams)
        pooling_task = io_processor.pooling_task
        if pooling_params.task is None:
            pooling_params.task = pooling_task
        elif pooling_params.task != pooling_task:
            msg = (
                f"You cannot overwrite {pooling_params.task=!r} with {pooling_task=!r}!"
            )
            raise ValueError(msg)

        ctx = OfflineScoringInputsContext(
            pooling_task=pooling_task,
            scoring_data=scoring_data,
            pooling_params=pooling_params,
            tokenization_kwargs=tokenization_kwargs,
            lora_request=lora_request,
            chat_template=chat_template,
            priorities=None,
        )

        request_factory, num_requests = io_processor.get_request_factory_offline(ctx)
        outputs = self._run_tiling_engine(
            io_processor, request_factory, num_requests, use_tqdm=use_tqdm
        )

        outputs = io_processor.post_process_offline(
            ctx=OfflineOutputsContext(outputs=outputs, n_queries=n_queries),
        )

        return [ScoringRequestOutput.from_base(item) for item in outputs]

    def _run_tiling_engine(
        self,
        io_processor: PoolingIOProcessor,
        request_factory: RequestFactory,
        num_requests: int,
        use_tqdm: bool | Callable[..., tqdm] = True,
    ):
        if num_requests == 0:
            raise ValueError("You must pass at least one prompt")

        # Keeping max_num_seqs * 2 requests in the core can already saturate the core.
        # Therefore, keep most requests waiting outside the core.
        max_requests_in_core = (
            self.llm_engine.vllm_config.scheduler_config.max_num_seqs * 2
        )
        num_requests_in_core = 0
        num_waited_requests = num_requests

        if use_tqdm:
            tqdm_func = use_tqdm if callable(use_tqdm) else tqdm
            pbar = tqdm_func(
                total=num_requests,
                desc="Processed prompts",
                dynamic_ncols=True,
                postfix=f"est. speed input: {0:.2f} toks/s, output: {0:.2f} toks/s",
            )

        outputs: list[PoolingRequestOutput] = []
        added_request_ids: set[str] = set()

        it = self._executor.map(io_processor.render, request_factory())

        try:
            while num_waited_requests or self.llm_engine.has_unfinished_requests():
                requests = []
                for _ in range(max_requests_in_core - num_requests_in_core):
                    if num_waited_requests == 0:
                        break
                    try:
                        request = next(it)
                        requests.append(request)
                    except StopIteration:
                        num_waited_requests = 0
                        break

                    num_waited_requests -= 1
                    num_requests_in_core += 1

                if requests:
                    request_ids = self._render_and_add_requests(
                        prompts=[x["prompts"] for x in requests],
                        params=[x["params"] for x in requests],
                        lora_requests=[x["lora_requests"] for x in requests],
                        priorities=[x["priorities"] for x in requests],
                    )

                    for request_id in request_ids:
                        # undo assign_request_id
                        request_id = request_id.split("-", 1)[0]
                        added_request_ids.add(request_id)

                step_outputs = self.llm_engine.step()
                for output in step_outputs:
                    assert isinstance(output, PoolingRequestOutput)
                    assert output.finished
                    outputs.append(output)
                    added_request_ids.discard(output.request_id)
                    num_requests_in_core -= 1

                    if use_tqdm:
                        pbar.update(1)
                        if pbar.n == num_requests:
                            pbar.refresh()

        except Exception:
            if added_request_ids:
                self.llm_engine.abort_request(list(added_request_ids))
            raise

        finally:
            if use_tqdm:
                pbar.close()

        return sorted(outputs, key=lambda x: int(x.request_id))
