# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable, Iterable, Sequence
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
    RequestOutput,
    ScoringRequestOutput,
)
from vllm.pooling_params import PoolingParams
from vllm.renderers.inputs.preprocess import prompt_to_seq
from vllm.tasks import SCORE_TYPE_MAP, PoolingTask, SupportedTask

from .base.io_processor import PoolingIOProcessor
from .factories import init_pooling_io_processors
from .scoring.io_processor import ScoringIOProcessor
from .scoring.typing import ScoreInput
from .typing import OfflineInputsContext, OfflineOutputsContext

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

        prompts_seq = prompt_to_seq(prompts)
        num_requests = len(prompts_seq)

        if isinstance(lora_request, Sequence):
            if len(lora_request) != num_requests:
                raise ValueError(
                    f"The lengths of prompts ({num_requests}) "
                    f"and lora_request ({len(lora_request)}) must be the same."
                )

        if isinstance(pooling_params, Sequence):
            if len(pooling_params) != num_requests:
                raise ValueError(
                    f"The lengths of prompts ({num_requests}) "
                    f"and params ({len(pooling_params)}) must be the same."
                )

        assert pooling_task is not None and pooling_task in self.pooling_io_processors

        io_processor = self.pooling_io_processors[pooling_task]

        if pooling_params is None:
            pooling_params = PoolingParams()

        def pre_process_iterator():
            for i in range(num_requests):
                if isinstance(pooling_params, Sequence):
                    param = pooling_params[i]
                else:
                    param = pooling_params

                if param.task is None:
                    param.task = pooling_task
                elif pooling_task == "plugin":
                    # `plugin` task uses io_processor.parse_request to verify inputs.
                    # We actually allow plugin to overwrite pooling_task.
                    pass
                elif param.task != pooling_task:
                    msg = (
                        f"You cannot overwrite {param.task=!r} with {pooling_task=!r}!"
                    )
                    raise ValueError(msg)

                if isinstance(lora_request, Sequence):
                    lora_requests = [lora_request[i]]
                else:
                    lora_requests = [lora_request]

                ctx = OfflineInputsContext(
                    prompts=[prompts_seq[i]],
                    pooling_params=[param],
                    tokenization_kwargs=tokenization_kwargs,
                    seq_lora_requests=lora_requests,
                    priorities=None,
                )
                yield ctx

        outputs = self._run_tiling_engine(
            io_processor, pre_process_iterator(), num_requests, use_tqdm=use_tqdm
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

        pooling_task = io_processor.pooling_task
        scoring_data = io_processor.valid_inputs(data_1, data_2)
        n_queries = len(scoring_data.data_1)

        if pooling_params is None:
            pooling_params = PoolingParams()

        ctx = OfflineInputsContext(
            prompts=scoring_data,
            pooling_params=pooling_params,
            tokenization_kwargs=tokenization_kwargs,
            chat_template=chat_template,
            n_queries=n_queries,
        )

        engine_inputs = io_processor.pre_process_offline(ctx)
        n_inputs = len(engine_inputs)

        seq_lora_requests = self._lora_request_to_seq(lora_request, n_inputs)
        params_seq = self._params_to_seq(ctx.pooling_params, n_inputs)

        for param in params_seq:
            if param.task is None:
                param.task = pooling_task
            elif param.task != pooling_task:
                msg = f"You cannot overwrite {param.task=!r} with {pooling_task=!r}!"
                raise ValueError(msg)

        seq_priority = self._priority_to_seq(None, n_inputs)

        self._render_and_add_requests(
            prompts=engine_inputs,
            params=params_seq,
            lora_requests=seq_lora_requests,
            priorities=seq_priority,
        )

        outputs = self._run_engine(use_tqdm=use_tqdm, output_type=PoolingRequestOutput)
        outputs = io_processor.post_process_offline(
            ctx=OfflineOutputsContext(outputs=outputs, n_queries=n_queries),
        )

        return [ScoringRequestOutput.from_base(item) for item in outputs]

    def _run_tiling_engine(
        self,
        io_processor: PoolingIOProcessor,
        iterator: Iterable[OfflineInputsContext],
        num_requests: int,
        use_tqdm: bool | Callable[..., tqdm] = True,
    ):
        # Keeping max_num_seqs * 2 requests in the core can already saturate the core.
        # Therefore, keep most requests waiting outside the core.
        max_requests_in_core = (
            self.llm_engine.vllm_config.scheduler_config.max_num_seqs * 2
        )
        n_requests_in_core = 0
        n_waited_requests = num_requests

        if use_tqdm:
            tqdm_func = use_tqdm if callable(use_tqdm) else tqdm
            pbar = tqdm_func(
                total=num_requests,
                desc="Processed prompts",
                dynamic_ncols=True,
                postfix=(f"est. speed input: {0:.2f} toks/s, output: {0:.2f} toks/s"),
            )

        outputs: list[PoolingRequestOutput] = []
        total_in_toks = 0
        total_out_toks = 0
        added_request_ids = set()

        def pre_process(ctx):
            engine_inputs = io_processor.pre_process_offline(ctx)

            request_kwargs = {
                "prompts": engine_inputs,
                "params": ctx.pooling_params,
                "lora_requests": ctx.seq_lora_requests,
                "priorities": ctx.priorities,
            }

            return request_kwargs

        it = self._executor.map(pre_process, iterator)

        try:
            while n_waited_requests or self.llm_engine.has_unfinished_requests():
                for i in range(max_requests_in_core - n_requests_in_core):
                    if n_waited_requests == 0:
                        break

                    requests = next(it)
                    n_waited_requests -= 1
                    n_requests_in_core += 1

                    request_ids = self._render_and_add_requests(**requests)

                    for request_id in request_ids:
                        # undo assign_request_id
                        request_id = request_id.split("-", 1)[0]
                        added_request_ids.update(request_id)

                step_outputs = self.llm_engine.step()

                for output in step_outputs:
                    assert isinstance(output, PoolingRequestOutput)
                    assert output.finished
                    outputs.append(output)
                    added_request_ids.discard(output.request_id)
                    n_requests_in_core -= 1

                    if use_tqdm:
                        if isinstance(output, RequestOutput):
                            # Calculate tokens only for RequestOutput
                            n = len(output.outputs)
                            assert output.prompt_token_ids is not None
                            total_in_toks += len(output.prompt_token_ids) * n
                            in_spd = total_in_toks / pbar.format_dict["elapsed"]
                            total_out_toks += sum(
                                len(stp.token_ids) for stp in output.outputs
                            )
                            out_spd = total_out_toks / pbar.format_dict["elapsed"]
                            pbar.postfix = (
                                f"est. speed input: {in_spd:.2f} toks/s, "
                                f"output: {out_spd:.2f} toks/s"
                            )
                            pbar.update(n)
                        else:
                            pbar.update(1)
                        if pbar.n == num_requests:
                            pbar.refresh()

        except Exception as e:
            if added_request_ids:
                self.llm_engine.abort_request(list(added_request_ids), internal=True)
            raise e

        if use_tqdm:
            pbar.close()
        # Sort the outputs by request ID.
        # This is necessary because some requests may be finished earlier than
        # its previous requests.
        return sorted(outputs, key=lambda x: int(x.request_id))
