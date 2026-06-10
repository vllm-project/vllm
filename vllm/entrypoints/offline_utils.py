# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable, Iterable, Sequence
from typing import Any

from tqdm import tqdm
from typing_extensions import TypeVar

from vllm import (
    PoolingParams,
    PoolingRequestOutput,
    PromptType,
    RequestOutput,
    SamplingParams,
)
from vllm.config import ModelConfig
from vllm.entrypoints.chat_utils import (
    ChatCompletionMessageParam,
    ChatTemplateContentFormatOption,
)
from vllm.inputs import EngineInput
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.renderers import BaseRenderer, ChatParams, merge_kwargs
from vllm.renderers.inputs.preprocess import (
    conversation_to_seq,
    parse_model_prompt,
    prompt_to_seq,
)
from vllm.sampling_params import RequestOutputKind
from vllm.utils.counter import Counter
from vllm.utils.mistral import is_mistral_tokenizer
from vllm.utils.tqdm_utils import maybe_tqdm
from vllm.v1.engine.llm_engine import LLMEngine

logger = init_logger(__name__)


_P = TypeVar("_P", bound=SamplingParams | PoolingParams | None)
_O = TypeVar(
    "_O",
    bound=RequestOutput | PoolingRequestOutput,
    default=RequestOutput | PoolingRequestOutput,
)
_R = TypeVar("_R", default=Any)


class OfflineInferenceMixin:
    """Offline inference utils"""

    request_counter: Counter
    renderer: BaseRenderer
    llm_engine: "LLMEngine"
    model_config: ModelConfig

    def _resolve_mm_lora(
        self,
        prompt: EngineInput,
        lora_request: LoRARequest | None,
    ) -> LoRARequest | None:
        if prompt["type"] != "multimodal":
            return lora_request

        lora_config = self.llm_engine.vllm_config.lora_config
        default_mm_loras = None if lora_config is None else lora_config.default_mm_loras
        if not default_mm_loras:
            return lora_request

        prompt_modalities = prompt["mm_placeholders"].keys()
        intersection = set(prompt_modalities).intersection(default_mm_loras.keys())
        if not intersection:
            return lora_request

        if len(intersection) > 1:
            # TODO: Would be nice to be able to have multiple loras per prompt
            logger.warning(
                "Multiple modality specific loras were registered and would be "
                "used by a single prompt consuming several modalities; "
                "currently we only support one lora per request; as such, "
                "lora(s) registered with modalities: %s will be skipped",
                intersection,
            )
            return lora_request

        # Build the LoRA request; the ID of the default mm lora is the
        # index of the modality name sorted alphabetically + 1.
        modality_name = intersection.pop()
        modality_lora_path = default_mm_loras[modality_name]
        modality_lora_id = sorted(default_mm_loras).index(modality_name) + 1

        # If we have a collision, warn if there is a collision,
        # but always send the explicitly provided request.
        if lora_request:
            if lora_request.lora_int_id != modality_lora_id:
                logger.warning(
                    "A modality with a registered lora and a lora_request "
                    "with a different ID were provided; falling back to the "
                    "lora_request as we only apply one LoRARequest per prompt"
                )
            return lora_request

        return LoRARequest(
            modality_name,
            modality_lora_id,
            modality_lora_path,
        )

    def _preprocess_cmpl(
        self,
        prompts: Sequence[PromptType],
        tokenization_kwargs: dict[str, Any] | None = None,
        mm_processor_kwargs: dict[str, Any] | None = None,
    ) -> Sequence[EngineInput]:
        """
        Convert prompt inputs from LLM APIs (other than [LLM.chat][]) into
        a format that can be passed to `_add_request`.

        Refer to [LLM.generate][] for a complete description of the arguments.

        Returns:
            A list of `EngineInput` objects ready to be passed into LLMEngine.
        """
        renderer = self.renderer
        model_config = self.model_config

        parsed_prompts = [
            parse_model_prompt(model_config, prompt) for prompt in prompts
        ]
        tok_params = renderer.default_cmpl_tok_params.with_kwargs(
            **(tokenization_kwargs or {})
        )
        prompt_extras = (
            None
            if mm_processor_kwargs is None
            else {"mm_processor_kwargs": mm_processor_kwargs}
        )

        return renderer.render_cmpl(
            parsed_prompts,
            tok_params,
            prompt_extras=prompt_extras,
        )

    def _preprocess_cmpl_one(
        self,
        prompt: PromptType,
        tokenization_kwargs: dict[str, Any] | None = None,
        mm_processor_kwargs: dict[str, Any] | None = None,
    ) -> EngineInput:
        (engine_input,) = self._preprocess_cmpl(
            [prompt],
            tokenization_kwargs,
            mm_processor_kwargs=mm_processor_kwargs,
        )
        return engine_input

    def _preprocess_chat(
        self,
        conversations: Sequence[list[ChatCompletionMessageParam]],
        chat_template: str | None = None,
        chat_template_content_format: ChatTemplateContentFormatOption = "auto",
        chat_template_kwargs: dict[str, Any] | None = None,
        add_generation_prompt: bool = True,
        continue_final_message: bool = False,
        tools: list[dict[str, Any]] | None = None,
        tokenization_kwargs: dict[str, Any] | None = None,
        mm_processor_kwargs: dict[str, Any] | None = None,
    ) -> Sequence[EngineInput]:
        """
        Convert a list of conversations into prompts so that they can then
        be used as input for other LLM APIs.

        Refer to [LLM.chat][] for a complete description of the arguments.

        Returns:
            A list of `EngineInput` objects ready to be passed into LLMEngine.
        """
        renderer = self.renderer

        chat_params = ChatParams(
            chat_template=chat_template,
            chat_template_content_format=chat_template_content_format,
            chat_template_kwargs=merge_kwargs(
                chat_template_kwargs,
                dict(
                    add_generation_prompt=add_generation_prompt,
                    continue_final_message=continue_final_message,
                    tools=tools,
                    tokenize=(
                        is_mistral_tokenizer(renderer.tokenizer)
                        or self.model_config.enable_prompt_embeds
                    ),
                ),
            ),
            mm_processor_kwargs=mm_processor_kwargs,
        )
        tok_params = renderer.default_chat_tok_params.with_kwargs(
            **(tokenization_kwargs or {})
        )
        prompt_extras = (
            None
            if mm_processor_kwargs is None
            else {"mm_processor_kwargs": mm_processor_kwargs}
        )

        _, engine_inputs = renderer.render_chat(
            conversations,
            chat_params,
            tok_params,
            prompt_extras=prompt_extras,
        )

        return engine_inputs

    def _preprocess_chat_one(
        self,
        conversation: list[ChatCompletionMessageParam],
        chat_template: str | None = None,
        chat_template_content_format: ChatTemplateContentFormatOption = "auto",
        chat_template_kwargs: dict[str, Any] | None = None,
        add_generation_prompt: bool = True,
        continue_final_message: bool = False,
        tools: list[dict[str, Any]] | None = None,
        tokenization_kwargs: dict[str, Any] | None = None,
        mm_processor_kwargs: dict[str, Any] | None = None,
    ) -> EngineInput:
        (engine_input,) = self._preprocess_chat(
            [conversation],
            chat_template=chat_template,
            chat_template_content_format=chat_template_content_format,
            chat_template_kwargs=chat_template_kwargs,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=continue_final_message,
            tools=tools,
            tokenization_kwargs=tokenization_kwargs,
            mm_processor_kwargs=mm_processor_kwargs,
        )

        return engine_input

    def _params_to_seq(
        self,
        params: _P | Sequence[_P],
        num_requests: int,
    ) -> Sequence[_P]:
        if isinstance(params, Sequence):
            if len(params) != num_requests:
                raise ValueError(
                    f"The lengths of prompts ({num_requests}) "
                    f"and params ({len(params)}) must be the same."
                )

            return params

        return [params] * num_requests

    def _lora_request_to_seq(
        self,
        lora_request: LoRARequest | None | Sequence[LoRARequest | None],
        num_requests: int,
    ) -> Sequence[LoRARequest | None]:
        if isinstance(lora_request, Sequence):
            if len(lora_request) != num_requests:
                raise ValueError(
                    f"The lengths of prompts ({num_requests}) "
                    f"and lora_request ({len(lora_request)}) must be the same."
                )

            return lora_request

        return [lora_request] * num_requests

    def _priority_to_seq(
        self,
        priority: list[int] | None,
        num_requests: int,
    ) -> Sequence[int]:
        if priority is not None:
            if len(priority) != num_requests:
                raise ValueError(
                    f"The lengths of prompts ({num_requests}) "
                    f"and priority ({len(priority)}) must be the same."
                )

            return priority

        return [0] * num_requests

    def _add_completion_requests(
        self,
        prompts: PromptType | Sequence[PromptType],
        params: SamplingParams
        | PoolingParams
        | Sequence[SamplingParams | PoolingParams],
        *,
        use_tqdm: bool | Callable[..., tqdm] = True,
        lora_request: Sequence[LoRARequest] | LoRARequest | None = None,
        priority: list[int] | None = None,
        tokenization_kwargs: dict[str, Any] | None = None,
        mm_processor_kwargs: dict[str, Any] | None = None,
    ) -> list[str]:
        seq_prompts = prompt_to_seq(prompts)
        seq_params = self._params_to_seq(params, len(seq_prompts))
        seq_lora_requests = self._lora_request_to_seq(lora_request, len(seq_prompts))
        seq_priority = self._priority_to_seq(priority, len(seq_prompts))

        return self._render_and_add_requests(
            prompts=(
                self._preprocess_cmpl_one(
                    prompt,
                    tokenization_kwargs,
                    mm_processor_kwargs=mm_processor_kwargs,
                )
                for prompt in maybe_tqdm(
                    seq_prompts,
                    use_tqdm=use_tqdm,
                    desc="Rendering prompts",
                )
            ),
            params=seq_params,
            lora_requests=seq_lora_requests,
            priorities=seq_priority,
        )

    def _run_completion(
        self,
        prompts: PromptType | Sequence[PromptType],
        params: SamplingParams
        | PoolingParams
        | Sequence[SamplingParams | PoolingParams],
        output_type: type[_O],
        *,
        use_tqdm: bool | Callable[..., tqdm] = True,
        lora_request: Sequence[LoRARequest] | LoRARequest | None = None,
        priority: list[int] | None = None,
        tokenization_kwargs: dict[str, Any] | None = None,
        mm_processor_kwargs: dict[str, Any] | None = None,
    ):
        self._add_completion_requests(
            prompts=prompts,
            params=params,
            use_tqdm=use_tqdm,
            lora_request=lora_request,
            priority=priority,
            tokenization_kwargs=tokenization_kwargs,
            mm_processor_kwargs=mm_processor_kwargs,
        )
        return self._run_engine(use_tqdm=use_tqdm, output_type=output_type)

    def _run_chat(
        self,
        messages: list[ChatCompletionMessageParam]
        | Sequence[list[ChatCompletionMessageParam]],
        params: SamplingParams
        | PoolingParams
        | Sequence[SamplingParams | PoolingParams],
        output_type: type[_O],
        *,
        use_tqdm: bool | Callable[..., tqdm] = True,
        lora_request: Sequence[LoRARequest] | LoRARequest | None = None,
        chat_template: str | None = None,
        chat_template_content_format: ChatTemplateContentFormatOption = "auto",
        add_generation_prompt: bool = True,
        continue_final_message: bool = False,
        tools: list[dict[str, Any]] | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
        tokenization_kwargs: dict[str, Any] | None = None,
        mm_processor_kwargs: dict[str, Any] | None = None,
    ):
        self._add_chat_requests(
            messages=messages,
            params=params,
            use_tqdm=use_tqdm,
            lora_request=lora_request,
            chat_template=chat_template,
            chat_template_content_format=chat_template_content_format,
            chat_template_kwargs=chat_template_kwargs,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=continue_final_message,
            tools=tools,
            tokenization_kwargs=tokenization_kwargs,
            mm_processor_kwargs=mm_processor_kwargs,
        )
        return self._run_engine(output_type=output_type, use_tqdm=use_tqdm)

    def _add_chat_requests(
        self,
        messages: list[ChatCompletionMessageParam]
        | Sequence[list[ChatCompletionMessageParam]],
        params: SamplingParams
        | PoolingParams
        | Sequence[SamplingParams | PoolingParams],
        *,
        use_tqdm: bool | Callable[..., tqdm] = True,
        lora_request: Sequence[LoRARequest] | LoRARequest | None = None,
        priority: list[int] | None = None,
        chat_template: str | None = None,
        chat_template_content_format: ChatTemplateContentFormatOption = "auto",
        add_generation_prompt: bool = True,
        continue_final_message: bool = False,
        tools: list[dict[str, Any]] | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
        tokenization_kwargs: dict[str, Any] | None = None,
        mm_processor_kwargs: dict[str, Any] | None = None,
    ) -> list[str]:
        seq_convs = conversation_to_seq(messages)
        seq_params = self._params_to_seq(params, len(seq_convs))
        seq_lora_requests = self._lora_request_to_seq(lora_request, len(seq_convs))
        seq_priority = self._priority_to_seq(priority, len(seq_convs))

        # When thinking is enabled or tools are provided, and the model
        # uses special tokens for structured output (e.g. Gemma4's
        # <|channel>, <|tool_call>, <|"|>), automatically set
        # skip_special_tokens=False so these tokens are preserved in
        # output.text for downstream parsing.
        needs_parsing = (
            chat_template_kwargs and chat_template_kwargs.get("enable_thinking")
        ) or tools
        if needs_parsing:
            self._adjust_params_for_parsing(seq_params)

        return self._render_and_add_requests(
            prompts=(
                self._preprocess_chat_one(
                    conversation,
                    chat_template=chat_template,
                    chat_template_content_format=chat_template_content_format,
                    chat_template_kwargs=chat_template_kwargs,
                    add_generation_prompt=add_generation_prompt,
                    continue_final_message=continue_final_message,
                    tools=tools,
                    tokenization_kwargs=tokenization_kwargs,
                    mm_processor_kwargs=mm_processor_kwargs,
                )
                for conversation in maybe_tqdm(
                    seq_convs,
                    use_tqdm=use_tqdm,
                    desc="Rendering conversations",
                )
            ),
            params=seq_params,
            lora_requests=seq_lora_requests,
            priorities=seq_priority,
        )

    def _adjust_params_for_parsing(
        self, params: Sequence[SamplingParams | PoolingParams]
    ) -> None:
        """Set ``skip_special_tokens=False`` when the model encodes
        structured output syntax as special tokens.

        Models like Gemma4 register thinking delimiters
        (``<|channel>``/``<channel|>``) and tool call tokens
        (``<|tool_call>``/``<tool_call|>``/``<|"|>``) as special tokens.
        The default ``skip_special_tokens=True`` strips them from
        ``output.text``, breaking parsing of both reasoning blocks and
        tool calls.

        This is a no-op for models whose structured tokens are regular
        text tokens (e.g. DeepSeek's ``<think>``/``</think>``).
        """
        # The offline API currently lacks a unified rendering pipeline.
        # Until the planned Renderer refactor is complete, we hardcode
        # this token preservation logic specifically for Gemma4 models
        # to avoid regressions on other models.
        hf_config = getattr(self.model_config, "hf_config", None)
        architectures = getattr(hf_config, "architectures", [])

        if any("Gemma4" in arch for arch in architectures):
            tokenizer = self.renderer.get_tokenizer()
            vocab = tokenizer.get_vocab()
            special_ids = set(getattr(tokenizer, "all_special_ids", []))

            # Tokens used for thinking delimiters and tool call syntax
            # that some models (Gemma4) register as special tokens.
            structured_tokens = (
                "<|channel>",
                "<channel|>",  # thinking delimiters
                "<|tool_call>",
                "<tool_call|>",  # tool call delimiters
                '<|"|>',  # string quoting in tool args
            )
            needs_special = any(
                vocab.get(tok) in special_ids
                for tok in structured_tokens
                if tok in vocab
            )
            if needs_special:
                for sp in params:
                    if isinstance(sp, SamplingParams) and sp.skip_special_tokens:
                        sp.skip_special_tokens = False

    def _render_and_run_requests(
        self,
        prompts: Iterable[EngineInput],
        params: Sequence[SamplingParams | PoolingParams],
        output_type: type[_O],
        *,
        lora_requests: Sequence[LoRARequest | None] | None = None,
        priorities: Sequence[int] | None = None,
        use_tqdm: bool | Callable[..., tqdm] = True,
    ):
        if isinstance(prompts, (list, tuple)):
            logger.warning_once(
                "Rendering all prompts before adding them to the engine "
                "is less efficient than performing both on the same prompt "
                "before processing the next prompt. You should instead pass "
                "a generator that renders one prompt per iteration, as that allows "
                "engine execution to begin for the first prompt while processing "
                "the next prompt."
            )

        self._render_and_add_requests(
            prompts=prompts,
            params=params,
            lora_requests=lora_requests,
            priorities=priorities,
        )

        return self._run_engine(output_type, use_tqdm=use_tqdm)

    def _render_and_add_requests(
        self,
        prompts: Iterable[EngineInput],
        params: Sequence[SamplingParams | PoolingParams],
        *,
        lora_requests: Sequence[LoRARequest | None] | None = None,
        priorities: Sequence[int] | None = None,
    ) -> list[str]:
        added_request_ids: list[str] = []

        try:
            for i, prompt in enumerate(prompts):
                request_id = self._add_request(
                    prompt,
                    params[i],
                    lora_request=self._resolve_mm_lora(
                        prompt,
                        None if lora_requests is None else lora_requests[i],
                    ),
                    priority=0 if priorities is None else priorities[i],
                )
                added_request_ids.append(request_id)
        except Exception as e:
            if added_request_ids:
                self.llm_engine.abort_request(added_request_ids, internal=True)
            raise e

        return added_request_ids

    def _add_request(
        self,
        prompt: EngineInput,
        params: SamplingParams | PoolingParams,
        lora_request: LoRARequest | None = None,
        priority: int = 0,
    ) -> str:
        if isinstance(params, SamplingParams):
            # We only care about the final output
            params.output_kind = RequestOutputKind.FINAL_ONLY

        request_id = str(next(self.request_counter))

        return self.llm_engine.add_request(
            request_id,
            prompt,
            params,
            lora_request=lora_request,
            priority=priority,
        )

    def _run_engine(
        self,
        output_type: type[_O] | tuple[type[_O], ...],
        *,
        use_tqdm: bool | Callable[..., tqdm] = True,
    ) -> list[_O]:
        # Initialize tqdm.
        if use_tqdm:
            num_requests = self.llm_engine.get_num_unfinished_requests()
            tqdm_func = use_tqdm if callable(use_tqdm) else tqdm
            pbar = tqdm_func(
                total=num_requests,
                desc="Processed prompts",
                dynamic_ncols=True,
                postfix=(f"est. speed input: {0:.2f} toks/s, output: {0:.2f} toks/s"),
            )

        # Run the engine.
        outputs: list[_O] = []
        total_in_toks = 0
        total_out_toks = 0
        while self.llm_engine.has_unfinished_requests():
            step_outputs = self.llm_engine.step()
            for output in step_outputs:
                assert isinstance(output, output_type)
                if output.finished:
                    outputs.append(output)  # type: ignore[arg-type]
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

        if use_tqdm:
            pbar.close()
        # Sort the outputs by request ID.
        # This is necessary because some requests may be finished earlier than
        # its previous requests.
        return sorted(outputs, key=lambda x: int(x.request_id))
