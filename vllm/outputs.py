# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import MutableSequence
from collections.abc import Sequence as GenericSequence
from dataclasses import dataclass
from typing import Any, Generic

import numpy as np
import torch
from typing_extensions import TypeVar

from vllm.logger import init_logger
from vllm.logprobs import PromptLogprobs, SampleLogprobs
from vllm.lora.request import LoRARequest
from vllm.multimodal.inputs import MultiModalPlaceholderDict
from vllm.v1.metrics.stats import RequestStateStats

logger = init_logger(__name__)


@dataclass
class CompletionOutput:
    """The output data of one completion output of a request.

    Args:
        index: The index of the output in the request.
        text: The generated output text.
        token_ids: The token IDs of the generated output text.
        cumulative_logprob: The cumulative log probability of the generated
            output text.
        logprobs: The log probabilities of the top probability words at each
            position if the logprobs are requested.
        finish_reason: The reason why the sequence is finished.
        stop_reason: The stop string or token id that caused the completion
            to stop, None if the completion finished for some other reason
            including encountering the EOS token.
        lora_request: The LoRA request that was used to generate the output.
    """

    index: int
    text: str
    token_ids: GenericSequence[int]
    cumulative_logprob: float | None
    logprobs: SampleLogprobs | None
    routed_experts: np.ndarray | None = None  # [seq_len,layer_num,topk]
    finish_reason: str | None = None
    stop_reason: int | str | None = None
    lora_request: LoRARequest | None = None
    num_preempted: int = 0

    def finished(self) -> bool:
        return self.finish_reason is not None

    def __repr__(self) -> str:
        return (
            f"CompletionOutput(index={self.index}, "
            f"text={self.text!r}, "
            f"token_ids={self.token_ids}, "
            f"routed_experts={self.routed_experts}, "
            f"cumulative_logprob={self.cumulative_logprob}, "
            f"logprobs={self.logprobs}, "
            f"finish_reason={self.finish_reason}, "
            f"stop_reason={self.stop_reason}, "
            f"num_preempted={self.num_preempted})"
        )


@dataclass
class PoolingOutput:
    """The output data of one pooling output of a request.

    Args:
        data: The extracted hidden states.
    """

    data: torch.Tensor

    def __repr__(self) -> str:
        return f"PoolingOutput(data={self.data})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, self.__class__) and bool(
            (self.data == other.data).all()
        )


class RequestOutput:
    """The output data of a completion request to the LLM.

    Args:
        request_id: The unique ID of the request.
        prompt: The prompt string of the request.
                For encoder/decoder models, this is the
                decoder input prompt.
        prompt_token_ids: The token IDs of the prompt.
                          For encoder/decoder models, this is the
                          decoder input prompt token ids.
        prompt_logprobs: The log probabilities to return per prompt token.
        outputs: The output sequences of the request.
        finished: Whether the whole request is finished.
        metrics: Metrics associated with the request.
        lora_request: The LoRA request that was used to generate the output.
        encoder_prompt: The encoder prompt string of the request.
                        None if decoder-only.
        encoder_prompt_token_ids: The token IDs of the encoder prompt.
                                  None if decoder-only.
        num_cached_tokens: The number of tokens with prefix cache hit.
        kv_transfer_params: The params for remote K/V transfer.
    """

    def __init__(
        self,
        request_id: str,
        prompt: str | None,
        prompt_token_ids: list[int] | None,
        prompt_logprobs: PromptLogprobs | None,
        outputs: list[CompletionOutput],
        finished: bool,
        metrics: RequestStateStats | None = None,
        lora_request: LoRARequest | None = None,
        encoder_prompt: str | None = None,
        encoder_prompt_token_ids: list[int] | None = None,
        num_cached_tokens: int | None = None,
        num_preempted: int = 0,
        *,
        multi_modal_placeholders: MultiModalPlaceholderDict | None = None,
        kv_transfer_params: dict[str, Any] | None = None,
        # Forward compatibility, code that uses args added in new release can
        # still run with older versions of vLLM without breaking.
        **kwargs: Any,
    ) -> None:
        if kwargs:
            logger.warning_once(
                "RequestOutput: Ignoring extra arguments: %s", str(kwargs)
            )
        self.request_id = request_id
        self.prompt = prompt
        self.prompt_token_ids = prompt_token_ids
        self.multi_modal_placeholders = multi_modal_placeholders or {}
        self.prompt_logprobs = prompt_logprobs
        self.outputs = outputs
        self.finished = finished
        self.metrics = metrics
        self.lora_request = lora_request
        self.encoder_prompt = encoder_prompt
        self.encoder_prompt_token_ids = encoder_prompt_token_ids
        self.num_cached_tokens = num_cached_tokens
        self.num_preempted = num_preempted
        self.kv_transfer_params = kv_transfer_params

    def add(self, next_output: "RequestOutput", aggregate: bool) -> None:
        """Merge subsequent RequestOutput into this one"""

        self.finished |= next_output.finished
        self.kv_transfer_params = next_output.kv_transfer_params

        for next_completion in next_output.outputs:
            for i, completion in enumerate(self.outputs):
                if completion.index == next_completion.index:
                    if aggregate:
                        # Merge outputs with same index
                        completion.text += next_completion.text
                        if not isinstance(completion.token_ids, MutableSequence):
                            completion.token_ids = list(completion.token_ids)
                        completion.token_ids.extend(next_completion.token_ids)
                        if next_completion.logprobs:
                            assert completion.logprobs is not None
                            completion.logprobs.extend(next_completion.logprobs)
                        completion.cumulative_logprob = (
                            next_completion.cumulative_logprob
                        )
                        completion.finish_reason = next_completion.finish_reason
                        completion.stop_reason = next_completion.stop_reason
                    else:
                        # Replace the output with the new one
                        self.outputs[i] = next_completion
                    break
            else:
                self.outputs.append(next_completion)

    def __repr__(self) -> str:
        return (
            f"RequestOutput(request_id={self.request_id}, "
            f"prompt={self.prompt!r}, "
            f"prompt_token_ids={self.prompt_token_ids}, "
            f"encoder_prompt={self.encoder_prompt!r}, "
            f"encoder_prompt_token_ids={self.encoder_prompt_token_ids}, "
            f"prompt_logprobs={self.prompt_logprobs}, "
            f"outputs={self.outputs}, "
            f"finished={self.finished}, "
            f"metrics={self.metrics}, "
            f"lora_request={self.lora_request}, "
            f"num_cached_tokens={self.num_cached_tokens}, "
            f"multi_modal_placeholders={self.multi_modal_placeholders})"
        )


# Sentinel to indicate request is finished, used with streaming inputs.
STREAM_FINISHED = RequestOutput(
    request_id="",
    prompt=None,
    prompt_token_ids=None,
    prompt_logprobs=None,
    outputs=[],
    finished=True,
)

_O = TypeVar("_O", default=PoolingOutput)


class PoolingRequestOutput(Generic[_O]):
    """
    The output data of a pooling request to the LLM.

    Args:
        request_id (str): A unique identifier for the pooling request.
        outputs (PoolingOutput): The pooling results for the given input.
        prompt_token_ids (list[int]): A list of token IDs used in the prompt.
        num_cached_tokens: The number of tokens with prefix cache hit.
        finished (bool): A flag indicating whether the pooling is completed.
    """

    def __init__(
        self,
        request_id: str,
        outputs: _O,
        prompt_token_ids: list[int],
        num_cached_tokens: int,
        finished: bool,
        num_preempted: int = 0,
    ):
        self.request_id = request_id
        self.prompt_token_ids = prompt_token_ids
        self.num_cached_tokens = num_cached_tokens
        self.finished = finished
        self.outputs = outputs
        self.num_preempted = num_preempted

    def __repr__(self):
        return (
            f"{type(self).__name__}(request_id={self.request_id!r}, "
            f"outputs={self.outputs!r}, "
            f"prompt_token_ids={self.prompt_token_ids}, "
            f"num_cached_tokens={self.num_cached_tokens}, "
            f"finished={self.finished}, "
            f"num_preempted={self.num_preempted})"
        )


@dataclass
class EmbeddingOutput:
    """The output data of one embedding output of a request.

    Args:
        embedding: The embedding vector, which is a list of floats.
            Its length depends on the hidden dimension of the model.
    """

    embedding: list[float]

    @staticmethod
    def from_base(pooling_output: PoolingOutput):
        pooled_data = pooling_output.data
        if pooled_data.ndim != 1:
            raise ValueError("pooled_data should be a 1-D embedding vector")

        return EmbeddingOutput(pooled_data.tolist())

    @property
    def hidden_size(self) -> int:
        return len(self.embedding)

    def __repr__(self) -> str:
        return f"EmbeddingOutput(hidden_size={self.hidden_size})"


class EmbeddingRequestOutput(PoolingRequestOutput[EmbeddingOutput]):
    @staticmethod
    def from_base(request_output: PoolingRequestOutput):
        return EmbeddingRequestOutput(
            request_id=request_output.request_id,
            outputs=EmbeddingOutput.from_base(request_output.outputs),
            prompt_token_ids=request_output.prompt_token_ids,
            num_cached_tokens=request_output.num_cached_tokens,
            finished=request_output.finished,
        )


@dataclass
class ClassificationOutput:
    """The output data of one classification output of a request.

    Args:
        probs: The probability vector, which is a list of floats.
            Its length depends on the number of classes.
    """

    probs: list[float]

    @staticmethod
    def from_base(pooling_output: PoolingOutput):
        # pooling_output shape: (num_classes)
        pooled_data = pooling_output.data
        if pooled_data.ndim != 1:
            raise ValueError("pooled_data should be a 1-D probability vector")

        return ClassificationOutput(pooled_data.tolist())

    @property
    def num_classes(self) -> int:
        return len(self.probs)

    def __repr__(self) -> str:
        return f"ClassificationOutput(num_classes={self.num_classes})"


class ClassificationRequestOutput(PoolingRequestOutput[ClassificationOutput]):
    @staticmethod
    def from_base(request_output: PoolingRequestOutput):
        return ClassificationRequestOutput(
            request_id=request_output.request_id,
            outputs=ClassificationOutput.from_base(request_output.outputs),
            prompt_token_ids=request_output.prompt_token_ids,
            num_cached_tokens=request_output.num_cached_tokens,
            finished=request_output.finished,
        )


@dataclass
class ScoringOutput:
    """The output data of one scoring output of a request.

    Args:
        score: The similarity score, which is a scalar value.
    """

    score: float

    @staticmethod
    def from_base(pooling_output: PoolingOutput):
        # pooling_output shape:
        #   classify task: (num_classes) num_classes == 1
        #   embed task: a scalar value
        pooled_data = pooling_output.data.squeeze()
        if pooled_data.ndim != 0:
            raise ValueError("pooled_data should be a scalar score")

        return ScoringOutput(pooled_data.item())

    def __repr__(self) -> str:
        return f"ScoringOutput(score={self.score})"


class ScoringRequestOutput(PoolingRequestOutput[ScoringOutput]):
    @staticmethod
    def from_base(request_output: PoolingRequestOutput):
        return ScoringRequestOutput(
            request_id=request_output.request_id,
            outputs=ScoringOutput.from_base(request_output.outputs),
            prompt_token_ids=request_output.prompt_token_ids,
            num_cached_tokens=request_output.num_cached_tokens,
            finished=request_output.finished,
        )
