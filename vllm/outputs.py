# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# [CN] 文件总览：面向用户的输出数据结构（RequestOutput 家族）
# [CN] 职责：定义生成 / 池化 / 嵌入 / 分类 / 打分五类请求的返回体。
# [CN] 链路：engine -> output_processor -> RequestOutput -> 用户 / OpenAI 服务层
# [CN] 三个层次不要混：
# [CN]   CompletionOutput      —— 一条候选序列（n>1 时一个请求会有多条，靠 index 区分）
# [CN]   RequestOutput         —— 一次请求的结果快照，里面装 n 个 CompletionOutput
# [CN]   PoolingRequestOutput  —— 池化类请求（embed/classify/score）的容器，泛型参数决定具体形态
# [CN] 易错点：这些类是「快照」而非累加器。流式情况下每步产出的是一个新 RequestOutput，
# [CN]         由调用方用 add() 决定是否合并；直接改属性很容易把新旧状态搅在一起。

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
from vllm.v1.metrics.stats import RequestSpecDecodeMetrics, RequestStateStats

logger = init_logger(__name__)


# [CN] 采样时的候选集记录：每个生成位置记下通过了 top-k/top-p/min-p 过滤的 token id 全集。
# [CN] 它不是「最终分布」，而是「过滤后的支撑集」 —— 用于体外约束（如结构化输出）的事后校验。

@dataclass
class SamplingMask:
    """Per-token sampling support sets aligned with completion token IDs.

    Each inner list contains the vocabulary token IDs that survived
    top-k / top-p / min-p filtering for the corresponding generated token.
    """

    # [CN] 外层按生成位置对齐 `completion.token_ids`，内层是该位置的候选 token id。

    token_ids: list[list[int]]


# [CN] 一条候选序列的完整结果。注意它由 dataclass 定义 —— 字段多且需要 repr/eq，
# [CN] 但不参与任何引擎内部逻辑，纯粹是跨进程边界给用户看的形状。

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
        sampling_mask: The post-processing token support set for each generated
            token, if requested.
        finish_reason: The reason why the sequence is finished.
        stop_reason: The stop string or token id that caused the completion
            to stop, None if the completion finished for some other reason
            including encountering the EOS token.
        lora_request: The LoRA request that was used to generate the output.
        spec_decode_metrics: Per-sequence speculative-decoding acceptance metrics,
            populated on finish when speculative decoding ran and
            ``--per-request-spec-decode-metrics`` is enabled; None otherwise.
            Surfaced in the response as ``metrics.speculative_decoding`` for
            single-sequence (``n == 1``) requests.
    """

    index: int
    # [CN] 已 detokenize 的文本。增量模式下这里可能是「本步新增的那段」而非全文，
    # [CN] 取决于调用方用 aggregate 还是 replace 合并。

    text: str
    # [CN] 用 Sequence 而非 list：允许底层给只读视图。想追加时得先转 list（见 RequestOutput.add）。

    token_ids: GenericSequence[int]
    # [CN] 累积对数概率。进行 n>1 采样时每条候选各自累积，是所有束线的比较依据。

    cumulative_logprob: float | None
    # [CN] 每个位置取 top-k 个 token 的对数概率；未请求时为 None（不是空列表）。

    logprobs: SampleLogprobs | None
    # [CN] 形状 [seq_len, layer_num, topk]，记录每层路由到了哪些专家 —— 用于 MoE 负载分析。

    routed_experts: np.ndarray | None = None  # [seq_len,layer_num,topk]
    # [CN] 结束原因：'stop'(遇到止词/EOS)、'length'(到上限)、'abort'。None 表示未结束。

    finish_reason: str | None = None
    # [CN] 具体触发停止的东西：可能是 stop string(s), 也可能是 stop token id。
    # [CN] 因为两者都可能，类型才写成 int | str；其它原因结束时为 None。

    stop_reason: int | str | None = None
    # [CN] 记录生成这条结果时用的是哪个 LoRA 适配器，便于多适配器混跑时归因。

    lora_request: LoRARequest | None = None
    sampling_mask: SamplingMask | None = None
    # [CN] 投机解码的接受率统计。只有开启 --per-request-spec-decode-metrics 且真的跑了
    # [CN] 投机解码才会填，否则为 None —— 相应的 SQL 字段也不要假设一定存在。

    spec_decode_metrics: RequestSpecDecodeMetrics | None = None

    # [CN] 用 finish_reason 是否为 None 来判结束 —— 而不是去看有没有 stop_reason。

    def finished(self) -> bool:
        return self.finish_reason is not None

    def __repr__(self) -> str:
        return (
            f"CompletionOutput(index={self.index}, "
            f"text={self.text!r}, "
            f"token_ids={self.token_ids}, "
            f"routed_experts={self.routed_experts}, "
            f"sampling_mask={self.sampling_mask}, "
            f"cumulative_logprob={self.cumulative_logprob}, "
            f"logprobs={self.logprobs}, "
            f"finish_reason={self.finish_reason}, "
            f"stop_reason={self.stop_reason})"
        )


# [CN] 池化任务的原始产出：直接装 hidden states 张量。
# [CN] 下游会按任务类型把它tolist()成 Embedding / Classification / Scoring 三种形态之一。

@dataclass
class PoolingOutput:
    """The output data of one pooling output of a request.

    Args:
        data: The extracted hidden states.
    """

    # [CN] 注意这里保留 torch.Tensor（不是 numpy/list）：跨进程/跨 stage 时不希望提前落地成 Python 对象。

    data: torch.Tensor

    def __repr__(self) -> str:
        return f"PoolingOutput(data={self.data})"

    # [CN] 张量相等不能直接用 Python 的 == 语义，所以手写：先类型判断再 all() 比较元素。

    def __eq__(self, other: object) -> bool:
        return isinstance(other, self.__class__) and bool(
            (self.data == other.data).all()
        )


# [CN] 一个完成请求的整个快照。刻意不用 dataclass 而手写 __init__：
# [CN] 一是要接受 **kwargs 做前向兼容，二是 kv/ec transfer 参数必须是 keyword-only。

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
        num_cache_creation_tokens: Prompt tokens currently counted as local
            prefix-cache writes for this request.
        kv_transfer_params: The params for remote K/V transfer.
        ec_transfer_params: The params for remote encoder-cache transfer.
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
        num_cache_creation_tokens: int | None = None,
        *,
        kv_transfer_params: dict[str, Any] | None = None,
        ec_transfer_params: dict[str, Any] | None = None,
        # Forward compatibility, code that uses args added in new release can
        # still run with older versions of vLLM without breaking.
        # [CN] 前向兼容口：新版本加的参数在旧版 vLLM 上不会被当成 TypeError，而是被吞掉并告警。

        **kwargs: Any,
    ) -> None:
        # [CN] warning_once 而非 raise：宁可降级也要让「新形态的调用方 + 旧引擎」能跑起来。

        if kwargs:
            logger.warning_once(
                "RequestOutput: Ignoring extra arguments: %s", str(kwargs)
            )
        self.request_id = request_id
        self.prompt = prompt
        self.prompt_token_ids = prompt_token_ids
        self.prompt_logprobs = prompt_logprobs
        self.outputs = outputs
        self.finished = finished
        self.metrics = metrics
        self.lora_request = lora_request
        self.encoder_prompt = encoder_prompt
        self.encoder_prompt_token_ids = encoder_prompt_token_ids
        self.num_cached_tokens = num_cached_tokens
        self.num_cache_creation_tokens = num_cache_creation_tokens
        self.kv_transfer_params = kv_transfer_params
        self.ec_transfer_params = ec_transfer_params

    # [CN] 流式合并的关键方法。aggregate 决定两种语义：
    # [CN]   True  —— 把同一条候选的新token追加进来（形成越来越长的文本）
    # [CN]   False —— 直接用新的替换旧的（每步只关心最新状态）

    def add(self, next_output: "RequestOutput", aggregate: bool) -> None:
        """Merge subsequent RequestOutput into this one"""

        # [CN] 用 |= 而不是赋值：一旦某步标记结束，后续任何一步都不能把它改回未完成。

        self.finished |= next_output.finished
        self.kv_transfer_params = next_output.kv_transfer_params
        self.ec_transfer_params = next_output.ec_transfer_params

        for next_completion in next_output.outputs:
            for i, completion in enumerate(self.outputs):
                if completion.index == next_completion.index:
                    # [CN] 按 index 找同条候选：n>1 时outputs列表的顺序不保证，只能靠 index 配对。

                    if aggregate:
                        # [CN] 累积模式：文本/logprobs 直接往后追加，而 cumulative_logprob 取较新值
                        # [CN] （它是累积量，新值已经包含旧值）。

                        # Merge outputs with same index
                        completion.text += next_completion.text
                        if not isinstance(completion.token_ids, MutableSequence):
                            completion.token_ids = list(completion.token_ids)
                        completion.token_ids.extend(next_completion.token_ids)
                        if next_completion.logprobs:
                            assert completion.logprobs is not None
                            completion.logprobs.extend(next_completion.logprobs)  # type: ignore[arg-type]
                        completion.cumulative_logprob = (
                            next_completion.cumulative_logprob
                        )
                        completion.finish_reason = next_completion.finish_reason
                        completion.stop_reason = next_completion.stop_reason
                    # [CN] 替换模式：整条候选换掉。delta 流式接口常用这种。

                    else:
                        # Replace the output with the new one
                        self.outputs[i] = next_completion
                    break
            # [CN] for 的自然出口：没找到同 index 的候选说明这是一条新的（n>1 首次到达）。

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
            f"num_cache_creation_tokens={self.num_cache_creation_tokens})"
        )


# [CN] 流结束哨兵。用一个「空但 finished=True」的 RequestOutput 表示结束，
# [CN] 这样消费方不需要额外的协议来判断「这是不是最后一块」。

# Sentinel to indicate request is finished, used with streaming inputs.
STREAM_FINISHED = RequestOutput(
    request_id="",
    prompt=None,
    prompt_token_ids=None,
    prompt_logprobs=None,
    outputs=[],
    finished=True,
)

# [CN] 带 default 的 TypeVar：未参数化时默认当作 PoolingOutput，省掉到处写泛型的负担。

_O = TypeVar("_O", default=PoolingOutput)


# [CN] 池化类输出的公共容器。具体形态由泛型参数实例化：
# [CN] EmbeddingRequestOutput / ClassificationRequestOutput / ScoringRequestOutput。

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

    # [CN] 池化类的构造：与生成类不同，它不支持 **kwargs 前向兼容，因为字段集很少变动。

    def __init__(
        self,
        request_id: str,
        outputs: _O,
        prompt_token_ids: list[int],
        num_cached_tokens: int,
        finished: bool,
    ):
        self.request_id = request_id
        self.prompt_token_ids = prompt_token_ids
        self.num_cached_tokens = num_cached_tokens
        self.finished = finished
        self.outputs = outputs

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(request_id={self.request_id!r}, "
            f"outputs={self.outputs!r}, "
            f"prompt_token_ids={self.prompt_token_ids}, "
            f"num_cached_tokens={self.num_cached_tokens}, "
            f"finished={self.finished})"
        )


# [CN] 向量形态：一个 request -> 一组 float。

@dataclass
class EmbeddingOutput:
    """The output data of one embedding output of a request.

    Args:
        embedding: The embedding vector, which is a list of floats.
            Its length depends on the hidden dimension of the model.
    """

    # [CN] 长度等于模型 hidden size，与 pooler 配置有关。

    embedding: list[float]

    @staticmethod
    # [CN] 统一适配器：把原始的 PoolingOutput 按任务类型投影成各自的形态。
    # [CN] 走 staticmethod 是因为转换不依赖任何实例状态。

    def from_base(pooling_output: PoolingOutput) -> "EmbeddingOutput":
        pooled_data = pooling_output.data
        if pooled_data.ndim != 1:
            raise ValueError("pooled_data should be a 1-D embedding vector")

        return EmbeddingOutput(pooled_data.tolist())

    @property
    # [CN] 从 embedding 长度反推 hidden size —— 避免额外存一份容易不一致的字段。

    def hidden_size(self) -> int:
        return len(self.embedding)

    def __repr__(self) -> str:
        return f"EmbeddingOutput(hidden_size={self.hidden_size})"


# [CN] 嵌入请求结果：outputs 是单个 EmbeddingOutput（注意不是列表）。

class EmbeddingRequestOutput(PoolingRequestOutput[EmbeddingOutput]):
    @staticmethod
    def from_base(
        request_output: PoolingRequestOutput,
    ) -> "EmbeddingRequestOutput":
        return EmbeddingRequestOutput(
            request_id=request_output.request_id,
            outputs=EmbeddingOutput.from_base(request_output.outputs),
            prompt_token_ids=request_output.prompt_token_ids,
            num_cached_tokens=request_output.num_cached_tokens,
            finished=request_output.finished,
        )


# [CN] 分类形态：probability 向量，长度等于类别数。

@dataclass
class ClassificationOutput:
    """The output data of one classification output of a request.

    Args:
        probs: The probability vector, which is a list of floats.
            Its length depends on the number of classes.
    """

    # [CN] 注意这里存的是「已经池化出来的向量」，是否做过 softmax 取决于模型 pooler 的定义。

    probs: list[float]

    @staticmethod
    def from_base(pooling_output: PoolingOutput) -> "ClassificationOutput":
        # pooling_output shape: (num_classes)
        pooled_data = pooling_output.data
        if pooled_data.ndim != 1:
            raise ValueError("pooled_data should be a 1-D probability vector")

        return ClassificationOutput(pooled_data.tolist())

    @property
    # [CN] 同上：类别数就是概率向量的长度。

    def num_classes(self) -> int:
        return len(self.probs)

    def __repr__(self) -> str:
        return f"ClassificationOutput(num_classes={self.num_classes})"


# [CN] 分类请求结果：outputs 是单个 ClassificationOutput。

class ClassificationRequestOutput(PoolingRequestOutput[ClassificationOutput]):
    @staticmethod
    def from_base(
        request_output: PoolingRequestOutput,
    ) -> "ClassificationRequestOutput":
        return ClassificationRequestOutput(
            request_id=request_output.request_id,
            outputs=ClassificationOutput.from_base(request_output.outputs),
            prompt_token_ids=request_output.prompt_token_ids,
            num_cached_tokens=request_output.num_cached_tokens,
            finished=request_output.finished,
        )


# [CN] 打分形态：单个标量（相似度或回归分数）。

@dataclass
class ScoringOutput:
    """The output data of one scoring output of a request.

    Args:
        score: The similarity score, which is a scalar value.
    """

    # [CN] squeeze() 过再取 item：兼容 classify(num_classes==1) 与 embed 两种输入的形状差异。

    score: float

    @staticmethod
    def from_base(pooling_output: PoolingOutput) -> "ScoringOutput":
        # pooling_output shape:
        #   classify task: (num_classes) num_classes == 1
        #   embed task: a scalar value
        pooled_data = pooling_output.data.squeeze()
        if pooled_data.ndim != 0:
            raise ValueError("pooled_data should be a scalar score")

        return ScoringOutput(pooled_data.item())

    def __repr__(self) -> str:
        return f"ScoringOutput(score={self.score})"


# [CN] 打分请求结果：outputs 是单个 ScoringOutput。三个类结构同构，
# [CN] 差别只在 from_base 里怎么把张量解释成 Python 值。

class ScoringRequestOutput(PoolingRequestOutput[ScoringOutput]):
    @staticmethod
    def from_base(
        request_output: PoolingRequestOutput,
    ) -> "ScoringRequestOutput":
        return ScoringRequestOutput(
            request_id=request_output.request_id,
            outputs=ScoringOutput.from_base(request_output.outputs),
            prompt_token_ids=request_output.prompt_token_ids,
            num_cached_tokens=request_output.num_cached_tokens,
            finished=request_output.finished,
        )
