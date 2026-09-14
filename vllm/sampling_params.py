# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sampling parameters for text generation."""

import copy
import json as json_mod
import math
from dataclasses import field
from enum import Enum, IntEnum
from functools import cached_property
from typing import Annotated, Any

import msgspec
from pydantic import BeforeValidator
from pydantic.dataclasses import dataclass

import vllm.envs as envs
from vllm.config import ModelConfig, SpeculativeConfig, StructuredOutputsConfig
from vllm.exceptions import VLLMValidationError
from vllm.logger import init_logger
from vllm.tokenizers import TokenizerLike
from vllm.utils.mistral import is_mistral_tokenizer
from vllm.v1.serial_utils import PydanticMsgspecMixin

logger = init_logger(__name__)

# [CN] 判断"是否等于零"的容差：温度小于它就当作贪心。

_SAMPLING_EPS = 1e-5
# [CN] 温度下限。低于它会在除法里放大到 inf/nan，所以强制抬到这个值。

_MAX_TEMP = 1e-2

# [CN] 指定 token 求 logprob 的数量上限，必须与采样器里预分配的行宽一致，
# [CN] 否则内核会写出界。改动这里要同步改 LogprobTokenIdsState。

MAX_LOGPROB_TOKEN_IDS = 128
"""Upper bound on `SamplingParams.logprob_token_ids` list length. Must match
the per-request row width allocated by the sampler's `LogprobTokenIdsState`."""


# [CN] n 与 beam_width 共用的范围检查，上限由环境变量控制而非硬编码。

def _verify_num_sequences(value: int, parameter_name: str) -> None:
    if not isinstance(value, int):
        raise VLLMValidationError(
            f"{parameter_name} must be an int, but is of type {type(value)}"
        )
    if value < 1:
        raise VLLMValidationError(f"{parameter_name} must be at least 1, got {value}.")
    max_n = envs.VLLM_MAX_N_SEQUENCES
    if value > max_n:
        raise VLLMValidationError(
            f"{parameter_name} must be at most {max_n}, got {value}. "
            "To increase this limit, set the VLLM_MAX_N_SEQUENCES "
            "environment variable."
        )


# [CN] -1 表示不限制，归一成 None 让下游少判断一种取值。
# [CN] bool 是 int 的子类，所以要显式排除 True/False 这类误传。

def validate_thinking_token_budget(value: int | float | bool | None) -> int | None:
    """Validate ``thinking_token_budget``; return ``None`` if unset."""
    if value is None:
        return None
    if isinstance(value, (bool, float)) or not isinstance(value, int):
        raise VLLMValidationError(
            "`thinking_token_budget` must be a non-negative integer "
            "or -1 for unlimited.",
            parameter="thinking_token_budget",
            value=value,
        )
    if value == -1:
        return None
    if value < 0:
        raise VLLMValidationError(
            "`thinking_token_budget` must be a non-negative integer "
            "or -1 for unlimited.",
            parameter="thinking_token_budget",
            value=value,
        )
    return value


ThinkingTokenBudget = Annotated[
    int | None,
    BeforeValidator(validate_thinking_token_budget),
]


# [CN] 采样类型三态。RANDOM_SEED 单独一档是因为它走 CPU 侧生成器，
# [CN] 无法与无种子的随机采样合并成同一条快路径。

class SamplingType(IntEnum):
    # [CN] 贪心：直接 argmax，不需要随机数。

    GREEDY = 0
    # [CN] 随机：用设备侧的共享随机数，速度最快。

    RANDOM = 1
    # [CN] 带种子：必须走每请求独立的 CPU 生成器，无法与上一档合并。

    RANDOM_SEED = 2


# maybe make msgspec?

# [CN] 结构化输出的约束描述：六选一，构造时就强制互斥。
@dataclass
class StructuredOutputsParams:
    # One of these fields will be used to build a logit processor.
    # [CN] 既接受 JSON Schema 字符串，也接受已经解析好的 dict。

    json: str | dict | None = None
    # [CN] 正则会被转成等价文法，因此并非所有语法特性都能支持。

    regex: str | None = None
    # [CN] 选项约束本质是"只这几个串"的文法，空列表没有意义。

    choice: list[str] | None = None
    # [CN] 直接给后端原生文法（如 GBNF），不做转换。

    grammar: str | None = None
    # [CN] 只要"输出是个合法 JSON 对象"，不约束具体 schema。

    json_object: bool | None = None
    # [CN] 这一组是"修饰项"：它们不改变约束种类，只调整约束的严格程度。

    # These are other options that can be set.
    # [CN] 关掉任意空白后，模型必须给出紧凑 JSON，能提速也更容易失败。

    disable_any_whitespace: bool = False
    # [CN] 对应 JSON Schema 的 additionalProperties: false，由后端翻译成文法。

    disable_additional_properties: bool = False
    # [CN] 自定义空白的定义，用于与特定后端的词表对齐。

    whitespace_pattern: str | None = None
    # [CN] 结构标签走独立的注入通道，因此"是否无约束"要单独判一次。

    structural_tag: str | None = None

    # [CN] init=False 表示不参与构造入参，只能由引擎侧回填，避免用户乱选后端。

    _backend: str | None = field(default=None, init=False)
    """CAUTION: Should only be set by Processor._validate_structured_output"""
    # [CN] 记住后端是自动挑的还是指定的，避免复用参数时误判冲突。

    _backend_was_auto: bool = field(default=False, init=False)
    """CAUTION: Should only be set by Processor._validate_structured_output"""

    # [CN] 结构化输出：构造时就强制"恰好一个约束"。

    def __post_init__(self):
        """Validate that some fields are mutually exclusive."""
        count = sum(
            [
                self.json is not None,
                self.regex is not None,
                self.choice is not None,
                self.grammar is not None,
                self.json_object is not None,
                self.structural_tag is not None,
            ]
        )
        # [CN] 多选一不是"与"的关系，同时给多个约束没有明确语义。

        if count > 1:
            raise VLLMValidationError(
                "You can only use one kind of structured outputs constraint "
                f"but multiple are specified: {self.__dict__}"
            )
        # [CN] 一个约束都不给等于没有启用，直接报错比静默忽略安全。

        if count < 1:
            raise VLLMValidationError(
                "You must use one kind of structured outputs constraint "
                f"but none are specified: {self.__dict__}"
            )

    # [CN] 判断整个对象是否等价于"没约束"，用于跳过结构化输出的处理分支。

    def all_constraints_none(self) -> bool:
        """
        Returns True if all structured-output constraint fields are None.
        """
        return all(
            getattr(self, field) is None
            for field in (
                "json",
                "regex",
                "choice",
                "grammar",
                "json_object",
                "structural_tag",
            )
        )

    # [CN] 与上一个的差别在于忽略 structural_tag：它走的是另一条注入通道。

    def all_non_structural_tag_constraints_none(self) -> bool:
        """
        Returns True if all structured-output constraint fields are None.
        """
        return all(
            getattr(self, field) is None
            for field in (
                "json",
                "regex",
                "choice",
                "grammar",
                "json_object",
            )
        )


# [CN] 重复 N-gram 检测：专门用来掐掉 'abcdabcdabcd...' 这类死循环输出。
@dataclass
class RepetitionDetectionParams:
    """Parameters for detecting repetitive N-gram patterns in output tokens."""

    # [CN] 0 表示关闭整个检测；它与 min_count 必须成对配置。

    max_pattern_size: int = 0
    """Maximum size of N-gram pattern to detect for sequence repetition.
    Set to 0 to disable. Must be used together with min_count."""

    # [CN] 为 0 时内部按 1 处理；真正生效的开关是 max_pattern_size。

    min_pattern_size: int = 0
    """Minimum N-gram pattern size to check for sequence repetition.
    If set to 0, it defaults to 1.
    Must be <= max_pattern_size."""

    min_count: int = 0
    """Minimum number of times an N-gram pattern must repeat to trigger
    detection. Must be >= 2. Example: 3 for detecting a phrase repeated
    3 times. Must be used together with max_pattern_size."""

    # [CN] 重复检测参数：把非法组合挡在构造期，而不是等到生成时才发现。

    def __post_init__(self):
        if (
            self.max_pattern_size < 0
            or self.min_pattern_size < 0
            or self.min_pattern_size > self.max_pattern_size
        ):
            raise VLLMValidationError(
                "max_pattern_size, min_pattern_size must be >=0, "
                "with min_pattern_size <= max_pattern_size. "
                "Set both to 0 to disable repetitive pattern detection."
            )
        # [CN] 至少要重复两次才谈得上"是重复模式"，1 次没有意义。

        if self.max_pattern_size > 0 and self.min_count < 2:
            raise VLLMValidationError(
                "min_count must be >= 2 to detect repetitive patterns "
                "in engine output. If you do not wish to detect repetitive "
                "patterns, set max_pattern_size to 0."
            )


# [CN] 输出形态：累计全量、增量、还是只给最终结果。

class RequestOutputKind(Enum):
    # Return entire output so far in every RequestOutput
    # [CN] 默认形态：每次都带完整文本，实现简单但重复传输。

    CUMULATIVE = 0
    # Return only deltas in each RequestOutput
    # [CN] 只回增量，省带宽但调用方要自己拼。

    DELTA = 1
    # Do not return intermediate RequestOutput
    # [CN] 完全不回中间结果，n>1 时想一次性拿全得用它。

    FINAL_ONLY = 2


# [CN] 老版 Mistral 分词器（非 tekken）在部分语法后端上不被支持。

def _is_non_tekken_mistral(tokenizer: TokenizerLike) -> bool:
    return is_mistral_tokenizer(tokenizer) and not tokenizer.is_tekken


# [CN] 只有 Mistral 分词器才带 llg_tokenizer，其余返回 None 交给后端默认处理。

def _get_llg_tokenizer(tokenizer: TokenizerLike) -> Any:
    return tokenizer.llg_tokenizer if is_mistral_tokenizer(tokenizer) else None


# [CN] 请求级采样参数。同时是 msgspec.Struct 与 pydantic dataclass：
# [CN] msgspec 负责进程间高速序列化，pydantic 负责入参校验。
# [CN] omit_defaults=True 让默认值不上线，显著降低传输体积。

class SamplingParams(
    PydanticMsgspecMixin,
    msgspec.Struct,
    omit_defaults=True,  # type: ignore[call-arg]
    # required for @cached_property.
    dict=True,
):  # type: ignore[call-arg]
    """Sampling parameters for text generation.

    Overall, we follow the sampling parameters from the OpenAI text completion
    API (https://platform.openai.com/docs/api-reference/completions/create).
    In addition, we support beam search, which is not supported by OpenAI.
    """

    # [CN] 一个 prompt 出几条结果；上限由 VLLM_MAX_N_SEQUENCES 控制。

    n: int = 1
    """Number of outputs to return for the given prompt request.

    The maximum allowed value is controlled by the ``VLLM_MAX_N_SEQUENCES``
    environment variable (default: 16384).

    NOTE:
        `AsyncLLM` streams outputs by default. When `n > 1`, all `n` outputs
        are generated and streamed cumulatively per request. To see all `n`
        outputs upon completion, use `output_kind=RequestOutputKind.FINAL_ONLY`
        in `SamplingParams`."""
    # [CN] 存在惩罚只看"出现过没有"，与出现次数无关，区间 [-2, 2]。

    presence_penalty: float = 0.0
    """Penalizes new tokens based on whether they appear in the generated text
    so far. Values > 0 encourage the model to use new tokens, while values < 0
    encourage the model to repeat tokens."""
    # [CN] 频率惩罚按出现次数累加，与 presence 的区别就在这点。

    frequency_penalty: float = 0.0
    """Penalizes new tokens based on their frequency in the generated text so
    far. Values > 0 encourage the model to use new tokens, while values < 0
    encourage the model to repeat tokens."""
    # [CN] 注意基准值是 1.0 而不是 0.0：它是除数，>1 惩罚、<1 鼓励重复。

    repetition_penalty: float = 1.0
    """Penalizes new tokens based on whether they appear in the prompt and the
    generated text so far. Values > 1 encourage the model to use new tokens,
    while values < 1 encourage the model to repeat tokens."""
    # [CN] 0 即贪心。过小的非零值会在 softmax 里溢出，所以有下限保护。

    temperature: float = 1.0
    """Controls the randomness of the sampling. Lower values make the model
    more deterministic, while higher values make the model more random. Zero
    means greedy sampling."""
    # [CN] 核采样的累积概率阈值，必须在 (0, 1]。

    top_p: float = 1.0
    """Controls the cumulative probability of the top tokens to consider. Must
    be in (0, 1]. Set to 1 to consider all tokens."""
    # [CN] 0 与 -1 都表示不限制；-1 只是为了兼容 OpenAI 客户端的写法。

    top_k: int = 0
    """Controls the number of top tokens to consider. Set to 0 (or -1) to
    consider all tokens."""
    # [CN] 相对阈值的动态截断：只保留概率 >= 最大概率 × min_p 的 token。

    min_p: float = 0.0
    """Represents the minimum probability for a token to be considered,
    relative to the probability of the most likely token. Must be in [0, 1].
    Set to 0 to disable this."""
    # [CN] 有种子就走可复现路径，代价是无法与其他请求共享随机数。

    seed: int | None = None
    """Random seed to use for the generation."""
    # [CN] 字符串与列表都收，构造时统一成列表。

    stop: str | list[str] | None = None
    """String(s) that stop the generation when they are generated. The returned
    output will not contain the stop strings."""
    # [CN] 停止 token 会留在输出里（除非是特殊 token），与 stop 字符串不同。

    stop_token_ids: list[int] | None = None
    """Token IDs that stop the generation when they are generated. The returned
    output will contain the stop tokens unless the stop tokens are special
    tokens."""
    # [CN] 生成到 EOS 也不停，用于需要强制凑满长度的场景。

    ignore_eos: bool = False
    """Whether to ignore the EOS token and continue generating
    tokens after the EOS token is generated."""
    # [CN] 默认 16 是历史遗留的保守值；None 表示不限（由模型上限兜底）。

    max_tokens: int | None = 16
    """Maximum number of tokens to generate per output sequence."""
    # [CN] 在达到这个长度前，EOS 与 stop_token_ids 的 logits 会被屏蔽。

    min_tokens: int = 0
    """Minimum number of tokens to generate per output sequence before EOS or
    `stop_token_ids` can be generated"""
    # [CN] 按 OpenAI 语义，返回的是 top-N 加被采样 token，所以可能是 N+1 个。
    # [CN] -1 表示返回整个词表，代价是显存与带宽都按词表大小增长。

    logprobs: int | None = None
    """Number of log probabilities to return per output token. When set to
    `None`, no probability is returned. If set to a non-`None` value, the
    result includes the log probabilities of the specified number of most
    likely tokens, as well as the chosen tokens. Note that the implementation
    follows the OpenAI API: The API will always return the log probability of
    the sampled token, so there may be up to `logprobs+1` elements in the
    response. When set to -1, return all `vocab_size` log probabilities."""
    # [CN] 请求 prompt 的 logprobs 会强制跳过前缀缓存，否则算出来的长度对不上。

    prompt_logprobs: int | None = None
    """Number of log probabilities to return per prompt token.
    When set to -1, return all `vocab_size` log probabilities."""
    # [CN] 只要少数几个 token 的 logprob 时用这个，比 logprobs=-1 便宜得多。

    logprob_token_ids: list[int] | None = None
    """Specific token IDs to return logprobs for. More efficient than
    logprobs=-1 when you only need logprobs for a small set of tokens.
    When set, logprobs for exactly these token IDs will be returned,
    in addition to the sampled token. This is useful for scoring tasks
    where you want to compare probabilities of specific label tokens."""
    # [CN] 扁平格式的 GC 压力远小于 list[dict]，长输出场景收益明显。

    flat_logprobs: bool = False
    """Whether to return logprobs in flatten format (i.e. FlatLogprob)
    for better performance.
    NOTE: GC costs of FlatLogprobs is significantly smaller than
    list[dict[int, Logprob]]. After enabled, PromptLogprobs and
    SampleLogprobs would populated as FlatLogprobs."""
    # NOTE: This parameter is only exposed at the engine level for now.
    # It is not exposed in the OpenAI API server, as the OpenAI API does
    # not support returning only a list of token IDs.
    # [CN] 关掉后只回 token id；注意此时 stop 字符串也无法使用。

    detokenize: bool = True
    """Whether to detokenize the output."""
    # [CN] 默认跳过特殊 token，做评测时可能要显式关掉。

    skip_special_tokens: bool = True
    """Whether to skip special tokens in the output."""
    # [CN] 影响特殊 token 之间的拼接方式，关掉会得到更原始的文本。

    spaces_between_special_tokens: bool = True
    """Whether to add spaces between special tokens in the output."""
    # [CN] 打开后 stop 字符串会出现在输出文本里，输出缓冲逻辑也随之变化。

    include_stop_str_in_output: bool = False
    """Whether to include the stop strings in output text."""
    # [CN] 唯一没有文档字符串的字段：它决定 output_processor 的组装方式。

    output_kind: RequestOutputKind = RequestOutputKind.CUMULATIVE
    # [CN] 只能把间隔调大不能调小：低于引擎配置的值会被向上截断。

    stream_interval: int | None = None
    """Number of newly generated tokens to batch into each streamed
    `RequestOutput`. Raises the interval above the engine-level
    `--stream-interval`. Values below engine setting are clamped up to it.
    The first and final outputs are always emitted immediately."""
    # [CN] 深拷贝 SamplingParams 在热路径上有成本，专用对象可以跳过。

    skip_clone: bool = False
    """Internal flag indicating that this SamplingParams instance is safe to
    reuse without cloning. When True, clone() will return self without
    performing a deep copy. This should only be set when the params object
    is guaranteed to be dedicated to a single request and won't be modified
    in ways that would affect other uses."""

    # [CN] 下划线开头 + 运行时回填：用户传了也会被覆盖，属于引擎内部状态。

    # The below fields are not supposed to be used as an input.
    # They are set in post_init.
    # [CN] 为判断 stop 字符串而扣留的字符数：最长 stop 串减一。

    output_text_buffer_length: int = 0
    # [CN] 由引擎在 update_from_generation_config 里回填，用户不应直接设置。

    _eos_token_id: int | None = None
    # [CN] 用集合去重，最终喂给采样器做一次性屏蔽。

    _all_stop_token_ids: set[int] = msgspec.field(default_factory=set)

    # [CN] 以下字段不会直接进内核，而是被用来构造对应的 logits processor。

    # Fields used to construct logits processors
    # [CN] 结构化输出参数由引擎侧进一步校验并回填 _backend。

    structured_outputs: StructuredOutputsParams | None = None
    """Parameters for configuring structured outputs."""
    # [CN] 键可以是字符串形式的 token id（兼容 OpenAI），构造时会转成 int。

    logit_bias: dict[int, float] | None = None
    """If provided, the engine will construct a logits processor that applies
    these logit biases."""
    # [CN] 白名单以外的 token 会被屏蔽；空列表是非法值（等于屏蔽一切）。

    allowed_token_ids: list[int] | None = None
    """If provided, the engine will construct a logits processor which only
    retains scores for the given token ids."""
    # [CN] 给插件与自定义采样逻辑用的逃生口，树内实现完全不读它。

    extra_args: dict[str, Any] | None = None
    """Arbitrary additional args, that can be used by custom sampling
    implementations, plugins, etc. Not used by any in-tree sampling
    implementations."""
    # [CN] 屏蔽词需要分词后才能用，所以存的是原始字符串。

    # Fields used for bad words
    # [CN] 屏蔽的是"能凑成这个词的最后一个 token"，而不是整段序列。

    bad_words: list[str] | None = None
    """Words that are not allowed to be generated. More precisely, only the
    last token of a corresponding token sequence is not allowed when the next
    generated token can complete the sequence."""
    _bad_words_token_ids: list[list[int]] | None = None

    # [CN] 三态：None 时由 __post_init__ 根据 prompt_logprobs 推断。

    skip_reading_prefix_cache: bool | None = None
    # [CN] 思考预算，-1 归一成 None 表示不限。

    thinking_token_budget: int | None = None
    """Maximum number of tokens allowed for thinking operations."""

    # [CN] 命中重复模式就提前结束，避免白烧 token 到 max_tokens。

    repetition_detection: RepetitionDetectionParams | None = None
    """Parameters for detecting repetitive N-gram patterns in output tokens.
    If such repetition is detected, generation will be ended early. LLMs can
    sometimes generate repetitive, unhelpful token patterns, stopping only
    when they hit the maximum output length (e.g. 'abcdabcdabcd...' or
    '\\emoji \\emoji \\emoji ...'). This feature can detect such behavior
    and terminate early, saving time and tokens."""

    # [CN] 这两个字段只服务于调试与 RL 训练，线上不应依赖其语义。

    # Debugging / RL-specific parameters. Not intended for production serving.
    # [CN] 多轮 agent 场景下跳过已返回过的路由数据，避免重复传输。

    routed_experts_prompt_start: int = 0
    """When enable_return_routed_experts is active, skip the first
    routed_experts_prompt_start prompt tokens from the returned routing
    data. In multi-turn agent scenarios, set this to the length of the
    already-returned prefix to avoid duplicating routing for prompt tokens
    covered by earlier turns. Default 0 returns routing for all prompt
    tokens."""
    # [CN] 强制回放固定序列但依然算真实 logprobs，主要用于 RL 与调试。

    trace_decode_token_ids: list[int] | None = None
    """If provided, forces the engine to emit this predetermined sequence of
    token IDs during decoding instead of sampling randomly. Real logprobs are
    still computed. Conflict checking is performed at the engine level."""

    # [CN] 接受全 None 的便利构造器：None 一律落到默认构造函数上。
    @staticmethod
    def from_optional(
        n: int | None = 1,
        presence_penalty: float | None = 0.0,
        frequency_penalty: float | None = 0.0,
        repetition_penalty: float | None = 1.0,
        temperature: float | None = 1.0,
        top_p: float | None = 1.0,
        top_k: int = 0,
        min_p: float = 0.0,
        seed: int | None = None,
        stop: str | list[str] | None = None,
        stop_token_ids: list[int] | None = None,
        bad_words: list[str] | None = None,
        thinking_token_budget: int | None = None,
        include_stop_str_in_output: bool = False,
        ignore_eos: bool = False,
        max_tokens: int | None = 16,
        min_tokens: int = 0,
        logprobs: int | None = None,
        prompt_logprobs: int | None = None,
        detokenize: bool = True,
        skip_special_tokens: bool = True,
        spaces_between_special_tokens: bool = True,
        output_kind: RequestOutputKind = RequestOutputKind.CUMULATIVE,
        stream_interval: int | None = None,
        structured_outputs: StructuredOutputsParams | None = None,
        logit_bias: dict[int, float] | dict[str, float] | None = None,
        allowed_token_ids: list[int] | None = None,
        extra_args: dict[str, Any] | None = None,
        skip_clone: bool = False,
        repetition_detection: RepetitionDetectionParams | None = None,
        logprob_token_ids: list[int] | None = None,
        routed_experts_prompt_start: int = 0,
        # Debugging / RL-specific parameters.
        trace_decode_token_ids: list[int] | None = None,
    ) -> "SamplingParams":
        # [CN] 先走一次快速的字典推导，失败才逐个条目重试以定位出错的键。
        # [CN] 偏置值被夹到 [-100, 100]，与 OpenAI 的行为保持一致。

        if logit_bias is not None:
            # Fast path uses a dict comprehension; on failure we iterate once
            # to identify the exact offending entry for the error message.
            try:
                logit_bias = {
                    int(token): min(100.0, max(-100.0, bias))
                    for token, bias in logit_bias.items()
                }
            except (ValueError, TypeError):
                invalid_keys = []
                converted_logit_bias = {}
                for token, bias in logit_bias.items():
                    try:
                        token_id = int(token)
                    except (ValueError, TypeError):
                        invalid_keys.append(token)
                        continue
                    converted_logit_bias[token_id] = min(100.0, max(-100.0, bias))
                # [CN] 只有定位到具体出错的键才报错，否则静默使用修好的字典。

                if invalid_keys:
                    raise VLLMValidationError(
                        f"logit_bias contains key(s) that cannot be "
                        f"converted to integer token IDs: {invalid_keys!r}",
                        parameter="logit_bias",
                        value=invalid_keys,
                    ) from None
                # [CN] 慢路径的产物重新赋回，后续类型就是 dict[int, float]。

                logit_bias = converted_logit_bias

        return SamplingParams(
            n=1 if n is None else n,
            presence_penalty=0.0 if presence_penalty is None else presence_penalty,
            frequency_penalty=0.0 if frequency_penalty is None else frequency_penalty,
            repetition_penalty=1.0
            if repetition_penalty is None
            else repetition_penalty,
            temperature=1.0 if temperature is None else temperature,
            top_p=1.0 if top_p is None else top_p,
            top_k=top_k,
            min_p=min_p,
            seed=seed,
            stop=stop,
            stop_token_ids=stop_token_ids,
            bad_words=bad_words,
            thinking_token_budget=thinking_token_budget,
            include_stop_str_in_output=include_stop_str_in_output,
            ignore_eos=ignore_eos,
            max_tokens=max_tokens,
            min_tokens=min_tokens,
            logprobs=logprobs,
            prompt_logprobs=prompt_logprobs,
            logprob_token_ids=logprob_token_ids,
            detokenize=detokenize,
            skip_special_tokens=skip_special_tokens,
            spaces_between_special_tokens=spaces_between_special_tokens,
            output_kind=output_kind,
            stream_interval=stream_interval,
            structured_outputs=structured_outputs,
            logit_bias=logit_bias,
            allowed_token_ids=allowed_token_ids,
            extra_args=extra_args,
            skip_clone=skip_clone,
            repetition_detection=repetition_detection,
            routed_experts_prompt_start=routed_experts_prompt_start,
            trace_decode_token_ids=trace_decode_token_ids,
        )

    # [CN] 归一化入口：把用户友好的输入统一成引擎内部稳定的表示。

    def __post_init__(self) -> None:
        # [CN] 只警告不报错：静默抬到下限比让用户在张量里收到 nan 友好。

        if 0 < self.temperature < _MAX_TEMP:
            logger.warning(
                "temperature %s is less than %s, which may cause numerical "
                "errors nan or inf in tensors. We have maxed it out to %s.",
                self.temperature,
                _MAX_TEMP,
                _MAX_TEMP,
            )
            self.temperature = max(self.temperature, _MAX_TEMP)

        # [CN] -1 是 OpenAI 客户端表示"不设种子"的惯例值。

        if self.seed == -1:
            self.seed = None

        self.thinking_token_budget = validate_thinking_token_budget(
            self.thinking_token_budget
        )

        # [CN] 统一成空列表：下游可以避免反复判断 None。

        if self.stop is None:
            self.stop = []
        elif isinstance(self.stop, str):
            self.stop = [self.stop]

        # [CN] 去重保序：dict.fromkeys 既去重又保留原顺序。

        if self.stop_token_ids is None:
            self.stop_token_ids = []
        else:
            self.stop_token_ids = list(dict.fromkeys(self.stop_token_ids))

        # [CN] 同样归一化成列表，屏蔽词为空是常见情况。

        if self.bad_words is None:
            self.bad_words = []
        else:
            self.bad_words = list(dict.fromkeys(self.bad_words))

        # [CN] 兼容 logprobs=True 这种布尔写法，转成数值 1。

        if self.logprobs is True:
            self.logprobs = 1

        # [CN] 与 logprobs 一样兼容布尔写法。

        if self.prompt_logprobs is True:
            self.prompt_logprobs = 1

        # Number of characters to hold back for stop string evaluation
        # until sequence is finished.
        # [CN] 需要扣留尾部字符，因为 stop 串可能横跨多步流式输出。

        if self.stop and not self.include_stop_str_in_output:
            self.output_text_buffer_length = max(len(s) for s in self.stop) - 1

        # [CN] 归一化之后才校验：先修形状再验范围，避免误报。

        self._verify_args()

        # [CN] 贪心时把 top_p/top_k/min_p 全部复位：留着会与 argmax 语义打架。

        if self.temperature < _SAMPLING_EPS:
            # Zero temperature means greedy sampling.
            self.top_p = 1.0
            self.top_k = 0
            self.min_p = 0.0
            self._verify_greedy_sampling()

        # eos_token_id is added to this by the engine
        # [CN] 真正的 EOS 由引擎后续再加进来，这里只并用户指定的。

        self._all_stop_token_ids.update(self.stop_token_ids)

        # [CN] prompt_logprobs 要求完整的 prompt 前向，命中缓存会导致结果变短。

        if self.skip_reading_prefix_cache is None:
            # If prefix caching is enabled,
            # the output of prompt logprobs may less than n_prompt_tokens,
            # we need to skip reading cache at this request.
            self.skip_reading_prefix_cache = self.prompt_logprobs is not None

    # [CN] 纯范围校验，不依赖模型配置；依赖词表的检查在 verify() 里做。

    def _verify_args(self) -> None:
        _verify_num_sequences(self.n, "n")
        # [CN] 区间与 OpenAI 对齐：超出后惩罚会压过原始 logits 的量级。

        if not -2.0 <= self.presence_penalty <= 2.0:
            raise VLLMValidationError(
                f"presence_penalty must be in [-2, 2], got {self.presence_penalty}."
            )
        # [CN] 与 presence 同区间，两者叠加时总惩罚上限是 4。

        if not -2.0 <= self.frequency_penalty <= 2.0:
            raise VLLMValidationError(
                f"frequency_penalty must be in [-2, 2], got {self.frequency_penalty}."
            )
        # [CN] 只要求有限：它是除数，非零且有限即可，上下界不限制。

        if not math.isfinite(self.repetition_penalty):
            raise VLLMValidationError(
                "repetition_penalty must be a finite number, "
                f"got {self.repetition_penalty}."
            )
        # [CN] 为零会除零、为负会反转 logits 符号，两种都必须拒绝。

        if self.repetition_penalty <= 0.0:
            raise VLLMValidationError(
                "repetition_penalty must be greater than zero, got "
                f"{self.repetition_penalty}."
            )
        # [CN] nan 温度会静默污染整个 batch，必须在入口就拦住。

        if not math.isfinite(self.temperature):
            raise VLLMValidationError(
                f"temperature must be a finite number, got {self.temperature}.",
                parameter="temperature",
                value=self.temperature,
            )
        # [CN] 负温度在数学上没有意义，会被解释成反向的 softmax。

        if self.temperature < 0.0:
            raise VLLMValidationError(
                f"temperature must be non-negative, got {self.temperature}.",
                parameter="temperature",
                value=self.temperature,
            )
        # [CN] 上界 2 是经验值，再高基本等价于均匀采样。

        if self.temperature > 2.0:
            raise VLLMValidationError(
                f"temperature must be in [0, 2], got {self.temperature}.",
                parameter="temperature",
                value=self.temperature,
            )
        # [CN] top_p=0 等于什么都不保留，所以开区间。

        if not 0.0 < self.top_p <= 1.0:
            raise VLLMValidationError(
                f"top_p must be in (0, 1], got {self.top_p}.",
                parameter="top_p",
                value=self.top_p,
            )
        # quietly accept -1 as disabled, but prefer 0
        # [CN] -1 与 0 都当作关闭，只有更小的负数才是真错误。

        if self.top_k < -1:
            raise VLLMValidationError(
                f"top_k must be 0 (disable), or at least 1, got {self.top_k}."
            )
        # [CN] float 的 top_k 会在切片时静默截断，宁可显式报错。

        if not isinstance(self.top_k, int):
            raise VLLMValidationError(
                f"top_k must be an integer, got {type(self.top_k).__name__}"
            )
        # [CN] min_p 是比例，天然闭合在 [0, 1]。

        if not 0.0 <= self.min_p <= 1.0:
            raise VLLMValidationError(f"min_p must be in [0, 1], got {self.min_p}.")
        # [CN] 0 个 token 的输出没有意义，而且会让停止条件无从触发。

        if self.max_tokens is not None and self.max_tokens < 1:
            raise VLLMValidationError(
                f"max_tokens must be at least 1, got {self.max_tokens}.",
                parameter="max_tokens",
                value=self.max_tokens,
            )
        # [CN] 负数没有语义：最少生成 0 个 token 等于不限制。

        if self.min_tokens < 0:
            raise VLLMValidationError(
                f"min_tokens must be greater than or equal to 0, got {self.min_tokens}."
            )
        # [CN] 下限超过上限会导致永远无法停止，必须在构造期就判掉。

        if self.max_tokens is not None and self.min_tokens > self.max_tokens:
            raise VLLMValidationError(
                f"min_tokens must be less than or equal to "
                f"max_tokens={self.max_tokens}, got {self.min_tokens}."
            )
        # [CN] 间隔 0 会造成每步都发包的退化行为。

        if self.stream_interval is not None and self.stream_interval < 1:
            raise VLLMValidationError(
                f"stream_interval must be at least 1, got {self.stream_interval}.",
                parameter="stream_interval",
                value=self.stream_interval,
            )
        # [CN] -1 是"全词表"的哨兵值，其余负数一律非法。

        if self.logprobs is not None and self.logprobs != -1 and self.logprobs < 0:
            raise VLLMValidationError(
                f"logprobs must be non-negative or -1, got {self.logprobs}.",
                parameter="logprobs",
                value=self.logprobs,
            )
        if (
            self.prompt_logprobs is not None
            and self.prompt_logprobs != -1
            and self.prompt_logprobs < 0
        ):
            raise VLLMValidationError(
                f"prompt_logprobs must be non-negative or -1, got "
                f"{self.prompt_logprobs}.",
                parameter="prompt_logprobs",
                value=self.prompt_logprobs,
            )
        # [CN] __post_init__ 已归一成列表，这里断言是为了挡住绕过构造器的改法。

        assert isinstance(self.stop_token_ids, list)
        # [CN] 非整数 id 会在比较时静默失配，导致请求永远不停。

        if not all(isinstance(st_id, int) for st_id in self.stop_token_ids):
            raise VLLMValidationError(
                f"stop_token_ids must contain only integers, got {self.stop_token_ids}."
            )
        assert isinstance(self.stop, list)
        # [CN] 空字符串会匹配到任何位置，等于立即停止。

        if any(not stop_str for stop_str in self.stop):
            raise VLLMValidationError("stop cannot contain an empty string.")
        # [CN] 不做 detokenize 就没有文本可比，stop 字符串无从判断。

        if self.stop and not self.detokenize:
            raise VLLMValidationError(
                "stop strings are only supported when detokenize is True. "
                "Set detokenize=True to use stop."
            )
        assert isinstance(self.bad_words, list)
        # [CN] 空串会匹配到任意位置，等价于禁掉整个词表。

        if any(not bad_word for bad_word in self.bad_words):
            raise VLLMValidationError(
                f"bad_words cannot contain an empty string. "
                f"Got bad_words={self.bad_words}"
            )

    # [CN] 贪心只有唯一解，n>1 没有意义。

    def _verify_greedy_sampling(self) -> None:
        # [CN] 贪心只有唯一结果，n>1 只会得到 n 份完全相同的输出。

        if self.n > 1:
            raise VLLMValidationError(
                f"n must be 1 when using greedy sampling, got {self.n}."
            )

    # [CN] 用模型自带的 generation_config 补全 EOS：不同模型可能配多个 EOS。

    def update_from_generation_config(
        self,
        generation_config: dict[str, Any],
        eos_token_id: int | None = None,
    ) -> None:
        """Update if there are non-default values from generation_config"""
        # [CN] 只有不忽略 EOS 时才记录主 EOS；ignore_eos 下它不参与停止判断。
        # [CN] 模型自带的 generation_config 可能配多个 EOS，其余的并进 stop_token_ids。

        if not self.ignore_eos:
            self._eos_token_id = eos_token_id

        if eos_token_id is not None:
            # Add the eos token id into the sampling_params to support
            # min_tokens processing.
            self._all_stop_token_ids.add(eos_token_id)

        # Update eos_token_id for generation
        if (eos_ids := generation_config.get("eos_token_id")) is not None:
            # it can be either int or list of int
            eos_ids = {eos_ids} if isinstance(eos_ids, int) else set(eos_ids)
            if eos_token_id is not None:
                # We don't need to include the primary eos_token_id in
                # stop_token_ids since it's handled separately for stopping
                # purposes.
                eos_ids.discard(eos_token_id)
            if eos_ids:
                self._all_stop_token_ids.update(eos_ids)
                if not self.ignore_eos:
                    assert self.stop_token_ids is not None
                    eos_ids.update(self.stop_token_ids)
                    self.stop_token_ids = list(eos_ids)

    # [CN] bad_words 必须在这里分词：参数是字符串，屏蔽发生在 token 维度。

    def update_from_tokenizer(self, tokenizer: TokenizerLike) -> None:
        # [CN] 没有屏蔽词时不做分词，省掉一次昂贵的 tokenizer 调用。

        if not self.bad_words:
            return
        self._bad_words_token_ids = []
        # [CN] 屏蔽词会被展开成多组 token 序列，数量必须设上限防止 FSM 爆炸。

        max_num_bad_words = envs.VLLM_MAX_NUM_BAD_WORDS
        for bad_word in self.bad_words:
            # To prohibit words both at the beginning
            # and in the middle of text
            # (related to add_prefix_space tokenizer parameter)
            # [CN] 词首与词中都可能出现目标词，两种分词结果都要屏蔽。

            for add_prefix_space in [False, True]:
                prefix = " " if add_prefix_space else ""
                prompt = prefix + bad_word.lstrip()
                prompt_token_ids = tokenizer.encode(
                    text=prompt, add_special_tokens=False
                )

                if not prompt_token_ids:
                    if not add_prefix_space:
                        raise VLLMValidationError(
                            "bad_words entries must tokenize to at least one token.",
                            parameter="bad_words",
                            value=self.bad_words,
                        )
                    # The unprefixed form is still enforceable when only the
                    # optional space-prefixed form tokenizes to nothing.
                    continue

                # If no space at the beginning
                # or if prefix space produces a new word token
                if (not add_prefix_space) or (
                    add_prefix_space
                    and prompt_token_ids[0] != self._bad_words_token_ids[-1][0]
                    and len(prompt_token_ids) == len(self._bad_words_token_ids[-1])
                ):
                    self._bad_words_token_ids.append(prompt_token_ids)
                    if len(self._bad_words_token_ids) > max_num_bad_words:
                        raise VLLMValidationError(
                            f"Too many bad words after tokenization: "
                            f"{len(self._bad_words_token_ids)}. "
                            f"The max number is {max_num_bad_words}.",
                            parameter="bad_words",
                            value=self.bad_words,
                        )

        # [CN] 越界的 token id 会让屏蔽落空，必须在分词后立刻查出。

        invalid_token_ids = [
            token_id
            for bad_words_token_ids in self._bad_words_token_ids
            for token_id in bad_words_token_ids
            if token_id < 0 or token_id > tokenizer.max_token_id
        ]
        if len(invalid_token_ids) > 0:
            raise VLLMValidationError(
                f"The model vocabulary size is {tokenizer.max_token_id + 1},"
                f" but the following tokens"
                f" were specified as bad: {invalid_token_ids}."
                f" All token id values should be integers satisfying:"
                f" 0 <= token_id <= {tokenizer.max_token_id}.",
                parameter="bad_words",
                value=self.bad_words,
            )

    # [CN] 缓存属性：采样类型是调度与快路径选择的高频判据。
    @cached_property
    def sampling_type(self) -> SamplingType:
        # [CN] 这里是采样器侧的判定，与 __post_init__ 里的归一化用的是同一个阈值。

        if self.temperature < _SAMPLING_EPS:
            # [CN] 贪心档优先级最高：温度近似为零时其他参数都已被复位。

            return SamplingType.GREEDY
        # [CN] 有种子就升级成 RANDOM_SEED，优先级高于普通随机。

        if self.seed is not None:
            return SamplingType.RANDOM_SEED
        # [CN] 默认档：无种子、非零温度，走共享随机数的快路径。

        return SamplingType.RANDOM

    @property
    def eos_token_id(self) -> int | None:
        return self._eos_token_id

    # [CN] 只读暴露：外部不能改这个集合，否则会绕过引擎的停止逻辑。
    @property
    def all_stop_token_ids(self) -> set[int]:
        return self._all_stop_token_ids

    # [CN] 内部专用：分词结果依赖具体 tokenizer，不保证跨版本稳定。
    @property
    def bad_words_token_ids(self) -> list[list[int]] | None:
        # For internal use only. Backward compatibility not guaranteed
        return self._bad_words_token_ids

    # [CN] 统一对外口径：只设了 logprob_token_ids 时，按它的长度算份数。
    @property
    def num_logprobs(self) -> int | None:
        """Number of sample logprobs to return per output token, or `None` if
        no sample logprobs were requested. Takes `logprob_token_ids` into
        account: when `logprobs` is unset but `logprob_token_ids` is set,
        returns `len(logprob_token_ids)`."""
        if self.logprobs is not None:
            return self.logprobs
        return len(self.logprob_token_ids) if self.logprob_token_ids else None

    # [CN] 深拷贝是默认行为：参数对象常被多个请求共享，改一处会串味。

    def clone(self) -> "SamplingParams":
        """If skip_clone is True, uses shallow copy instead of deep copy."""
        # [CN] 浅拷贝就够了的前提是调用方保证不再修改这个对象。

        if self.skip_clone:
            return copy.copy(self)

        # [CN] 默认深拷贝：cached_property 与集合字段都不能被共享。

        return copy.deepcopy(self)

    # [CN] 需要模型配置的二次校验集中在这里，按依赖顺序逐个调用。

    def verify(
        self,
        model_config: ModelConfig,
        speculative_config: SpeculativeConfig | None,
        structured_outputs_config: StructuredOutputsConfig | None,
        tokenizer: TokenizerLike | None,
    ) -> None:
        self._validate_logprobs(model_config)
        self._validate_logit_bias(model_config)
        self._validate_trace_replay(model_config, speculative_config)
        self._validate_stop_token_ids(model_config)
        self._validate_logits_processors(model_config)
        self._validate_allowed_token_ids(model_config)
        self._validate_spec_decode(speculative_config)
        self._validate_diffusion(model_config)
        self._validate_structured_outputs(
            model_config, structured_outputs_config, tokenizer
        )

    # [CN] logprobs 的上限来自模型配置，-1 要展开成完整词表再比较。

    def _validate_logprobs(self, model_config: ModelConfig) -> None:
        max_logprobs = model_config.max_logprobs
        # [CN] 引擎侧关闭限制时，上限就是词表大小。

        if max_logprobs == -1:
            max_logprobs = model_config.get_vocab_size()

        # Validate sample logprobs.
        # [CN] 用海象运算符：logprobs 为 0 或 None 时直接跳过整段校验。

        if num_logprobs := self.logprobs:
            # [CN] -1 展开成完整词表后再与上限比较。

            if num_logprobs == -1:
                num_logprobs = model_config.get_vocab_size()
            # [CN] 超过上限的请求直接拒绝，防止显存被 logprobs 缓冲打爆。

            if num_logprobs > max_logprobs:
                raise VLLMValidationError(
                    f"Requested sample logprobs of {num_logprobs}, "
                    f"which is greater than max allowed: {max_logprobs}",
                    parameter="logprobs",
                    value=num_logprobs,
                )

        # Validate logprob_token_ids.
        # [CN] 指定 token 与 logprobs 同时出现时，两者数量必须相等，否则语义不明。

        if self.logprob_token_ids is not None:
            n = len(self.logprob_token_ids)
            if n > MAX_LOGPROB_TOKEN_IDS:
                raise VLLMValidationError(
                    f"Requested logprob_token_ids of length {n}, "
                    f"which is greater than max allowed: {MAX_LOGPROB_TOKEN_IDS}",
                    parameter="logprob_token_ids",
                    value=n,
                )
            vocab_size = model_config.get_vocab_size()
            invalid_token_ids = [
                token_id
                for token_id in self.logprob_token_ids
                if token_id < 0 or token_id >= vocab_size
            ]
            if invalid_token_ids:
                raise VLLMValidationError(
                    f"token_id(s) {invalid_token_ids} in logprob_token_ids "
                    f"contain out-of-vocab token ids. Vocabulary size: "
                    f"{vocab_size}",
                    parameter="logprob_token_ids",
                    value=invalid_token_ids,
                )
            if self.logprobs is not None and self.logprobs != n:
                raise VLLMValidationError(
                    f"When both logprobs and logprob_token_ids are set, "
                    f"logprobs must equal len(logprob_token_ids). Got "
                    f"logprobs={self.logprobs}, len(logprob_token_ids)={n}.",
                    parameter="logprob_token_ids",
                    value=n,
                )

        # Validate prompt logprobs.
        if num_prompt_logprobs := self.prompt_logprobs:
            # [CN] prompt 的 -1 同样展开成完整词表，代价是显存按 prompt 长度放大。

            if num_prompt_logprobs == -1:
                num_prompt_logprobs = model_config.get_vocab_size()
            # [CN] prompt logprobs 的缓冲比采样 logprobs 更大，因此更依赖这个上限。

            if num_prompt_logprobs > max_logprobs:
                raise VLLMValidationError(
                    f"Requested prompt logprobs of {num_prompt_logprobs}, "
                    f"which is greater than max allowed: {max_logprobs}",
                    parameter="prompt_logprobs",
                    value=num_prompt_logprobs,
                )

    # [CN] 停止 token 会被当作 logits 的列下标，越界就是越界访问。

    def _validate_stop_token_ids(self, model_config: ModelConfig) -> None:
        """Validate stop_token_ids are within vocabulary range."""
        # [CN] 空集合直接返回：越界检查对空列表没有意义。

        if not self.stop_token_ids:
            return

        # stop_token_ids are used as column indices into the logits tensor,
        # whose width is the model's vocab size (LogitsProcessor is built from
        # config.vocab_size, InputBatch.vocab_size comes from
        # model_config.get_vocab_size()), so use the same bound here — like
        # _validate_logit_bias, which indexes the same tensor.
        vocab_size = model_config.get_vocab_size()
        invalid_token_ids = [
            token_id
            for token_id in self.stop_token_ids
            if token_id < 0 or token_id >= vocab_size
        ]

        if invalid_token_ids:
            raise VLLMValidationError(
                f"token_id(s) {invalid_token_ids} in stop_token_ids contain "
                f"out-of-vocab token ids. Vocabulary size: {vocab_size}",
                parameter="stop_token_ids",
                value=invalid_token_ids,
            )

    # [CN] 与 stop_token_ids 同构：本质都是对 logits 张量按列索引。

    def _validate_logit_bias(self, model_config: ModelConfig) -> None:
        """Validate logit_bias token IDs are within vocabulary range."""
        # [CN] 空字典等价于没设置，直接返回省掉一次遍历。

        if not self.logit_bias:
            return

        vocab_size = model_config.get_vocab_size()
        invalid_token_ids = [
            token_id
            for token_id in self.logit_bias
            if token_id < 0 or token_id >= vocab_size
        ]

        if invalid_token_ids:
            raise VLLMValidationError(
                f"token_id(s) {invalid_token_ids} in logit_bias contain "
                f"out-of-vocab token ids. Vocabulary size: {vocab_size}",
                parameter="logit_bias",
                value=invalid_token_ids,
            )

    # [CN] trace 回放与几乎所有"改变分布"的特性互斥，因为序列是预先定死的。

    def _validate_trace_replay(
        self,
        model_config: ModelConfig,
        speculative_config: SpeculativeConfig | None,
    ) -> None:
        """Validate trace replay request compatibility."""
        if self.trace_decode_token_ids is None:
            return

        # [CN] 空列表与 None 语义不同：None 是没开，空是开了但没内容。

        if len(self.trace_decode_token_ids) == 0:
            raise ValueError("trace_decode_token_ids must be a non-empty list.")
        # [CN] 回放固定序列只能产生一条结果，n>1 无意义。

        if self.n != 1:
            raise ValueError("trace_decode_token_ids requires n=1.")
        # [CN] 布尔是 int 的子类，这里只挡负数与非整数，True 会被当成 1。

        if not all(isinstance(t, int) and t >= 0 for t in self.trace_decode_token_ids):
            raise ValueError(
                "trace_decode_token_ids must contain non-negative integers."
            )

        # [CN] 回放场景下 prompt 也要跑前向，与 prompt_logprobs 的语义冲突。

        if self.prompt_logprobs is not None:
            raise ValueError(
                "trace_decode_token_ids is not supported with prompt_logprobs."
            )
        if speculative_config is not None:
            raise ValueError(
                "trace_decode_token_ids is not supported with speculative decoding."
            )
        # [CN] 回放的 token 序列未必满足文法，两者不能同时使用。

        if self.structured_outputs is not None:
            raise ValueError(
                "trace_decode_token_ids is not supported with structured outputs."
            )
        # [CN] 重复检测可能提前终止，与"必须生成完固定序列"的要求冲突。

        if self.repetition_detection is not None:
            raise ValueError(
                "trace_decode_token_ids is not supported with repetition_detection."
            )
        # [CN] 思考预算也会改变停止时机，同样与回放互斥。

        if self.thinking_token_budget is not None:
            raise ValueError(
                "trace_decode_token_ids is not supported with thinking_token_budget."
            )
        if self.bad_words:
            raise ValueError("trace_decode_token_ids is not supported with bad_words.")

        vocab_size = model_config.get_vocab_size()
        invalid_token_ids = [
            token_id
            for token_id in self.trace_decode_token_ids
            if token_id < 0 or token_id >= vocab_size
        ]
        if invalid_token_ids:
            raise VLLMValidationError(
                f"token_id(s) {invalid_token_ids} in trace_decode_token_ids "
                f"contain out-of-vocab token ids. Vocabulary size: {vocab_size}",
                parameter="trace_decode_token_ids",
                value=invalid_token_ids,
            )

    # [CN] 延迟导入：避免顶层引入采样器实现造成循环依赖。

    def _validate_logits_processors(self, model_config: ModelConfig) -> None:
        from vllm.v1.sample.logits_processor import (
            validate_logits_processors_parameters,
        )

        validate_logits_processors_parameters(model_config.logits_processors, self)

    # [CN] 空列表要单独拦：它会把整行 logits 屏蔽成 -inf。

    def _validate_allowed_token_ids(self, model_config: ModelConfig) -> None:
        allowed_token_ids = self.allowed_token_ids
        if allowed_token_ids is None:
            return

        # [CN] 空白名单会把整行 logits 屏蔽成 -inf，是典型的配置错误。

        if len(allowed_token_ids) == 0:
            raise VLLMValidationError(
                "allowed_token_ids is not None and empty!",
                parameter="allowed_token_ids",
                value=allowed_token_ids,
            )

        # allowed_token_ids are client-supplied ids used as column indices
        # into the logits tensor (the mask in InputBatch is sized by
        # model_config.get_vocab_size(), and the ids are written as
        # mask[req_index][allowed_token_ids]), so use the same bound here —
        # like _validate_stop_token_ids and _validate_logit_bias, which
        # index the same tensor.
        vocab_size = model_config.get_vocab_size()
        invalid_token_ids = [
            token_id
            for token_id in allowed_token_ids
            if token_id < 0 or token_id >= vocab_size
        ]
        if invalid_token_ids:
            raise VLLMValidationError(
                "allowed_token_ids contains out-of-vocab token id!",
                parameter="allowed_token_ids",
                value=invalid_token_ids,
            )

    # [CN] min_p 与 logit_bias 会改变草稿模型的分布，当前验证逻辑不支持。

    def _validate_spec_decode(
        self,
        speculative_config: SpeculativeConfig | None,
    ) -> None:
        # [CN] 没开投机解码就没有兼容性约束。

        if speculative_config is None:
            return

        # Some sampling parameters are not yet compatible with spec decoding.
        # [CN] 这两项会改变目标分布，当前的草稿验证实现还没覆盖。

        if self.min_p > _SAMPLING_EPS or self.logit_bias:
            raise VLLMValidationError(
                "The min_p and logit_bias sampling parameters "
                "are not yet supported with speculative decoding."
            )

    # [CN] 扩散语言模型整体去噪，没有逐 token 的采样自由度。

    def _validate_diffusion(self, model_config: ModelConfig) -> None:
        # [CN] 非扩散模型直接放行，扩散模型才需要逐项拒绝。

        if not model_config.is_diffusion:
            return

        # Diffusion models denoise a whole canvas per step with a fixed
        # temperature schedule, so per-request sampling parameters are not
        # supported. Penalties are ignored by the sampler with a warning.
        if (
            self.temperature != 1.0
            or self.min_p > _SAMPLING_EPS
            or self.seed is not None
            or self.min_tokens > 0
            or self.logit_bias
            or self.bad_words
            or self.allowed_token_ids
        ):
            raise VLLMValidationError(
                "The temperature, min_p, seed, min_tokens, logit_bias, "
                "bad_words, and allowed_token_ids sampling parameters "
                "are not yet supported with diffusion models."
            )

    # [CN] 结构化输出校验最重：要先排除非法输入，再按后端分派做语法校验。

    def _validate_structured_outputs(
        self,
        model_config: ModelConfig,
        structured_outputs_config: StructuredOutputsConfig | None,
        tokenizer: TokenizerLike | None,
    ) -> None:
        if structured_outputs_config is None or self.structured_outputs is None:
            return

        # [CN] 扩散模型的文法 FSM 需要从左到右采样，整体去噪满足不了。

        if model_config.is_diffusion:
            # Diffusion LLMs denoise a whole canvas of tokens in parallel
            # rather than sampling left-to-right, which the grammar FSM
            # requires. Without this check, requests fail mid-generation
            # with an FSM rejection (HTTP 500). See issue #45436.
            raise VLLMValidationError(
                "Structured outputs are not yet supported for diffusion "
                "language models. Remove the structured output constraint "
                "(e.g. `response_format`, `structured_outputs`) from the "
                "request."
            )

        # [CN] skip_tokenizer_init 下没有分词器，结构化输出无从校验。

        if tokenizer is None:
            raise VLLMValidationError(
                "Structured outputs requires a tokenizer so it can't be used with 'skip_tokenizer_init'"  # noqa: E501
            )

        # [CN] 后端在引擎启动时确定，请求级选择不被支持。

        backend = structured_outputs_config.backend
        # [CN] 请求带 _backend 说明它可能来自上一次请求的对象复用。

        if _backend := self.structured_outputs._backend:
            # Request-level backend selection is not supported.
            # The values may differ if `params` is reused and was set
            # to a specific backend based on `auto` behavior in a previous
            # request. We remember that it was set as a result of `auto`
            # using the `_backend_was_auto` field set in the params.
            # [CN] 例外是"请求是 auto 挑出来的、引擎也是 auto"，这时可以覆盖。

            if backend != _backend and not (
                backend == "auto" and self.structured_outputs._backend_was_auto
            ):
                raise VLLMValidationError(
                    "Request-level structured output backend selection is not "
                    f"supported. The request specified '{_backend}', but vLLM "
                    f"was initialised with '{backend}'. This error can be "
                    "resolved by removing '_backend' from the request."
                )
        else:
            self.structured_outputs._backend = backend

        # Request content validation
        if (
            # [CN] 空 choice 列表会在后端构造 FSM 时炸掉，提前拦成 400。

            isinstance(self.structured_outputs.choice, list)
            and not self.structured_outputs.choice
        ):
            # It is invalid for choice to be an empty list
            raise VLLMValidationError(
                f"Choice '{self.structured_outputs.choice}' cannot be an empty list"  # noqa: E501
            )
        # Reject empty string grammar early to avoid engine-side crashes
        if (
            isinstance(self.structured_outputs.grammar, str)
            and self.structured_outputs.grammar.strip() == ""
        ):
            raise VLLMValidationError(
                "structured_outputs.grammar cannot be an empty string"
            )
        # Reject empty string json schema early to avoid engine-side crashes
        if (
            isinstance(self.structured_outputs.json, str)
            and self.structured_outputs.json.strip() == ""
        ):
            raise VLLMValidationError(
                "structured_outputs.json cannot be an empty string"
            )
        # Reject json_object=False early to avoid engine-side crashes
        # [CN] False 是无意义的取值：要禁用就应该整个字段留空。

        if self.structured_outputs.json_object is False:
            raise VLLMValidationError(
                "structured_outputs.json_object must be True if set; omit "
                "structured_outputs to disable structured outputs"
            )
        # Reject a regex containing a NUL byte early, in every backend mode. A
        # NUL is never meaningful in a regex pattern and is not handled by the
        # regex-to-grammar conversion. Checked here, before backend selection,
        # so it is a clean 400 rather than a silent fallback in the default
        # "auto" mode.
        # [CN] NUL 字节在任何正则里都无意义，且会绕过后续所有转换，必须前置拦掉。

        if self.structured_outputs.regex and "\x00" in self.structured_outputs.regex:
            raise VLLMValidationError(
                "structured_outputs.regex must not contain a NUL character ('\\x00')"
            )

        from vllm.v1.structured_output.backend_guidance import (
            has_guidance_unsupported_json_features,
            validate_guidance_grammar,
        )
        from vllm.v1.structured_output.backend_lm_format_enforcer import (
            validate_structured_output_request_lm_format_enforcer,
        )
        from vllm.v1.structured_output.backend_outlines import (
            validate_structured_output_request_outlines,
        )
        from vllm.v1.structured_output.backend_xgrammar import validate_xgrammar_grammar

        # [CN] 指定了具体后端就不做回退：失败即报错，避免悄悄换后端导致语义漂移。

        if backend.startswith("xgrammar"):
            # xgrammar with no fallback
            validate_xgrammar_grammar(self)
        # [CN] guidance 走 llguidance，需要真实分词器才能处理特殊 token。

        elif backend.startswith("guidance"):
            # [CN] 老版 Mistral 分词器在 guidance 下会失败，需要提前换后端或换分词模式。

            if _is_non_tekken_mistral(tokenizer=tokenizer):
                raise VLLMValidationError(
                    "Non-tekken Mistral tokenizers are not supported for the 'guidance'"
                    " structured output backend. Please either use a more recent "
                    "Mistral model, the ['xgrammar', 'outlines'] "
                    "backends or tokenizer_mode='hf' instead."
                )
            # TODO: ideally we would have the LLTokenizer here as Lark syntax
            # allows <|special_token|> and similar, see
            # https://github.com/guidance-ai/llguidance/blob/main/docs/syntax.md#special-tokens
            # Without tokenizer these are disallowed in grammars.
            validate_guidance_grammar(
                self,
                tokenizer=_get_llg_tokenizer(tokenizer),
            )
        # [CN] outlines 走另一套有限状态机实现，支持的语法特性与 xgrammar 不同。

        elif backend == "outlines":
            # outlines backend
            validate_structured_output_request_outlines(self)
        # [CN] 这个后端对 Mistral 分词器完全不支持，需要显式报错。

        elif backend == "lm-format-enforcer":
            # lm format enforcer backend
            # [CN] 用分词器类型而非模型名判断：同一个模型可以配不同的分词模式。

            if is_mistral_tokenizer(tokenizer):
                raise VLLMValidationError(
                    "Mistral tokenizer is not supported for the 'lm-format-enforcer' "
                    "structured output backend. Please use ['xgrammar', 'outlines'] "
                    "backends or tokenizer_mode='hf' instead."
                )
            validate_structured_output_request_lm_format_enforcer(self)
        else:
            # NOTE: backend must be "auto" here, because we have
            # checked supported_backends above.
            # In this mode, we set opinionated defaults based on what we think
            # will satisfy the most use cases without having to worry about
            # this setting. We include fallback behavior here, but not with any
            # other setting where a specific backend was specified.
            try:
                validate_xgrammar_grammar(self)
                self.structured_outputs._backend = "xgrammar"
            except VLLMValidationError:
                # The request either failed validation
                # or includes some jsonschema feature(s) that
                # are not supported in xgrammar.

                skip_guidance = _is_non_tekken_mistral(tokenizer)

                # Check if schema has features unsupported by guidance
                so_params = self.structured_outputs
                if not skip_guidance and so_params.json:
                    if isinstance(so_params.json, str):
                        try:
                            schema = json_mod.loads(so_params.json)
                        except json_mod.JSONDecodeError as e:
                            raise VLLMValidationError(
                                "Invalid JSON grammar specification."
                            ) from e
                    else:
                        schema = so_params.json
                    skip_guidance = has_guidance_unsupported_json_features(schema)

                # [CN] 回退顺序是 xgrammar -> guidance -> outlines，这里决定走后两者之一。

                if skip_guidance:
                    # Fall back to outlines if the tokenizer is non-tekken Mistral or
                    # the schema contains features unsupported by guidance
                    validate_structured_output_request_outlines(self)
                    self.structured_outputs._backend = "outlines"
                else:
                    # Fall back to guidance by default.
                    validate_guidance_grammar(
                        self,
                        tokenizer=_get_llg_tokenizer(tokenizer),
                    )
                    self.structured_outputs._backend = "guidance"
            # Remember that this backend was set automatically
            self.structured_outputs._backend_was_auto = True

        # Run post-init validation. This is also important to ensure subsequent
        # roundtrip serialization/deserialization won't fail.
        # [CN] 末尾再跑一次校验，保证序列化往返之后仍然构造得出来。

        self.structured_outputs.__post_init__()

    # [CN] 手写的 repr 只列关键字段，避免把超长的结构化输出参数打进日志。

    def __repr__(self) -> str:
        return (
            f"SamplingParams(n={self.n}, "
            f"presence_penalty={self.presence_penalty}, "
            f"frequency_penalty={self.frequency_penalty}, "
            f"repetition_penalty={self.repetition_penalty}, "
            f"temperature={self.temperature}, "
            f"top_p={self.top_p}, "
            f"top_k={self.top_k}, "
            f"min_p={self.min_p}, "
            f"seed={self.seed}, "
            f"stop={self.stop}, "
            f"stop_token_ids={self.stop_token_ids}, "
            f"bad_words={self.bad_words}, "
            f"thinking_token_budget={self.thinking_token_budget}, "
            f"include_stop_str_in_output={self.include_stop_str_in_output}, "
            f"ignore_eos={self.ignore_eos}, "
            f"max_tokens={self.max_tokens}, "
            f"min_tokens={self.min_tokens}, "
            f"logprobs={self.logprobs}, "
            f"prompt_logprobs={self.prompt_logprobs}, "
            f"skip_special_tokens={self.skip_special_tokens}, "
            "spaces_between_special_tokens="
            f"{self.spaces_between_special_tokens}, "
            f"structured_outputs={self.structured_outputs}, "
            f"extra_args={self.extra_args})"
        )

    # [CN] 故意把每种特性都打开一点，让预热覆盖到全部采样分支。
    @staticmethod
    def for_sampler_warmup() -> "SamplingParams":
        """Set parameters to exercise all sampler logic."""
        return SamplingParams(
            temperature=0.9,
            top_p=0.9,
            top_k=50,
            min_p=0.1,
            frequency_penalty=0.5,
            presence_penalty=0.5,
            repetition_penalty=1.2,
            min_tokens=2,
            logit_bias={0: -1.0, 1: 0.5},
            _bad_words_token_ids=[[0], [1, 2]],
            logprobs=5,
            prompt_logprobs=1,
        )


# [CN] 集束搜索参数独立成类：它与采样参数没有可共享的默认语义。

class BeamSearchParams(
    msgspec.Struct,
    omit_defaults=True,  # type: ignore[call-arg]
    # required for @cached_property.
    dict=True,
):  # type: ignore[call-arg]
    """Beam search parameters for text generation."""

    # [CN] 集束宽度没有默认值：它决定了整次搜索的形状，必须由调用方明确。

    beam_width: int
    max_tokens: int
    ignore_eos: bool = False
    # [CN] 集束搜索默认是确定性的，温度置 0 走贪心扩展。

    temperature: float = 0.0
    # [CN] 长度惩罚作用于归一化得分，>1 偏好长序列、<1 偏好短序列。

    length_penalty: float = 1.0
    include_stop_str_in_output: bool = False
    structured_outputs: StructuredOutputsParams | None = None

    # [CN] 集束宽度复用与 n 相同的上限检查。

    def __post_init__(self) -> None:
        _verify_num_sequences(self.beam_width, "beam_width")
