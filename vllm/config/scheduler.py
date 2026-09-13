# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# [CN] 文件总览：调度行为的全部开关（SchedulerConfig）
# [CN] 职责：定义「一个 step 能做多少事」的上界 —— token 预算、seq 预算、是否切块 prefill、
# [CN]       排队上限、抢占水位、流控粒度等。这里只描述**规则**，执行在 v1/core/sched/。
# [CN] 链路：EngineArgs -> create_engine_config() -> SchedulerConfig
# [CN]        -> Scheduler.schedule()（真正的实现）
# [CN] @config 装饰器：把普通 dataclass 升级为带校验能力的 pydantic dataclass，
# [CN]   因此 Field(ge=1) 这类约束会在构造时生效，而不是等到运行才炸。
# [CN] 最需要分清的三个「上限」：
# [CN]   max_num_batched_tokens —— 单步 token 预算（吞吐/延迟的主旋钮）
# [CN]   max_num_seqs          —— 单步并发 seq 数（persistent batch 的槽位数）
# [CN]   max_num_queued_reqs   —— 在途请求总数（含 waiting），由 API server 侧执行
# [CN] 易错点：默认值（2048 / 128）是给单测用的，真实部署由 create_engine_config 覆盖。
# [CN]         直接构造 SchedulerConfig 而不走 EngineArgs，会拿到的并不是生产默认值。

from collections.abc import Callable
from dataclasses import InitVar
from typing import TYPE_CHECKING, Any, ClassVar, Literal, cast

from pydantic import Field, field_validator
from typing_extensions import Self

from vllm.config.utils import config
from vllm.logger import init_logger
from vllm.utils.hashing import safe_hash
from vllm.utils.import_utils import resolve_obj_by_qualname

if TYPE_CHECKING:
    from vllm.v1.core.sched.interface import SchedulerInterface

logger = init_logger(__name__)

# [CN] runner_type 决定这次跑的是生成 / 池化（embedding）/ 草稿（投机 decoding）三种 engine 之一。

RunnerType = Literal["generate", "pooling", "draft"]
SchedulerPolicy = Literal["fcfs", "priority"]


# [CN] @config 同时给类加上 __post_init__ 钩子与 pydantic 校验；
# [CN] InitVar（下面两个字段）因此能在 post_init 里被消费但不作为真实字段存储。

@config
class SchedulerConfig:
    """Scheduler configuration."""

    # [CN] InitVar 而非普通字段：真正的归属是 ModelConfig，这里借来只为了做校验与兜底。

    max_model_len: InitVar[int]
    """Maximum length of a sequence (including prompt and generated text).

    Note: This is stored in the ModelConfig, and is used only here to
    provide fallbacks and validate other attributes."""

    # [CN] 同样来自 ModelConfig。它唯一的作用是关掉 encoder-decoder 不支持的两件事：
    # [CN] chunked prefill 与 prefix caching（见 __post_init__）。

    is_encoder_decoder: InitVar[bool]
    """True if the model is an encoder-decoder model.

    Note: This is stored in the ModelConfig, and is used only here to
    disable chunked prefill and prefix caching for encoder-decoder models.
    """

    # [CN] ClassVar：不是实例字段，不参与 hash / 序列化，仅作默认值来源。
    # [CN] 注意 DP 场景的默认预算被压到 256 —— 因为 DP 下每个 rank 独立 batch，
    # [CN] 预算太大会让单步时间抖动剧烈。

    DEFAULT_MAX_NUM_BATCHED_TOKENS: ClassVar[int] = 2048
    DEFAULT_MAX_NUM_BATCHED_TOKENS_FOR_BATCHED_DP: ClassVar[int] = 256
    DEFAULT_MAX_NUM_SEQS: ClassVar[int] = 128

    runner_type: RunnerType = "generate"
    """The runner type to launch for the model."""

    # [CN] 单步 token 预算。它同时也是 KV cache 单步写入量的上界，
    # [CN] 因此会被写进 compute_hash —— 形状会进 torch.compile 图。

    max_num_batched_tokens: int = Field(default=DEFAULT_MAX_NUM_BATCHED_TOKENS, ge=1)
    """Maximum number of tokens that can be processed in a single iteration.

    The default value here is mainly for convenience when testing.
    In real usage, this should be set in `EngineArgs.create_engine_config`.
    """

    # [CN] 调度器实际可下发的 token 数，默认等于 max_num_batched_tokens。
    # [CN] 投机解码等「会往 batch 里追加 token」的场景需要设得更小，留出余量给追加部分。

    max_num_scheduled_tokens: int | None = Field(default=None, ge=0)
    """Maximum number of tokens that the scheduler may issue in a single iteration.
    
    This is usually equal to max_num_batched_tokens, but can be smaller in cases
    when the model might append tokens into the batch (such as speculative decoding).
    Defaults to max_num_batched_tokens."""

    # [CN] 并发 seq 上界，直接决定 persistent batch 里 slot 的数量，因此也进 compute_hash。

    max_num_seqs: int = Field(default=DEFAULT_MAX_NUM_SEQS, ge=1)
    """Maximum number of sequences to be processed in a single iteration.

    The default value here is mainly for convenience when testing.
    In real usage, this should be set in `EngineArgs.create_engine_config`.
    """

    # [CN] 长 prefill 判定门槛：超过它的 prompt 才算「长」，用于分离长/短 prefill 的调度策略。
    # [CN] 0 表示不启用这层区分。

    long_prefill_token_threshold: int = Field(default=0, ge=0)
    """For chunked prefill, a request is considered long if the prompt is
    longer than this number of tokens. 0 disables the cap (default)."""

    # [CN] 在途请求上限，超限直接回 HTTP 503 让客户端去别的实例重试 —— 比无限排队更有用。
    # [CN] 与 max_num_seqs 的关键差别：这个是 API server 进程跨所有 DP rank 统一计数，
    # [CN] 而 max_num_seqs 是单 rank 的。配置时应取 dp_size * max_num_seqs 再加上期望排队深度。

    max_num_queued_reqs: int | None = Field(default=None, ge=0)
    """Maximum number of requests that can be in-flight (waiting or running)
    at the same time, or None for no limit. When the limit is reached, new
    requests are rejected with HTTP 503 so the client can retry on another
    instance. This bounds vLLM's otherwise unbounded request queue and is
    primarily a coarse capacity valve.

    Unlike ``max_num_seqs``, which applies per data-parallel rank, this
    limit is enforced in the API server process and counts in-flight
    requests across all DP ranks it routes to. Size it as roughly
    ``data_parallel_size * max_num_seqs`` plus the desired queue depth if
    it should not bind before per-rank admission does."""

    # [CN] prefill 积压上限，这是 TTFT 的 QoS 手段：设为 目标TTFT x prefill吞吐，
    # [CN] 即可在「积压会导致超时」时提前拒绝。
    # [CN] 注意计数偏保守：分块 prefill 期间 EngineCoreOutput 还没产出，
    # [CN] API server 侧看不到已完成的进度，请求会按完整 prompt_len 计入 ——
    # [CN] 于是会比真实积压更早拒绝。这是刻意的「宁可早拒绝」。

    max_num_queued_tokens: int | None = Field(default=None, ge=0)
    """Maximum total prompt tokens of requests currently in the prefill
    phase, or None for no limit. When the limit is reached, new requests
    are rejected with HTTP 503.

    This is a TTFT QoS mechanism: by setting it to
    ``target_TTFT * prefill_throughput`` you reject requests when the
    prefill backlog would exceed the latency target.  In a disaggregated
    prefill-decode setup this maps directly to the prefill pool's
    capacity.

    Like ``max_num_queued_reqs``, this limit is enforced in the API
    server process and covers the prefill backlog across all DP ranks it
    routes to, so ``prefill_throughput`` in the formula above is the
    aggregate throughput of the deployment.

    Note: the count is conservative.  A partially prefilled request
    still contributes its full ``prompt_len`` until it transitions out
    of the prefill phase, because the scheduler's per-iteration
    ``num_computed_tokens`` progress is not propagated to the API
    server process during prefill (``EngineCoreOutput`` is only
    emitted once the request starts producing tokens).  Similarly,
    prefix-cache hits (``num_cached_tokens``) are only known to the
    OutputProcessor after prefill completes.  This overestimates the
    real backlog, causing earlier rejection than strictly necessary
    — the safe direction for QoS.  The impact is limited to long
    prompts under chunked prefill; short prompts that prefill in a
    single iteration are unaffected."""

    # [CN] 是否允许把长 prompt 切成多步。关掉的话，必须保证
    # [CN] max_num_batched_tokens >= max_model_len，否则长序列根本跑不起来（见 verify_max_model_len）。

    enable_chunked_prefill: bool = True
    """If True, prefill requests can be chunked based
    on the remaining `max_num_batched_tokens`.

    The default value here is mainly for convenience when testing.
    In real usage, this should be set in `EngineArgs.create_engine_config`.
    """

    is_multimodal_model: bool = False
    """True if the model is multimodal."""

    # TODO (ywang96): Make this configurable.
    max_num_encoder_input_tokens: int = Field(init=False)
    """Multimodal encoder compute budget, only used in V1.

    NOTE: This is not currently configurable. It will be overridden by
    max_num_batched_tokens in case max multimodal embedding size is larger."""

    # TODO (ywang96): Make this configurable.
    encoder_cache_size: int = Field(init=False)
    """Multimodal encoder cache size, only used in V1.

    NOTE: This is not currently configurable. It will be overridden by
    max_num_batched_tokens in case max multimodal embedding size is larger."""

    policy: SchedulerPolicy = "fcfs"
    """The scheduling policy to use:

    - "fcfs" means first come first served, i.e. requests are handled in order 
      of arrival.
    - "priority" means requests are handled based on given priority (lower
      value means earlier handling) and time of arrival deciding any ties)."""

    # [CN] 多模态输入不切块：一个图片/音频项要么整体进 batch，要么下一步再进，
    # [CN] 避免出现「图片被切一半」导致 embeddings 对不上。

    disable_chunked_mm_input: bool = False
    """If set to true and chunked prefill is enabled, we do not want to
    partially schedule a multimodal item. Only used in V1
    This ensures that if a request has a mixed prompt
    (like text tokens TTTT followed by image tokens IIIIIIIIII) where only
    some image tokens can be scheduled (like TTTTIIIII, leaving IIIII),
    it will be scheduled as TTTT in one step and IIIIIIIIII in the next."""

    # scheduler class or path. "vllm.v1.core.sched.scheduler.Scheduler"
    # (default) or "mod.custom_class".
    scheduler_cls: str | type[object] | None = None
    """The scheduler class to use. "vllm.v1.core.sched.scheduler.Scheduler" is
    the default scheduler. Can be a class directly or the path to a class of
    form "mod.custom_class"."""

    disable_hybrid_kv_cache_manager: bool | None = None
    """If set to True, KV cache manager will allocate the same size of KV cache
    for all attention layers even if there are multiple type of attention layers
    like full attention and sliding window attention.
    If set to None, the default value will be determined based on the environment
    and starting configuration.
    """

    # [CN] 准入前先检查「完整输入长度」能否装进 KV cache，而不是只看第一个 chunk。
    # [CN] 这是防止 chunked prefill 下过度准入 —— 抢着进来又装不下会被反复抢占，白白浪费算力。

    scheduler_reserve_full_isl: bool = True
    """If True, the scheduler checks whether the full input sequence length
    fits in the KV cache before admitting a new request, rather than only
    checking the first chunk. Prevents over-admission and KV cache thrashing
    with chunked prefill."""

    # [CN] 预留多少比例的 block 保持空闲。留水位可以避免「刚准入就抢占」的抖动。
    # [CN] 显存不紧张时设 0（默认）换取更多并发；抢占频繁时调大。

    watermark: float = Field(default=0.0, ge=0.0, lt=1.0)
    """Fraction of total KV cache blocks to keep free (the watermark) when
    admitting waiting or preempted requests into the running queue. This headroom
    helps avoid frequent KV cache eviction and the resulting repeated preemption
    of requests when GPU memory is scarce. Must be in the range [0.0, 1.0); 0.0
    (the default) disables the watermark."""

    # [CN] 数据并行下每 N 步才准入一批新 prefill，且各 DP rank 对齐。
    # [CN] 目的是让各 rank 的单步耗时更均衡（否则 rank 间时钟差会被同步等待放大）。

    prefill_schedule_interval: int = Field(default=1, ge=1)
    """For data-parallel deployments, only admit new prefill requests
    once every N engine steps, aligned across DP ranks, to better balance
    per-step forward-pass times."""

    # [CN] 异步调度：让 CPU 侧的下一步调度与 GPU 上的当前步重叠，消除 GPU 空档。
    # [CN] 三态(None 未显式设置) —— None 时不强制，由 get_scheduler_cls 决定走哪个 Scheduler。

    async_scheduling: bool | None = None
    """If set to False, disable async scheduling. Async scheduling helps to
    avoid gaps in GPU utilization, leading to better latency and throughput.
    """

    # [CN] 流式输出的发送粒度（按 token 计）。1 = 每 token 立刻发（最平滑、开销最大），
    # [CN] 调大则是攒若干 token 再发，省主机开销、换更高吞吐。

    stream_interval: int = Field(default=1, ge=1)
    """The interval (or buffer size) for streaming in terms of token length.
    A smaller value (1) makes streaming smoother by sending each token immediately,
    while a larger value (e.g., 10) reduces host overhead and may increase throughput
    by batching multiple tokens before sending."""

    @staticmethod
    def default_factory(**kwargs):
        """
        Factory method to create `SchedulerConfig` with default values for `InitVar`s.
        """
        if "max_model_len" not in kwargs:
            kwargs["max_model_len"] = 8192
        if "is_encoder_decoder" not in kwargs:
            kwargs["is_encoder_decoder"] = False
        return SchedulerConfig(**kwargs)

    # [CN] 决定用哪个 Scheduler 实现：显式指定了就用它；没指定则按 async_scheduling 分流到
    # [CN] AsyncScheduler（一步 prev forward 与下一轮调度重叠）或普通 Scheduler。

    def get_scheduler_cls(self) -> type["SchedulerInterface"]:
        if self.scheduler_cls is None:
            if self.async_scheduling:
                from vllm.v1.core.sched.async_scheduler import AsyncScheduler

                return AsyncScheduler
            from vllm.v1.core.sched.scheduler import Scheduler

            return Scheduler

        # [CN] 自定义 scheduler 走的是非公开接口。若继承的是 Scheduler 而非 AsyncScheduler，
        # [CN] 异步调度会被自动关闭 —— 性能下降但不是错误，所以这里只 warning_once。

        # The first half of this warning can be removed once the Scheduler interface is
        # finalized and we can maintain support for scheduler classes that implement it
        logger.warning_once(
            "Using custom scheduler class %s. This scheduler interface is not public "
            "and compatibility may not be maintained. If you have subclassed Scheduler "
            "instead of AsyncScheduler, you will see degraded performance due to async "
            "scheduling being disabled.",
            self.scheduler_cls,  # type: ignore[arg-type]
        )
        if not isinstance(self.scheduler_cls, str):
            return cast(type["SchedulerInterface"], self.scheduler_cls)
        return resolve_obj_by_qualname(self.scheduler_cls)

    # [CN] 计算图的指纹：只包含影响「形状/结构」的字段。
    # [CN] 加新字段时必须判断它是否影响 compile 图，影响就要进 factors，否则会命中错误的缓存。

    def compute_hash(self) -> str:
        """
        WARNING: Whenever a new field is added to this config,
        ensure that it is included in the factors list if
        it affects the computation graph.

        Provide a hash that uniquely identifies all the configs
        that affect the structure of the computation
        graph from input ids/embeddings to the final hidden states,
        excluding anything before input ids/embeddings and after
        the final hidden states.
        """
        factors: list[Any] = []

        # max_num_batched_tokens need to be included in the hash due
        # to two reasons:
        # 1. LoRA creates static buffers based on max_num_batched_tokens.
        #   The tensor sizes and strides get captured in the torch.compile
        #   graph explicitly.
        # 2. Inductor decides whether using 32-bit or 64-bit indexing integer
        #   based on the data sizes. `max_num_batched_tokens` has an
        #   impact on that. For more details, please check
        #   https://github.com/vllm-project/vllm/issues/29585
        factors.append(self.max_num_batched_tokens)

        # PLE and other model components allocate static per-request buffers.
        # Their shapes are captured in compiled graphs.
        factors.append(self.max_num_seqs)

        # [CN] usedforsecurity=False：这里只要稳定快速的摘要，不需要密码学强度。

        hash_str = safe_hash(str(factors).encode(), usedforsecurity=False).hexdigest()
        return hash_str

    @field_validator("scheduler_cls", "async_scheduling", mode="wrap")
    @classmethod
    def _skip_none_validation(cls, value: Any, handler: Callable) -> Any:
        """Skip validation if the value is `None` when initialisation is delayed."""
        return None if value is None else handler(value)

    # [CN] 后处理：把 encoder-decoder 的三个不支持项关掉，并给多模态编码器预算兜底。

    def __post_init__(self, max_model_len: int, is_encoder_decoder: bool) -> None:
        # [CN] encoder-decoder 模型的 prompt 要一次性喂给 encoder，无法分块；
        # [CN] prefix caching 也依赖逐 token 的块哈希，同样不适用。

        if is_encoder_decoder:
            # Chunked prefill should be disabled for encoder-decoder models.
            self.disable_chunked_mm_input = True
            self.enable_chunked_prefill = False
            self.long_prefill_token_threshold = 0
            logger.info(
                "Encoder-decoder models do not support chunked prefill nor"
                " prefix caching; disabling both."
            )

        # [CN] 多模态编码器的两项预算当前不可单独配，直接对齐到 token 预算
        # [CN] （多模态 embedding 尺寸可能更大，必要时会被下游覆盖）。

        self.max_num_encoder_input_tokens = self.max_num_batched_tokens
        self.encoder_cache_size = self.max_num_batched_tokens

        if self.enable_chunked_prefill:
            logger.info_once(
                "Chunked prefill is enabled with max_num_batched_tokens=%d.",
                self.max_num_batched_tokens,
            )

        self.verify_max_model_len(max_model_len)

    # [CN] 参数自洽性校验。这里拦的都是「配了但必然跑不对」的组合。

    def verify_max_model_len(self, max_model_len: int) -> Self:
        if (
            self.max_num_batched_tokens < max_model_len
            and not self.enable_chunked_prefill
        ):
            raise ValueError(
                f"max_num_batched_tokens ({self.max_num_batched_tokens}) is "
                f"smaller than max_model_len ({max_model_len}). "
                "This effectively limits the maximum sequence length to "
                "max_num_batched_tokens and makes vLLM reject longer "
                "sequences. Please increase max_num_batched_tokens or "
                "decrease max_model_len."
            )

        # [CN] 至少要保证每个 seq 能分到 1 个 token，否则一定会有 seq 拿不到预算永远排在后面。

        if self.max_num_batched_tokens < self.max_num_seqs:
            raise ValueError(
                f"max_num_batched_tokens ({self.max_num_batched_tokens}) must "
                "be greater than or equal to max_num_seqs "
                f"({self.max_num_seqs})."
            )

        if self.max_num_batched_tokens > self.max_num_seqs * max_model_len:
            logger.warning(
                "max_num_batched_tokens (%d) exceeds max_num_seqs "
                "* max_model_len (%d). This may lead to unexpected behavior.",
                self.max_num_batched_tokens,
                self.max_num_seqs * max_model_len,
            )

        if self.long_prefill_token_threshold > max_model_len:
            raise ValueError(
                "long_prefill_token_threshold "
                f"({self.long_prefill_token_threshold}) cannot be greater "
                f"than the max_model_len ({max_model_len})."
            )

        return self
