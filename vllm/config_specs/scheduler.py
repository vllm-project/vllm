# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduler field definitions shared by CLI metadata and runtime config."""

from dataclasses import InitVar, dataclass, field
from typing import ClassVar, Literal

from . import CLI

RunnerType = Literal["generate", "pooling", "draft"]
SchedulerPolicy = Literal["fcfs", "priority"]


@dataclass
class SchedulerFields:
    """Scheduler configuration."""

    max_model_len: InitVar[int] = field(
        metadata={
            "doc": (
                "Maximum length of a sequence (including prompt and generated "
                "text).\n"
                "\n"
                "Note: This is stored in the ModelConfig, and is used only "
                "here to\n"
                "provide fallbacks and validate other attributes."
            ),
        },
    )

    is_encoder_decoder: InitVar[bool] = field(
        metadata={
            "doc": (
                "True if the model is an encoder-decoder model.\n"
                "\n"
                "Note: This is stored in the ModelConfig, and is used only "
                "here to\n"
                "disable chunked prefill and prefix caching for "
                "encoder-decoder models."
            ),
        },
    )

    DEFAULT_MAX_NUM_BATCHED_TOKENS: ClassVar[int] = 2048

    DEFAULT_MAX_NUM_BATCHED_TOKENS_FOR_BATCHED_DP: ClassVar[int] = 256

    DEFAULT_MAX_NUM_SEQS: ClassVar[int] = 128

    runner_type: RunnerType = field(
        default="generate",
        metadata={
            "doc": ("The runner type to launch for the model."),
        },
    )

    max_num_batched_tokens: int = field(
        default=DEFAULT_MAX_NUM_BATCHED_TOKENS,
        metadata={
            "ge": 1,
            "cli": CLI(
                0,
                default=None,
                human_readable=True,
                python_after="kv_cache_memory_bytes",
            ),
            "doc": (
                "Maximum number of tokens that can be processed in a single "
                "iteration.\n"
                "\n"
                "The default value here is mainly for convenience when "
                "testing.\n"
                "In real usage, this should be set in "
                "`EngineArgs.create_engine_config`."
            ),
        },
    )

    max_num_scheduled_tokens: int | None = field(
        default=None,
        metadata={
            "ge": 0,
            "cli": CLI(
                1,
                default=None,
                human_readable=True,
                python_after="max_num_batched_tokens",
            ),
            "doc": (
                "Maximum number of tokens that the scheduler may issue in a "
                "single iteration.\n"
                "\n"
                "This is usually equal to max_num_batched_tokens, but can be "
                "smaller in cases\n"
                "when the model might append tokens into the batch (such as "
                "speculative decoding).\n"
                "Defaults to max_num_batched_tokens."
            ),
        },
    )

    max_num_seqs: int = field(
        default=DEFAULT_MAX_NUM_SEQS,
        metadata={
            "ge": 1,
            "cli": CLI(2, default=None, python_after="long_prefill_token_threshold"),
            "doc": (
                "Maximum number of sequences to be processed in a single "
                "iteration.\n"
                "\n"
                "The default value here is mainly for convenience when "
                "testing.\n"
                "In real usage, this should be set in "
                "`EngineArgs.create_engine_config`."
            ),
        },
    )

    long_prefill_token_threshold: int = field(
        default=0,
        metadata={
            "ge": 0,
            "cli": CLI(5, python_after="max_num_scheduled_tokens"),
            "doc": (
                "For chunked prefill, a request is considered long if the "
                "prompt is\n"
                "longer than this number of tokens. 0 disables the cap "
                "(default)."
            ),
        },
    )

    max_num_queued_reqs: int | None = field(
        default=None,
        metadata={
            "ge": 0,
            "cli": CLI(3, python_after="max_num_seqs"),
            "doc": (
                "Maximum number of requests that can be in-flight (waiting or "
                "running)\n"
                "at the same time, or None for no limit. When the limit is "
                "reached, new\n"
                "requests are rejected with HTTP 503 so the client can retry "
                "on another\n"
                "instance. This bounds vLLM's otherwise unbounded request "
                "queue and is\n"
                "primarily a coarse capacity valve.\n"
                "\n"
                "Unlike ``max_num_seqs``, which applies per data-parallel "
                "rank, this\n"
                "limit is enforced in the API server process and counts "
                "in-flight\n"
                "requests across all DP ranks it routes to. Size it as roughly\n"
                "``data_parallel_size * max_num_seqs`` plus the desired queue "
                "depth if\n"
                "it should not bind before per-rank admission does."
            ),
        },
    )

    max_num_queued_tokens: int | None = field(
        default=None,
        metadata={
            "ge": 0,
            "cli": CLI(4, human_readable=True, python_after="max_num_queued_reqs"),
            "doc": (
                "Maximum total prompt tokens of requests currently in the "
                "prefill\n"
                "phase, or None for no limit. When the limit is reached, new "
                "requests\n"
                "are rejected with HTTP 503.\n"
                "\n"
                "This is a TTFT QoS mechanism: by setting it to\n"
                "``target_TTFT * prefill_throughput`` you reject requests when "
                "the\n"
                "prefill backlog would exceed the latency target.  In a "
                "disaggregated\n"
                "prefill-decode setup this maps directly to the prefill pool's\n"
                "capacity.\n"
                "\n"
                "Like ``max_num_queued_reqs``, this limit is enforced in the "
                "API\n"
                "server process and covers the prefill backlog across all DP "
                "ranks it\n"
                "routes to, so ``prefill_throughput`` in the formula above is "
                "the\n"
                "aggregate throughput of the deployment.\n"
                "\n"
                "Note: the count is conservative.  A partially prefilled "
                "request\n"
                "still contributes its full ``prompt_len`` until it "
                "transitions out\n"
                "of the prefill phase, because the scheduler's per-iteration\n"
                "``num_computed_tokens`` progress is not propagated to the API\n"
                "server process during prefill (``EngineCoreOutput`` is only\n"
                "emitted once the request starts producing tokens).  "
                "Similarly,\n"
                "prefix-cache hits (``num_cached_tokens``) are only known to "
                "the\n"
                "OutputProcessor after prefill completes.  This overestimates "
                "the\n"
                "real backlog, causing earlier rejection than strictly "
                "necessary\n"
                "— the safe direction for QoS.  The impact is limited to long\n"
                "prompts under chunked prefill; short prompts that prefill in "
                "a\n"
                "single iteration are unaffected."
            ),
        },
    )

    enable_chunked_prefill: bool = field(
        default=True,
        metadata={
            "cli": CLI(7, default=None, python_after="ignore_patterns"),
            "doc": (
                "If True, prefill requests can be chunked based\n"
                "on the remaining `max_num_batched_tokens`.\n"
                "\n"
                "The default value here is mainly for convenience when "
                "testing.\n"
                "In real usage, this should be set in "
                "`EngineArgs.create_engine_config`."
            ),
        },
    )

    is_multimodal_model: bool = field(
        default=False,
        metadata={
            "doc": ("True if the model is multimodal."),
        },
    )

    # TODO (ywang96): Make this configurable.
    max_num_encoder_input_tokens: int = field(
        init=False,
        metadata={
            "doc": (
                "Multimodal encoder compute budget, only used in V1.\n\n"
                "NOTE: This is not currently configurable. It will be overridden by\n"
                "max_num_batched_tokens in case max multimodal embedding size "
                "is larger."
            ),
        },
    )

    # TODO (ywang96): Make this configurable.
    encoder_cache_size: int = field(
        init=False,
        metadata={
            "doc": (
                "Multimodal encoder cache size, only used in V1.\n\n"
                "NOTE: This is not currently configurable. It will be overridden by\n"
                "max_num_batched_tokens in case max multimodal embedding size "
                "is larger."
            ),
        },
    )

    policy: SchedulerPolicy = field(
        default="fcfs",
        metadata={
            "cli": CLI(
                6, dest="scheduling_policy", python_after="enable_mm_processor_stats"
            ),
            "doc": (
                "The scheduling policy to use:\n"
                "\n"
                '- "fcfs" means first come first served, i.e. requests are '
                "handled in order \n"
                "  of arrival.\n"
                '- "priority" means requests are handled based on given '
                "priority (lower\n"
                "  value means earlier handling) and time of arrival deciding "
                "any ties)."
            ),
        },
    )

    disable_chunked_mm_input: bool = field(
        default=False,
        metadata={
            "cli": CLI(8, python_after="enable_chunked_prefill"),
            "doc": (
                "If set to true and chunked prefill is enabled, we do not want "
                "to\n"
                "partially schedule a multimodal item. Only used in V1\n"
                "This ensures that if a request has a mixed prompt\n"
                "(like text tokens TTTT followed by image tokens IIIIIIIIII) "
                "where only\n"
                "some image tokens can be scheduled (like TTTTIIIII, leaving "
                "IIIII),\n"
                "it will be scheduled as TTTT in one step and IIIIIIIIII in "
                "the next."
            ),
        },
    )

    scheduler_cls: str | type[object] | None = field(
        default=None,
        metadata={
            "cli": CLI(9, python_after="scheduling_policy"),
            "doc": (
                "The scheduler class to use. "
                '"vllm.v1.core.sched.scheduler.Scheduler" is\n'
                "the default scheduler. Can be a class directly or the path to "
                "a class of\n"
                'form "mod.custom_class".'
            ),
        },
    )

    disable_hybrid_kv_cache_manager: bool | None = field(
        default=None,
        metadata={
            "cli": CLI(13, python_after="watermark"),
            "doc": (
                "If set to True, KV cache manager will allocate the same size "
                "of KV cache\n"
                "for all attention layers even if there are multiple type of "
                "attention layers\n"
                "like full attention and sliding window attention.\n"
                "If set to None, the default value will be determined based on "
                "the environment\n"
                "and starting configuration."
            ),
        },
    )

    scheduler_reserve_full_isl: bool = field(
        default=True,
        metadata={
            "cli": CLI(10, python_after="disable_chunked_mm_input"),
            "doc": (
                "If True, the scheduler checks whether the full input sequence "
                "length\n"
                "fits in the KV cache before admitting a new request, rather "
                "than only\n"
                "checking the first chunk. Prevents over-admission and KV "
                "cache thrashing\n"
                "with chunked prefill."
            ),
        },
    )

    watermark: float = field(
        default=0.0,
        metadata={
            "ge": 0.0,
            "lt": 1.0,
            "cli": CLI(11, python_after="prefill_schedule_interval"),
            "doc": (
                "Fraction of total KV cache blocks to keep free (the "
                "watermark) when\n"
                "admitting waiting or preempted requests into the running "
                "queue. This headroom\n"
                "helps avoid frequent KV cache eviction and the resulting "
                "repeated preemption\n"
                "of requests when GPU memory is scarce. Must be in the range "
                "[0.0, 1.0); 0.0\n"
                "(the default) disables the watermark."
            ),
        },
    )

    prefill_schedule_interval: int = field(
        default=1,
        metadata={
            "ge": 1,
            "cli": CLI(12, python_after="scheduler_reserve_full_isl"),
            "doc": (
                "For data-parallel deployments, only admit new prefill "
                "requests\n"
                "once every N engine steps, aligned across DP ranks, to better "
                "balance\n"
                "per-step forward-pass times."
            ),
        },
    )

    async_scheduling: bool | None = field(
        default=None,
        metadata={
            "cli": CLI(14, python_after="logits_processors"),
            "doc": (
                "If set to False, disable async scheduling. Async scheduling "
                "helps to\n"
                "avoid gaps in GPU utilization, leading to better latency and "
                "throughput."
            ),
        },
    )

    stream_interval: int = field(
        default=1,
        metadata={
            "ge": 1,
            "cli": CLI(15, python_after="async_scheduling"),
            "doc": (
                "The interval (or buffer size) for streaming in terms of token "
                "length.\n"
                "A smaller value (1) makes streaming smoother by sending each "
                "token immediately,\n"
                "while a larger value (e.g., 10) reduces host overhead and may "
                "increase throughput\n"
                "by batching multiple tokens before sending."
            ),
        },
    )
