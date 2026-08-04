# SPDX-License-Identifier: Apache-2.0
"""Workload definitions and factory for ``lmcache bench engine``.

Each workload module defines its own config dataclass and workload
class. The ``create_workload`` factory selects the right workload
based on ``EngineBenchConfig.workload``, resolves the workload-specific
config from CLI args, and returns the workload instance.
"""

# Standard
import argparse

# First Party
from lmcache.cli.commands.bench.engine_bench.config import EngineBenchConfig
from lmcache.cli.commands.bench.engine_bench.progress import ProgressMonitor
from lmcache.cli.commands.bench.engine_bench.request_sender import (
    RequestSender,
)
from lmcache.cli.commands.bench.engine_bench.stats import StatsCollector
from lmcache.cli.commands.bench.engine_bench.workloads.base import BaseWorkload
from lmcache.cli.commands.bench.engine_bench.workloads.long_doc_permutator import (
    LongDocPermutatorConfig,
    LongDocPermutatorWorkload,
)
from lmcache.cli.commands.bench.engine_bench.workloads.long_doc_qa import (
    LongDocQAConfig,
    LongDocQAWorkload,
)
from lmcache.cli.commands.bench.engine_bench.workloads.multi_round_chat import (
    MultiRoundChatConfig,
    MultiRoundChatWorkload,
)
from lmcache.cli.commands.bench.engine_bench.workloads.prefix_suffix_tuner import (
    PrefixSuffixTunerConfig,
    PrefixSuffixTunerWorkload,
)
from lmcache.cli.commands.bench.engine_bench.workloads.random_prefill import (
    RandomPrefillConfig,
    RandomPrefillWorkload,
)

__all__ = [
    "BaseWorkload",
    "LongDocPermutatorConfig",
    "LongDocPermutatorWorkload",
    "LongDocQAConfig",
    "LongDocQAWorkload",
    "MultiRoundChatConfig",
    "MultiRoundChatWorkload",
    "PrefixSuffixTunerConfig",
    "PrefixSuffixTunerWorkload",
    "RandomPrefillConfig",
    "RandomPrefillWorkload",
    "create_workload",
    "validate_max_output_length_supported",
]

_WORKLOAD_NAMES = (
    "long-doc-permutator",
    "long-doc-qa",
    "multi-round-chat",
    "prefix-suffix-tuner",
    "random-prefill",
)

# Workloads that expose a user-configurable max output length, via a
# ``max_output_length`` arg (``--ldqa-max-output-length`` / ``--mrc-output-length``).
_WORKLOADS_WITH_MAX_OUTPUT_LENGTH: frozenset[str] = frozenset(
    {"long-doc-qa", "multi-round-chat"}
)


def validate_max_output_length_supported(workload: str) -> None:
    """Validate that a max output length can be specified for ``workload``.

    Only workloads with a max-output-length parameter (``long-doc-qa``,
    ``multi-round-chat``) support setting it; every other workload fixes its
    generation length internally, so requesting one is rejected.

    Args:
        workload: The selected workload name (``EngineBenchConfig.workload``).

    Raises:
        ValueError: If ``workload`` has no max-output-length parameter.
    """
    if workload in _WORKLOADS_WITH_MAX_OUTPUT_LENGTH:
        return
    supported = ", ".join(sorted(_WORKLOADS_WITH_MAX_OUTPUT_LENGTH))
    raise ValueError(
        f"max output length cannot be specified for the {workload!r} workload: "
        f"it has no max-output-length parameter. Supported workloads: {supported}."
    )


def create_workload(
    config: EngineBenchConfig,
    args: argparse.Namespace,
    request_sender: RequestSender,
    stats_collector: StatsCollector,
    progress_monitor: ProgressMonitor,
) -> BaseWorkload:
    """Resolve workload-specific config and create the workload instance.

    Dispatches on ``config.workload`` to the appropriate workload module,
    resolves the workload-specific config from ``args`` and ``config``,
    and returns the workload instance ready to ``run()``.

    Args:
        config: Fully-resolved general benchmark config.
        args: Raw CLI args namespace (contains workload-specific flags).
        request_sender: Shared request sender instance.
        stats_collector: Shared stats collector instance.
        progress_monitor: Shared progress monitor instance.

    Returns:
        A concrete BaseWorkload instance.

    Raises:
        ValueError: If the workload name is not recognized.
    """
    if config.workload == "long-doc-permutator":
        ldp_workload_config = LongDocPermutatorConfig.resolve(
            num_contexts=args.ldp_num_contexts,
            context_length=args.ldp_context_length,
            system_prompt_length=args.ldp_system_prompt_length,
            num_permutations=args.ldp_num_permutations,
            vocab_size=8000,
            num_inflight_requests=args.ldp_num_inflight_requests,
        )
        return LongDocPermutatorWorkload(
            config=ldp_workload_config,
            request_sender=request_sender,
            stats_collector=stats_collector,
            progress_monitor=progress_monitor,
            seed=config.seed,
        )

    if config.workload == "long-doc-qa":
        ld_workload_config = LongDocQAConfig.resolve(
            kv_cache_volume_gb=config.kv_cache_volume_gb,
            tokens_per_gb_kvcache=config.tokens_per_gb_kvcache,
            document_length=args.ldqa_document_length,
            query_per_document=args.ldqa_query_per_document,
            shuffle_policy=args.ldqa_shuffle_policy,
            num_inflight_requests=args.ldqa_num_inflight_requests,
            max_output_length=args.ldqa_max_output_length,
        )
        return LongDocQAWorkload(
            config=ld_workload_config,
            request_sender=request_sender,
            stats_collector=stats_collector,
            progress_monitor=progress_monitor,
            seed=config.seed,
        )

    if config.workload == "multi-round-chat":
        mr_workload_config = MultiRoundChatConfig.resolve(
            kv_cache_volume_gb=config.kv_cache_volume_gb,
            tokens_per_gb_kvcache=config.tokens_per_gb_kvcache,
            shared_prompt_length=args.mrc_shared_prompt_length,
            chat_history_length=args.mrc_chat_history_length,
            user_input_length=args.mrc_user_input_length,
            output_length=args.mrc_output_length,
            qps=args.mrc_qps,
            duration=args.mrc_duration,
        )
        return MultiRoundChatWorkload(
            config=mr_workload_config,
            request_sender=request_sender,
            stats_collector=stats_collector,
            progress_monitor=progress_monitor,
            seed=config.seed,
        )

    if config.workload == "prefix-suffix-tuner":
        psf_workload_config = PrefixSuffixTunerConfig.resolve(
            tokens_per_gb_kvcache=config.tokens_per_gb_kvcache,
            context_length=args.psf_context_length,
            prefix_ratio=args.psf_prefix_ratio,
            thrash=args.psf_thrash,
        )
        return PrefixSuffixTunerWorkload(
            config=psf_workload_config,
            request_sender=request_sender,
            stats_collector=stats_collector,
            progress_monitor=progress_monitor,
            seed=config.seed,
            model_name=config.model,
        )

    if config.workload == "random-prefill":
        rp_workload_config = RandomPrefillConfig.resolve(
            request_length=args.rp_request_length,
            num_requests=args.rp_num_requests,
        )
        return RandomPrefillWorkload(
            config=rp_workload_config,
            request_sender=request_sender,
            stats_collector=stats_collector,
            progress_monitor=progress_monitor,
            seed=config.seed,
        )

    raise ValueError(
        f"Unknown workload {config.workload!r}. Available: {', '.join(_WORKLOAD_NAMES)}"
    )
