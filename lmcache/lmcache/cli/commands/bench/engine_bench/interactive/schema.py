# SPDX-License-Identifier: Apache-2.0
"""Centralized config item definitions for interactive configuration.

Each ``ConfigItem`` declaratively describes one configurable parameter:
its key, display name, description, input type, default, and when it
should be shown.  The ``ALL_ITEMS`` list is the single source of truth
for descriptions, ordering, and defaults.
"""

# Standard
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Phases
# ---------------------------------------------------------------------------

PHASE_REQUIRED = 1
PHASE_GENERAL = 2
PHASE_WORKLOAD = 3


# ---------------------------------------------------------------------------
# ConfigItem
# ---------------------------------------------------------------------------


@dataclass
class ConfigItem:
    """Declarative description of a single configurable parameter.

    Attributes:
        key: State dict key (matches argparse attr name, e.g., ``"engine_url"``).
        display_name: Heading shown in the prompt.
        description: One-sentence explanation shown below the heading.
        input_type: One of ``"text"``, ``"int"``, ``"float"``, ``"bool"``,
            ``"choice"``.
        default: Default value.  ``None`` means required (no default).
        required: If True, this item must have a value before the benchmark
            can start.
        choices: For ``"choice"`` type — list of ``(value, description)`` tuples.
        condition: Callable ``(state_dict) -> bool`` that determines whether
            this item should be shown.  ``None`` means always shown.
        phase: Which interactive phase this item belongs to.
    """

    key: str
    display_name: str
    description: str
    input_type: str  # "text", "int", "float", "bool", "choice"
    default: Any = None
    required: bool = False
    choices: list[tuple[str, str]] = field(default_factory=list)
    condition: Callable[[dict[str, Any]], bool] | None = None
    phase: int = PHASE_GENERAL


# ---------------------------------------------------------------------------
# Condition helpers
# ---------------------------------------------------------------------------


def _has_lmcache(state: dict[str, Any]) -> bool:
    """Show this item only when the user said they have LMCache."""
    return bool(state.get("has_lmcache"))


def _no_lmcache_url(state: dict[str, Any]) -> bool:
    """Show this item only when lmcache_url is not set."""
    return not state.get("lmcache_url")


def _workload_is(name: str) -> Callable[[dict[str, Any]], bool]:
    """Return a condition that checks the workload value."""

    def check(state: dict[str, Any]) -> bool:
        return state.get("workload") == name

    return check


# ---------------------------------------------------------------------------
# ALL_ITEMS — the centralized registry
# ---------------------------------------------------------------------------

ALL_ITEMS: list[ConfigItem] = [
    # ── Phase 1: Required ─────────────────────────────────────────────
    ConfigItem(
        key="engine_url",
        display_name="Engine URL",
        description=(
            "URL of the inference engine. Enter just a port (e.g. 8000) to "
            "use http://localhost:8000. "
            "Set OPENAI_API_KEY env var if authentication is needed."
        ),
        input_type="text",
        default="http://localhost:8000",
        required=True,
        phase=PHASE_REQUIRED,
    ),
    ConfigItem(
        key="workload",
        display_name="Workload",
        description="The type of benchmark workload to run.",
        input_type="choice",
        default=None,
        required=True,
        choices=[
            (
                "long-doc-permutator",
                "Query the same set of long documents with different orders",
            ),
            ("long-doc-qa", "Repeated Q&A over long documents (tests KV cache reuse)"),
            ("multi-round-chat", "Multi-turn chat with stateful sessions"),
            (
                "prefix-suffix-tuner",
                "Two-pass sequential workload demonstrating tiered KV cache reuse",
            ),
            ("random-prefill", "Prefill-only requests fired simultaneously"),
        ],
        phase=PHASE_REQUIRED,
    ),
    ConfigItem(
        key="has_lmcache",
        display_name="LMCache Server",
        description=(
            "Do you have a running LMCache server? "
            "It can auto-detect KV cache size information."
        ),
        input_type="bool",
        default=True,
        required=False,
        phase=PHASE_REQUIRED,
    ),
    ConfigItem(
        key="lmcache_url",
        display_name="LMCache Server URL",
        description=(
            "URL of the running LMCache HTTP server. Enter just a port "
            "(e.g. 8080) to use http://localhost:8080."
        ),
        input_type="text",
        default="http://localhost:8080",
        required=False,
        condition=_has_lmcache,
        phase=PHASE_REQUIRED,
    ),
    ConfigItem(
        key="tokens_per_gb_kvcache",
        display_name="Tokens per GB KV cache",
        description=(
            "How many tokens fit in 1 GB of KV cache for your model.\n"
            "  If using vLLM, look for these lines in the startup log:\n"
            '    "Available KV cache memory: XX.XX GiB"\n'
            '    "GPU KV cache size: XXX,XXX tokens"\n'
            "  Then compute: tokens_per_gb = "
            "GPU_KV_cache_tokens / Available_KV_cache_GiB"
        ),
        input_type="int",
        default=None,
        required=True,
        condition=_no_lmcache_url,
        phase=PHASE_REQUIRED,
    ),
    # ── Phase 2: General ──────────────────────────────────────────────
    ConfigItem(
        key="model",
        display_name="Model name",
        description=(
            "The model served by the engine. "
            "Leave empty to auto-detect from the engine."
        ),
        input_type="text",
        default="",
        phase=PHASE_GENERAL,
    ),
    ConfigItem(
        key="kv_cache_volume",
        display_name="KV cache volume (GB)",
        description="Target active KV cache size for the benchmark.",
        input_type="float",
        default=100.0,
        phase=PHASE_GENERAL,
    ),
    ConfigItem(
        key="ignore_eos",
        display_name="Ignore EOS",
        description=(
            "Force generation to run for the full output length by ignoring "
            "the model's EOS token (vLLM extension). Makes decode throughput "
            "reproducible."
        ),
        input_type="bool",
        default=False,
        phase=PHASE_GENERAL,
    ),
    # ── Phase 3: long-doc-permutator ─────────────────────────────────
    ConfigItem(
        key="ldp_num_contexts",
        display_name="Number of contexts",
        description="Number of unique context documents to generate.",
        input_type="int",
        default=5,
        condition=_workload_is("long-doc-permutator"),
        phase=PHASE_WORKLOAD,
    ),
    ConfigItem(
        key="ldp_context_length",
        display_name="Context length (tokens)",
        description="Token length of each context document.",
        input_type="int",
        default=5000,
        condition=_workload_is("long-doc-permutator"),
        phase=PHASE_WORKLOAD,
    ),
    ConfigItem(
        key="ldp_system_prompt_length",
        display_name="System prompt length (tokens)",
        description="Token length of the shared system prompt. Use 0 for none.",
        input_type="int",
        default=1000,
        condition=_workload_is("long-doc-permutator"),
        phase=PHASE_WORKLOAD,
    ),
    ConfigItem(
        key="ldp_num_permutations",
        display_name="Number of permutations",
        description="Distinct permutations to send. Capped at N! (N = num_contexts).",
        input_type="int",
        default=10,
        condition=_workload_is("long-doc-permutator"),
        phase=PHASE_WORKLOAD,
    ),
    ConfigItem(
        key="ldp_num_inflight_requests",
        display_name="Max inflight requests",
        description="Maximum concurrent in-flight requests.",
        input_type="int",
        default=1,
        condition=_workload_is("long-doc-permutator"),
        phase=PHASE_WORKLOAD,
    ),
    # ── Phase 3: long-doc-qa ──────────────────────────────────────────
    ConfigItem(
        key="ldqa_document_length",
        display_name="Document length (tokens)",
        description="Token length of each synthetic document.",
        input_type="int",
        default=10000,
        condition=_workload_is("long-doc-qa"),
        phase=PHASE_WORKLOAD,
    ),
    ConfigItem(
        key="ldqa_query_per_document",
        display_name="Queries per document",
        description="Number of questions asked per document.",
        input_type="int",
        default=2,
        condition=_workload_is("long-doc-qa"),
        phase=PHASE_WORKLOAD,
    ),
    ConfigItem(
        key="ldqa_shuffle_policy",
        display_name="Shuffle policy",
        description="How benchmark requests are ordered.",
        input_type="choice",
        default="random",
        choices=[
            ("random", "Shuffle all (doc, query) pairs randomly"),
            ("tile", "Process queries round by round across all documents"),
        ],
        condition=_workload_is("long-doc-qa"),
        phase=PHASE_WORKLOAD,
    ),
    ConfigItem(
        key="ldqa_num_inflight_requests",
        display_name="Max inflight requests",
        description="Maximum concurrent in-flight requests.",
        input_type="int",
        default=3,
        condition=_workload_is("long-doc-qa"),
        phase=PHASE_WORKLOAD,
    ),
    ConfigItem(
        key="ldqa_max_output_length",
        display_name="Max output length (tokens)",
        description="Max tokens to generate per benchmark query.",
        input_type="int",
        default=128,
        condition=_workload_is("long-doc-qa"),
        phase=PHASE_WORKLOAD,
    ),
    # ── Phase 3: multi-round-chat ─────────────────────────────────────
    ConfigItem(
        key="mrc_shared_prompt_length",
        display_name="System prompt length (tokens)",
        description="Token length of the system prompt per session.",
        input_type="int",
        default=2000,
        condition=_workload_is("multi-round-chat"),
        phase=PHASE_WORKLOAD,
    ),
    ConfigItem(
        key="mrc_chat_history_length",
        display_name="Chat history length (tokens)",
        description="Token length of pre-filled conversation history.",
        input_type="int",
        default=10000,
        condition=_workload_is("multi-round-chat"),
        phase=PHASE_WORKLOAD,
    ),
    ConfigItem(
        key="mrc_user_input_length",
        display_name="User input length (tokens)",
        description="Tokens per user query in each round.",
        input_type="int",
        default=50,
        condition=_workload_is("multi-round-chat"),
        phase=PHASE_WORKLOAD,
    ),
    ConfigItem(
        key="mrc_output_length",
        display_name="Output length (tokens)",
        description="Max tokens to generate per response.",
        input_type="int",
        default=200,
        condition=_workload_is("multi-round-chat"),
        phase=PHASE_WORKLOAD,
    ),
    ConfigItem(
        key="mrc_qps",
        display_name="Queries per second",
        description="Target request dispatch rate.",
        input_type="float",
        default=1.0,
        condition=_workload_is("multi-round-chat"),
        phase=PHASE_WORKLOAD,
    ),
    ConfigItem(
        key="mrc_duration",
        display_name="Duration (seconds)",
        description="How long the benchmark runs.",
        input_type="float",
        default=60.0,
        condition=_workload_is("multi-round-chat"),
        phase=PHASE_WORKLOAD,
    ),
    # ── Phase 3: prefix-suffix-tuner ──────────────────────────────────
    ConfigItem(
        key="psf_context_length",
        display_name="Context length (tokens)",
        description="Total tokens per request (prefix + breaker + suffix).",
        input_type="int",
        default=8000,
        condition=_workload_is("prefix-suffix-tuner"),
        phase=PHASE_WORKLOAD,
    ),
    ConfigItem(
        key="psf_prefix_ratio",
        display_name="Prefix ratio",
        description=(
            "Fraction of context-length used by the prefix. Must be in "
            "(0.0, 1.0). The remainder (minus a 32-token breaker) is the "
            "shared suffix."
        ),
        input_type="float",
        default=0.8,
        condition=_workload_is("prefix-suffix-tuner"),
        phase=PHASE_WORKLOAD,
    ),
    ConfigItem(
        key="psf_thrash",
        display_name="Target tier size (GB)",
        description=(
            "Size in GB of the KV-cache tier to overflow. The prefix pool "
            "is sized to slightly more than this, so every pass-2 request "
            "misses the targeted tier. Use the L0 (HBM) size for vanilla "
            "vLLM, or the L1 (LMCache DRAM) size for tiered baselines."
        ),
        input_type="float",
        default=20.0,
        condition=_workload_is("prefix-suffix-tuner"),
        phase=PHASE_WORKLOAD,
    ),
    # ── Phase 3: random-prefill ───────────────────────────────────────
    ConfigItem(
        key="rp_request_length",
        display_name="Request length (tokens)",
        description="Token length of each prefill request.",
        input_type="int",
        default=10000,
        condition=_workload_is("random-prefill"),
        phase=PHASE_WORKLOAD,
    ),
    ConfigItem(
        key="rp_num_requests",
        display_name="Number of requests",
        description="Total prefill requests to fire simultaneously.",
        input_type="int",
        default=50,
        condition=_workload_is("random-prefill"),
        phase=PHASE_WORKLOAD,
    ),
]


def get_items_by_phase(phase: int) -> list[ConfigItem]:
    """Return all items belonging to a given phase."""
    return [item for item in ALL_ITEMS if item.phase == phase]


def get_item(key: str) -> ConfigItem:
    """Look up a ConfigItem by key.  Raises KeyError if not found."""
    for item in ALL_ITEMS:
        if item.key == key:
            return item
    raise KeyError(f"No ConfigItem with key {key!r}")
