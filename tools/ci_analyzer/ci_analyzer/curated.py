# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Everything a human maintains in this analyzer lives here.

The map (which tests cover which code) is derived from the repo per run,
never written down. Written here are facts about the vLLM repo's mechanisms
and CI generator: things that change when the OUTSIDE world moves, not when
the analyzer's algorithms do. Each table lists its update trigger and guard;
every guard is a test that fails loudly when reality moves.

Sections:
  1. Hardware taxonomy
  2. Dynamic-import census
  3. Command vocabulary
  4. Generator replica facts
  5. Selection policy
  6. Parser anchors
  7. Detector thresholds
"""

from __future__ import annotations

# =========================================================================
# 1. HARDWARE TAXONOMY
# Update when: a new device type appears in the job YAML, a new hardware
# family lands, or a hardware-exclusive source namespace is created/removed.
# Guard: tests/test_hardware.py re-derives the device set from the job YAML.
# =========================================================================

# device string -> family, by prefix (covers sized variants: h200_35gb,
# b200-k8s, dgx-spark, mi300_4).
FAMILY_DEVICE_PREFIXES: dict[str, tuple[str, ...]] = {
    "cuda": ("h100", "h200", "a100", "b200", "gh200", "dgx"),
    "amd": ("mi",),
    # No tpu devices in the job YAML today; the prefix makes a future one
    # classify loudly (re-enabling tpu selection) instead of landing family-less.
    "tpu": ("tpu",),
}
FAMILY_DEVICE_EXACT: dict[str, frozenset[str]] = {
    "cpu": frozenset({"intel_cpu", "arm_cpu", "amd_cpu"}),
    "xpu": frozenset({"intel_gpu"}),
    "hpu": frozenset({"intel_hpu"}),
    "npu": frozenset({"ascend_npu"}),
}
# Build-runner sizes, not test hardware; deliberately family-less.
INFRA_DEVICES = frozenset({"cpu-small", "cpu-medium"})

# path token -> family, ADDITIVE tagging only (first match wins).
PATH_TOKEN_FAMILIES: tuple[tuple[frozenset[str], str], ...] = (
    (frozenset({"rocm", "aiter", "hip", "amd"}), "amd"),
    (frozenset({"xpu"}), "xpu"),
    (frozenset({"hpu"}), "hpu"),
    (frozenset({"npu", "ascend"}), "npu"),
    (frozenset({"tpu", "pallas"}), "tpu"),
    (frozenset({"cpu"}), "cpu"),
)

# String literals that name a platform/device, derived from the tables above.
# A guard comparing against one of these is platform dispatch, not config-key
# dispatch: routing tests by the literal would under-select the platform case,
# so the demotion pass refuses such guards (graph/dispatch.py).
PLATFORM_GUARD_LITERALS: frozenset[str] = (
    frozenset(FAMILY_DEVICE_PREFIXES)
    | frozenset(FAMILY_DEVICE_EXACT)
    | frozenset(t for tokens, _ in PATH_TOKEN_FAMILIES for t in tokens)
)

# The basename hardware-token heuristic ("rocm" in basename -> amd-exclusive)
# is a Python/shell source convention; foreign workspaces (rust/) name files
# freely, so the subtractive heuristic applies only to these extensions.
# An allowlist fails safe: a future workspace's files are exempt by default.
BASENAME_TOKEN_EXTENSIONS: tuple[str, ...] = (".py", ".sh")

# Namespaces whose files can only execute on one family (SUBTRACTIVE).
# Curated NAMESPACES only, not generic tokens: kv_offload/cpu/common.py runs
# inside CUDA jobs. NO entry for vllm/v1/worker/gpu*: CPU/XPU workers
# subclass the gpu worker via module-level imports.
EXCLUSIVE_NAMESPACES: tuple[tuple[tuple[str, ...], tuple[str, ...], str], ...] = (
    # (path prefixes, exact paths, allowed family)
    (
        ("vllm/v1/worker/cpu", "csrc/cpu/"),
        ("vllm/platforms/cpu.py", "vllm/platforms/zen_cpu.py"),
        "cpu",
    ),
    (("csrc/rocm/",), (), "amd"),
    # tpu lists an exact file, not a worker prefix: lora_model_runner_mixin.py
    # is not tpu-only but imports vllm/v1/worker/tpu_input_batch.py at module
    # level, so a prefix would wrongly mark that file tpu-only. A test guards this.
    ((), ("vllm/platforms/tpu.py",), "tpu"),
    (("vllm/v1/worker/xpu",), ("vllm/platforms/xpu.py",), "xpu"),
)

# Module-level cross-family imports that are provably runtime-guarded in
# source: (importer, namespace member) -> the guard, cited. Update when:
# the exclusivity oracle test reports a new importer -- either the import
# is guarded (add it here with the guard cited) or the member's exclusion
# is unsound (leave it; selection disables it at build time, fail-open).
# NO entry may be added without reading the importer's source. try/except
# ImportError is NOT a guard: the import still executes on every platform.
EXCLUSIVE_IMPORT_EXCEPTIONS: dict[tuple[str, str], str] = {
    (
        "vllm/compilation/passes/pass_manager.py",
        "vllm/compilation/passes/fusion/rocm_aiter_fusion.py",
    ): (
        "inside `if rocm_aiter_ops.is_enabled() "
        "or rocm_aiter_ops.is_rdna_aiter_enabled():`"
    ),
    (
        "tests/lora/test_worker.py",
        "vllm/v1/worker/xpu_worker.py",
    ): "inside `if current_platform.is_xpu():` (test_worker.py:24)",
}

# =========================================================================
# 2. DYNAMIC-IMPORT CENSUS
# Update when: dynamic-sites reports an UNCLASSIFIED site (new dynamism
# entered the repo): classify it here or teach a parser.
# Guard: tests/test_dynamic.py asserts zero unclassified at HEAD.
# =========================================================================

# EXACT paths only: a directory blanket would auto-bless new dynamic imports
# appearing under it, which is what this census exists to prevent. A moved
# file or a new site anywhere goes UNCLASSIFIED loudly.
#
# The two lists are checked differently. DYNAMIC_IMPORT_FILES are dynamic by
# nature (qualnames from user config, entry points, wire data,
# trust_remote_code), so each must still HAVE a live site: one whose site
# vanished would silently pre-approve the next. WALL_PARSER_FILES hold a
# literal table a parser reads and have no site at all, so only their
# existence is checked.
DYNAMIC_IMPORT_FILES = (
    "vllm/v1/serial_utils.py",  # msgpack wire (mod, name)
    "vllm/v1/executor/abstract.py",  # distributed_executor_backend qualname
    "vllm/v1/worker/worker_base.py",  # worker_cls / worker_extension_cls
    "vllm/config/compilation.py",  # config qualname fields
    "vllm/config/ec_manager_config.py",
    "vllm/config/scheduler.py",
    "vllm/entrypoints/openai/api_server.py",  # --middleware qualname
    "vllm/entrypoints/openai/engine/protocol.py",  # logits proc qualname
    "vllm/compilation/backends.py",  # inductor pass / backend qualnames
    "vllm/compilation/decorators.py",
    "vllm/plugins/io_processors/__init__.py",  # entry-point groups
    "vllm/platforms/__init__.py",  # platform plugin entry points
    "vllm/v1/sample/logits_processor/__init__.py",  # module:qualname plugins
    "vllm/v1/spec_decode/custom_class_proposer.py",
    "vllm/model_executor/layers/pooler/activations.py",  # HF config attr
    "vllm/model_executor/models/llava_onevision2.py",  # trust_remote_code
    "vllm/model_executor/models/kanana_v.py",  # trust_remote_code
    "vllm/transformers_utils/processor.py",  # transformers_modules
    "vllm/distributed/nixl_utils.py",  # external lib, computed name
    "vllm/distributed/kv_transfer/kv_connector/factory.py",  # module_path override
    "vllm/distributed/ec_transfer/ec_connector/factory.py",
    "vllm/v1/kv_offload/factory.py",
    "vllm/tool_parsers/abstract_tool_parser.py",  # import_from_path plugin
    "vllm/reasoning/abs_reasoning_parsers.py",
    "vllm/utils/import_utils.py",  # the shared resolver itself
    "vllm/utils/flashinfer.py",  # external lib submodules
    "vllm/v1/attention/ops/rocm_aiter_mla_sparse.py",  # find_spec picks
    "tests/transformers_utils/test_processor.py",  # trust_remote_code
    "tests/utils.py",  # test-local qualname helper
    "tests/v1/kv_connector/nixl_integration/test_nixl_imports.py",
)

WALL_PARSER_FILES = (
    "vllm/__init__.py",  # MODULE_ATTRS
    "vllm/model_executor/models/registry.py",
    "vllm/renderers/registry.py",
    "vllm/tokenizers/registry.py",
    "vllm/transformers_utils/configs/__init__.py",  # _CLASS_TO_MODULE (parsed)
    "vllm/transformers_utils/processors/__init__.py",  # _CLASS_TO_MODULE (parsed)
    "vllm/utils/humming.py",  # _EXPORTS
    "vllm/models/inkling/amd/ops/__init__.py",  # _LAZY_EXPORTS
    "vllm/models/inkling/nvidia/ops/__init__.py",  # _LAZY_EXPORTS
    "vllm/kernels/helion/ops/__init__.py",  # pkgutil enumerator (parsed)
    "vllm/model_executor/layers/quantization/__init__.py",
    "vllm/v1/attention/backends/registry.py",  # backend enum (parsed)
    "vllm/v1/attention/backends/mla/prefill/registry.py",
    "vllm/v1/attention/selector.py",  # resolves enum qualnames
    "vllm/distributed/weight_transfer/factory.py",
    "vllm/device_allocator/sleep_mode_backend.py",
    "vllm/model_executor/model_loader/modelexpress_loader.py",
    # platform-method consumers: resolve() fed by traced get_* literals
    "vllm/distributed/parallel_state.py",
    "vllm/distributed/stateless_coordinator.py",
    "vllm/lora/punica_wrapper/punica_selector.py",
    "vllm/v1/kv_offload/tiering/factory.py",  # register_tier literal table
    "vllm/v1/kv_offload/cpu/policies/factory.py",  # register_cache_policy table
)

AUDITED_DYNAMIC_FILES = DYNAMIC_IMPORT_FILES + WALL_PARSER_FILES

# =========================================================================
# 3. COMMAND VOCABULARY
# Update when: the UNPARSABLE test fails on a new benign command, or a new
# pytest plugin flag that consumes a value appears in job commands.
# Guard: tests/test_testmap.py::test_unparsable_empty_at_head.
# =========================================================================

# Commands that never invoke tests. Anything not matched, not a test shape,
# and not a script invocation is UNPARSABLE by design.
BENIGN_CMDS = {
    "export",
    "set",
    "echo",
    "cd",
    "mkdir",
    "rm",
    "cp",
    "mv",
    "ln",
    "chmod",
    "cat",
    "ls",
    "which",
    "env",
    "true",
    "sleep",
    "nproc",
    "free",
    "df",
    "pip",
    "pip3",
    "uv",
    "apt",
    "apt-get",
    "dpkg",
    "yum",
    "nvidia-smi",
    "amd-smi",
    "rocm-smi",
    "rocminfo",
    "xpu-smi",
    "npu-smi",
    "docker",
    "git",
    "wget",
    "curl",
    "tar",
    "unzip",
    "ray",
    "hf",
    "huggingface-cli",
    "aws",
    "buildkite-agent",
    "sudo",
    "pkill",
    "kill",
    "sccache",
    "ccache",
    "pip-compile",
    "printenv",
    "exit",
    "return",
    "wait",
    "trap",
    "source",
    ".",
}

# Commands that wrap another: drop the wrapper and parse the rest, so
# `uv run pytest ...` still finds its targets. Value = the subcommand that
# must follow (None = none). docker is left out because its flags and image
# name cannot be skipped reliably: a docker line naming pytest goes UNPARSABLE.
UNWRAP_CMDS = {"uv": "run", "sudo": None}
# Wrapper flags that consume the next token as a value.
UNWRAP_VALUE_FLAGS = {
    "--with",
    "--python",
    "-p",
    "--group",
    "--extra",
    "-u",
    "--directory",
    "--project",
}
# Wrapper flags safe to skip bare; an unknown bare flag stops unwrapping
# and the remainder falls through to UNPARSABLE (loud).
UNWRAP_BARE_FLAGS = {
    "-E",
    "-H",
    "-n",
    "-q",
    "--quiet",
    "--frozen",
    "--offline",
    "--no-sync",
    "--isolated",
    "--active",
    "--no-project",
}

# pytest flags that consume the following token as a value (space form).
PYTEST_VALUE_FLAGS = {
    "-m",
    "-k",
    "-n",
    "-p",
    "-W",
    "-o",
    "-c",
    "--ignore",
    "--deselect",
    "--config-list-file",
    "--shard-id",
    "--num-shards",
    "--timeout",
    "--durations",
    "--dist",
    "--max-worker-restart",
    "--forked-timeout",
    "--models",  # rust_frontend.yaml's custom model-filter plugin flag
}

# =========================================================================
# 4. GENERATOR REPLICA FACTS
# Facts about vllm-project/ci-infra's pipeline generator. Update when:
# ci-infra's generator semantics change.
# Guards: unknown-step-fields test.
# =========================================================================

# step.py read_steps_from_job_dir: applies only under truthy group depends_on
DEFAULT_WORKING_DIR = "/vllm-workspace/tests"

# ci-infra amd.py, verbatim.
AMD_ALWAYS_RUN_STEP_KEYS = frozenset({"ensure-ci-base-amd", "refresh-rocm-base-amd"})
AMD_NATIVE_RUNTIME_SOURCE_DEPENDENCIES = (
    ".buildkite/scripts/hardware_ci/run-amd-test.sh",
)

# Fields we understand on a step; anything else lands in Step.extra and
# fails the unknown-fields test so new generator features surface loudly.
KNOWN_STEP_FIELDS = {
    "label",
    "key",
    "commands",
    "command",
    "source_file_dependencies",
    "device",
    "gpu",
    "num_devices",
    "num_gpus",
    "num_nodes",
    "working_dir",
    "timeout_in_minutes",
    "optional",
    "soft_fail",
    "autorun_on_main",
    "no_plugin",
    "parallelism",
    "depends_on",
    "env",
    "mirror",
    "retry",
    "agent_tags",
    "dind",
    "no_gpu",
    "agent_pool",
    "concurrency",
    "concurrency_group",
}

# mirror.<hw> fields the generator honors (ci-infra buildkite_step.py).
MIRROR_OVERRIDABLE = {
    "device",
    "timeout_in_minutes",
    "depends_on",
    "dind",
    "source_file_dependencies",
    "commands",
    "working_dir",
    "env",
    "optional",
    "soft_fail",
    "no_plugin",
    "no_gpu",
    "num_devices",
    "num_gpus",
    "num_nodes",
    "agent_tags",
}

# =========================================================================
# 5. SELECTION POLICY
# Update when: a deliberate policy decision changes (with the team).
# Guard: policy divergences from today's run_all are stated in output.
# =========================================================================

# Files the analyzer runs everything for even though the repo's own
# run_all_patterns do not list them.
# The extra breadth is not unconditional: for a structured config file,
# worldtables.py reads WHICH tables the diff touched and drops the policy half
# when every one of them configures a tool no live step runs. What remains is
# the repo's own run_all match, so a narrowed selection is never below CI.
EXTRA_WORLD_FILES = ("pyproject.toml",)

# Dep prefixes treated as catch-all noise when the harnesses triage
# ran-but-not-selected jobs (today's broadest hand-list entries).
CATCH_ALL_DEP_PREFIXES = frozenset({"vllm", "tests"})

# CI files no live pipeline reads, so an edit selects nothing. test-amd.yaml
# is in no ci_config job_dir (ci-infra's legacy Jinja pipeline consumes it);
# calling it unknown infra ran the whole pipeline on AMD housekeeping PRs.
# Guard: the uninvoked report calls its exclusive coverage "legacy-only" and
# fails if the file rejoins a job_dir. Rejoining is safe anyway: the live-step
# rules run first, so the file's real steps claim it.
LEGACY_CI_FILES = (".buildkite/test-amd.yaml",)

# CI trees no live pipeline consumes, so an edit selects nothing.
# Do NOT derive this as "yaml in no live job_dir": 31 lm-eval-harness configs fit
# that shape, but a live step does read them, through a manifest (lm_eval.yaml
# passes --config-list-file, and the .txt it names lists the yamls). They fail
# open to run-all today; a complement rule would give zero jobs to the file that
# changes an accuracy threshold.
# Update when: the inert-tree oracle test fails (a live step now reaches into
# the tree) -- DELETE the entry, the referencing rule takes over.
INERT_CI_PREFIXES = (
    ".buildkite/performance-benchmarks/",  # external nightly perf pipeline only
    ".buildkite/test-pipeline.yaml",  # deprecated stub (its own header says so)
    ".buildkite/amd-disagg/",  # native SLURM pipeline, in no job_dir
)

# The standalone release/nightly publish pipeline. Buildkite runs it directly
# rather than through the generator configs we model, so it is in no ci_config
# job_dir. The scripts it references are re-derived every build (externals.py);
# one a live step also uses is claimed by the live-step rules first.
# Listed by hand, not globbed from unmodeled .buildkite/*.yaml, so a pipeline
# that is live but not yet modeled fails open to run-all instead of selecting
# nothing. Update when: a new non-modeled publish pipeline yaml appears.
RELEASE_PIPELINE_FILES = (".buildkite/release-pipeline.yaml",)

# =========================================================================
# 6. PARSER ANCHORS
# Where vLLM keeps the tables the wall parsers read. Update when: a table
# file moves. Guard: each parser's oracle test (entry counts re-derived
# from the repo) fails on a dead anchor.
# =========================================================================

REGISTRY_FILE = "vllm/model_executor/models/registry.py"
TEST_REGISTRY_FILE = "tests/models/registry.py"
MODEL_MODULE_PREFIX = "vllm.model_executor.models"
QUANT_INIT_FILE = "vllm/model_executor/layers/quantization/__init__.py"
REASONING_INIT = "vllm/reasoning/__init__.py"
TOOL_PARSER_INIT = "vllm/tool_parsers/__init__.py"
# _CLASS_TO_MODULE tables (class name -> full module qualname). The consumer
# is an HF checkpoint picking by model_type, which we cannot see, so entries
# carry claims and self-materializing leaf edges, not scoped routing: the
# overshoot is deliberate.
TRANSFORMERS_CONFIGS_INIT = "vllm/transformers_utils/configs/__init__.py"
TRANSFORMERS_PROCESSORS_INIT = "vllm/transformers_utils/processors/__init__.py"
ATTN_REGISTRY = "vllm/v1/attention/backends/registry.py"
VLLM_INIT = "vllm/__init__.py"
PLATFORMS_DIR = "vllm/platforms"
CLI_ENTRYPOINT_MODULE = "vllm.entrypoints.cli.main"
ENGINE_ENTRY_MODULES = (
    "vllm.entrypoints.llm",
    "vllm.entrypoints.cli.main",
    "vllm.entrypoints.openai.api_server",
    "vllm.v1.engine.llm_engine",
    "vllm.v1.engine.async_llm",
    "vllm.v1.engine.core",
)
PACKAGE_ROOTS = ("vllm", "tests", "benchmarks")
SKIP_DIRS = {"__pycache__", ".git", "node_modules", ".venv", "build"}
# Conftests that import an engine entrypoint but must NOT mark their whole
# directory as engine-starting. tests/conftest.py does `from vllm import LLM`,
# so counting it would make every test look like it boots an engine, and the
# checks that turn on that distinction would stop discriminating. Real boots
# are still caught per-test through the vllm_runner fixture argument.
# Update when: a second repo-wide conftest appears (its oracle test fails) or
# the root conftest stops defining vllm_runner.
CONFTESTS_NOT_ENGINE_STARTING = ("tests/conftest.py",)

# =========================================================================
# 7. DETECTOR THRESHOLDS
# Measured judgment knobs (risk-tolerance), not derivable. Update when: the
# measured gap they sit in moves (re-measure, don't guess).
# =========================================================================

# keys.py registered-key matching. A substring key shorter than this risks
# matching inside unrelated words; raw table-diff archs are 8-11 chars, so the
# raw bar stays at 8 (raising it drops them -> under-selection).
SUBSTRING_KEY_MIN_LEN = 12
RAW_KEY_MIN_LEN = 8

# dispatch.py demotion routing. Demotion cuts a plugin's import edge, so its
# coverage is restored by routing on the config word that enables it: a test
# mentioning "eagle3" links to the eagle3 plugin. Ordinary words break that --
# "auto" appears in 170 test files, so routing on it links noise. A word in
# more files than this counts as ordinary. ("Files" are tests plus the
# examples/ and benchmarks/ scripts that steps run directly.)
# Why 32: the counts are bimodal, kept words top out at 23 and the rest start
# at 41, so anything in 24..40 behaves identically.
# It drops real config keys too ("pooling" is registered, at 76). Safe, because
# it only filters which words add routing edges: the import demotes either way,
# the key index is untouched, and the member still routes by its own filename
# stem and parent dir. Too high over-selects, too low under-selects.
CONFIG_KEY_MAX_TEST_FILES = 32
