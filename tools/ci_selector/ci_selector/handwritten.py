# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Facts about the world outside this tool, which moves on its own schedule.

Two kinds live here: what vllm-project/ci-infra's pipeline generator does, and
which dynamic imports leave the repo. Neither is something the tool can read for
itself. Facts about vLLM's own tree do NOT belong here, because the tool already
reads that tree; they sit beside the code that reads them, and stay central only
when two modules share one.

  1. Devices and hardware families   5. Our own selection decisions
  2. Imports that load outside vLLM  6. Where vLLM keeps the tables we read
  3. Commands in job scripts         7. Coverage recorder
  4. What ci-infra's generator does

Each section says when to update it, and every constant that can go stale is
watched by a drift-marked test (`pytest tests -m drift -q`), all of which run
offline. The ci-infra values are compared against a downloaded copy of that
generator under `tests/ci_infra_snapshot/`, and the functions we reproduce are
run against it directly; `pytest tests --sync` refreshes the copy and is the
only thing here that needs network.
"""

from __future__ import annotations

# =========================================================================
# 1. DEVICES AND HARDWARE FAMILIES
# Update when: a new device appears in the job yaml, or a hardware-exclusive
# namespace is added or removed.
# Guard: drift tests re-derive the device set from the yaml, check every
# exclusive namespace still matches a file, and cross-check that each device
# family has path tokens.
# =========================================================================

# device string -> family, by prefix (covers sized variants: h200_35gb,
# b200-k8s, dgx-spark, mi300_4).
FAMILY_DEVICE_PREFIXES: dict[str, tuple[str, ...]] = {
    "cuda": ("h100", "h200", "a100", "b200", "gh200", "dgx"),
    "amd": ("mi",),
    "tpu": ("tpu",),
}
FAMILY_DEVICE_EXACT: dict[str, frozenset[str]] = {
    # Exact, not a prefix: "l4" would also match filename tokens.
    "cuda": frozenset({"l4"}),
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

# Every literal naming a platform or device, from the three tables above. A
# guard comparing against one of these is platform dispatch, not config-key
# dispatch, and routing tests by the literal would under-select. demote.py
# refuses such guards.
PLATFORM_GUARD_LITERALS: frozenset[str] = (
    frozenset(FAMILY_DEVICE_PREFIXES)
    | frozenset(FAMILY_DEVICE_EXACT)
    | frozenset(t for tokens, _ in PATH_TOKEN_FAMILIES for t in tokens)
)

# Namespaces whose files can only execute on one family (SUBTRACTIVE).
# Curated NAMESPACES only, not generic tokens: kv_offload/cpu/common.py runs
# inside CUDA jobs. NO entry for vllm/v1/worker/gpu*: CPU/XPU workers
# subclass the gpu worker via module-level imports.
EXCLUSIVE_NAMESPACES: tuple[tuple[tuple[str, ...], tuple[str, ...], str], ...] = (
    (
        ("vllm/v1/worker/cpu", "csrc/cpu/"),
        ("vllm/platforms/cpu.py", "vllm/platforms/zen_cpu.py"),
        "cpu",
    ),
    (("csrc/rocm/",), (), "amd"),
    ((), ("vllm/platforms/tpu.py",), "tpu"),
    (("vllm/v1/worker/xpu",), ("vllm/platforms/xpu.py",), "xpu"),
)

# Cross-family imports at module level that a runtime check really does guard:
# (importer, member) -> the guard, quoted. Never add one without reading the
# importer. try/except ImportError does NOT count, because the import still
# runs on every platform. If an importer is not guarded, leave it out and
# selection disables that exclusion by itself.
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
# 2. IMPORTS THAT LOAD SOMETHING OUTSIDE VLLM
# Update when: a drift test reports an UNCLASSIFIED site. Either add the file
# here, or teach a parser to read the table behind the import.
# Guard: drift tests, both directions -- a new site and a dead entry.
# =========================================================================

# Files whose dynamic import lands outside the tree: qualnames from user
# config, entry points, wire data, trust_remote_code, and two targets that are
# built rather than checked in. A checkout cannot tell you where an import
# lands, which is why this stays typed. The other half of the question, "does
# a parser already read this file's table", IS derived (graph.table_files).
#
# Exact paths only. A directory blanket would bless new imports appearing under
# it, which is what this exists to stop. An entry whose import disappears is
# just as bad: it pre-approves whatever lands in that file next.
DYNAMIC_IMPORT_FILES = (
    "vllm/v1/serial_utils.py",  # msgpack wire (mod, name)
    "vllm/v1/executor/abstract.py",  # distributed_executor_backend qualname
    "vllm/v1/worker/worker_base.py",  # worker_cls / worker_extension_cls
    "vllm/config/compilation.py",  # config qualname fields
    "vllm/config/ec_manager_config.py",
    "vllm/config/scheduler.py",
    "vllm/entrypoints/serve/middleware/register.py",  # --middleware qualname
    "vllm/entrypoints/generate/base/protocol.py",  # logits proc qualname
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
    "vllm/tool_parsers/rust_tool_parser.py",
    "vllm/utils/deep_gemm.py",
    "vllm/utils/flashinfer.py",  # external lib submodules
    "vllm/utils/b12x.py",
    "vllm/v1/attention/ops/rocm_aiter_mla_sparse.py",  # find_spec picks
    "tests/transformers_utils/test_processor.py",  # trust_remote_code
    "tests/utils.py",  # test-local qualname helper
    "tests/v1/kv_connector/nixl_integration/test_nixl_imports.py",
)


# =========================================================================
# 3. COMMANDS IN JOB SCRIPTS
# Update when: a drift test reports a command shape the target parser does not
# understand.
# Guard: that same test, which fails on anything left unparsed.
# =========================================================================

# Commands that never run tests. Anything left over that is not a test shape or
# a script call is UNPARSABLE, which is deliberate and loud.
BENIGN_CMDS = {
    "export",
    "set",
    "unset",
    "echo",
    "cd",
    "mkdir",
    "rm",
    "cp",
    "mv",
    "ln",
    "chmod",
    "cat",
    "sed",
    "ls",
    "which",
    "env",
    "true",
    "test",
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
# `uv run pytest ...` still finds its targets. Value is the subcommand that must
# follow, None if there is none. docker is left out on purpose, because its
# flags and image name cannot be skipped reliably.
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
# Wrapper flags safe to skip. An unknown one stops unwrapping and the rest goes
# to UNPARSABLE rather than being guessed at.
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
# 4. WHAT CI-INFRA'S GENERATOR DOES
# What ci-infra's pipeline generator does. Every entry is a hand copy, so it
# can only be checked against that repo. Update when: it changes.
# Guard: the values down to ONLY_STEP_KEYS_ENV are compared against what
# ci-infra's generator actually assigns, offline, using the copy
# `pytest tests --sync` downloads. The two field sets have a second guard as
# well: they fail when vLLM's yaml uses a name we do not model. Everything
# below ONLY_STEP_KEYS_ENV is a vLLM-tree or own-policy fact, not a generator
# copy, and carries its own Guard: line.
# =========================================================================

# Where the generator mounts the checkout inside the test container. Step
# working_dirs are absolute paths under it, mapped back to repo-relative.
CONTAINER_WORKSPACE = "/vllm-workspace"

# step.py read_steps_from_job_dir: applies only under truthy group depends_on
DEFAULT_WORKING_DIR = f"{CONTAINER_WORKSPACE}/tests"

# Keys starting with this always run: they build the images everything else
# depends on. The AMD ones have no shared prefix, so they are listed below.
IMAGE_BUILD_KEY_PREFIX = "image-build"

# What a mirror step depends on when its override declares nothing.
MIRROR_DEFAULT_DEPENDS_ON = ("image-build-amd",)

# ci-infra amd.py, verbatim.
AMD_ALWAYS_RUN_STEP_KEYS = frozenset({"ensure-ci-base-amd", "refresh-rocm-base-amd"})
AMD_NATIVE_RUNTIME_SOURCE_DEPENDENCIES = (
    ".buildkite/scripts/hardware_ci/run-amd-test.sh",
)

# Step fields we model. Anything else lands in Step.extra, fails a drift test,
# and is force-selected at run time, so a new generator feature cannot pass
# unnoticed.
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
    # Buildkite built-in, absent from the generator's Step model, so it is
    # dropped before reaching a command. Governs ordering after a failure.
    "allow_dependency_failure",
    # Buildkite built-in, also absent from the generator's Step model, so it
    # never reaches the emitted step.
    "if",
    # In the generator's Step model but unused by vLLM's yaml, which sets the
    # group at file level. Both are display or plugin wiring; listed so the
    # first step to use one is not forced for it.
    "group",
    "mount_buildkite_agent",
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
    "concurrency",
    "concurrency_group",
    # Declared per mirror and used verbatim as the step's Buildkite label, so
    # it is the spelling real job contexts carry.
    "label",
    # Listed, never honored: the generator passes the parent's parallelism, so
    # a mirror asking for its own is silently ignored upstream.
    "parallelism",
}

# The env var the generator reads. Absent means "your rules"; present means
# "exactly these, plus what they depend on".
ONLY_STEP_KEYS_ENV = "VLLM_CI_ONLY_STEP_KEYS"

# The opt-in gates for the rust binary: vllm-rs never launches unless one is
# set, so a step that exports one runs rust code.
# Update when: vllm/envs.py renames the gates or grows a third.
# Guard: tests/test_handwritten.py::test_rust_gate_env_vars_exist.
RUST_GATE_ENV_VARS = ("VLLM_USE_RUST_FRONTEND", "VLLM_USE_RUST_BENCH")

# The two shipped rust artifacts' crate roots, in (binary, cdylib) order.
# Update when: build_rust.py adds, drops, or moves an artifact.
# Guard: tests/test_handwritten.py::test_build_rust_still_builds_exactly_two_artifacts.
RUST_ARTIFACT_ROOTS = ("rust/src/cmd", "rust/src/parser/python")

# The single vllm/ import site of the PyO3 cdylib. Cdylib rust files borrow
# this file's whole claim, since that parser is NOT opt-in behind an env gate.
# Update when: a second file imports vllm._rust_tool_parser.
# Guard: tests/test_handwritten.py::test_pyo3_bridge_file_is_the_only_import_site.
RUST_PYO3_BRIDGE_FILE = "vllm/tool_parsers/rust_tool_parser.py"

# Build-defining rust files outside rust/. A change to any can change both
# artifacts, so they take the widest rust bucket.
# Update when: the rust build entry points move.
# Guard: tests/test_handwritten.py::test_rust_toolchain_files_exist.
RUST_TOOLCHAIN_FILES = ("rust-toolchain.toml", "build_rust.sh", "tools/build_rust.py")

# Family tokens that hold only under requirements/, where the filename names
# the device. `cuda` cannot join PATH_TOKEN_FAMILIES: it would misfire on vllm/
# paths and on the generic Dockerfile, which must stay family-less.
# Update when: a requirements file appears for a family with no global token.
# Guard: tests/test_classify.py::test_requirements_family_map_pins_the_tree.
REQUIREMENTS_EXTRA_TOKEN_FAMILIES = ((frozenset({"cuda"}), "cuda"),)

# Requirements files that exist only for tooling no test imports. Their honest
# reach is the steps that declare them plus the always-run builds, so they skip
# the docker-image widening.
# Update when: a new tooling-only requirements file appears.
# Guard: tests/test_classify.py::test_requirements_build_validated_members_exist.
REQUIREMENTS_BUILD_VALIDATED = frozenset(
    {"requirements/lint.txt", "requirements/dev.txt"}
)

# The pipeline we emit keys for, which is why the variable above governs it and
# why the crosscheck scores it. AMD coverage arrives as mirror steps inside it.
#
# NOT the only pipeline a PR can trigger: a `rocm` or `intel-gpu` label also
# fires a mirror pipeline. Those are independent builds that subscribe to the
# repo's webhooks themselves, and the key variable is set per build on one named
# pipeline, so a mirror build never sees ours. We scope here because it is the
# only pipeline we emit for, not because the others never run.
# Update when: vLLM renames its PR pipeline in .buildkite/ci_config.yaml.
# Guard: tests/test_buildkite.py::test_pr_pipeline_is_what_the_pr_config_calls_itself.
PR_PIPELINE = "vllm_ci"

# =========================================================================
# 5. OUR OWN SELECTION DECISIONS
# Mostly our own decisions, which cannot drift, only be re-decided. Update
# when: the team decides differently. Every escalation here is named as
# analyzer policy in the output.
# Two exceptions are outside facts and are guarded as such: DOCS_ONLY_* is
# copied from ci-infra and compared against its generator, and the CI-file
# lists are checked against the live tree by drift tests.
# =========================================================================

# Dep prefixes so broad they say nothing, treated as noise when the harnesses
# triage a job that ran but was not selected.
CATCH_ALL_DEP_PREFIXES = frozenset({"vllm", "tests"})

# CI files no live pipeline reads, so editing one selects nothing. test-amd.yaml
# sits in no job_dir because ci-infra's legacy Jinja pipeline consumes it, and
# treating it as unknown infra used to run everything on AMD housekeeping PRs.
# If it rejoins a job_dir a drift test catches it.
LEGACY_CI_FILES = (".buildkite/test-amd.yaml",)

# CI trees no live pipeline consumes. Do NOT try to derive this as "yaml in no
# job_dir": the lm-eval-harness configs fit that shape and a live step does read
# them, through a manifest. Deriving it would give zero jobs to the file that
# changes an accuracy threshold.
# Update when: a drift test says a live step now reaches into the tree. Delete
# the entry and the ordinary reference rule takes over.
INERT_CI_PREFIXES = (
    ".buildkite/performance-benchmarks/",  # external nightly perf pipeline only
    ".buildkite/test-pipeline.yaml",  # deprecated stub (its own header says so)
    ".buildkite/amd-disagg/",  # native SLURM pipeline, in no job_dir
)

# The standalone release pipeline. Buildkite runs it directly, not through the
# generator configs we model, so it sits in no job_dir. Listed by hand rather
# than globbed, so a pipeline that is live but not yet modeled runs everything
# instead of selecting nothing.
RELEASE_PIPELINE_FILES = (".buildkite/release-pipeline.yaml",)

# ci-infra's is_docs_only_change, verbatim. See the warning above before
# merging these with NO_CODE_*.
DOCS_ONLY_PREFIXES = ("docs/",)
DOCS_ONLY_SUFFIXES = (".md",)
DOCS_ONLY_EXACT = ("mkdocs.yaml",)

# Config the docs build itself reads: an edit here affects every rendered page,
# so the docs job is selected without tracing a single reference.
DOCS_INFRA_FILES = frozenset(
    {
        "mkdocs.yaml",
        ".readthedocs.yaml",
        "requirements/docs.in",
        "requirements/docs.txt",
        "requirements/test/cuda.txt",
    }
)
# Trees where every file is a docs dependency, reference-tracing or not.
DOCS_FLOOR_PREFIXES = ("docs/", "examples/")

# =========================================================================
# 6. WHERE VLLM KEEPS THE TABLES WE READ
# Where vLLM keeps the tables the parsers read, and the names they search for.
# Only the ones two modules share are here; the rest sit with their parser.
# Update when: a table moves or is renamed.
# Guard: a dead anchor parses zero entries, and preflight escalates on that.
# =========================================================================

REGISTRY_FILE = "vllm/model_executor/models/registry.py"
TEST_REGISTRY_FILE = "tests/models/registry.py"
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
# Conftests that import an engine entrypoint but must NOT mark everything below
# them as engine-starting. tests/conftest.py does `from vllm import LLM`, so
# counting it would make the whole suite look like it boots an engine and the
# gate would stop discriminating. Real boots are still caught per test, through
# the vllm_runner fixture.
CONFTESTS_NOT_ENGINE_STARTING = ("tests/conftest.py",)

# Table names read out of vLLM source by the edge parsers.
MODEL_REGISTRY_DICTS = (
    "_VLLM_MODELS",
    "_PREVIOUSLY_SUPPORTED_MODELS",
    "_OOT_SUPPORTED_MODELS",
)
TEST_REGISTRY_CALL = "_HfExamplesInfo"


# =========================================================================
# 7. COVERAGE RECORDER
# Update when: Buildkite's job schema changes.
# The GitHub names and job-metadata fields moved to their single readers,
# crosscheck.py and coverage/model.py.
# =========================================================================

# The recorder only sees the installed vllm package, so no row can ever name a
# file outside this prefix. That is the recorder's behaviour, not our policy,
# and it is why a changed file outside it can never be dropped on.
RECORDER_SCOPE = "vllm/"
