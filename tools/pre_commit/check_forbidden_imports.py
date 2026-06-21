#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import sys
from dataclasses import dataclass, field

import regex as re


@dataclass
class ForbiddenImport:
    pattern: str
    tip: str
    allowed_pattern: re.Pattern = re.compile(r"^$")  # matches nothing by default
    allowed_files: set[str] = field(default_factory=set)
    applies_to: re.Pattern | None = None


CHECK_IMPORTS = {
    "pickle/cloudpickle": ForbiddenImport(
        pattern=(
            r"^\s*(import\s+(pickle|cloudpickle)(\s|$|\sas)"
            r"|from\s+(pickle|cloudpickle)\s+import\b)"
        ),
        tip=(
            "Avoid using pickle or cloudpickle or add this file to "
            "tools/pre_commit/check_forbidden_imports.py."
        ),
        allowed_files={
            # pickle
            "vllm/multimodal/hasher.py",
            "vllm/transformers_utils/config.py",
            "vllm/model_executor/models/registry.py",
            "vllm/compilation/caching.py",
            "vllm/env_override.py",
            "vllm/compilation/piecewise_backend.py",
            "vllm/distributed/utils.py",
            "vllm/distributed/parallel_state.py",
            "vllm/distributed/device_communicators/all_reduce_utils.py",
            "vllm/distributed/device_communicators/shm_broadcast.py",
            "vllm/distributed/device_communicators/shm_object_storage.py",
            "vllm/distributed/weight_transfer/ipc_engine.py",
            "tests/distributed/test_weight_transfer.py",
            "vllm/utils/hashing.py",
            "tests/multimodal/media/test_base.py",
            "tests/tokenizers_/test_hf.py",
            "tests/utils_/test_hashing.py",
            "tests/compile/test_aot_compile.py",
            "benchmarks/kernels/graph_machete_bench.py",
            "benchmarks/kernels/benchmark_lora.py",
            "benchmarks/kernels/benchmark_machete.py",
            "benchmarks/fused_kernels/layernorm_rms_benchmarks.py",
            "benchmarks/cutlass_benchmarks/w8a8_benchmarks.py",
            "benchmarks/cutlass_benchmarks/sparse_benchmarks.py",
            # cloudpickle
            "vllm/v1/executor/multiproc_executor.py",
            "vllm/v1/executor/ray_executor.py",
            "vllm/entrypoints/llm.py",
            "tests/utils.py",
            # pickle and cloudpickle
            "vllm/v1/serial_utils.py",
        },
    ),
    "base64": ForbiddenImport(
        pattern=r"^\s*(?:import\s+base64(?:$|\s|,)|from\s+base64\s+import)",
        tip=(
            "Replace 'import base64' with 'import pybase64' "
            "or 'import pybase64 as base64'."
        ),
        allowed_pattern=re.compile(r"^\s*import\s+pybase64(\s*|\s+as\s+base64\s*)$"),
    ),
    "re": ForbiddenImport(
        pattern=r"^\s*(?:import\s+re(?:$|\s|,)|from\s+re\s+import)",
        tip="Replace 'import re' with 'import regex as re' or 'import regex'.",
        allowed_pattern=re.compile(r"^\s*import\s+regex(\s*|\s+as\s+re\s*)$"),
        allowed_files={"setup.py"},
    ),
    "triton": ForbiddenImport(
        pattern=r"^(from|import)\s+triton(\s|\.|$)",
        tip="Use 'from vllm.triton_utils import triton' instead.",
        allowed_pattern=re.compile(
            "from vllm.triton_utils import (triton|tl|tl, triton)"
        ),
        allowed_files={"vllm/triton_utils/importing.py"},
    ),
    "hw_agnostic_isolation": ForbiddenImport(
        pattern=(
            r"^\s*from\s+vllm\."
            r"(?:"
            r"model_executor\.layers(?!\.utils\b)(?!\.quantization\.base_config\b)(?!\.quantization\.utils\.fp8_utils\b)(?!\.quantization\.utils\.quant_utils\b)(?!\.quantization\.utils\.w8a8_utils\b)(?!\.quantization\.utils\.layer_utils\b)(?!\.quantization\.input_quant_fp8\b)(?!\.quantization\.compressed_tensors\.triton_scaled_mm\b)(?!\.fusion\.quant_activation\b)(?!\.fused_moe\.activation\b)(?!\.fused_moe\.fused_moe_method_base\b)(?!\.fused_moe\.modular_kernel\b)(?!\.fused_moe\.runner\.moe_runner_interface\b)(?!\.sparse_attn_indexer\b)(?!\.attention_layer_base\b)"
            r"|model_executor\.kernels\b"
            r"|model_executor\.models(?!\.utils\b)"
            r"|models\.[^.]+(?!\.hw_agnostic\b)"
            r"|v1\.attention\.backends(?!\.utils\b)(?!\.mla\.indexer\b)"
            r")"
            r"(?:\.|\s+import\b)"
        ),
        tip=(
            "hardware-agnostic modelling code must not import from "
            "non-hardware=angnostic parts. The only exceptions are general "
            "utils.py files such as vllm.model_executor.layers.utils.py."
        ),
        applies_to=re.compile(r"^vllm/models/[^/]+/hw_agnostic/.*\.py$"),
        allowed_files={
            "vllm/models/deepseek_v4/hw_agnostic/quantization/quant_config.py",
        },
    ),
}


def check_file(path: str) -> int:
    with open(path, encoding="utf-8") as f:
        content = f.read()
    return_code = 0
    # Check all patterns in the whole file
    for import_name, forbidden_import in CHECK_IMPORTS.items():
        # Path-scoped rules: skip files that don't match the rule's scope.
        if (
            forbidden_import.applies_to is not None
            and not forbidden_import.applies_to.search(path)
        ):
            continue
        # Skip files that are allowed for this import
        if path in forbidden_import.allowed_files:
            continue
        # Search for forbidden imports
        for match in re.finditer(forbidden_import.pattern, content, re.MULTILINE):
            # Check if it's allowed
            if forbidden_import.allowed_pattern.match(match.group()):
                continue
            # Calculate line number from match position
            line_num = content[: match.start() + 1].count("\n") + 1
            print(
                f"{path}:{line_num}: "
                "\033[91merror:\033[0m "  # red color
                f"Found forbidden import: {import_name}. {forbidden_import.tip}"
            )
            return_code = 1
    return return_code


def main():
    returncode = 0
    for path in sys.argv[1:]:
        returncode |= check_file(path)
    return returncode


def test_regex():
    test_cases = [
        # Should match
        ("import pickle", True),
        ("import cloudpickle", True),
        ("import pickle as pkl", True),
        ("import cloudpickle as cpkl", True),
        ("from pickle import *", True),
        ("from cloudpickle import dumps", True),
        ("from pickle import dumps, loads", True),
        ("from cloudpickle import (dumps, loads)", True),
        ("    import pickle", True),
        ("\timport cloudpickle", True),
        ("from   pickle   import   loads", True),
        # Should not match
        ("import somethingelse", False),
        ("from somethingelse import pickle", False),
        ("# import pickle", False),
        ("print('import pickle')", False),
        ("import pickleas as asdf", False),
    ]
    pickle_pattern = re.compile(CHECK_IMPORTS["pickle/cloudpickle"].pattern)
    for i, (line, should_match) in enumerate(test_cases):
        result = bool(pickle_pattern.match(line))
        assert result == should_match, (
            f"Test case {i} failed: '{line}' (expected {should_match}, got {result})"
        )

    hw_agnostic_cases = [
        ("from vllm.model_executor.layers.activation import SiluAndMul", True),
        ("from vllm.model_executor.layers.layernorm import RMSNorm", True),
        ("from vllm.model_executor.layers.layernorm import LayerNorm, RMSNorm", True),
        ("from vllm.model_executor.layers.rotary_embedding import get_rope", True),
        # sparse_attn_indexer carve-out: the upstream class has the OOT
        # torch-fallback wired in; the vendored copy re-exports it.
        ("from vllm.model_executor.layers.sparse_attn_indexer import X", False),
        ("from vllm.model_executor.layers.mhc import HCHeadOp", True),
        ("from vllm.model_executor.layers.linear import ColumnParallelLinear", True),
        ("from vllm.model_executor.layers.fused_moe import FusedMoE", True),
        (
            "from vllm.model_executor.layers.logits_processor import LogitsProcessor",
            True,
        ),
        # attention_layer_base carve-out: the worker walks static_forward_context
        # with isinstance(layer, upstream.AttentionLayerBase) to find KV-cache
        # owners. Vendored DSv4 layers must inherit from the same upstream class
        # or kv_cache_specs comes back empty.
        ("from vllm.model_executor.layers.attention_layer_base import X", False),
        (
            "from vllm.model_executor.layers.quantization import QuantizationConfig",
            True,
        ),
        # Other quantization.utils.* modules (int8_utils, marlin_utils, etc.)
        # remain forbidden — only fp8_utils is carved out as the FP8 path's
        # generic quant primitive (DSv4 needs ``per_token_group_quant_fp8``
        # in the vendored fused_moe.utils).
        (
            "from vllm.model_executor.layers.quantization.utils.int8_utils import x",
            True,
        ),
        (
            "from vllm.model_executor.layers.quantization.utils.marlin_utils import x",
            True,
        ),
        # Concrete quantization implementations stay forbidden — only the
        # abstract bases get a carve-out (further down).
        (
            "from vllm.model_executor.layers.quantization.fp8 import X",
            True,
        ),
        ("from vllm.model_executor.layers.vocab_parallel_embedding import X", True),
        ("from vllm.model_executor.models.deepseek_v2 import foo", True),
        ("from vllm.model_executor.models.deepseek_mtp import SharedHead", True),
        ("from vllm.model_executor.models.interfaces import SupportsPP", True),
        ("from vllm.models.deepseek_v4.compressor import DeepseekCompressor", True),
        ("from vllm.models.deepseek_v4.sparse_mla import X", True),
        ("from vllm.models.deepseek_v4.common.ops import foo", True),
        ("from vllm.models.deepseek_v4.common.ops.fused_indexer_q import bar", True),
        ("from vllm.models.minimax_m3.compressor import X", True),
        ("from vllm.models.llama4.experts import X", True),
        # mla.indexer carve-out: the indexer's metadata builder is
        # framework-driven (called per kv-cache group at request time) but
        # does not contain hardware-specific kernel calls — it only
        # composes Triton helpers and shape/index math. Re-exporting
        # rather than vendoring keeps the per-step metadata logic in
        # sync with upstream.
        ("from vllm.v1.attention.backends.mla.indexer import X", False),
        ("from vllm.v1.attention.backends.mla.sparse_swa import Y", True),
        ("    from vllm.model_executor.layers.activation import SiluAndMul", True),
        ("from vllm.model_executor.custom_op import PluggableLayer", False),
        ("from vllm.model_executor.custom_op import CustomOp", False),
        ("from vllm.model_executor.models.utils import maybe_prefix", False),
        ("from vllm.model_executor.models.utils import make_layers", False),
        ("from vllm.v1.attention.backends.utils import split_decodes", False),
        # Quantization abstract-bases carve-out: vendored linear/embedding
        # code must inherit from the registry's QuantizeMethodBase to keep
        # quant-method dispatch working. The base ABCs aren't HW-specific.
        (
            "from vllm.model_executor.layers.quantization.base_config import "
            "QuantizeMethodBase",
            False,
        ),
        (
            "from vllm.model_executor.layers.quantization.base_config import "
            "QuantizationConfig, QuantizeMethodBase",
            False,
        ),
        # FP8 quant primitives carve-out: per_token_group_quant_fp8 is the
        # generic FP8 quant op the vendored fused_moe.utils needs.
        (
            "from vllm.model_executor.layers.quantization.utils.fp8_utils import "
            "per_token_group_quant_fp8",
            False,
        ),
        # Vendored kernels.linear carve-outs: the kernel ABCs reach a small
        # set of pure-data / pure-helper modules that don't pull in
        # platform-specific kernel code.
        (
            "from vllm.model_executor.layers.quantization.utils.quant_utils import X",
            False,
        ),
        (
            "from vllm.model_executor.layers.quantization.utils.w8a8_utils import X",
            False,
        ),
        (
            "from vllm.model_executor.layers.quantization.utils.layer_utils import X",
            False,
        ),
        (
            "from vllm.model_executor.layers.quantization.input_quant_fp8 import "
            "QuantFP8",
            False,
        ),
        (
            "from vllm.model_executor.layers.quantization.compressed_tensors."
            "triton_scaled_mm import triton_scaled_mm",
            False,
        ),
        (
            "from vllm.model_executor.layers.fusion.quant_activation import X",
            False,
        ),
        # model_executor.kernels: forbidden everywhere on hw_agnostic.
        # The vendored kernel selector lives under hw_agnostic/shared/kernels/.
        (
            "from vllm.model_executor.kernels.linear import init_fp8_linear_kernel",
            True,
        ),
        (
            "from vllm.model_executor.kernels.linear.scaled_mm import X",
            True,
        ),
        # Vendored kernel subtree inside hw_agnostic must not be flagged.
        (
            "from vllm.models.deepseek_v4.hw_agnostic.shared.kernels.linear "
            "import init_fp8_linear_kernel",
            False,
        ),
        (
            "from vllm.model_executor.model_loader.weight_utils import "
            "default_weight_loader",
            False,
        ),
        ("from vllm.v1.attention.backend import AttentionBackend", False),
        ("from vllm.v1.kv_cache_interface import KVCacheSpec", False),
        ("from vllm.v1.worker.workspace import current_workspace_manager", False),
        ("from vllm.config import VllmConfig", False),
        ("from vllm.distributed import get_pp_group", False),
        ("from vllm.compilation.decorators import support_torch_compile", False),
        ("from vllm.platforms import current_platform", False),
        ("from vllm.forward_context import get_forward_context", False),
        ("# from vllm.model_executor.layers.layernorm import RMSNorm", False),
        (
            "from vllm.models.deepseek_v4.hw_agnostic.shared.layers.layernorm import X",
            False,
        ),
        (
            "from vllm.models.minimax_m3.hw_agnostic.shared.layers.layernorm import X",
            False,
        ),
        ("from vllm.model_executor.layers_extra import x", False),
        ("from vllm.model_executor.models_extra import x", False),
    ]
    rule = CHECK_IMPORTS["hw_agnostic_isolation"]
    rule_pattern = re.compile(rule.pattern, re.MULTILINE)
    for i, (line, should_match) in enumerate(hw_agnostic_cases):
        result = bool(rule_pattern.match(line))
        assert result == should_match, (
            f"hw_agnostic test case {i} failed: '{line}' "
            f"(expected {should_match}, got {result})"
        )

    assert rule.applies_to is not None
    accept_paths = [
        "vllm/models/deepseek_v4/hw_agnostic/model.py",
        "vllm/models/deepseek_v4/hw_agnostic/attention/attention.py",
        "vllm/models/deepseek_v4/hw_agnostic/shared/layers/linear.py",
        "vllm/models/deepseek_v4/hw_agnostic/tests/test_hw_agnostic_e2e.py",
        "vllm/models/minimax_m3/hw_agnostic/model.py",
        "vllm/models/llama4/hw_agnostic/attention/attention.py",
    ]
    reject_paths = [
        "vllm/models/deepseek_v4/attention.py",
        "vllm/models/deepseek_v4/sparse_mla.py",
        "vllm/models/minimax_m3/model.py",
        "vllm/model_executor/layers/activation.py",
        "tests/some_other.py",
    ]
    for p in accept_paths:
        assert rule.applies_to.search(p), f"applies_to should match {p}"
    for p in reject_paths:
        assert not rule.applies_to.search(p), f"applies_to should NOT match {p}"

    print("All regex tests passed.")


if __name__ == "__main__":
    if "--test-regex" in sys.argv:
        test_regex()
    else:
        sys.exit(main())
