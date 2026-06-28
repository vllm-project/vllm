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
            r"model_executor\.layers(?!\.utils\b)(?!\.quantization\.base_config\b)(?!\.quantization\.kv_cache\b)(?!\.attention_layer_base\b)"
            r"|model_executor\.kernels\b"
            r"|model_executor\.models(?!\.utils\b)"
            r"|model_executor\.model_loader\.reload\.layerwise\b"
            r"|models\.[^.]+(?!\.hw_agnostic\b)"
            r"|v1\.attention\.backends?\b"
            r"|v1\.kv_cache_interface\b"
            r")"
            r"(?:\.|\s+import\b)"
        ),
        tip=(
            "hardware-agnostic modelling code must not import from "
            "non-hardware-agnostic parts. The only exceptions are general "
            "utils.py files such as vllm.model_executor.layers.utils.py."
        ),
        applies_to=re.compile(
            r"^vllm/(?:models/[^/]+/hw_agnostic|model_executor/hw_agnostic)"
            r"/.*\.py$"
        ),
        allowed_files={
            # Re-export modules whose only job is to expose an upstream
            # symbol under a hw_agnostic-shaped path (identity match for
            # framework isinstance checks, or a process-wide registry).
            "vllm/model_executor/hw_agnostic/layers/attention.py",
            "vllm/model_executor/hw_agnostic/model_loader/reload/layerwise.py",
            "vllm/model_executor/hw_agnostic/quantization/quant_keys.py",
            "vllm/model_executor/hw_agnostic/quantization/input_quant_fp8.py",
            "vllm/model_executor/hw_agnostic/quantization/quant_activation.py",
            # Attention-backend / KV-cache-spec shims: the V1 framework
            # keys group identity on the same class object, so these
            # re-exports must resolve to the upstream classes verbatim.
            "vllm/model_executor/hw_agnostic/v1/attention/backend.py",
            "vllm/model_executor/hw_agnostic/v1/kv_cache_interface.py",
        },
    ),
    "hw_agnostic_no_vendor_utils": ForbiddenImport(
        pattern=(
            r"^\s*(?:from\s+vllm\.utils\."
            r"(?:flashinfer|deep_gemm|cutlass|trtllm|deep_ep|aiter)\b"
            r"|import\s+vllm\.utils\."
            r"(?:flashinfer|deep_gemm|cutlass|trtllm|deep_ep|aiter)\b)"
        ),
        tip=(
            "hardware-agnostic code must not import from vendor-specific "
            "vllm.utils.* modules (flashinfer/deep_gemm/cutlass/trtllm/"
            "deep_ep/aiter)."
        ),
        applies_to=re.compile(
            r"^vllm/(?:models/[^/]+/hw_agnostic|model_executor/hw_agnostic)"
            r"/.*\.py$"
        ),
    ),
    "protected_upstream_no_oot_leak": ForbiddenImport(
        # The principle: HW-specific attention authors must be able to
        # change these files without breaking the hw-agnostic path. The
        # corollary: these files must not contain hw-agnostic carve-outs
        # ("is_out_of_tree" branches) and must not import from any
        # hw_agnostic tree. The agnostic counterpart lives under
        # vllm/models/deepseek_v4/hw_agnostic/attention/sparse_attn_indexer.py
        # (registered as torch.ops.vllm.dsv4_sparse_attn_indexer) so no
        # OOT branch is needed here.
        pattern=(
            r"(?:"
            r"current_platform\.is_out_of_tree\("
            r"|^\s*from\s+vllm\.(?:model_executor|models\.[^.]+)\.hw_agnostic\b"
            r"|^\s*import\s+vllm\.(?:model_executor|models\.[^.]+)\.hw_agnostic\b"
            r")"
        ),
        tip=(
            "Protected upstream attention files must not depend on "
            "is_out_of_tree() branches or import from any hw_agnostic "
            "subtree. The agnostic path keeps its own implementation; "
            "do not couple the two."
        ),
        applies_to=re.compile(r"^vllm/model_executor/layers/sparse_attn_indexer\.py$"),
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
        # sparse_attn_indexer is now vendored under
        # ``hw_agnostic/attention/sparse_attn_indexer.py`` — the upstream
        # module is forbidden from hw_agnostic.
        ("from vllm.model_executor.layers.sparse_attn_indexer import X", True),
        ("from vllm.model_executor.layers.mhc import HCHeadOp", True),
        ("from vllm.model_executor.layers.linear import ColumnParallelLinear", True),
        ("from vllm.model_executor.layers.fused_moe import FusedMoE", True),
        # Upstream Attention is forbidden — the vendored re-export at
        # ``hw_agnostic/layers/attention.py`` is used instead.
        ("from vllm.model_executor.layers.attention import Attention", True),
        # initialize_online_processing is forbidden via upstream — the
        # re-export at ``hw_agnostic/model_loader/reload/layerwise.py``
        # is used instead.
        (
            "from vllm.model_executor.model_loader.reload.layerwise "
            "import initialize_online_processing",
            True,
        ),
        # The upstream Triton experts kernel is now vendored under
        # ``hw_agnostic/layers/fused_moe/experts/triton_moe.py``; the
        # upstream module is forbidden.
        (
            "from vllm.model_executor.layers.fused_moe.experts.triton_moe "
            "import TritonExperts",
            True,
        ),
        # All ``fused_moe.*`` upstream submodules are now vendored — the
        # upstream paths are forbidden.
        (
            "from vllm.model_executor.layers.fused_moe.activation import MoEActivation",
            True,
        ),
        (
            "from vllm.model_executor.layers.fused_moe.modular_kernel "
            "import FusedMoEKernel",
            True,
        ),
        (
            "from vllm.model_executor.layers.fused_moe.fused_moe_method_base "
            "import FusedMoEMethodBase",
            True,
        ),
        (
            "from vllm.model_executor.layers.fused_moe.runner.moe_runner_interface "
            "import MoERunnerInterface",
            True,
        ),
        (
            "from vllm.model_executor.layers.fused_moe.oracle.fp8 import Fp8MoeBackend",
            True,
        ),
        (
            "from vllm.model_executor.layers.fused_moe.routed_experts "
            "import RoutedExperts",
            True,
        ),
        (
            "from vllm.model_executor.layers.fused_moe.runner.shared_experts "
            "import SharedExperts",
            True,
        ),
        (
            "from vllm.model_executor.layers.fused_moe.config "
            "import FusedMoEQuantConfig",
            True,
        ),
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
        # ``v1.attention.backends`` is forbidden whole-subtree: the indexer
        # metadata dataclasses are vendored under hw_agnostic and
        # ``split_decodes_and_prefills`` lives in
        # ``hw_agnostic/attention/_metadata_utils.py``.
        ("from vllm.v1.attention.backends.mla.indexer import X", True),
        ("from vllm.v1.attention.backends.mla.sparse_swa import Y", True),
        ("from vllm.v1.attention.backends.utils import split_decodes", True),
        # ``v1.attention.backend`` (singular) and ``v1.kv_cache_interface``
        # carry the abstract framework types. They are reached only
        # through the re-export shims at
        # ``hw_agnostic/v1/attention/backend.py`` and
        # ``hw_agnostic/v1/kv_cache_interface.py``; direct imports from
        # the agnostic tree are forbidden so the HW-specific framework
        # path can evolve without touching us.
        ("from vllm.v1.attention.backend import AttentionBackend", True),
        ("from vllm.v1.attention.backend import CommonAttentionMetadata", True),
        ("from vllm.v1.kv_cache_interface import KVCacheSpec", True),
        ("from vllm.v1.kv_cache_interface import MLAAttentionSpec", True),
        # The shim itself is exempt (see allowed_files).
        (
            "from vllm.model_executor.hw_agnostic.v1.attention.backend "
            "import AttentionBackend",
            False,
        ),
        (
            "from vllm.model_executor.hw_agnostic.v1.kv_cache_interface "
            "import MLAAttentionSpec",
            False,
        ),
        ("    from vllm.model_executor.layers.activation import SiluAndMul", True),
        ("from vllm.model_executor.custom_op import PluggableLayer", False),
        ("from vllm.model_executor.custom_op import CustomOp", False),
        ("from vllm.model_executor.models.utils import maybe_prefix", False),
        ("from vllm.model_executor.models.utils import make_layers", False),
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
        # FP8 helpers / Triton kernels are now vendored under
        # ``hw_agnostic/quantization/fp8_utils.py``; the upstream module is
        # forbidden everywhere except its own re-export shim.
        (
            "from vllm.model_executor.layers.quantization.utils.fp8_utils import "
            "per_token_group_quant_fp8",
            True,
        ),
        # ``quant_utils``, ``input_quant_fp8`` and ``fusion.quant_activation``
        # are reached only via the re-export shims in
        # ``hw_agnostic/quantization/{quant_keys,input_quant_fp8,
        # quant_activation}.py`` (those files are on the per-file allowlist).
        (
            "from vllm.model_executor.layers.quantization.utils.quant_utils import X",
            True,
        ),
        (
            "from vllm.model_executor.layers.quantization.input_quant_fp8 import "
            "QuantFP8",
            True,
        ),
        (
            "from vllm.model_executor.layers.fusion.quant_activation import X",
            True,
        ),
        # ``w8a8_utils``, ``layer_utils``, ``triton_scaled_mm`` are no longer
        # consumed by hw_agnostic code at all.
        (
            "from vllm.model_executor.layers.quantization.utils.w8a8_utils import X",
            True,
        ),
        (
            "from vllm.model_executor.layers.quantization.utils.layer_utils import X",
            True,
        ),
        (
            "from vllm.model_executor.layers.quantization.compressed_tensors."
            "triton_scaled_mm import triton_scaled_mm",
            True,
        ),
        # model_executor.kernels: forbidden everywhere on hw_agnostic.
        # The vendored kernel selector lives under
        # vllm/model_executor/hw_agnostic/kernels/.
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
            "from vllm.model_executor.hw_agnostic.kernels.linear "
            "import init_fp8_linear_kernel",
            False,
        ),
        # ``quantization.kv_cache`` is on the carve-out (the FP8 KV-cache
        # method inherits from upstream ``BaseKVCacheMethod`` for registry
        # identity).
        (
            "from vllm.model_executor.layers.quantization.kv_cache import "
            "BaseKVCacheMethod",
            False,
        ),
        # All ``fused_moe.experts.*`` upstream submodules are forbidden;
        # the Triton path is vendored locally.
        (
            "from vllm.model_executor.layers.fused_moe.experts.flashinfer_cutlass_moe "
            "import X",
            True,
        ),
        (
            "from vllm.model_executor.model_loader.weight_utils import "
            "default_weight_loader",
            False,
        ),
        ("from vllm.v1.worker.workspace import current_workspace_manager", False),
        ("from vllm.config import VllmConfig", False),
        ("from vllm.distributed import get_pp_group", False),
        ("from vllm.compilation.decorators import support_torch_compile", False),
        ("from vllm.platforms import current_platform", False),
        ("from vllm.forward_context import get_forward_context", False),
        ("# from vllm.model_executor.layers.layernorm import RMSNorm", False),
        (
            "from vllm.model_executor.hw_agnostic.layers.layernorm import X",
            False,
        ),
        (
            "from vllm.models.minimax_m3.hw_agnostic.layers.layernorm import X",
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

    no_vendor_utils_cases = [
        ("from vllm.utils.flashinfer import nvfp4_block_scale_interleave", True),
        ("from vllm.utils.deep_gemm import is_deep_gemm_e8m0_used", True),
        ("from vllm.utils.cutlass import foo", True),
        ("from vllm.utils.trtllm import foo", True),
        ("import vllm.utils.flashinfer", True),
        ("from vllm.utils.math_utils import cdiv", False),
        ("from vllm.utils.torch_utils import aux_stream", False),
        ("from vllm.utils.import_utils import has_triton_kernels", False),
    ]
    no_vendor_rule = CHECK_IMPORTS["hw_agnostic_no_vendor_utils"]
    no_vendor_pattern = re.compile(no_vendor_rule.pattern, re.MULTILINE)
    for i, (line, should_match) in enumerate(no_vendor_utils_cases):
        result = bool(no_vendor_pattern.match(line))
        assert result == should_match, (
            f"no_vendor_utils test case {i} failed: '{line}' "
            f"(expected {should_match}, got {result})"
        )

    protected_upstream_cases = [
        ("if current_platform.is_out_of_tree():", True),
        ("        elif current_platform.is_out_of_tree():", True),
        (
            "from vllm.model_executor.hw_agnostic.v1.attention.backend import X",
            True,
        ),
        (
            "from vllm.models.deepseek_v4.hw_agnostic.attention.sparse_mla import X",
            True,
        ),
        ("import vllm.model_executor.hw_agnostic", True),
        ("if current_platform.is_cuda():", False),
        ("from vllm.platforms import current_platform", False),
        ("# is_out_of_tree() — historical note", False),
    ]
    protected_rule = CHECK_IMPORTS["protected_upstream_no_oot_leak"]
    protected_pattern = re.compile(protected_rule.pattern, re.MULTILINE)
    for i, (line, should_match) in enumerate(protected_upstream_cases):
        result = bool(protected_pattern.search(line))
        assert result == should_match, (
            f"protected_upstream test case {i} failed: '{line}' "
            f"(expected {should_match}, got {result})"
        )
    assert protected_rule.applies_to is not None
    assert protected_rule.applies_to.search(
        "vllm/model_executor/layers/sparse_attn_indexer.py"
    )
    assert not protected_rule.applies_to.search(
        "vllm/models/deepseek_v4/hw_agnostic/attention/sparse_attn_indexer.py"
    )

    assert rule.applies_to is not None
    accept_paths = [
        "vllm/models/deepseek_v4/hw_agnostic/model.py",
        "vllm/models/deepseek_v4/hw_agnostic/attention/attention.py",
        "vllm/models/deepseek_v4/hw_agnostic/tests/test_hw_agnostic_e2e.py",
        "vllm/models/minimax_m3/hw_agnostic/model.py",
        "vllm/models/llama4/hw_agnostic/attention/attention.py",
        # Lifted shared subtree at vllm/model_executor/hw_agnostic/ is
        # subject to the same isolation lint as the per-model trees.
        "vllm/model_executor/hw_agnostic/layers/linear.py",
        "vllm/model_executor/hw_agnostic/layers/fused_moe/layer.py",
        "vllm/model_executor/hw_agnostic/custom_op.py",
    ]
    reject_paths = [
        "vllm/models/deepseek_v4/attention.py",
        "vllm/models/deepseek_v4/sparse_mla.py",
        "vllm/models/minimax_m3/model.py",
        "vllm/model_executor/layers/activation.py",
        "vllm/model_executor/custom_op.py",
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
