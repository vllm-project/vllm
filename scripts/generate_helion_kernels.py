#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Generate standalone Triton modules from tuned Helion kernels."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import os
import sys
import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from pprint import pformat
from typing import Any

import regex as re
import torch
from tqdm.auto import tqdm

from vllm.kernels.helion.case_key import CaseKey
from vllm.kernels.helion.config_manager import ConfigManager

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "vllm/kernels/helion_generated/kernels"
PLATFORM_DEVICE_NAMES = {
    "nvidia_b200": "nvidia_b200",
    "nvidia_h100": "nvidia_h100",
    "nvidia_h100_80gb_hbm3": "nvidia_h100",
    "nvidia_h100_nvl": "nvidia_h100",
    "nvidia_h100_pcie": "nvidia_h100",
    "nvidia_h100_sxm5": "nvidia_h100",
}


@dataclass(frozen=True)
class KernelSpec:
    name: str
    case_fields: tuple[tuple[str, str], ...]
    input_factory: Callable[[CaseKey], tuple[Any, ...]]
    always_use_factory: bool = False


def _fp8_empty(*shape: int) -> torch.Tensor:
    return torch.empty(shape, device="cuda", dtype=torch.float8_e4m3fn)


def _dynamic_quant_inputs(case: CaseKey) -> tuple[Any, ...]:
    shape = (case["num_tokens"], case["hidden_size"])
    input = torch.empty(shape, device="cuda", dtype=torch.bfloat16)
    return (
        _fp8_empty(*shape),
        input,
        torch.empty((shape[0], 1), device="cuda", dtype=torch.float32),
        torch.ones((), device="cuda", dtype=torch.float32),
    )


def _fused_qk_norm_rope_inputs(case: CaseKey) -> tuple[Any, ...]:
    num_tokens = case["num_tokens"]
    q_heads = case["q_heads"]
    kv_heads = case["kv_heads"]
    head_dim = 128
    total_dim = (q_heads + 2 * kv_heads) * head_dim
    dtype = torch.bfloat16
    return (
        torch.empty((num_tokens, total_dim), device="cuda", dtype=dtype),
        q_heads,
        kv_heads,
        kv_heads,
        head_dim,
        1e-6,
        torch.empty(head_dim, device="cuda", dtype=dtype),
        torch.empty(head_dim, device="cuda", dtype=dtype),
        torch.empty((40960, head_dim), device="cuda", dtype=dtype),
        True,
        torch.arange(num_tokens, device="cuda", dtype=torch.int64),
    )


def _per_token_group_fp8_quant_inputs(case: CaseKey) -> tuple[Any, ...]:
    num_tokens = case["num_tokens"]
    hidden_size = case["hidden_size"]
    group_size = case["group_size"]
    shape = (num_tokens, hidden_size)
    input = torch.empty(shape, device="cuda", dtype=torch.bfloat16)
    return (
        input,
        _fp8_empty(*shape),
        torch.empty(
            (num_tokens, hidden_size // group_size),
            device="cuda",
            dtype=torch.float32,
        ),
        group_size,
        1e-10,
        -448.0,
        448.0,
        False,
        False,
        False,
    )


def _rms_norm_dynamic_quant_inputs(case: CaseKey) -> tuple[Any, ...]:
    shape = (case["num_tokens"], case["hidden_size"])
    input = torch.empty(shape, device="cuda", dtype=torch.bfloat16)
    return (
        _fp8_empty(*shape),
        input,
        torch.empty(shape[1], device="cuda", dtype=input.dtype),
        torch.empty((shape[0], 1), device="cuda", dtype=torch.float32),
        1e-6,
        torch.ones((), device="cuda", dtype=torch.float32),
        torch.empty_like(input),
    )


def _rms_norm_per_block_quant_inputs(case: CaseKey) -> tuple[Any, ...]:
    num_tokens = case["num_tokens"]
    hidden_size = case["hidden_size"]
    group_size = case["group_size"]
    shape = (num_tokens, hidden_size)
    input = torch.empty(shape, device="cuda", dtype=torch.bfloat16)
    groups_per_row = hidden_size // group_size
    return (
        _fp8_empty(*shape),
        input,
        torch.empty(hidden_size, device="cuda", dtype=input.dtype),
        torch.empty(
            (groups_per_row, num_tokens), device="cuda", dtype=torch.float32
        ).t(),
        1e-6,
        None,
        None,
        group_size,
        True,
    )


def _silu_and_mul_per_block_quant_inputs(case: CaseKey) -> tuple[Any, ...]:
    num_tokens = case["num_tokens"]
    intermediate_size = case["intermediate_size"]
    group_size = case["group_size"]
    groups_per_row = intermediate_size // group_size
    return (
        _fp8_empty(num_tokens, intermediate_size),
        torch.empty(
            (num_tokens, 2 * intermediate_size),
            device="cuda",
            dtype=torch.bfloat16,
        ),
        torch.empty(
            (groups_per_row, num_tokens), device="cuda", dtype=torch.float32
        ).t(),
        group_size,
        None,
        True,
    )


def _silu_mul_fp8_inputs(case: CaseKey) -> tuple[Any, ...]:
    return (
        torch.empty(
            (case["numtokens"], 2 * case["intermediate"]),
            device="cuda",
            dtype=torch.bfloat16,
        ),
        torch.ones(1, device="cuda", dtype=torch.float32),
    )


KERNEL_REGISTRY = {
    spec.name: spec
    for spec in (
        KernelSpec(
            "dynamic_per_token_scaled_fp8_quant",
            (("hidden_size", "h"), ("num_tokens", "t")),
            _dynamic_quant_inputs,
        ),
        KernelSpec(
            "fused_qk_norm_rope",
            (("q_heads", "qh"), ("kv_heads", "kvh"), ("num_tokens", "t")),
            _fused_qk_norm_rope_inputs,
        ),
        KernelSpec(
            "per_token_group_fp8_quant",
            (("hidden_size", "h"), ("group_size", "g"), ("num_tokens", "t")),
            _per_token_group_fp8_quant_inputs,
        ),
        KernelSpec(
            "rms_norm_dynamic_per_token_quant",
            (("hidden_size", "h"), ("num_tokens", "t")),
            _rms_norm_dynamic_quant_inputs,
        ),
        KernelSpec(
            "rms_norm_per_block_quant",
            (("hidden_size", "h"), ("group_size", "g"), ("num_tokens", "t")),
            _rms_norm_per_block_quant_inputs,
            always_use_factory=True,
        ),
        KernelSpec(
            "silu_and_mul_per_block_quant",
            (
                ("intermediate_size", "i"),
                ("group_size", "g"),
                ("num_tokens", "t"),
            ),
            _silu_and_mul_per_block_quant_inputs,
            always_use_factory=True,
        ),
        KernelSpec(
            "silu_mul_fp8",
            (("intermediate", "i"), ("numtokens", "t")),
            _silu_mul_fp8_inputs,
        ),
    )
}

_LAUNCHER = """def _get_num_sm(device: torch.device) -> int:
    return torch.cuda.get_device_properties(device).multi_processor_count


def _default_launcher(
    triton_kernel: object,
    grid: tuple[int, ...],
    *args: object,
    num_warps: int,
    num_stages: int,
    ptx_options: str | None = None,
    launch_cooperative_grid: bool = False,
    **kwargs: object,
) -> object:
    run_kwargs = {
        "grid": grid,
        "warmup": False,
        "num_warps": num_warps,
        "num_stages": num_stages,
        "launch_cooperative_grid": launch_cooperative_grid,
        **kwargs,
    }
    if ptx_options is not None:
        run_kwargs["ptx_options"] = ptx_options
    return triton_kernel.run(*args, **run_kwargs)
"""

_SOURCE_HELPERS = """def _get_fp8_dtype() -> torch.dtype:
    return torch.float8_e4m3fn


def _get_fp8_min_max() -> tuple[float, float]:
    info = torch.finfo(_get_fp8_dtype())
    return info.min, info.max


def _get_int8_min_max() -> tuple[int, int]:
    info = torch.iinfo(torch.int8)
    return info.min, info.max


def _get_int8_min_scaling_factor() -> float:
    return torch.finfo(torch.float32).eps
"""


def _canonical_device_name(name: str) -> str:
    return name.lower().replace(" ", "_").replace("-", "_").replace("/", "_")


def _require_matching_hardware(platform: str) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(f"Generating {platform} kernels requires a CUDA GPU")
    actual_name = _canonical_device_name(torch.cuda.get_device_name())
    actual = PLATFORM_DEVICE_NAMES.get(actual_name)
    if actual != platform:
        raise RuntimeError(
            f"Generating {platform} kernels requires matching hardware; "
            f"found {torch.cuda.get_device_name()!r}"
        )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _replace_binary_helper(code: str, helper: str, replacement: str) -> str:
    pattern = rf"\b{re.escape(helper)}\(([^,\n]+),\s*([^\)\n]+)\)"
    return re.sub(
        pattern,
        rf"{replacement}(\1, \2, tl.PropagateNan.ALL)",
        code,
    )


def _postprocess(
    code: str,
    spec: KernelSpec,
    case: CaseKey,
    expected_args: list[str],
) -> str:
    uses_source_helpers = "_source_module." in code
    code = (
        code.replace("from torch._inductor.runtime import triton_helpers\n", "")
        .replace(
            "from torch._inductor.runtime.triton_helpers import math as tl_math\n", ""
        )
        .replace("from torch._inductor.runtime.triton_compat import libdevice\n", "")
        .replace(
            "from helion.runtime import default_launcher as _default_launcher\n", ""
        )
        .replace("import helion\n", "")
    )
    code = re.sub(
        rf"^import vllm\.kernels\.helion\.ops\.{spec.name} "
        r"as _source_module\n",
        "",
        code,
        flags=re.MULTILINE,
    )
    code = code.replace("tl_math.", "tl.").replace("libdevice.", "tl.")
    code = code.replace("helion.runtime.get_num_sm", "_get_num_sm")
    code = code.replace("_source_module.get_fp8_dtype", "_get_fp8_dtype")
    code = code.replace("_source_module.get_fp8_min_max", "_get_fp8_min_max")
    code = code.replace("_source_module.get_int8_min_max", "_get_int8_min_max")
    code = code.replace(
        "_source_module.get_int8_min_scaling_factor",
        "_get_int8_min_scaling_factor",
    )
    code = _replace_binary_helper(code, "triton_helpers.maximum", "tl.maximum")
    code = _replace_binary_helper(code, "triton_helpers.minimum", "tl.minimum")
    code = code.replace(f"_helion_{spec.name}", f"_triton_{spec.name}")
    code = code.replace(f"def {spec.name}(", "def call(")
    code = re.sub(r"^\s*# src\[[^\n]*\n", "", code, flags=re.MULTILINE)
    if spec.name == "per_token_group_fp8_quant" and case["num_tokens"] == 1:
        code = re.sub(
            r"(\* )1(,\),)",
            r"\1num_tokens\2",
            code,
            count=1,
        )
    import_end = code.index("\n\n", code.index("import triton.language as tl"))
    support_code = _LAUNCHER
    if uses_source_helpers:
        support_code += "\n\n" + _SOURCE_HELPERS
    code = code[: import_end + 2] + support_code + "\n" + code[import_end + 2 :]
    code = (
        "# SPDX-License-Identifier: Apache-2.0\n"
        "# SPDX-FileCopyrightText: Copyright contributors to the vLLM project\n"
        "# This file is generated by scripts/generate_helion_kernels.py.\n"
        "# ruff: noqa\n"
        "# mypy: ignore-errors\n"
        "# fmt: off\n" + code
    )
    _validate_artifact(code, expected_args)
    return code.rstrip() + "\n"


def _validate_artifact(code: str, expected_args: list[str]) -> None:
    tree = ast.parse(code)
    allowed_roots = {"__future__", "torch", "triton"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported = {alias.name.split(".", 1)[0] for alias in node.names}
            if not imported <= allowed_roots:
                raise ValueError(f"forbidden generated import: {sorted(imported)}")
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".", 1)[0]
            if root not in allowed_roots:
                raise ValueError(f"forbidden generated import: {node.module}")
    call = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "call"
    )
    if [arg.arg for arg in call.args.args] != expected_args:
        raise ValueError("generated artifact has an unstable call(...) contract")
    if "_source_module" in code:
        raise ValueError("generated artifact retains a source-module dependency")


def _case_tuple(spec: KernelSpec, case: CaseKey) -> tuple[Any, ...]:
    return tuple(case[field] for field, _ in spec.case_fields)


def _module_name(spec: KernelSpec, case: CaseKey) -> str:
    suffix = "_".join(f"{prefix}{case[field]}" for field, prefix in spec.case_fields)
    return f"{spec.name}_{suffix}"


def _manifest(
    spec: KernelSpec,
    platform: str,
    cases: list[CaseKey],
    configs: dict[CaseKey, Any],
) -> str:
    source_path = ROOT / f"vllm/kernels/helion/ops/{spec.name}.py"
    config_path = ROOT / f"vllm/kernels/helion/configs/{spec.name}" / f"{platform}.json"
    kernels = {
        _case_tuple(spec, case): (
            f"vllm.kernels.helion_generated.kernels.{spec.name}."
            f"{platform}.{_module_name(spec, case)}"
        )
        for case in cases
    }
    serialized_configs = {
        _case_tuple(spec, case): json.loads(configs[case].to_json()) for case in cases
    }
    provenance = {
        "config_path": str(config_path.relative_to(ROOT)),
        "config_sha256": _sha256(config_path),
        "generator": "scripts/generate_helion_kernels.py",
        "helion_version": importlib.metadata.version("helion"),
        "kernel": spec.name,
        "platform": platform,
        "source_path": str(source_path.relative_to(ROOT)),
        "source_sha256": _sha256(source_path),
        "torch_version": torch.__version__,
        "triton_version": importlib.metadata.version("triton"),
    }
    return (
        "# SPDX-License-Identifier: Apache-2.0\n"
        "# SPDX-FileCopyrightText: Copyright contributors to the vLLM project\n"
        "# This file is generated by scripts/generate_helion_kernels.py.\n\n"
        "# ruff: noqa\n"
        "# mypy: ignore-errors\n"
        "# fmt: off\n"
        f"KERNELS = {pformat(kernels, sort_dicts=True, width=88)}\n\n"
        f"CONFIGS = {pformat(serialized_configs, sort_dicts=True, width=88)}\n\n"
        "PRESERVES_SPECIALIZATIONS = True\n\n"
        f"PROVENANCE = {pformat(provenance, sort_dicts=True, width=88)}\n"
    )


def _init_file() -> str:
    return (
        "# SPDX-License-Identifier: Apache-2.0\n"
        "# SPDX-FileCopyrightText: Copyright contributors to the vLLM project\n"
    )


def _write_or_check(path: Path, content: str, check: bool, errors: list[str]) -> None:
    if check:
        if not path.exists() or path.read_text() != content:
            errors.append(str(path.relative_to(ROOT)))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def _load_op(spec: KernelSpec) -> Any:
    module = importlib.import_module(f"vllm.kernels.helion.ops.{spec.name}")
    return getattr(module, spec.name)


@contextmanager
def _suppress_helion_output() -> Iterator[None]:
    failed = False
    with tempfile.TemporaryFile() as captured:
        saved_stdout = os.dup(1)
        saved_stderr = os.dup(2)
        try:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(captured.fileno(), 1)
            os.dup2(captured.fileno(), 2)
            yield
        except BaseException:
            failed = True
            raise
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(saved_stdout, 1)
            os.dup2(saved_stderr, 2)
            os.close(saved_stdout)
            os.close(saved_stderr)
            if failed:
                captured.seek(0)
                diagnostics = captured.read()
                if diagnostics:
                    sys.stderr.write(diagnostics.decode(errors="replace"))


def generate(kernel_name: str, platform: str, check: bool) -> None:
    _require_matching_hardware(platform)
    spec = KERNEL_REGISTRY[kernel_name]
    op = _load_op(spec)
    configs = ConfigManager().get_platform_configs(spec.name, platform)
    if not configs:
        raise RuntimeError(f"No {platform} configs found for {spec.name}")
    inputs = op.get_inputs()
    case_fields = {field for field, _ in spec.case_fields}
    concrete_configs = {
        case: config for case, config in configs.items() if not case.is_default()
    }
    invalid_cases = [
        case for case in concrete_configs if set(case.keys()) != case_fields
    ]
    if invalid_cases:
        raise RuntimeError(
            f"Unexpected config keys for {spec.name}: "
            f"{', '.join(map(str, invalid_cases))}"
        )
    cases = sorted(
        concrete_configs,
        key=lambda case: _case_tuple(spec, case),
    )

    kernel_dir = OUTPUT_ROOT / spec.name
    output_dir = kernel_dir / platform
    errors: list[str] = []
    expected = {"__init__.py", "manifest.py"}
    configured_op = op.get_configured_op()
    configured = configured_op._decorated_kernel
    if "preserve_specializations" not in inspect.signature(configured.bind).parameters:
        raise RuntimeError(
            "Kernel generation requires a Helion version whose Kernel.bind() "
            "supports preserve_specializations=True"
        )
    expected_args = list(inspect.signature(configured_op.raw_kernel_func).parameters)
    action = "Check" if check else "Generate"
    hardware = platform.removeprefix("nvidia_").upper()
    progress = tqdm(
        cases,
        bar_format=(
            "{desc}: {percentage:3.0f}%|{bar:20}| "
            "{n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
        ),
        desc=f"{action} {hardware} {spec.name}",
        unit="case",
    )
    for case in progress:
        filename = f"{_module_name(spec, case)}.py"
        expected.add(filename)
        args = None if spec.always_use_factory else inputs.get(case)
        if args is None:
            args = spec.input_factory(case)
        with _suppress_helion_output():
            code = configured.bind(
                args,
                preserve_specializations=True,
            ).to_code(concrete_configs[case])
        content = _postprocess(code, spec, case, expected_args)
        _write_or_check(output_dir / filename, content, check, errors)

    _write_or_check(output_dir / "__init__.py", _init_file(), check, errors)
    _write_or_check(kernel_dir / "__init__.py", _init_file(), check, errors)
    _write_or_check(
        output_dir / "manifest.py",
        _manifest(spec, platform, cases, concrete_configs),
        check,
        errors,
    )
    if output_dir.exists():
        stale = sorted(
            path for path in output_dir.glob("*.py") if path.name not in expected
        )
        if check:
            errors.extend(str(path.relative_to(ROOT)) for path in stale)
        else:
            for path in stale:
                path.unlink()
    if errors:
        paths = "\n".join(f"  {path}" for path in sorted(errors))
        raise SystemExit(f"Generated Helion kernels are stale:\n{paths}")
    operation = "checked" if check else "generated"
    print(f"Done: {operation} {len(cases)} {spec.name} cases for {platform}.")
    print(f"Output: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kernel",
        choices=sorted(KERNEL_REGISTRY),
        default="per_token_group_fp8_quant",
    )
    parser.add_argument(
        "--platform",
        choices=sorted(set(PLATFORM_DEVICE_NAMES.values())),
        required=True,
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail instead of writing when generated files differ.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate(args.kernel, args.platform, args.check)


if __name__ == "__main__":
    main()
