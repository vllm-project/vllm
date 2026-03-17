#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Autotune registered Helion kernels for optimal configurations.

Usage:
    # Autotune all registered kernels (latest version of each)
    python scripts/autotune_helion_kernels.py

    # Autotune specific kernel (latest version)
    python scripts/autotune_helion_kernels.py --kernels silu_mul_fp8

    # Autotune specific version of a kernel
    python scripts/autotune_helion_kernels.py --kernels silu_mul_fp8:1

    # Autotune multiple kernels with different versions
    python scripts/autotune_helion_kernels.py --kernels silu_mul_fp8:1 rms_norm_fp8:2

    # Force re-autotuning
    python scripts/autotune_helion_kernels.py --force

    # List available kernels
    python scripts/autotune_helion_kernels.py --list
"""

import argparse
import sys
import time
from dataclasses import dataclass

import torch
from torch._subclasses.fake_tensor import FakeTensorMode

try:
    import helion

    from vllm.kernels.helion import (
        ConfigManager,
        get_kernel,
        get_registered_kernels,
    )
    from vllm.kernels.helion.utils import get_canonical_gpu_name
    from vllm.logger import init_logger
    from vllm.utils.import_utils import has_helion
except ImportError as e:
    print(f"Error importing vLLM: {e}")
    print("Please ensure vLLM is installed and in your Python path")
    sys.exit(1)

logger = init_logger("vllm.scripts.autotune_helion_kernels")


@dataclass
class AutotuneResult:
    status: str  # "success" | "partial" | "error" | "skipped"
    successful: int
    failed: int
    configs: dict[str, "helion.Config"]
    message: str = ""


def list_kernels() -> None:
    kernels = get_registered_kernels()

    if not kernels:
        print("No Helion kernels found in registry.")
        return

    print("Available Helion kernels:")
    print("=" * 50)

    for name in sorted(kernels.keys()):
        versions = kernels[name]
        newest = max(versions)
        for ver in sorted(versions):
            suffix = "" if ver == newest else " [superseded]"
            print(f"  {name} v{ver}{suffix}")

    print(f"\nTotal: {len(kernels)} kernels")


def check_requirements() -> bool:
    if not torch.cuda.is_available():
        logger.error("CUDA is not available. Helion autotuning requires GPU.")
        return False

    if not has_helion():
        logger.error("Helion is not installed. Please install Helion package.")
        return False

    return True


def autotune_kernel(
    kernel_name: str,
    version: int,
    platform: str,
    config_manager: ConfigManager,
    force: bool = False,
    autotune_effort: str = "quick",
) -> AutotuneResult:
    kernel_wrapper = get_kernel(kernel_name, version)
    logger.debug(
        "Starting autotune for kernel '%s' v%d with effort='%s'",
        kernel_name,
        version,
        autotune_effort,
    )
    if kernel_wrapper is None:
        error_msg = f"Kernel '{kernel_name}' version {version} not found in registry"
        logger.error(error_msg)
        return AutotuneResult(
            status="error",
            message=error_msg,
            successful=0,
            failed=0,
            configs={},
        )

    # TODO(gmagogsfm): get_inputs() allocates real tensors just to derive
    # config keys.  Consider a lightweight get_config_keys() that returns
    # keys without materialising tensors, so we can skip already-tuned
    # configs before doing any GPU work.
    try:
        with FakeTensorMode():
            all_config_keys = list(kernel_wrapper.get_inputs().keys())
    except NotImplementedError:
        error_msg = f"Kernel '{kernel_name}' has no input generator registered"
        logger.error(error_msg)
        return AutotuneResult(
            status="error",
            message=error_msg,
            successful=0,
            failed=0,
            configs={},
        )

    # From here on, use the versioned name for config lookup and logging.
    kernel_name = kernel_wrapper.versioned_name

    try:
        logger.info(
            "Autotuning kernel '%s' for platform '%s' with %d configs",
            kernel_name,
            platform,
            len(all_config_keys),
        )

        if not force:
            existing_configs = config_manager.get_platform_configs(
                kernel_name, platform
            )
            keys_to_autotune = []
            for config_key in all_config_keys:
                if config_key in existing_configs:
                    logger.debug(
                        "Config '%s' already exists for platform '%s', skipping",
                        config_key,
                        platform,
                    )
                else:
                    keys_to_autotune.append(config_key)
        else:
            logger.debug("Force mode enabled, will re-autotune all configs")
            keys_to_autotune = all_config_keys

        if not keys_to_autotune:
            logger.info(
                "All configs already exist for kernel '%s' on platform '%s'. "
                "Use --force to re-autotune.",
                kernel_name,
                platform,
            )
            return AutotuneResult(
                status="skipped",
                message="All configs already exist",
                successful=0,
                failed=0,
                configs={},
            )

        inputs_dict = kernel_wrapper.get_inputs()
        configs_to_autotune = {k: inputs_dict[k] for k in keys_to_autotune}

        total_start_time = time.time()
        autotuned_configs = {}
        failed_configs = []

        for config_key, inputs in configs_to_autotune.items():
            logger.info("Autotuning config: %s", config_key)
            logger.debug(
                "Input shapes: %s",
                [getattr(inp, "shape", type(inp).__name__) for inp in inputs],
            )

            try:
                config_start_time = time.time()
                config = kernel_wrapper.run_autotune(inputs, autotune_effort)
                config_duration = time.time() - config_start_time

                # Save immediately for checkpointing
                config_manager.save_configs(kernel_name, platform, {config_key: config})

                autotuned_configs[config_key] = config
                logger.debug("Config details: %s", config)

                logger.info(
                    "✓ Autotuned and saved config '%s' (%.2fs)",
                    config_key,
                    config_duration,
                )

            except (RuntimeError, ValueError, OSError) as e:
                logger.exception(
                    "Failed to autotune config '%s': %s",
                    config_key,
                    e,
                )
                failed_configs.append(config_key)

        total_duration = time.time() - total_start_time
        successful = len(autotuned_configs)
        failed = len(failed_configs)

        logger.info(
            "Completed autotuning for kernel '%s': %d successful, %d failed (%.2fs)",
            kernel_name,
            successful,
            failed,
            total_duration,
        )

        status = "success" if failed == 0 else "partial"
        return AutotuneResult(
            status=status,
            successful=successful,
            failed=failed,
            configs=autotuned_configs,
        )

    except (KeyError, RuntimeError, ValueError, OSError) as e:
        error_msg = f"Unexpected error: {e}"
        logger.exception("Failed to autotune kernel '%s': %s", kernel_name, e)
        return AutotuneResult(
            status="error",
            message=error_msg,
            successful=0,
            failed=0,
            configs={},
        )


def summarize_results(results: dict[str, AutotuneResult]) -> bool:
    logger.info("=" * 50)
    logger.info("Autotuning Results Summary")
    logger.info("=" * 50)

    total_successful = 0
    total_failed = 0
    success_kernels = []
    partial_kernels = []
    error_kernels = []
    skipped_kernels = []

    for kernel_name, result in results.items():
        total_successful += result.successful
        total_failed += result.failed

        if result.status == "success":
            success_kernels.append(f"{kernel_name} ({result.successful} configs)")
            logger.info("✓ %s: %d configs successful", kernel_name, result.successful)
        elif result.status == "partial":
            partial_kernels.append(
                f"{kernel_name} ({result.successful} ok, {result.failed} failed)"
            )
            logger.warning(
                "⚠ %s: %d successful, %d failed",
                kernel_name,
                result.successful,
                result.failed,
            )
        elif result.status == "error":
            error_kernels.append(f"{kernel_name}: {result.message or 'Unknown error'}")
            logger.error("✗ %s: %s", kernel_name, result.message or "Unknown error")
        elif result.status == "skipped":
            skipped_kernels.append(f"{kernel_name}: {result.message or 'Skipped'}")
            logger.info("- %s: %s", kernel_name, result.message or "Skipped")

    logger.info("=" * 50)
    logger.info(
        "Summary: %d total configs (%d successful, %d failed)",
        total_successful + total_failed,
        total_successful,
        total_failed,
    )
    logger.info(
        "Kernels: %d success, %d partial, %d error, %d skipped",
        len(success_kernels),
        len(partial_kernels),
        len(error_kernels),
        len(skipped_kernels),
    )

    has_failures = bool(error_kernels or partial_kernels)

    if not has_failures:
        if total_successful > 0:
            logger.info("All configs autotuned successfully!")
        else:
            logger.info("No new configs were generated (all may already exist)")

    return not has_failures


def parse_kernel_spec(spec: str) -> tuple[str, int | None]:
    """Parse 'name' or 'name:version' into (name, version|None)."""
    if ":" in spec:
        name, ver_str = spec.rsplit(":", 1)
        try:
            ver = int(ver_str)
        except ValueError:
            logger.error(
                "Invalid version in '%s': '%s' is not an integer", spec, ver_str
            )
            sys.exit(1)
        return name, ver
    return spec, None


def get_kernels_to_autotune(
    requested: list[str] | None,
) -> dict[str, int]:
    """Parse kernel specs and resolve versions.

    Returns dict mapping kernel name to resolved version number.
    """
    all_kernels = get_registered_kernels()
    if not all_kernels:
        logger.error("No Helion kernels found in registry")
        sys.exit(1)

    if not requested:
        return {name: max(versions) for name, versions in all_kernels.items()}

    parsed = [parse_kernel_spec(spec) for spec in requested]

    names = [name for name, _ in parsed]
    if len(names) != len(set(names)):
        duplicates = [k for k in set(names) if names.count(k) > 1]
        logger.error("Duplicate kernel names in --kernels flag: %s", duplicates)
        sys.exit(1)

    result = {}
    errors = []
    for name, ver in parsed:
        if name not in all_kernels:
            errors.append(f"Kernel '{name}' not found")
            continue
        versions = all_kernels[name]
        if ver is not None:
            if ver not in versions:
                errors.append(
                    f"Version {ver} not found for kernel '{name}'. "
                    f"Available: {sorted(versions.keys())}"
                )
                continue
            result[name] = ver
        else:
            result[name] = max(versions)

    if errors:
        for err in errors:
            logger.error(err)
        logger.error("Available kernels: %s", list(all_kernels.keys()))
        sys.exit(1)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Autotune Helion kernels",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage:")[1] if "Usage:" in __doc__ else "",
    )

    parser.add_argument(
        "--kernels",
        nargs="+",
        help=(
            "Kernel(s) to autotune, with optional version suffix "
            "(e.g. 'silu_mul_fp8' or 'silu_mul_fp8:1'). "
            "Default: latest version of all kernels."
        ),
    )

    parser.add_argument(
        "--config-dir",
        type=str,
        help="Config directory for config files (default: vLLM helion configs dir)",
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="List available Helion kernels and exit",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Force re-autotuning even if configs already exist for the "
            "platform and config keys"
        ),
    )

    parser.add_argument(
        "--autotune-effort",
        type=str,
        default="quick",
        help=(
            "Helion autotune effort level: 'quick' (smaller search) or "
            "'full' (full search budget) (default: quick)"
        ),
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    import logging

    if args.verbose:
        logging.getLogger("vllm").setLevel(logging.DEBUG)
        logger.debug("Verbose mode enabled")
        logger.debug("Arguments: %s", vars(args))
    else:
        logging.getLogger("vllm").setLevel(logging.INFO)

    if args.list:
        list_kernels()
        return

    if not check_requirements():
        sys.exit(1)

    platform = get_canonical_gpu_name()
    logger.info("Detected GPU platform: %s", platform)

    config_manager = (
        ConfigManager(args.config_dir) if args.config_dir else ConfigManager()
    )

    try:
        config_manager.ensure_base_dir_writable()
    except OSError as e:
        logger.error("Failed to access config directory: %s", e)
        sys.exit(1)

    kernels_to_autotune = get_kernels_to_autotune(args.kernels)

    logger.info(
        "Will autotune %d kernel(s) for platform '%s': %s",
        len(kernels_to_autotune),
        platform,
        [f"{name}:v{ver}" for name, ver in kernels_to_autotune.items()],
    )

    results = {}
    for kernel_name, version in kernels_to_autotune.items():
        result = autotune_kernel(
            kernel_name,
            version,
            platform,
            config_manager,
            args.force,
            args.autotune_effort,
        )
        results[kernel_name] = result

    success = summarize_results(results)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
