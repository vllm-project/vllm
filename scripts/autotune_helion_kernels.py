#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Script to autotune Helion kernels using the kernel registry.

This script discovers all registered Helion kernels via @register_kernel decorator
and runs autotuning to generate optimized configurations for different input shapes.

Usage:
    # Autotune all Helion kernels (default behavior)
    python scripts/autotune_helion_kernels.py

    # Explicitly autotune all Helion kernels
    python scripts/autotune_helion_kernels.py --kernels all

    # Autotune specific kernel by name
    python scripts/autotune_helion_kernels.py --kernels silu_mul_fp8

    # Autotune multiple specific kernels
    python scripts/autotune_helion_kernels.py --kernels silu_mul_fp8 rms_norm_fp8

    # Autotune with custom output directory
    python scripts/autotune_helion_kernels.py --kernels all --output-dir ./my_configs

    # List available Helion kernels
    python scripts/autotune_helion_kernels.py --list

Requirements:
    - CUDA GPU available
    - Helion package installed
    - vLLM environment setup
"""

import argparse
import os
import sys
import time

import torch

# Add vLLM to path if not already available
try:
    from vllm.compilation.helion.config_manager import ConfigManager
    from vllm.compilation.helion.register import (
        HELION_AVAILABLE,
        get_kernel_by_name,
        get_registered_kernels,
    )
    from vllm.config import VllmConfig
    from vllm.config.compilation import CompilationConfig
    from vllm.config.vllm import set_current_vllm_config
    from vllm.logger import init_logger
except ImportError as e:
    print(f"Error importing vLLM: {e}")
    print("Please ensure vLLM is installed and in your Python path")
    sys.exit(1)

logger = init_logger("vllm.scripts.autotune_helion_kernels")


def get_default_config_dir() -> str:
    """
    Get the default configuration directory using ConfigManager.

    Returns:
        Default path for Helion configs
    """
    config_manager = ConfigManager.get_instance()
    return str(config_manager.get_base_dir())


def get_helion_kernels() -> dict[str, "HelionKernelWrapper"]:
    """
    Discover all registered Helion kernels.

    Returns:
        Dictionary mapping kernel names to HelionKernelWrapper instances
    """
    return get_registered_kernels()


def list_kernels():
    """List all available Helion kernels."""
    kernels = get_helion_kernels()

    if not kernels:
        print("No Helion kernels found in registry.")
        return

    print("Available Helion kernels:")
    print("=" * 50)

    for name, kernel_wrapper in kernels.items():
        # Get description from the kernel wrapper
        base_op_name = kernel_wrapper.base_op_name
        print(f"  {name:<30} - {base_op_name}")

    print(f"\nTotal: {len(kernels)} kernels")


def check_requirements() -> bool:
    """
    Check if all requirements are met for autotuning.

    Returns:
        True if requirements are met, False otherwise
    """
    # Check CUDA availability
    if not torch.cuda.is_available():
        logger.error("CUDA is not available. Helion autotuning requires GPU.")
        return False

    # Check Helion availability
    if not HELION_AVAILABLE:
        logger.error("Helion is not installed. Please install Helion package.")
        return False

    return True


def autotune_kernel(
    kernel_name: str,
    kernel_wrapper: "HelionKernelWrapper",
    output_dir: str,
    force: bool = False,
) -> bool:
    """
    Autotune a specific Helion kernel.

    Args:
        kernel_name: Name of the kernel
        kernel_wrapper: HelionKernelWrapper instance
        output_dir: Output directory for configs
        force: Force re-autotuning even if configs exist

    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info("Autotuning kernel: %s", kernel_name)

        # Initialize config manager once for the entire function
        config_manager = ConfigManager(output_dir)
        config_manager.ensure_base_dir_exists()

        # Get autotune inputs to check what will be generated
        autotune_inputs = kernel_wrapper.get_autotune_inputs()
        logger.info(
            "Will generate %d configs: %s",
            len(autotune_inputs),
            list(autotune_inputs.keys()),
        )

        # Filter out existing configs (unless forcing)
        configs_to_autotune = autotune_inputs
        if not force:
            existing_configs = []
            configs_to_autotune = {}

            for config_key, inputs in autotune_inputs.items():
                if config_manager.config_exists(kernel_name, config_key):
                    existing_configs.append(config_key)
                    logger.info("Config %s already exists, skipping", config_key)
                else:
                    configs_to_autotune[config_key] = inputs

            if existing_configs and configs_to_autotune:
                logger.info(
                    "Found existing configs for %s, will autotune remaining: %s",
                    existing_configs,
                    list(configs_to_autotune.keys()),
                )
            elif existing_configs and not configs_to_autotune:
                logger.info(
                    "All configs already exist for %s, use --force to re-generate",
                    existing_configs,
                )
                return True
            elif not existing_configs:
                logger.info(
                    "No existing configs found, will autotune all: %s",
                    list(configs_to_autotune.keys()),
                )

        if not configs_to_autotune:
            logger.info("No configs to autotune for %s", kernel_name)
            return True

        saved_configs = []
        failed_configs = []
        total_start_time = time.time()

        logger.info("Starting autotuning for %d configs", len(configs_to_autotune))

        # Autotune and save each config immediately
        for config_key, inputs in configs_to_autotune.items():
            logger.info("Autotuning config: %s", config_key)

            try:
                # Autotune this single config
                config_start_time = time.time()
                config = kernel_wrapper.run_autotune(inputs)
                config_end_time = time.time()

                # Save immediately after successful autotuning
                try:
                    config_path = config_manager.save_config(
                        kernel_name, config_key, config
                    )
                    saved_configs.append(config_key)

                    config_duration = config_end_time - config_start_time
                    logger.info(
                        "✓ Autotuned and saved config %s (%.2fs) to: %s",
                        config_key,
                        config_duration,
                        config_path,
                    )

                except Exception as e:
                    logger.error("Failed to save config %s: %s", config_key, e)
                    failed_configs.append(config_key)

            except Exception as e:
                logger.error("Autotuning failed for config %s: %s", config_key, e)
                failed_configs.append(config_key)

        total_end_time = time.time()
        total_duration = total_end_time - total_start_time

        logger.info("All autotuning completed in %.2fs", total_duration)
        logger.info(
            "Successfully generated and saved %d/%d configs for %s: %s",
            len(saved_configs),
            len(configs_to_autotune),
            kernel_name,
            saved_configs,
        )

        if failed_configs:
            logger.warning("Failed configs for %s: %s", kernel_name, failed_configs)

        return True

    except Exception as e:
        logger.error("Failed to autotune %s: %s", kernel_name, e)
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Autotune Helion kernels",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage:")[1] if "Usage:" in __doc__ else "",
    )

    parser.add_argument(
        "--kernels",
        nargs="+",
        help="Kernel(s) to autotune. Can specify individual kernel names or 'all' (default: all kernels)"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for config files "
        "(default: <vllm_repo>/vllm/compilation/helion/configs)",
    )

    parser.add_argument(
        "--list", action="store_true", help="List available Helion kernels and exit"
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-autotuning even if config files already exist",
    )

    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Set up logging
    if args.verbose:
        import logging

        logging.getLogger("vllm").setLevel(logging.DEBUG)

    # List kernels if requested
    if args.list:
        list_kernels()
        return

    # Check requirements
    if not check_requirements():
        sys.exit(1)

    # Configure vLLM to enable all custom ops for autotuning
    # This overrides the default "none" setting when using inductor backend
    vllm_config = VllmConfig(
        compilation_config=CompilationConfig(
            custom_ops=["all"]  # Enable all custom ops including Helion kernels
        )
    )

    # Set the config context for this autotuning session
    set_current_vllm_config(vllm_config)
    logger.info("Enabled all custom ops for autotuning")

    # Get available kernels
    helion_kernels = get_helion_kernels()

    if not helion_kernels:
        logger.error("No Helion kernels found in registry")
        sys.exit(1)

    # Filter to specific kernels if requested
    if args.kernels:
        # Handle 'all' as a special case
        if len(args.kernels) == 1 and args.kernels[0].lower() == 'all':
            # Keep all kernels - no filtering needed
            logger.info("Autotuning all %d kernels", len(helion_kernels))
        else:
            # Filter to specified kernels
            filtered_kernels = {}
            missing_kernels = []

            for kernel_name in args.kernels:
                if kernel_name in helion_kernels:
                    filtered_kernels[kernel_name] = helion_kernels[kernel_name]
                else:
                    missing_kernels.append(kernel_name)

            if missing_kernels:
                logger.error("Kernel(s) not found: %s", missing_kernels)
                logger.error("Available kernels:")
                for name in helion_kernels:
                    logger.error("  - %s", name)
                sys.exit(1)

            helion_kernels = filtered_kernels
            logger.info("Selected %d kernel(s): %s", len(helion_kernels), list(helion_kernels.keys()))

    # Determine output directory
    output_dir = args.output_dir if args.output_dir else get_default_config_dir()

    # Create output directory
    try:
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Output directory: %s", output_dir)

        # Verify directory is writable
        test_file = os.path.join(output_dir, ".write_test")
        try:
            with open(test_file, "w") as f:
                f.write("test")
            os.remove(test_file)
        except Exception as e:
            logger.error("Output directory is not writable: %s", e)
            sys.exit(1)

    except Exception as e:
        logger.error("Failed to create output directory '%s': %s", output_dir, e)
        sys.exit(1)

    # Autotune kernels
    total_kernels = len(helion_kernels)
    successful = 0

    logger.info("Starting autotuning for %d kernel(s)", total_kernels)

    for kernel_name, kernel_wrapper in helion_kernels.items():
        if autotune_kernel(kernel_name, kernel_wrapper, output_dir, args.force):
            successful += 1
        else:
            logger.warning("Skipped or failed: %s", kernel_name)

    # Summary
    logger.info("=" * 50)
    logger.info(
        "Autotuning complete: %d/%d kernels successful", successful, total_kernels
    )

    if successful < total_kernels:
        logger.warning("%d kernels failed or were skipped", total_kernels - successful)
        sys.exit(1)
    else:
        logger.info("All kernels autotuned successfully!")


if __name__ == "__main__":
    main()
