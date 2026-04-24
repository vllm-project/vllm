# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
apply_hardware_profiles.py

Emit shell `export` statements for hardware-specific environment variables and
CLI arguments based on a YAML config. Designed to be *sourced* via process
substitution so the exports affect the caller's shell.

Environment variables from the ``env:`` section are exported directly.
CLI arguments from the ``args:`` section are collected into a single
``VLLM_HARDWARE_PROFILE_ARGS`` variable (e.g. ``--attention-backend FLASHINFER``)
that can be appended to ``vllm serve`` or ``vllm bench`` commands.

Typical usage:
    source <(python3 apply_hardware_profiles.py --gpu-type "$GPU_TYPE")
"""

import argparse
import shlex
from pathlib import Path

import yaml

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="apply_hardware_profiles.py",
        description=(
            "Print shell export statements for env vars defined by a "
            "hardware profile.\n"
            "Intended to be sourced so exports affect the caller's environment."
        ),
        epilog=(
            "Example:\n"
            "  GPU_TYPE=h100 source <(python3 apply_hardware_profiles.py "
            '--gpu-type "$GPU_TYPE")\n\n'
            "Precedence: existing environment > profile value."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--gpu-type",
        type=str,
        required=True,
        choices=["h100", "b200", "gb200", "mi300x"],
        help="Target GPU type whose profile to load "
        "(applies profile named vllm-<gpu-type>).",
    )
    args = parser.parse_args()

    # Load hardware profiles from YAML file
    script_dir = Path(__file__).resolve().parent
    config_path = script_dir / "../configs/hardware_profiles.yaml"
    with open(config_path) as f:
        hardware_profiles = yaml.safe_load(f)

    # Collect env vars and CLI args from matching profiles.
    # ``vllm-default`` args are always applied as baseline defaults.
    # GPU-specific profiles (``vllm-<gpu_type>``) override/extend them.
    # NOTE: ``vllm-default`` *env* vars are NOT applied here — they are
    # managed separately (e.g. VLLM_USE_V1 is set in setup_tests.sh).
    env_vars: dict[str, str] = {}
    cli_args: dict[str, str] = {}
    gpu_key = f"vllm-{args.gpu_type}"
    for profile in hardware_profiles.get("profiles", []):
        name = profile.get("name")
        if name == "vllm-default":
            cli_args.update(profile.get("args", {}))
        elif name == gpu_key:
            env_vars.update(profile.get("env", {}))
            cli_args.update(profile.get("args", {}))

    # Print exports in a way that preserves any pre-existing user overrides.
    # The ${KEY:-VALUE} form uses VALUE only when $KEY is unset or null.
    # IMPORTANT: Values are emitted verbatim; if they contain spaces or special chars,
    # users may need to quote/escape in the YAML to ensure shell safety.
    for key, value in env_vars.items():
        print(f'export {key}="${{{key}:-{value}}}"')

    # Collect CLI args (from the ``args:`` sections) into a single variable
    # that can be appended to vllm commands. Keys in YAML should already match
    # vllm CLI flag names in kebab-case.
    #
    # Value handling:
    #   ""          → boolean flag, emit ``--key`` only
    #   list        → ``--key val1 val2 val3`` (for nargs='+' arguments)
    #   other       → ``--key value``
    #
    # Values are shell-quoted with shlex.quote() so that special characters
    # (e.g. JSON strings with braces/quotes) survive two levels of shell
    # parsing: once when this output is *sourced*, and again when the
    # variable is expanded inside a ``bash -c`` command string.
    if cli_args:
        parts: list[str] = []
        for k, v in cli_args.items():
            flag = f"--{k}"
            if v == "":
                parts.append(flag)
            elif isinstance(v, list):
                parts.append(f"{flag} {' '.join(shlex.quote(str(item)) for item in v)}")
            else:
                parts.append(f"{flag} {shlex.quote(str(v))}")
        args_str = " ".join(parts)
        print(f"export VLLM_HARDWARE_PROFILE_ARGS={shlex.quote(args_str)}")
