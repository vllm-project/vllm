# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
vLLM CLI completion generator.

This script extracts subcommands and CLI arguments from the `vllm` CLI,
including nested commands and fixed-value options, and uses them to
populate a bash completion script template.

It supports structured rendering of:
- Subcommands
- Arguments for each subcommand
- Options with fixed value sets (for value completion)

Output is written to a rendered bash file using Jinja-style placeholders.

Usage:
    python cli_args_completion_generator.py

NOTE: 
Make sure the template `vllm-completion.template.bash` and the script
in the same directory.
"""
import shutil
import subprocess
import sys
from collections import defaultdict

import regex as re


def extract_all_options(subcommand: str):
    """
    Extracts all CLI options from all sections of `vllm <subcommand> --help`.
    Handles nested sections like 'Options:', 'ModelConfig:', etc.
    """
    try:
        res = subprocess.run(["vllm"] + subcommand.split() + ["--help"],
                             capture_output=True,
                             text=True,
                             check=True)
        lines = res.stdout.splitlines()
    except Exception as e:
        print(f"# Error: {e}")
        return []

    options = set()
    current_section = None

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Match section headers like "Options:", "ModelConfig:", etc.
        if re.match(r"^[\w\s-]+:$", line):
            current_section = line.rstrip(":")
            continue

        # Only extract from known parameter lines
        if current_section:
            # Extract options like -x, --xxx from lines like:
            # "-i INPUT, --input-file INPUT"
            matches = re.findall(
                r"(?<![\w-])(--[a-zA-Z](?:[\w-]*[a-zA-Z0-9])?|-[a-zA-Z])(?=\s|,|$)",
                line)

            options.update(matches)

    return options


def format_all_subcommand_options(subcommands: list[str]):
    """
    Generates a dict mapping subcommand names to quoted,
    space-separated option strings.

    Returns:
        dict[str, str]: Keys are like "serve_args", 
        values are e.g. '"--opt1 --opt2"'
    """
    result = {}
    for sub in subcommands:
        opts = extract_all_options(sub)
        key = sub.replace(" ", "_").replace("-", "_") + "_args"
        result[key] = f'"{" ".join(sorted(opts))}"'
    return result


def extract_subcommands_str(cmd_str: str):
    """
    Returns a space-separated string of subcommands from `vllm --help`.
    Example: "chat complete serve bench collect-env run-batch"
    """
    try:
        parts = cmd_str.split()
        result = subprocess.run(parts + ["--help"],
                                capture_output=True,
                                text=True,
                                check=True)
        lines = result.stdout.splitlines()
    except Exception as e:
        print(f"# Error running vllm --help: {e}")
        return ""

    for i, line in enumerate(lines):
        # Match line like: {chat,complete,serve,...}
        if (line.strip().lower().startswith("positional arguments:")
                and i + 1 < len(lines)
                and (match := re.search(r"\{([^}]+)\}", lines[i + 1]))):
            subcommands = " ".join(cmd.strip()
                                   for cmd in match.group(1).split(","))
            return f'"{subcommands}"'

    return ""


def extract_option_value(subcommand: str):
    """
    Extracts CLI options that provide a fixed set of values, like:
        --option {val1,val2,val3}
    """
    try:
        result = subprocess.run(["vllm"] + subcommand.split() + ["--help"],
                                capture_output=True,
                                text=True,
                                check=True)
        lines = result.stdout.splitlines()
    except Exception as e:
        print(f"# Error while fetching help for '{subcommand}': {e}")
        return {}

    option_values = defaultdict(list)

    for line in lines:
        line = line.strip()

        # Match pattern like: --option-name {val1,val2,...}
        match = re.match(r"(--[\w-]+)\s+\{([^}]+)\}", line)
        if match:
            option = match.group(1)
            values = [v.strip() for v in match.group(2).split(",")]
            option_values[option] = values

    return dict(option_values)


def extract_all_option_values(subcommands: list[str]):
    all_options = {}
    for sub in subcommands:
        opt_vals = extract_option_value(sub)
        for opt, vals in opt_vals.items():
            if opt in all_options:
                all_options[opt] = sorted(set(all_options[opt] + vals))
            else:
                all_options[opt] = vals
    return all_options


def format_option_value_map(option_value_dict: dict[str, list[str]]):
    """
    Format the dictionary into Bash-compatible declare syntax entries.
    """
    lines = []
    for opt, values in sorted(option_value_dict.items()):
        joined = " ".join(values)
        lines.append(f'    [{opt}]="{joined}"')
    return "\n".join(lines)


def render_bash_template(template_path: str, output_path: str,
                         replacements: dict):
    """
    Replace placeholders like {{ var }} in a bash template with given values.
    """
    with open(template_path, encoding="utf-8") as f:
        content = f.read()

    for placeholder, value in replacements.items():
        content = content.replace(placeholder, value)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)


if __name__ == "__main__":
    if shutil.which("vllm") is None:
        print("Warning: 'vllm' command not found. "
              "Skipping CLI completion generation.\n"
              "To enable auto-generation, "
              "ensure 'vllm' is installed and on your PATH.")
        sys.exit(0)

    # Used for shell completion of vLLM subcommands
    print("Generating subcommands...")
    vllm_subcommands = extract_subcommands_str("vllm")
    bench_subcommands = extract_subcommands_str("vllm bench")

    # Used for shell completion of CLI options for each vLLM subcommand
    print("Generating options...")
    subcommand_list = [
        "chat", "complete", "collect-env", "serve", "run-batch",
        "bench latency", "bench throughput", "bench serve"
    ]
    all_options = format_all_subcommand_options(subcommand_list)

    # Used for option value completion
    print("Generating option and value map...")
    all_opts = extract_all_option_values(subcommand_list)
    option_value_map = format_option_value_map(all_opts)

    render_bash_template(template_path="scripts/vllm-completion.template.bash",
                         output_path="scripts/vllm-completion.bash",
                         replacements={
                             "{{ subcommands }}":
                             vllm_subcommands,
                             "{{ bench_subcommands }}":
                             bench_subcommands,
                             "{{ option_value_map_entries }}":
                             option_value_map,
                             "{{ chat_args }}":
                             all_options["chat_args"],
                             "{{ complete_args }}":
                             all_options["complete_args"],
                             "{{ serve_args }}":
                             all_options["serve_args"],
                             "{{ run_batch_args }}":
                             all_options["run_batch_args"],
                             "{{ bench_latency_args }}":
                             all_options["bench_latency_args"],
                             "{{ bench_serve_args }}":
                             all_options["bench_serve_args"],
                             "{{ bench_throughput_args }}":
                             all_options["bench_throughput_args"],
                         })
