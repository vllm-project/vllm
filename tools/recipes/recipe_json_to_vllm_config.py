#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Convert a vLLM Recipes per-hardware JSON rendering into:

  1) config.yml - native `vllm serve --config` YAML
  2) env.sh     - environment variables required by the recipe

Examples:
  # Interactive discovery
  python3 recipe_json_to_vllm_config.py

  # Direct JSON URL
  python3 recipe_json_to_vllm_config.py \
    https://recipes.vllm.ai/meta-llama/Llama-3.1-8B-Instruct/hw/xeon6.json

  # Non-interactive discovery (recommended strategy)
  python3 recipe_json_to_vllm_config.py \
    --model meta-llama/Llama-3.1-8B-Instruct --hardware xeon6

  # Promoted variant + explicit strategy
  python3 recipe_json_to_vllm_config.py \
    --model nvidia/Llama-3.1-8B-Instruct-FP8 \
    --hardware arc_pro_b70 \
    --strategy single_node_tp

Then:
  source env.sh
  vllm serve --config config.yml

Variants with a distinct Hugging Face model ID are promoted by the Recipes API
and selected through --model; this converter does not need a separate --variant
argument. Strategy selection follows the exact `alternatives` JSON links emitted
by the Recipes API instead of synthesizing strategy URLs locally.

The converter intentionally targets a single `vllm serve` process. If the
recipe rendering is multi-node, PD-disaggregated, or another multi-process
deployment, it exits instead of silently generating an incomplete config.
"""

from __future__ import annotations

import argparse
import difflib
import json
import shlex
import sys
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:
    raise SystemExit("PyYAML is required. Install it with: pip install pyyaml") from exc

DEFAULT_API_BASE = "https://recipes.vllm.ai"

SHORT_ALIASES = {
    "-tp": "tensor-parallel-size",
    "-pp": "pipeline-parallel-size",
    "-dp": "data-parallel-size",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert vLLM Recipes JSON to native vllm serve config.yml + env.sh"
    )
    p.add_argument(
        "source",
        nargs="?",
        help=(
            "Per-hardware recipe JSON URL or local JSON file. "
            "If omitted, discover a recipe from the Recipes API."
        ),
    )
    p.add_argument(
        "--model",
        help=(
            "Model hf_id or search text for recipe discovery. Promoted recipe "
            "variants are selected by their own hf_id. Use with --hardware for "
            "non-interactive operation."
        ),
    )
    p.add_argument(
        "--hardware",
        help=(
            "Hardware ID for recipe discovery, for example xeon6, b200, or arc_pro_b70."
        ),
    )
    p.add_argument(
        "--strategy",
        help=(
            "Serving strategy for recipe discovery, for example single_node_tp. "
            "When omitted with --model and --hardware, use the Recipes-recommended "
            "strategy. Interactive discovery offers the recommended strategy and "
            "all generated alternatives."
        ),
    )
    p.add_argument(
        "--api-base",
        default=DEFAULT_API_BASE,
        help=f"Recipes API base URL (default: {DEFAULT_API_BASE})",
    )
    p.add_argument(
        "--config-out",
        default="config.yml",
        help="Output vLLM YAML config (default: config.yml)",
    )
    p.add_argument(
        "--env-out",
        default="env.sh",
        help="Output shell environment file (default: env.sh)",
    )

    tuning = p.add_argument_group(
        "optional runtime tuning",
        (
            "Refine the recipe baseline only when additional information is supplied. "
            "Runtime tuning is enabled only for recipe hardware with a registered "
            "policy (currently: xeon6)."
        ),
    )
    tuning.add_argument(
        "--detect-hardware",
        action="store_true",
        help=(
            "Detect effective CPU/NUMA/memory resources and allow the selected "
            "recipe-hardware policy to override recipe runtime arguments."
        ),
    )
    tuning.add_argument(
        "--input-tokens",
        type=int,
        help="Expected input-token length. Optional workload hint.",
    )
    tuning.add_argument(
        "--output-tokens",
        type=int,
        help="Expected output-token length. Optional workload hint.",
    )
    tuning.add_argument(
        "--concurrency",
        type=int,
        help="Expected maximum concurrent requests. Optional workload hint.",
    )
    tuning.add_argument(
        "--ttft-sla-ms",
        type=float,
        help="Optional time-to-first-token objective in milliseconds.",
    )
    tuning.add_argument(
        "--tpot-sla-ms",
        type=float,
        help="Optional time-per-output-token objective in milliseconds.",
    )
    tuning.add_argument(
        "--target-qps",
        type=float,
        help="Optional capacity target for future DP/capacity tuning.",
    )

    sweep = p.add_argument_group(
        "optional performance sweep",
        ("Generate benchmark files after creating one initial runtime suggestion."),
    )
    sweep.add_argument(
        "--generate-sweep",
        action="store_true",
        help=(
            "Generate an optional vllm bench sweep package around the single "
            "initial runtime suggestion."
        ),
    )
    sweep.add_argument(
        "--sweep-out-dir",
        default="sweep",
        help="Output directory for optional sweep files (default: sweep).",
    )
    return p.parse_args()


def api_url(base: str, path: str) -> str:
    return urllib.parse.urljoin(base.rstrip("/") + "/", path.lstrip("/"))


def load_json(source: str) -> dict[str, Any] | list[Any]:
    if source.startswith(("http://", "https://")):
        req = urllib.request.Request(
            source,
            headers={"User-Agent": "vllm-recipe-config-converter/1.0"},
        )
        with urllib.request.urlopen(req, timeout=30) as response:
            return json.load(response)

    with open(source, encoding="utf-8") as f:
        return json.load(f)


def prompt(text: str) -> str:
    try:
        return input(text).strip()
    except EOFError as exc:
        raise ValueError(
            "Interactive input is unavailable. Pass --model and --hardware, "
            "or provide a recipe JSON URL/file."
        ) from exc


def model_label(model: dict[str, Any]) -> str:
    label = str(model.get("hf_id", ""))
    title = model.get("title")
    provider = model.get("provider")
    if title and title != label:
        label += f" — {title}"
    if provider:
        label += f" [{provider}]"
    return label


def search_models(
    models: list[dict[str, Any]], query: str, limit: int = 20
) -> list[dict[str, Any]]:
    q = query.strip().lower()
    if not q:
        return []

    exact = [model for model in models if str(model.get("hf_id", "")).lower() == q]
    if exact:
        return exact

    tokens = [token for token in q.replace("/", " ").replace("-", " ").split() if token]
    scored: list[tuple[float, dict[str, Any]]] = []

    for model in models:
        hf_id = str(model.get("hf_id", "")).lower()
        title = str(model.get("title", "")).lower()
        provider = str(model.get("provider", "")).lower()
        text = f"{hf_id} {title} {provider}"

        if q in hf_id:
            score = 1000.0
        elif q in title:
            score = 900.0
        elif tokens and all(token in text for token in tokens):
            score = 800.0
        elif tokens and any(token in text for token in tokens):
            score = 500.0
        else:
            ratio = difflib.SequenceMatcher(None, q, hf_id).ratio()
            if ratio < 0.30:
                continue
            score = ratio * 100.0

        scored.append((score, model))

    scored.sort(key=lambda pair: (-pair[0], str(pair[1].get("hf_id", "")).lower()))
    return [model for _, model in scored[:limit]]


def choose_from_menu(items: list[Any], label_fn, prompt_text: str) -> Any:
    if not items:
        raise ValueError("No selectable items found.")

    if len(items) == 1:
        print(f"Selected: {label_fn(items[0])}")
        return items[0]

    for index, item in enumerate(items, start=1):
        print(f"  [{index}] {label_fn(item)}")

    while True:
        answer = prompt(prompt_text)
        try:
            index = int(answer)
        except ValueError:
            print(f"Enter a number from 1 to {len(items)}.")
            continue
        if 1 <= index <= len(items):
            return items[index - 1]
        print(f"Enter a number from 1 to {len(items)}.")


def select_model(models: list[dict[str, Any]], requested: str | None) -> dict[str, Any]:
    query = requested
    while True:
        if not query:
            query = prompt("Model search (for example: llama 3.1): ")

        matches = search_models(models, query)
        if matches:
            print("\nMatching models:")
            return choose_from_menu(matches, model_label, "Select model: ")

        if requested:
            raise ValueError(f"No recipe model matched {requested!r}.")

        print(f"No recipe model matched {query!r}. Try again.")
        query = None


def select_hardware(
    by_hardware: dict[str, str], requested: str | None
) -> tuple[str, str]:
    hardware_ids = sorted(by_hardware)

    if requested:
        selected = next(
            (hw for hw in hardware_ids if hw.lower() == requested.lower()),
            None,
        )
        if selected is None:
            raise ValueError(
                f"Hardware {requested!r} is not available for this model. "
                f"Available: {', '.join(hardware_ids)}"
            )
        return selected, by_hardware[selected]

    print("\nAvailable hardware:")
    selected = choose_from_menu(hardware_ids, lambda value: value, "Select hardware: ")
    return selected, by_hardware[selected]


def strategy_sources(
    api_base: str, hardware_json_url: str, recipe: dict[str, Any]
) -> tuple[str, dict[str, str]]:
    recommended = recipe.get("strategy")
    if not isinstance(recommended, str) or not recommended:
        raise ValueError(
            "Hardware recipe JSON does not contain a usable `strategy` field."
        )

    sources = {recommended: hardware_json_url}
    raw_alternatives = recipe.get("alternatives") or {}
    if not isinstance(raw_alternatives, dict):
        raise ValueError(
            "Hardware recipe JSON `alternatives` must be an object when present."
        )

    for strategy, path in raw_alternatives.items():
        if isinstance(strategy, str) and strategy and isinstance(path, str) and path:
            sources[strategy] = api_url(api_base, path)

    return recommended, sources


def select_strategy(
    api_base: str,
    hardware_json_url: str,
    recipe: dict[str, Any],
    requested: str | None,
    interactive: bool,
) -> tuple[str, str]:
    recommended, sources = strategy_sources(api_base, hardware_json_url, recipe)
    strategies = [recommended, *sorted(s for s in sources if s != recommended)]

    if requested:
        selected = next(
            (
                strategy
                for strategy in strategies
                if strategy.lower() == requested.lower()
            ),
            None,
        )
        if selected is None:
            raise ValueError(
                f"Strategy {requested!r} is not available for this model/hardware. "
                f"Available: {', '.join(strategies)}"
            )
        return selected, sources[selected]

    if not interactive:
        return recommended, hardware_json_url

    print("\nAvailable strategies:")

    def strategy_label(strategy: str) -> str:
        if strategy == recommended:
            return f"{strategy} (recommended)"
        return strategy

    selected = choose_from_menu(strategies, strategy_label, "Select strategy: ")
    return selected, sources[selected]


def discover_recipe_source(
    api_base: str,
    requested_model: str | None,
    requested_hardware: str | None,
    requested_strategy: str | None,
) -> str:
    print("No recipe JSON supplied; starting Recipes API discovery.")

    models_url = api_url(api_base, "/models.json")
    models_data = load_json(models_url)
    if not isinstance(models_data, list):
        raise ValueError(f"{models_url} did not return a model list.")

    models = [model for model in models_data if isinstance(model, dict)]
    model = select_model(models, requested_model)

    model_json_path = model.get("json")
    if not isinstance(model_json_path, str) or not model_json_path:
        raise ValueError(f"Selected model {model.get('hf_id')!r} has no JSON API path.")

    model_json_url = api_url(api_base, model_json_path)
    model_data = load_json(model_json_url)
    if not isinstance(model_data, dict):
        raise ValueError(f"{model_json_url} did not return a JSON object.")

    recommended = model_data.get("recommended_command")
    if not isinstance(recommended, dict):
        raise ValueError(
            f"Model {model.get('hf_id')!r} has no rendered "
            "recommended_command in the Recipes API."
        )

    raw_by_hardware = recommended.get("by_hardware")
    if not isinstance(raw_by_hardware, dict) or not raw_by_hardware:
        raise ValueError(
            f"Model {model.get('hf_id')!r} has no per-hardware renderings "
            "in recommended_command.by_hardware."
        )

    by_hardware = {
        str(hw): path
        for hw, path in raw_by_hardware.items()
        if isinstance(path, str) and path
    }
    if not by_hardware:
        raise ValueError(
            f"Model {model.get('hf_id')!r} has no usable hardware JSON paths."
        )

    hardware, path = select_hardware(by_hardware, requested_hardware)
    hardware_json_url = api_url(api_base, path)
    hardware_data = load_json(hardware_json_url)
    if not isinstance(hardware_data, dict):
        raise ValueError(f"{hardware_json_url} did not return a JSON object.")

    interactive = requested_model is None or requested_hardware is None
    strategy, resolved = select_strategy(
        api_base,
        hardware_json_url,
        hardware_data,
        requested_strategy,
        interactive,
    )

    print("\nResolved recipe:")
    print(f"  Model:    {model.get('hf_id')}")
    print(f"  Hardware: {hardware}")
    print(f"  Strategy: {strategy}")
    print(f"  JSON:     {resolved}")
    print()

    return resolved


def coerce(value: str) -> Any:
    """Convert CLI string values to useful YAML scalar/object types."""
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return value


def is_option_token(token: str) -> bool:
    if token.startswith("--"):
        return True
    if token in SHORT_ALIASES or token == "-O":
        return True
    return token.startswith("-O") and len(token) > 2


def merge_value(dst: dict[str, Any], path: list[str], value: Any) -> None:
    """Merge dotted CLI args into nested YAML dictionaries."""
    cur = dst
    for part in path[:-1]:
        existing = cur.get(part)
        if existing is None:
            cur[part] = {}
        elif not isinstance(existing, dict):
            raise ValueError(
                f"Cannot merge nested option {'.'.join(path)!r}: "
                f"{part!r} is already a scalar"
            )
        cur = cur[part]

    leaf = path[-1]
    if leaf not in cur:
        cur[leaf] = value
        return

    # Repeated CLI option. Preserve all values.
    old = cur[leaf]
    if not isinstance(old, list):
        old = [old]
    if isinstance(value, list):
        old.extend(value)
    else:
        old.append(value)
    cur[leaf] = old


def normalize_key(raw_key: str) -> list[str]:
    """
    Convert the CLI key to config-file spelling.

    Only the top-level CLI option name gets underscore -> dash normalization.
    Nested JSON field names after a dot are preserved.
    """
    parts = raw_key.split(".")
    parts[0] = parts[0].replace("_", "-")
    return parts


def argv_to_config(argv: list[Any]) -> dict[str, Any]:
    argv = [str(x) for x in argv]

    if len(argv) < 3 or argv[0:2] != ["vllm", "serve"]:
        raise ValueError(
            "Expected recipe argv to start with: ['vllm', 'serve', MODEL, ...]. "
            f"Got: {argv[:4]!r}"
        )

    model = argv[2]
    if model.startswith("-"):
        raise ValueError(f"Expected model after 'vllm serve', got {model!r}")

    config: dict[str, Any] = {"model": model}

    i = 3
    while i < len(argv):
        token = argv[i]

        # -O3 / -O=3
        if token.startswith("-O") and token != "-O":
            value = token[3:] if token.startswith("-O=") else token[2:]
            merge_value(config, ["optimization-level"], coerce(value))
            i += 1
            continue

        # -O 3
        if token == "-O":
            if i + 1 >= len(argv):
                raise ValueError("-O is missing its value")
            merge_value(config, ["optimization-level"], coerce(argv[i + 1]))
            i += 2
            continue

        # Selected common short aliases.
        if token in SHORT_ALIASES:
            if i + 1 >= len(argv):
                raise ValueError(f"{token} is missing its value")
            merge_value(
                config,
                [SHORT_ALIASES[token]],
                coerce(argv[i + 1]),
            )
            i += 2
            continue

        if not token.startswith("--"):
            raise ValueError(
                f"Unexpected positional/short argument {token!r}. "
                "The converter expects Recipes to emit long-form vLLM serve options."
            )

        # --key=value
        if "=" in token:
            key, raw_value = token[2:].split("=", 1)
            merge_value(config, normalize_key(key), coerce(raw_value))
            i += 1
            continue

        key = token[2:]
        i += 1

        # Gather values until the next option. This supports both scalar and
        # nargs-style vLLM options.
        raw_values: list[str] = []
        while i < len(argv) and not is_option_token(argv[i]):
            raw_values.append(argv[i])
            i += 1

        if not raw_values:
            # Store flags exactly by their long-form spelling.
            # Example:
            #   --trust-remote-code        -> trust-remote-code: true
            #   --no-enable-prefix-caching -> no-enable-prefix-caching: true
            value: Any = True
        elif len(raw_values) == 1:
            value = coerce(raw_values[0])
        else:
            value = [coerce(v) for v in raw_values]

        merge_value(config, normalize_key(key), value)

    return config


def recipe_argv(recipe: dict[str, Any]) -> list[Any]:
    deploy_type = recipe.get("deploy_type")

    # Current Recipes API's single-node rendering exposes `argv`.
    if isinstance(recipe.get("argv"), list):
        if deploy_type not in (None, "single_node"):
            raise ValueError(
                f"Recipe deploy_type={deploy_type!r} is not a single-node deployment. "
                "A single config.yml would not fully represent this deployment."
            )
        return recipe["argv"]

    # Give a useful failure for other known rendered shapes.
    multi_process_fields = [
        k
        for k in (
            "head_argv",
            "worker_argv",
            "worker_argvs",
            "prefill",
            "decode",
            "vllm_argv",
        )
        if k in recipe
    ]
    if multi_process_fields:
        raise ValueError(
            "This recipe is a multi-process deployment and cannot be represented "
            "by one config.yml. Found fields: " + ", ".join(multi_process_fields)
        )

    raise ValueError("Recipe JSON does not contain an `argv` field")


def write_config(
    path: str,
    source: str,
    recipe: dict[str, Any],
    config: dict[str, Any],
) -> None:
    metadata = [
        "# Generated from vLLM Recipes JSON.",
        f"# Source: {source}",
    ]
    for key in ("hardware", "strategy", "variant", "deploy_type"):
        if recipe.get(key) is not None:
            metadata.append(f"# {key}: {recipe[key]}")

    body = yaml.safe_dump(
        config,
        sort_keys=False,
        default_flow_style=False,
        allow_unicode=True,
    )
    Path(path).write_text("\n".join(metadata) + "\n" + body, encoding="utf-8")


def write_env(path: str, source: str, recipe: dict[str, Any]) -> None:
    env = recipe.get("env") or {}
    if not isinstance(env, dict):
        raise ValueError(f"Recipe `env` must be an object, got {type(env).__name__}")

    lines = [
        "#!/usr/bin/env bash",
        "# Generated from vLLM Recipes JSON.",
        f"# Source: {source}",
        "",
    ]

    if env:
        for key, value in env.items():
            lines.append(f"export {key}={shlex.quote(str(value))}")
    else:
        lines.append("# No recipe-specific environment variables.")

    lines.append("")
    Path(path).write_text("\n".join(lines), encoding="utf-8")
    Path(path).chmod(Path(path).stat().st_mode | 0o111)


def main() -> int:
    args = parse_args()

    try:
        source = args.source
        if source is None:
            source = discover_recipe_source(
                args.api_base, args.model, args.hardware, args.strategy
            )
        elif args.model or args.hardware or args.strategy:
            raise ValueError(
                "Do not combine a positional recipe JSON source with "
                "--model/--hardware/--strategy discovery options."
            )

        recipe = load_json(source)
        if not isinstance(recipe, dict):
            raise ValueError("Recipe JSON must be a JSON object.")

        argv = recipe_argv(recipe)
        config = argv_to_config(argv)

        tuning_requested = (
            args.detect_hardware
            or args.generate_sweep
            or any(
                value is not None
                for value in (
                    args.input_tokens,
                    args.output_tokens,
                    args.concurrency,
                    args.ttft_sla_ms,
                    args.tpot_sla_ms,
                    args.target_qps,
                )
            )
        )

        tuning = None
        workload = None
        sweep_writer = None
        if tuning_requested:
            # Keep plain Recipes conversion lightweight. vLLM-specific modules
            # are imported only for optional runtime tuning or sweep generation.
            from runtime_tuning import (
                WorkloadHints,
                finetune_runtime_config,
                get_runtime_tuning_policies,
            )

            workload = WorkloadHints(
                input_tokens=args.input_tokens,
                output_tokens=args.output_tokens,
                concurrency=args.concurrency,
                ttft_sla_ms=args.ttft_sla_ms,
                tpot_sla_ms=args.tpot_sla_ms,
                target_qps=args.target_qps,
            )

            if args.generate_sweep:
                from sweep_generation import (
                    validate_sweep_workload,
                    write_sweep_files,
                )

                validate_sweep_workload(workload)
                sweep_writer = write_sweep_files

            recipe_hardware = recipe.get("hardware")
            policies = get_runtime_tuning_policies(recipe_hardware)

            hardware = None
            if args.detect_hardware:
                from hardware_detection import detect_hardware

                hardware = detect_hardware()

            tuning = finetune_runtime_config(
                config,
                hardware=hardware,
                workload=workload,
                policies=policies,
            )
            config.update(tuning.overrides)

        write_config(args.config_out, source, recipe, config)
        write_env(args.env_out, source, recipe)

        sweep_files: list[Path] = []
        if args.generate_sweep:
            assert workload is not None
            assert sweep_writer is not None
            sweep_files = sweep_writer(
                args.sweep_out_dir,
                config_path=args.config_out,
                env_path=args.env_out,
                config=config,
                workload=workload,
            )

        if tuning is not None:
            if tuning.overrides:
                print("Initial runtime suggestion:")
                for key, value in tuning.overrides.items():
                    print(f"  {key}: {value}")
            for note in tuning.notes:
                print(f"  tuning: {note}")
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"Wrote {args.config_out}")
    print(f"Wrote {args.env_out}")
    if sweep_files:
        print(f"Wrote optional sweep package under {args.sweep_out_dir}/")
    print()
    print("Run:")
    print(f"  source {shlex.quote(args.env_out)}")
    print(f"  vllm serve --config {shlex.quote(args.config_out)}")
    if sweep_files:
        sweep_dir = Path(args.sweep_out_dir)
        run_sweep = sweep_dir / "run_sweep.sh"
        recommend = sweep_dir / "recommend.py"
        print()
        print("Optional performance sweep:")
        print(f"  {shlex.quote(str(run_sweep))} --dry-run")
        print(f"  {shlex.quote(str(run_sweep))}")
        print()
        print("After the sweep:")
        print(f"  {shlex.quote(str(recommend))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
