#!/usr/bin/env python3
"""Assert an fp8 overlay will actually be served as fp8, before burning a run.

The tensors and the config that tells vLLM how to read them fail
independently. On 2026-09-19 the o_proj overlay held 93 byte-perfect fp8
weights with correct scales -- and served every one of them as bf16, because
`re:.*self_attn.*` was still in the `ignore` list and `should_ignore_layer` is
consulted BEFORE `config_groups`. The run loaded, served, and would have
scored, as the control. An hour of GSM8k, measuring nothing.

Nothing downstream can catch that: the server log says "93 delegated", the
accuracy is fine because the model is unchanged, and the perf number looks
like a null result rather than a bug.

This checks the actual question -- for every tensor the overlay quantized,
does the layer resolve to the fp8 group? -- using vLLM's own resolution
functions, on CPU, in about a second. It derives the layer names from the
overlay's index rather than hardcoding them, so it stays honest as targets
change.

    python3 verify_overlay.py --overlay <path> --target shared_experts
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from vllm.model_executor.layers.quantization.compressed_tensors.utils import (
    find_matched_target,
    should_ignore_layer,
)

_BUILDER = Path(__file__).with_name("build_overlay.py")


def _load_builder():
    """The builder owns the checkpoint->runtime name mapping; reuse it.

    Duplicating the fusion table here would let the two drift, and a verifier
    that disagrees with the builder is worse than no verifier.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location("_k3_overlay_builder", _BUILDER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def quantized_layer_names(overlay: Path) -> list[str]:
    """Layer names the overlay actually wrote fp8 weights for.

    A weight is only meaningfully quantized if it has a scale beside it, so
    require both. That also skips the untouched bf16 tensors the overlay
    symlinks straight through.
    """
    index = sorted(overlay.glob("*.safetensors.index.json"))
    if not index:
        raise SystemExit(f"no safetensors index under {overlay}")
    weight_map = json.loads(index[0].read_text())["weight_map"]
    return sorted(
        name[: -len(".weight")]
        for name in weight_map
        if name.endswith(".weight") and name + "_scale" in weight_map
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overlay", type=Path, required=True)
    parser.add_argument(
        "--target",
        required=True,
        nargs="+",
        help=(
            "one or more targets. A COMBINED overlay carries several fp8 "
            "groups; checking it against a single target reports the layers "
            "owned by the other groups as 'wrong group' and fails a correct "
            "overlay."
        ),
    )
    parser.add_argument(
        "--expect-min",
        type=int,
        default=1,
        help="fail if fewer than this many layers resolve to fp8",
    )
    args = parser.parse_args()

    quant = json.loads((args.overlay / "config.json").read_text())
    quant = quant["text_config"]["quantization_config"]
    groups = quant["config_groups"]
    expected_groups = {"group_fp8_" + one for one in args.target}
    missing = sorted(g for g in expected_groups if g not in groups)
    if missing:
        print(f"FAIL: {missing} missing from config_groups", file=sys.stderr)
        return 1
    target_to_group = {t: g for g, spec in groups.items() for t in spec["targets"]}

    layers = quantized_layer_names(args.overlay)
    # Fused Linears are resolved under the merged name, not the checkpoint
    # names its shards came from.
    builder = _load_builder()
    layers = builder.runtime_layer_names(
        [n + ".weight" for n in layers], args.target
    )  # args.target is a list; fusion maps from every target are applied
    if not layers:
        print(f"FAIL: no fp8 weight/scale pairs in {args.overlay}", file=sys.stderr)
        return 1

    ignored: list[str] = []
    wrong_group: list[tuple[str, str]] = []
    per_group: dict[str, int] = {}
    ok = 0
    for name in layers:
        if should_ignore_layer(name, ignore=quant["ignore"]):
            ignored.append(name)
            continue
        try:
            matched = find_matched_target(name, None, list(target_to_group), {})
        except Exception:
            wrong_group.append((name, "no matching target"))
            continue
        group = target_to_group[matched]
        if group not in expected_groups:
            wrong_group.append((name, group))
            continue
        per_group[group] = per_group.get(group, 0) + 1
        ok += 1

    if ignored:
        print(
            f"FAIL: {len(ignored)} of {len(layers)} quantized layers are in the "
            "`ignore` list, so they will be served as bf16 despite having fp8 "
            "tensors on disk. The enclosing family pattern most likely still "
            "covers them -- narrow it with a negative lookahead.",
            file=sys.stderr,
        )
        for name in ignored[:5]:
            print(f"    ignored: {name}", file=sys.stderr)
        return 1
    if wrong_group:
        print(
            f"FAIL: {len(wrong_group)} layers resolve to the wrong config group; "
            "they would be built with the wrong scheme.",
            file=sys.stderr,
        )
        for name, group in wrong_group[:5]:
            print(f"    {name} -> {group}", file=sys.stderr)
        return 1
    if ok < args.expect_min:
        print(
            f"FAIL: only {ok} fp8 layers, expected >= {args.expect_min}",
            file=sys.stderr,
        )
        return 1

    for group in sorted(expected_groups):
        weights = groups[group]["weights"]
        acts = groups[group].get("input_activations")
        scheme = f"w{weights['num_bits']}"
        scheme += (
            f"a{acts['num_bits']}{'-dynamic' if acts.get('dynamic') else ''}"
            if acts
            else "a16"
        )
        print(
            f"OK: {per_group.get(group, 0):>4} layers -> {group} "
            f"({scheme}, {weights['strategy']})"
        )
    print(f"OK: {ok} fp8 layers total; none ignored, none in an unexpected group")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
