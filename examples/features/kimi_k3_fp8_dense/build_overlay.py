#!/usr/bin/env python3
"""Build an fp8 dense-projection overlay for Kimi-K3, without copying weights.

Lever 1b asks what the dense bf16 projections cost. `shared_experts` is the
cheapest part of that question to answer: 276 tensors, 3 per layer across 92
MoE layers, 88.08 MB each, 24.3 GB in total -- about 3.0 GB per rank per
forward at TP8, all of it weight traffic on skinny decode GEMMs that are
bandwidth-bound in the weights at M=7.

It cannot be answered by editing the checkpoint's `ignore` list. Those modules
have no quantized tensors to fall back on: the index holds them as plain bf16
`.weight` with zero `weight_scale` and zero `weight_packed`, against 247,296 of
each for the routed experts. Removing a regex just makes vLLM look for tensors
that do not exist.

The obvious alternative -- write a requantized copy -- does not fit. The 276
tensors are spread across 92 of the 96 shards, so a conventional rewrite
touches 1553 GB of a 1561 GB checkpoint, and this node has ~208 GB free.

So do not copy the shards. A sharded checkpoint is loaded through its index,
and the index is the authority: a tensor present in a shard but absent from the
weight map is never read. This builds an output directory that

  1. symlinks all 96 original shards, costing nothing,
  2. writes the fp8 `shared_experts` weights and scales into a few small new
     shards (~12 GB, and it fits),
  3. emits an index whose `shared_experts` entries point at the new shards, so
     the original bf16 copies are still on disk and simply unreferenced,
  4. emits a config with a second `config_group` for fp8 and `shared_experts`
     dropped from `ignore`.

vLLM resolves `format` per config group, falling back to the global format
(compressed_tensors.py:343-350), so mxfp4 routed experts and fp8 dense
projections coexist in one checkpoint.

The original checkpoint is never written to. Deleting the output directory
reverts everything.

    python3 build_overlay.py \\
        --src /path/to/Kimi-K3 \\
        --dst /path/to/Kimi-K3-fp8se

VERIFY BEFORE TRUSTING A NUMBER FROM THIS:
  * that the serving path loads strictly through the index. `--load-format
    fastsafetensors` is the one the agentic arms use and it has not been
    checked here; a loader that enumerates shard files instead would read the
    stale bf16 tensors and silently measure the control.
    `--check-unreferenced` asserts the property this relies on.
  * accuracy. This changes what the model computes. The routed-expert topk=8
    arm was rejected at GSM8k 0.992 -> 0.976, so run the gate before the perf
    arm, not after.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path

# compressed-tensors names the quantized weight and its scale this way, and
# vLLM's loader looks them up by exactly these suffixes.
WEIGHT_SUFFIX = ".weight"
SCALE_SUFFIX = ".weight_scale"
# Only bf16 *_proj weights are Linear. Matching a bare module substring is not
# enough: self_attn also holds F32 q/k/v_conv1d, o_norm, A_log and dt_bias, and
# -- less obviously -- bf16 q_a_layernorm and kv_a_layernorm, which are norms
# rather than projections. A dtype filter alone would still let those through.
# Verified: this drops nothing for shared_experts (276 -> 276) and 348 non-
# Linear tensors for self_attn (1020 -> 672).
PROJ_SUFFIX = "_proj.weight"
# Some targets must be narrowed to a single leaf. Quantizing all of self_attn
# is correct but catastrophically slow -- ~1.7 questions/min against 53 for
# shared_experts, measured 2026-09-19 -- because in_proj_qkvgfab and f_b_proj
# feed fused_recurrent_kda_fwd_kernel on the KDA linear-attention path, 69
# layers deep, every token. o_proj is the pure output projection on all 93
# layers and is not on that path, so it is the safe subset.
# Maps leaf -> (tensor suffix, enclosing family whose ignore entry covers it).
LEAF_TARGETS = {"o_proj": (".o_proj.weight", "self_attn")}

# Composite targets quantize several modules together, because two things stop
# a module-at-a-time approach from being correct here:
#
#   * vLLM CONCATENATES q_a_proj and kv_a_proj_with_mqa into one Linear,
#     `fused_qkv_a_proj`. A fused layer cannot have one shard fp8 and another
#     bf16 -- compressed-tensors raises "Found different quantization schemes
#     for the shards" -- so they move together, and the name to release from
#     `ignore` is the RUNTIME name, not either checkpoint name.
#   * g_proj has 93 checkpoint tensors but is only a standalone Linear on the
#     24 MLA layers. On the other 69 it is loaded as shard 3 of
#     `in_proj_qkvgfab`, whose output feeds fused_recurrent_kda_fwd_kernel --
#     the path that made the full self_attn arm ~44x slower. Quantizing those
#     69 on disk would also push an fp8 shard into a bf16 fused module. So
#     g_proj is layer-filtered to the MLA layers.
#
# `tensor_modules` selects what to quantize ON DISK; `runtime_modules` is what
# vLLM will actually name the resulting Linears. They deliberately differ.
COMPOSITE_TARGETS = {
    "mla_proj": {
        "family": "self_attn",
        "tensor_modules": (
            "g_proj",
            "q_b_proj",
            "kv_b_proj",
            "q_a_proj",
            "kv_a_proj_with_mqa",
        ),
        "runtime_modules": ("g_proj", "q_b_proj", "kv_b_proj", "fused_qkv_a_proj"),
        # Modules that also exist on KDA layers and must be restricted.
        "mla_only_modules": ("g_proj",),
        # Checkpoint module -> the Linear vLLM actually builds from it, per
        # amd/linear.py:982-983. Anything that resolves a layer by name -- the
        # `ignore` list, `config_groups` targets, the overlay verifier -- must
        # use the RIGHT-hand name; the left-hand names exist only on disk.
        "fused_as": {
            "q_a_proj": "fused_qkv_a_proj",
            "kv_a_proj_with_mqa": "fused_qkv_a_proj",
        },
    }
}


def runtime_layer_names(tensor_names, target) -> list[str]:
    """Map checkpoint tensor names to the layer names vLLM will resolve.

    For most targets these are the same. They are NOT for fused Linears:
    q_a_proj and kv_a_proj_with_mqa are concatenated into fused_qkv_a_proj, so
    checking the checkpoint names against `ignore` reports a correct overlay as
    broken.
    """
    targets = [target] if isinstance(target, str) else list(target)
    fused = {}
    for one in targets:
        fused.update(COMPOSITE_TARGETS.get(one, {}).get("fused_as", {}))
    out = []
    for name in tensor_names:
        layer = name[: -len(WEIGHT_SUFFIX)] if name.endswith(WEIGHT_SUFFIX) else name
        head, _, module = layer.rpartition(".")
        mapped = fused.get(module)
        layer = f"{head}.{mapped}" if mapped else layer
        if layer not in out:
            out.append(layer)
    return out


def mla_layer_indices(src: Path) -> set[int]:
    """Zero-based indices of the full-attention (MLA) layers.

    `full_attn_layers` in config.json is ONE-INDEXED -- it contains 93 while
    the layers run 0..92. Reading it as zero-based silently selects the wrong
    layers, which is the kind of off-by-one that produces a checkpoint that
    loads fine and computes the wrong thing.
    """
    config = json.loads((src / "config.json").read_text())["text_config"]
    listed = config["linear_attn_config"]["full_attn_layers"]
    return {index - 1 for index in listed}


def select_tensor_names(src: Path, weight_map: dict, target: str) -> list[str]:
    """Checkpoint tensor names this target should quantize."""
    if target not in COMPOSITE_TARGETS:
        suffix = match_suffix_for(target)
        return sorted(
            name
            for name in weight_map
            if target in name and name.endswith(suffix)
        )

    spec = COMPOSITE_TARGETS[target]
    mla = mla_layer_indices(src)
    restricted = set(spec["mla_only_modules"])
    chosen = []
    for name in weight_map:
        matched = re.match(
            r".*\.layers\.(\d+)\.%s\.([A-Za-z0-9_]+)\.weight$" % spec["family"], name
        )
        if not matched:
            continue
        layer, module = int(matched.group(1)), matched.group(2)
        if module not in spec["tensor_modules"]:
            continue
        if module in restricted and layer not in mla:
            continue
        chosen.append(name)
    return sorted(chosen)


def match_suffix_for(target: str) -> str:
    """The exact name suffix a tensor must have to be quantized for *target*."""
    if target in LEAF_TARGETS:
        return LEAF_TARGETS[target][0]
    return PROJ_SUFFIX


def narrowed_ignore_pattern(family: str, leaf: str) -> str:
    """The family ignore regex with *leaf* carved out via negative lookahead.

    `should_ignore_layer` runs BEFORE `config_groups` is consulted, so a family
    entry like `re:.*self_attn.*` swallows `...self_attn.o_proj` and the layer
    resolves to UnquantizedLinearMethod no matter what the fp8 group says. That
    fails silently: the run loads, serves and scores -- as the control.
    Measured 2026-09-19, 93 o_proj layers "delegated to UnquantizedLinearMethod".

    Patterns are anchored with re.fullmatch, and Python regex supports
    lookaheads, so this releases exactly the leaf and keeps every sibling
    (q_b_proj, in_proj_qkvgfab, kv_a_proj_with_mqa) ignored.
    """
    return rf"re:.*{family}\.(?!{leaf}).*"


def _existing_narrowed(patterns, family: str) -> tuple[str | None, list[str]]:
    """Find an already-narrowed entry for *family* and its released modules.

    Building a COMBINED overlay chains the builder, and two targets can narrow
    the same family entry (o_proj and mla_proj both live under self_attn). The
    second pass would look for the pristine `re:.*self_attn.*`, not find it,
    and fail closed -- correct, but it makes the combined overlay unbuildable.
    Composing the lookaheads is the fix.
    """
    prefix = rf"re:.*{family}\.(?!"
    for pattern in patterns:
        if pattern.startswith(prefix) and pattern.endswith(").*"):
            inner = pattern[len(prefix) : -len(").*")]
            return pattern, [m for m in inner.split("|") if m]
    return None, []
# Default target. self_attn is the other one worth quantizing: 672 bf16 *_proj
# weights, 72.16 GB -> 36.08 GB, roughly 3x shared_experts' surface, with the
# same measured fp8 error (0.02634 vs 0.02646).
DEFAULT_TARGET = "shared_experts"


def ignore_pattern_for(target: str) -> str:
    """The `ignore` regex the checkpoint uses for this module family."""
    return f"re:.*{target}.*"


def shard_prefix_for(target: str) -> str:
    """Output shard prefix; must be unique per target so overlays can coexist."""
    return f"model-fp8-{target.replace('_', '-')}-"
# float8_e4m3fn saturates at 448; scales map each row's max onto that.
FP8_MAX = 448.0
# One output shard per this many tensors, to keep each file a sane size.
TENSORS_PER_SHARD = 24


def _index_path(root: Path) -> Path:
    hits = sorted(root.glob("*.safetensors.index.json"))
    if not hits:
        raise SystemExit(f"no safetensors index under {root}")
    return hits[0]


def is_fp8_dtype(dtype: str) -> bool:
    """Whether a safetensors dtype string names an 8-bit float.

    safetensors spells these ``F8_E4M3``/``F8_E5M2``, not ``float8_*`` as torch
    does. Matching on "float8" made the check reject a correctly built overlay,
    which is the less dangerous direction but still wrong -- and a checker
    nobody trusts is a checker that gets skipped.
    """
    return dtype.upper().replace("-", "_") in {"F8_E4M3", "F8_E5M2", "F8_E4M3FN"}


def rewrite_quant_config(config: dict, target: str = DEFAULT_TARGET) -> dict:
    """Add the fp8 dense group and drop shared_experts from `ignore`.

    Mutates and returns ``config``. Kept separate from the tensor work so the
    part that decides what vLLM will believe about the checkpoint can be
    checked without a GPU, a copy of the weights, or torch.
    """
    ignore_pattern = ignore_pattern_for(target)
    quant = config["text_config"]["quantization_config"]
    groups = quant["config_groups"]
    if "group_fp8_" + target in groups:
        raise SystemExit("source config already carries an fp8 dense group")
    # The ignore list is authoritative and is consulted FIRST, before any
    # config_groups target, so a layer left in it cannot be quantized however
    # the fp8 group is written. Two shapes of edit:
    #
    #   family target (shared_experts, self_attn)  drop the entry outright
    #   leaf target   (o_proj)                     narrow the ENCLOSING family
    #                                              entry with a lookahead, so
    #                                              the siblings stay bf16
    #
    # Either way the entry must exist. If it does not, this is not the
    # checkpoint the probe was written against -- fail closed rather than emit
    # an overlay that quietly serves the control.
    composite = COMPOSITE_TARGETS.get(target)
    leaf_family = LEAF_TARGETS[target][1] if target in LEAF_TARGETS else None
    if composite is not None:
        leaf_family = composite["family"]
    if leaf_family is not None:
        family_pattern = ignore_pattern_for(leaf_family)
        prior, already = _existing_narrowed(quant["ignore"], leaf_family)
        if prior is not None:
            # Chained build: compose with what an earlier target released.
            family_pattern = prior
        elif family_pattern not in quant["ignore"]:
            raise SystemExit(
                f"{family_pattern!r} (the family entry enclosing {target!r}) is not "
                "in the source ignore list; the checkpoint is not the one this "
                "probe was written against"
            )
        released = list(composite["runtime_modules"] if composite else (target,))
        overlap = sorted(set(already) & set(released))
        if overlap:
            raise SystemExit(
                f"{', '.join(overlap)} already released from {leaf_family}'s ignore "
                "entry by an earlier target; this overlay would double-quantize"
            )
        narrowed = narrowed_ignore_pattern(
            leaf_family, "|".join(already + released)
        )
        quant["ignore"] = [
            narrowed if p == family_pattern else p for p in quant["ignore"]
        ]
        print(
            f"  narrowed {family_pattern!r} -> {narrowed!r} so "
            f"{', '.join(released)} are released from ignore while the other "
            f"{leaf_family} modules stay bf16"
        )
        drop_ignore = False
    else:
        drop_ignore = ignore_pattern in quant["ignore"]
        if not drop_ignore:
            raise SystemExit(
                f"{ignore_pattern!r} is not in the source ignore list; the checkpoint "
                "is not the one this probe was written against"
            )
    groups["group_fp8_" + target] = {
        # A layer-name regex, NOT "Linear". find_matched_target resolves in the
        # order (layer-name regex) -> (fused mapping) -> (module class name), so
        # a class-name target here would tie with group_0's ["Linear"] and the
        # module would resolve to the mxfp4 scheme instead. That failure is
        # silent until load, where it surfaces as
        #   KeyError: '...shared_experts.down_proj.weight_scale'
        # because the module was built with weight_packed, not weight_scale.
        "targets": (
            [
                r"re:.*%s\.(%s)"
                % (composite["family"], "|".join(composite["runtime_modules"]))
            ]
            if composite
            else [ignore_pattern_for(target)]
        ),
        "weights": {
            "num_bits": 8,
            "type": "float",
            "strategy": "channel",
            "symmetric": True,
            "dynamic": False,
        },
        # Dynamic per-token fp8 activations, NOT weight-only. ROCm has no
        # ScaledMM kernel for W8A16 fp8 -- CompressedTensorsW8A16Fp8 fails in
        # create_weights with an empty reasons list -- so weight-only cannot
        # run here at all. Dynamic activations need no stored scales, so the
        # tensors on disk are unchanged; only this config differs.
        "input_activations": {
            "num_bits": 8,
            "type": "float",
            "strategy": "token",
            "symmetric": True,
            "dynamic": True,
        },
        "output_activations": None,
        # Per-group format; the global one stays mxfp4 for the routed experts.
        "format": "float-quantized",
    }
    if drop_ignore:
        quant["ignore"] = [p for p in quant["ignore"] if p != ignore_pattern]
    return config


def quantize_rowwise(weight):
    """Return (fp8 weight, fp32 per-output-channel scale).

    Symmetric per-channel along the output dimension, which is what
    compressed-tensors' ``strategy: channel`` means and what vLLM's fp8 path
    expects. Per-channel rather than per-tensor because one outlier row would
    otherwise crush the resolution of every other row in a 6144x7168 matrix.
    """
    import torch

    weight = weight.to(torch.float32)
    scale = weight.abs().amax(dim=-1, keepdim=True) / FP8_MAX
    # A zero row would divide by zero and poison the tensor with NaN.
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    quantized = (weight / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return quantized, scale.to(torch.float32)


def build(src: Path, dst: Path, target: str = DEFAULT_TARGET) -> int:
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    index_file = _index_path(src)
    index = json.loads(index_file.read_text())
    weight_map = dict(index["weight_map"])

    targets = select_tensor_names(src, weight_map, target)
    if not targets:
        raise SystemExit(f"no {target} *_proj weights in {index_file}")

    dst.mkdir(parents=True, exist_ok=True)

    # 1. Symlink the originals. Nothing is copied and nothing is modified.
    for shard in sorted(src.glob("*.safetensors")):
        link = dst / shard.name
        if not link.exists():
            link.symlink_to(shard.resolve())

    # 2. Requantize into new shards, holding only one shard in memory at a time.
    print(f"quantizing {len(targets)} {target} tensors to fp8")
    written = 0
    for shard_index, start in enumerate(range(0, len(targets), TENSORS_PER_SHARD)):
        batch = targets[start : start + TENSORS_PER_SHARD]
        tensors = {}
        for name in batch:
            source = src / weight_map[name]
            with safe_open(source, framework="pt") as handle:
                weight = handle.get_tensor(name)
            quantized, scale = quantize_rowwise(weight)
            tensors[name] = quantized
            tensors[name[: -len(WEIGHT_SUFFIX)] + SCALE_SUFFIX] = scale
            del weight

        shard_name = f"{shard_prefix_for(target)}{shard_index:05d}.safetensors"
        save_file(tensors, str(dst / shard_name), metadata={"format": "pt"})
        for key in tensors:
            weight_map[key] = shard_name
        written += len(batch)
        print(f"  {shard_name}: {len(batch)} tensors ({written}/{len(targets)})")
        del tensors

    # 3. Index: the repointed entries are what make the bf16 copies dead.
    index["weight_map"] = weight_map
    index.pop("metadata", None)
    (dst / index_file.name).write_text(json.dumps(index, indent=2))

    # 4. Config: add the fp8 group, drop shared_experts from ignore.
    config = rewrite_quant_config(json.loads((src / "config.json").read_text()), target)
    (dst / "config.json").write_text(json.dumps(config, indent=2))

    # 5. Everything else the tokenizer and loader expect.
    for extra in src.iterdir():
        if extra.suffix in {".safetensors"} or extra.name == "config.json":
            continue
        if extra.name == index_file.name or (dst / extra.name).exists():
            continue
        if extra.is_file():
            shutil.copy2(extra, dst / extra.name)

    print(f"\noverlay written to {dst}")
    new_bytes = sum(f.stat().st_size for f in dst.glob("model-fp8-*"))
    print(f"  new bytes: {new_bytes / 1e9:.1f} GB")
    print("  originals untouched; delete the output directory to revert")
    return 0


def check_unreferenced(
    dst: Path, src: Path, target: str = DEFAULT_TARGET, samples: int = 3
) -> int:
    """Assert the properties the overlay depends on.

    Two independent failures would each make a run against this overlay
    silently measure something other than what it claims:

      * a `shared_experts` entry still resolving to an original shard, so the
        model serves bf16 and the "fp8 arm" is the control;
      * fp8 bytes that do not dequantize back to the original weights, which a
        dtype check cannot see -- a transposed or misaligned scale produces
        perfectly well-typed garbage.
    """
    import torch
    from safetensors import safe_open

    weight_map = json.loads(_index_path(dst).read_text())["weight_map"]
    # The exact tensor set this target is responsible for. Deriving it the same
    # way the builder does keeps the checker honest when the selection rule is
    # not a plain suffix -- mla_proj filters g_proj by layer, so a suffix test
    # would demand 69 KDA tensors be repointed and fail a correct overlay.
    expected = set(select_tensor_names(src, weight_map, target))
    problems = []
    for name, shard in sorted(weight_map.items()):
        # Only *_proj weights (and their scales) are quantized. A module can
        # hold tensors the filter deliberately skips -- self_attn keeps bf16
        # q_a_layernorm / kv_a_layernorm and the fused kv_a_proj_with_mqa in the
        # original shards -- and demanding those be repointed reports a correct
        # overlay as broken.
        if name.endswith(SCALE_SUFFIX):
            base = name[: -len(SCALE_SUFFIX)] + WEIGHT_SUFFIX
            if base not in expected:
                continue
        elif name not in expected:
            continue
        if not shard.startswith(shard_prefix_for(target)):
            problems.append(f"{name} still resolves to {shard}")
            continue
        with safe_open(dst / shard, framework="pt") as handle:
            dtype = str(handle.get_slice(name).get_dtype())
        if name.endswith(WEIGHT_SUFFIX) and not is_fp8_dtype(dtype):
            problems.append(f"{name} is {dtype}, expected 8-bit float")

    weights = sorted(expected)
    scales = [
        n
        for n in weight_map
        if n.endswith(SCALE_SUFFIX)
        and n[: -len(SCALE_SUFFIX)] + WEIGHT_SUFFIX in expected
    ]
    if len(scales) != len(weights):
        problems.append(f"{len(weights)} quantized weights but {len(scales)} scales")

    source_map = json.loads(_index_path(src).read_text())["weight_map"]
    step = max(1, len(weights) // samples)
    for name in weights[::step][:samples]:
        with safe_open(src / source_map[name], framework="pt") as handle:
            reference = handle.get_tensor(name).float()
        with safe_open(dst / weight_map[name], framework="pt") as handle:
            quantized = handle.get_tensor(name)
        scale_name = name[: -len(WEIGHT_SUFFIX)] + SCALE_SUFFIX
        with safe_open(dst / weight_map[scale_name], framework="pt") as handle:
            scale = handle.get_tensor(scale_name).float()

        if scale.shape != (reference.shape[0], 1):
            problems.append(f"{scale_name} is {tuple(scale.shape)}, expected per-row")
            continue
        error = quantized.float() * scale - reference
        relative = (error.pow(2).sum() / reference.pow(2).sum()).sqrt().item()
        label = name.split("layers.")[-1][:44]
        print(f"  {label:46s} dequant relRMS {relative:.5f}")
        # The offline estimate on these tensors was 0.0265; anything far above
        # means the scale is not being applied the way the weights were built.
        if relative > 0.05:
            problems.append(
                f"{name} dequantizes at relRMS {relative:.4f}, expected ~0.027"
            )

    for problem in problems:
        print(f"  FAIL {problem}", file=sys.stderr)
    if problems:
        return 1
    print(
        f"  ok: {len(weights)} fp8 weights, {len(scales)} scales, no bf16 left "
        f"reachable, {min(samples, len(weights))} sampled tensors dequantize correctly"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument(
        "--target",
        default=DEFAULT_TARGET,
        help="module family to quantize: shared_experts (default) or self_attn",
    )
    parser.add_argument("--dst", type=Path, required=True)
    parser.add_argument(
        "--check-unreferenced",
        action="store_true",
        help="verify an existing overlay instead of building one",
    )
    parser.add_argument(
        "--config-only",
        action="store_true",
        help=(
            "rewrite only dst/config.json from src, leaving the tensors in "
            "place. The quantized weights and the config that tells vLLM how to "
            "read them fail independently -- a wrong ignore list serves the "
            "control from byte-perfect fp8 tensors -- so fixing the config must "
            "not require rebuilding gigabytes of weights."
        ),
    )
    args = parser.parse_args()

    if args.check_unreferenced:
        return check_unreferenced(args.dst, args.src, args.target)
    if args.config_only:
        if not (args.dst / "config.json").exists():
            raise SystemExit(f"no existing overlay config at {args.dst}/config.json")
        config = rewrite_quant_config(
            json.loads((args.src / "config.json").read_text()), args.target
        )
        (args.dst / "config.json").write_text(json.dumps(config, indent=2))
        print(f"  rewrote {args.dst}/config.json for target {args.target}")
        return 0
    if args.dst.exists() and any(args.dst.iterdir()):
        raise SystemExit(f"{args.dst} exists and is not empty; refusing to overwrite")
    return build(args.src, args.dst, args.target)


if __name__ == "__main__":
    raise SystemExit(main())
