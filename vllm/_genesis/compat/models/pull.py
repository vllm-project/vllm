# SPDX-License-Identifier: Apache-2.0
"""Genesis models pull — `python3 -m vllm._genesis.compat.models.pull <key>`.

Downloads a registered model from HuggingFace, verifies it, and
generates a personalized launch script that engages the right Genesis
patches for the user's hardware.

Workflow:
  1. Pre-flight: disk space, HF reachability, optional HF token
  2. Download via huggingface_hub (resume-capable, robust)
  3. Verify (file count + sizes; SHA optional if pinned)
  4. Generate launch script in scripts/launch/start_<key>_<workload>.sh
  5. Print recommended next steps

Usage:
  python3 -m vllm._genesis.compat.models.pull qwen3_6_27b_int4_autoround
  python3 -m vllm._genesis.compat.models.pull qwen3_6_27b_int4_autoround \
      --models-dir /nfs/genesis/models \
      --workload long_ctx_tool_call \
      --tp 2

Env overrides:
  GENESIS_MODELS_DIR       — where to download (default: ~/.cache/huggingface/hub
                              or /nfs/genesis/models if it exists)
  HF_TOKEN                 — for gated repos
  HUGGINGFACE_HUB_CACHE    — standard HF lib override

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
"""
from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
from pathlib import Path

log = logging.getLogger("genesis.compat.models.pull")


# ─── Pre-flight ───────────────────────────────────────────────────────────


def _resolve_models_dir(override: str | None = None) -> Path:
    """Resolve where to put downloaded models.

    Precedence:
      1. CLI --models-dir (passed in `override`)
      2. GENESIS_MODELS_DIR env
      3. /nfs/genesis/models if it exists (Sander's homelab default)
      4. HUGGINGFACE_HUB_CACHE env
      5. ~/.cache/huggingface/hub (HF default)
    """
    if override:
        return Path(override).expanduser().resolve()
    env_genesis = os.environ.get("GENESIS_MODELS_DIR")
    if env_genesis:
        return Path(env_genesis).expanduser().resolve()
    if Path("/nfs/genesis/models").is_dir():
        return Path("/nfs/genesis/models")
    env_hf = os.environ.get("HUGGINGFACE_HUB_CACHE")
    if env_hf:
        return Path(env_hf).expanduser().resolve()
    return Path("~/.cache/huggingface/hub").expanduser()


def _check_disk_space(target_dir: Path, needed_gb: float, headroom: float = 1.2) -> tuple[bool, str]:
    """Verify the target dir has enough free space (with `headroom` factor)."""
    target_dir.mkdir(parents=True, exist_ok=True)
    stat = shutil.disk_usage(target_dir)
    free_gb = stat.free / 1e9
    need_with_headroom = needed_gb * headroom
    if free_gb < need_with_headroom:
        return False, (
            f"insufficient disk: {free_gb:.1f} GB free at {target_dir}, "
            f"need ~{need_with_headroom:.1f} GB (model {needed_gb:.1f} GB × {headroom:.1f})"
        )
    return True, f"{free_gb:.1f} GB free at {target_dir} (need ~{need_with_headroom:.1f} GB)"


def _check_hf_reachable() -> tuple[bool, str]:
    """Best-effort connectivity check to huggingface.co."""
    try:
        import urllib.request
        urllib.request.urlopen("https://huggingface.co", timeout=5)
        return True, "huggingface.co reachable"
    except Exception as e:
        return False, f"huggingface.co not reachable: {e}"


def _check_hf_token_for_gated(model_entry) -> tuple[bool, str]:
    """If the model is gated, verify HF_TOKEN is set."""
    if not model_entry.gated:
        return True, "public repo (no token required)"
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        return False, (
            "gated model — set HF_TOKEN env var or run `huggingface-cli login` "
            "first. Visit the model card to request access."
        )
    return True, "HF_TOKEN present"


# ─── Download ─────────────────────────────────────────────────────────────


def download_model(
    model_entry,
    models_dir: Path,
    *,
    revision: str | None = None,
    progress: bool = True,
) -> Path:
    """Download via huggingface_hub.snapshot_download. Returns local path."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise RuntimeError(
            "huggingface_hub not installed. Run: "
            "pip install huggingface_hub"
        )

    rev = revision or model_entry.hf_revision
    log.info("Downloading %s (revision=%s) to %s",
             model_entry.hf_id, rev or "latest", models_dir)

    local_path = snapshot_download(
        repo_id=model_entry.hf_id,
        revision=rev,
        cache_dir=str(models_dir),
        # Allow resume on partial downloads
        resume_download=True,
        # Skip files we don't need (tokenizer files we always need;
        # safetensors over .bin where both exist)
        ignore_patterns=["*.bin", "*.h5", "*.msgpack", "tf_*", "flax_*"],
        local_files_only=False,
    )
    return Path(local_path)


def _verify_download(local_path: Path, model_entry) -> tuple[bool, str]:
    """Sanity-check that essential files are present + non-empty."""
    if not local_path.is_dir():
        return False, f"download path not a directory: {local_path}"
    safetensors = list(local_path.rglob("*.safetensors"))
    if not safetensors:
        return False, "no .safetensors files found in downloaded path"
    config = list(local_path.glob("config.json"))
    if not config:
        return False, "config.json missing"
    total_gb = sum(p.stat().st_size for p in safetensors) / 1e9
    expected_gb = model_entry.size_gb
    drift = abs(total_gb - expected_gb) / expected_gb if expected_gb else 0.0
    if drift > 0.20:  # 20% tolerance for community quants that re-roll
        return False, (
            f"size drift {drift*100:.0f}% — got {total_gb:.1f} GB, "
            f"expected ~{expected_gb:.1f} GB. Could be a different version."
        )
    return True, f"verified ({len(safetensors)} shards, {total_gb:.1f} GB)"


# ─── Launch script generation ─────────────────────────────────────────────


def _select_config(model_entry, workload: str | None, tp: int | None):
    """Pick a TestedConfig based on workload preference + TP."""
    configs = list(model_entry.tested_configs)
    if not configs:
        return None
    if workload:
        for c in configs:
            if workload in c.name.lower().replace(" ", "_") \
                    or workload in c.name.lower():
                return c
    if tp:
        for c in configs:
            if c.tensor_parallel_size == tp:
                return c
    return configs[0]  # default to first


def generate_launch_script(
    model_entry,
    config,
    local_model_path: Path,
    out_path: Path,
) -> None:
    """Write a bash launch script tailored to (model, config, hardware)."""
    served_name = model_entry.key.replace("_", "-")
    env_lines = []
    for patch_id in config.recommended_genesis_patches:
        # Convert patch_id "P67" → env flag (look up in PATCH_REGISTRY).
        try:
            from vllm._genesis.dispatcher import PATCH_REGISTRY
            meta = PATCH_REGISTRY.get(patch_id)
            if meta and meta.get("env_flag"):
                env_lines.append(f"  -e {meta['env_flag']}=1 \\")
        except Exception:
            # Fallback: best-guess name
            env_lines.append(f"  -e GENESIS_ENABLE_{patch_id.upper()}=1 \\")

    spec_json = "null"
    if config.speculative_config:
        import json
        spec_json = json.dumps(config.speculative_config)
    spec_arg = (
        f'  --speculative-config \'{spec_json}\' \\\n'
        if config.speculative_config else ""
    )

    additional = ""
    if config.additional_args:
        additional = "\n".join(f"  {a} \\" for a in config.additional_args) + "\n"

    quirks_block = ""
    if model_entry.quirks:
        quirks_block = (
            "# ───────────────────────────────────────────────────────────\n"
            "# Known quirks for this model:\n"
        )
        for q in model_entry.quirks:
            quirks_block += f"#   - {q}\n"
        quirks_block += "# ───────────────────────────────────────────────────────────\n\n"

    expected_block = ""
    if config.expected:
        e = config.expected
        expected_block = (
            f"# Expected metrics on {e.hardware_class} (captured {e.captured_at}):\n"
            f"#   wall_TPS  ≈ {e.wall_tps_median}  (CV ~5%)\n"
            f"#   TPOT      ≈ {e.decode_tpot_ms} ms\n"
            f"#   TTFT      ≈ {e.ttft_ms} ms\n"
            f"#   VRAM      ≈ {e.vram_gb_per_rank} GB per rank\n"
            f"#   tool-call ≈ {e.tool_call_pass_rate * 100:.0f}% pass rate\n"
            "#\n"
        )

    cache_pref_arg = " --enable-prefix-caching" if config.enable_prefix_caching else ""

    script = f"""#!/bin/bash
# ════════════════════════════════════════════════════════════════════════
# Genesis launch script for {model_entry.key}
# Workload: {config.name}
# vLLM pin: {config.vllm_pin}
# Generated by: python3 -m vllm._genesis.compat.models.pull
# ════════════════════════════════════════════════════════════════════════

{quirks_block}{expected_block}set -euo pipefail
docker stop vllm-{served_name} 2>/dev/null || true
docker rm   vllm-{served_name} 2>/dev/null || true

docker run -d \\
  --name vllm-{served_name} \\
  --shm-size=8g --memory=64g -p 8000:8000 --gpus all \\
  --security-opt label=disable --entrypoint /bin/bash \\
  -v {local_model_path}:/models/{model_entry.key}:ro \\
  -v $HOME/.cache/huggingface:/root/.cache/huggingface:ro \\
  -v $HOME/genesis-vllm-patches/vllm/_genesis:/usr/local/lib/python3.12/dist-packages/vllm/_genesis:ro \\
  -e VLLM_NO_USAGE_STATS=1 -e VLLM_LOGGING_LEVEL=WARNING \\
  -e PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:512" \\
{chr(10).join(env_lines)}
  vllm/vllm-openai:nightly -c \\
  "set -e; \\
python3 -m vllm._genesis.patches.apply_all ; \\
exec vllm serve --model /models/{model_entry.key} \\
  --tensor-parallel-size {config.tensor_parallel_size} \\
  --gpu-memory-utilization {config.gpu_memory_utilization} \\
  --max-model-len {config.max_model_len} \\
  --max-num-seqs {config.max_num_seqs} \\
  --max-num-batched-tokens {config.max_num_batched_tokens} \\
  --kv-cache-dtype {config.kv_cache_dtype}{cache_pref_arg} \\
{spec_arg}{additional}\\
  --trust-remote-code --language-model-only \\
  --served-model-name {served_name} \\
  --host 0.0.0.0 --port 8000 --disable-log-stats"

sleep 5
docker logs --tail 5 vllm-{served_name} 2>&1 | sed "s/^/  /"
echo "[{model_entry.key}] container started; tail logs with: docker logs -f vllm-{served_name}"
"""
    out_path.write_text(script)
    out_path.chmod(0o755)


# ─── CLI ─────────────────────────────────────────────────────────────────


def _parse_args():
    p = argparse.ArgumentParser(
        prog="python3 -m vllm._genesis.compat.models.pull",
        description="Download a Genesis-supported model from HuggingFace + "
                    "generate a launch script tailored to the chosen workload.",
    )
    p.add_argument("model_key", help="model key from `genesis list-models`")
    p.add_argument("--models-dir", default=None,
                   help="Where to put weights (default: GENESIS_MODELS_DIR / "
                        "/nfs/genesis/models / ~/.cache/huggingface/hub)")
    p.add_argument("--workload", default=None,
                   help="Workload preference: long_ctx_tool_call / interactive / throughput")
    p.add_argument("--tp", type=int, default=None,
                   help="Tensor parallel size override")
    p.add_argument("--launch-out", default="scripts/launch/",
                   help="Directory to write the generated launch script")
    p.add_argument("--no-launch", action="store_true",
                   help="Skip launch-script generation (just download)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print pre-flight + plan, do not actually download")
    p.add_argument("--revision", default=None,
                   help="HF revision (commit/tag) override")
    p.add_argument("--hf-id-override", default=None,
                   help="Override the registry's hf_id (e.g. use Lorbus's "
                        "Qwen3.6-27B variant instead of Intel's). Use the "
                        "exact 'org/repo' string accepted by huggingface_hub.")
    return p.parse_args()


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _parse_args()

    from vllm._genesis.compat.models.registry import get_model

    entry = get_model(args.model_key)
    if entry is None:
        print(f"unknown model key: {args.model_key!r}", file=sys.stderr)
        print("Run `python3 -m vllm._genesis.compat.models.list` to see available models.",
              file=sys.stderr)
        return 2

    # --hf-id-override: replace the registry's hf_id with operator-supplied
    # alternate (e.g. Lorbus/* instead of Intel/*). Recipe metadata
    # (size, quant_format, expected metrics) carries over from the
    # registry entry — operator's responsibility to ensure the override
    # is shape-compatible.
    if args.hf_id_override:
        from dataclasses import replace
        entry = replace(entry, hf_id=args.hf_id_override)
        print(f"[hf-id-override] using {args.hf_id_override!r} "
              f"(registry default: {get_model(args.model_key).hf_id!r})")

    print("=" * 64)
    print(f"Genesis model pull — {entry.title}")
    print("=" * 64)
    print(f"  HF id: {entry.hf_id}")
    print(f"  Size:  {entry.size_gb:.1f} GB")
    print(f"  Quant: {entry.quant_format}")
    print(f"  Status: {entry.status}")

    if entry.status == "PLANNED":
        print(f"\n[!] This model is PLANNED — not yet validated.")
        print("    Genesis won't auto-block but expect rough edges.")

    models_dir = _resolve_models_dir(args.models_dir)
    print(f"\n[1/4] Pre-flight checks")

    ok_disk, msg_disk = _check_disk_space(models_dir, entry.size_gb)
    print(f"  disk:    {'✓' if ok_disk else '✗'} {msg_disk}")
    if not ok_disk:
        return 3

    ok_net, msg_net = _check_hf_reachable()
    print(f"  network: {'✓' if ok_net else '✗'} {msg_net}")
    if not ok_net and not args.dry_run:
        return 3

    ok_tok, msg_tok = _check_hf_token_for_gated(entry)
    print(f"  token:   {'✓' if ok_tok else '✗'} {msg_tok}")
    if not ok_tok:
        return 3

    if args.dry_run:
        print("\n[dry-run] would download to:", models_dir)
        cfg = _select_config(entry, args.workload, args.tp)
        if cfg:
            print(f"[dry-run] would generate launch script for: {cfg.name}")
        return 0

    print(f"\n[2/4] Downloading {entry.hf_id}")
    try:
        local_path = download_model(entry, models_dir, revision=args.revision)
    except Exception as e:
        print(f"  ✗ download failed: {e}", file=sys.stderr)
        return 4

    print(f"  ✓ downloaded to {local_path}")

    print(f"\n[3/4] Verify download")
    ok_v, msg_v = _verify_download(local_path, entry)
    print(f"  {'✓' if ok_v else '⚠'} {msg_v}")
    if not ok_v:
        print("  (download retained in case operator wants to inspect manually)")

    if args.no_launch:
        print(f"\nSkipping launch-script generation (--no-launch).")
        print(f"\nModel ready at: {local_path}")
        return 0

    print(f"\n[4/4] Generate launch script")
    cfg = _select_config(entry, args.workload, args.tp)
    if cfg is None:
        print("  ⚠ no tested config available — skipping launch-script generation")
        return 0

    out_dir = Path(args.launch_out)
    out_dir.mkdir(parents=True, exist_ok=True)
    workload_slug = cfg.name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    out_file = out_dir / f"start_{entry.key}_{workload_slug}.sh"
    generate_launch_script(entry, cfg, local_path, out_file)
    print(f"  ✓ {out_file}")

    print()
    print("=" * 64)
    print("Next steps:")
    print(f"  bash {out_file}")
    if entry.quirks:
        print()
        print("Heads-up — known quirks for this model:")
        for q in entry.quirks:
            print(f"  - {q}")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
