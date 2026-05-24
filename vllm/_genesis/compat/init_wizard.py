# SPDX-License-Identifier: Apache-2.0
"""Genesis init — first-run interactive setup wizard.

Detects hardware, asks about model preference, generates a personalized
launch script.

Usage:
  python3 -m vllm._genesis.compat.init_wizard
  python3 -m vllm._genesis.compat.init_wizard --non-interactive --model qwen3_6_27b_int4_autoround

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
"""
from __future__ import annotations

import argparse
import sys


def _detect_gpu_envelope() -> tuple[float, int, str | None]:
    """Returns (total_vram_gb, num_gpus, hardware_class_hint)."""
    try:
        import torch
        if not torch.cuda.is_available():
            return 0.0, 0, None
        total = 0.0
        n = torch.cuda.device_count()
        first_name = None
        for i in range(n):
            props = torch.cuda.get_device_properties(i)
            total += props.total_memory / 1e9
            if first_name is None:
                first_name = props.name
        # Best-effort hardware-class hint from the GPU name
        hint = None
        if first_name:
            lower = first_name.lower()
            if "a5000" in lower:
                hint = "rtx_a5000"
            elif "4090" in lower:
                hint = "rtx_4090"
            elif "5090" in lower:
                hint = "rtx_5090"
            elif "h100" in lower:
                hint = "h100"
            elif "3090" in lower:
                hint = "rtx_3090"
            elif "a4000" in lower:
                hint = "rtx_a4000"
            elif "rtx pro 6000" in lower or "blackwell" in lower:
                hint = "rtx_pro_6000_blackwell"
        return total, n, hint
    except Exception:
        return 0.0, 0, None


def _ask_choice(prompt: str, options: list[str], default_idx: int = 0) -> int:
    """Interactive 1-based choice prompt. Returns 0-based index."""
    print(prompt)
    for i, opt in enumerate(options, 1):
        marker = " (default)" if i - 1 == default_idx else ""
        print(f"  {i}. {opt}{marker}")
    while True:
        raw = input(f"Choice [1-{len(options)}, default {default_idx + 1}]: ").strip()
        if not raw:
            return default_idx
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(options):
                return idx
        except ValueError:
            # Non-numeric input — fall through to retry prompt below
            pass
        print(f"  Please enter a number 1-{len(options)}")


def _ask_yesno(prompt: str, default: bool = True) -> bool:
    yn = "Y/n" if default else "y/N"
    while True:
        raw = input(f"{prompt} [{yn}]: ").strip().lower()
        if not raw:
            return default
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False
        print("  Please answer y or n")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="python3 -m vllm._genesis.compat.init_wizard",
        description="Genesis first-run setup — picks a model + generates "
                    "a launch script tailored to detected hardware.",
    )
    p.add_argument("--non-interactive", action="store_true",
                   help="Run without prompts (use defaults + flags only)")
    p.add_argument("--model", default=None,
                   help="Skip model picker, use this model key")
    p.add_argument("--workload", default=None,
                   help="Workload preference: long_ctx_tool_call / interactive / throughput")
    p.add_argument("--no-pull", action="store_true",
                   help="Skip the actual download step")
    args = p.parse_args(argv)

    print("=" * 70)
    print("Genesis first-run wizard")
    print("=" * 70)

    # Step 1: hardware detection
    print()
    print("[1/4] Detecting hardware…")
    vram_total, num_gpus, hw_hint = _detect_gpu_envelope()
    if num_gpus == 0:
        print("  ✗ no NVIDIA GPU detected (torch.cuda.is_available() == False)")
        print()
        print("Genesis is designed for vllm + CUDA. If you're testing the")
        print("CLI without GPU, you can still browse the model registry:")
        print("  python3 -m vllm._genesis.compat.models.list_cli")
        return 1
    print(f"  ✓ {num_gpus} GPU{'s' if num_gpus > 1 else ''}, "
          f"{vram_total:.1f} GB total VRAM"
          + (f" (class: {hw_hint})" if hw_hint else ""))

    # Step 2: model selection
    print()
    print("[2/4] Selecting model…")
    from vllm._genesis.compat.models.registry import (
        list_recommended_for_hardware, get_model,
    )

    if args.model:
        chosen = get_model(args.model)
        if not chosen:
            print(f"  ✗ unknown model key: {args.model}")
            return 2
    else:
        recommended = list_recommended_for_hardware(
            vram_gb_total=vram_total,
            num_gpus=num_gpus,
            hardware_class=hw_hint,
        )
        if not recommended:
            print(f"  ✗ no model fits a {vram_total:.1f} GB / {num_gpus} GPU envelope")
            print("  → consider using a smaller pre-quantized variant")
            return 2

        if args.non_interactive:
            chosen = recommended[0]
            print(f"  → auto-selected (non-interactive): {chosen.title}")
        else:
            display = [
                f"{m.title}  [{m.size_gb:.1f}GB, {m.quant_format}, {m.status}]"
                for m in recommended
            ]
            idx = _ask_choice(
                f"  Models that fit your hardware ({len(recommended)} options):",
                display,
                default_idx=0,
            )
            chosen = recommended[idx]
    print(f"  ✓ selected: {chosen.title}")

    # Step 3: workload
    print()
    print("[3/4] Workload preference…")
    workload = args.workload
    if not workload and not args.non_interactive:
        workloads = list(chosen.recommended_workloads) or ["interactive"]
        labels = {
            "interactive": "Interactive (single user, low latency)",
            "throughput":  "Throughput (batch, multi-user)",
            "long_ctx_tool_call": "Long context + tool-call (256K+, agent workflows)",
        }
        display = [labels.get(w, w) for w in workloads]
        idx = _ask_choice("  Pick a workload:", display, default_idx=0)
        workload = workloads[idx]
    elif not workload:
        workload = chosen.recommended_workloads[0] if chosen.recommended_workloads else "interactive"
    print(f"  ✓ workload: {workload}")

    # Step 4: pull + generate
    print()
    print("[4/4] Pull model + generate launch script…")
    if args.no_pull:
        print("  (skipping pull per --no-pull)")
        print()
        print("Manual steps:")
        print(f"  python3 -m vllm._genesis.compat.models.pull {chosen.key} "
              f"--workload {workload}")
        return 0

    if not args.non_interactive:
        if not _ask_yesno(
            f"  Proceed with download of {chosen.title} ({chosen.size_gb:.1f} GB)?",
            default=True,
        ):
            print("  cancelled. Run later with:")
            print(f"    python3 -m vllm._genesis.compat.models.pull {chosen.key}")
            return 0

    from vllm._genesis.compat.models.pull import main as pull_main
    return pull_main([chosen.key, "--workload", workload])


if __name__ == "__main__":
    sys.exit(main())
