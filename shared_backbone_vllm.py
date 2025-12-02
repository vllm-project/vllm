#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Nitin Gupta (ngupta@inferno.sh)
"""
Shared Backbone Multi-Model Demo

Loads multiple models simultaneously, keeps them in the same process, and proves
that they reuse a single set of shared tensors when generating text.

Usage:
    VLLM_ENABLE_V1_MULTIPROCESSING=0 python shared_backbone_vllm.py

Optional:
    VERIFY_SHARED_PREFIXES=1 VLLM_ENABLE_V1_MULTIPROCESSING=0 python shared_backbone_vllm.py
    (Runs the offline tensor verification before loading the models.)
"""

from __future__ import annotations

import os
import sys
from typing import List, Tuple

import logging

import torch
from vllm import LLM, SamplingParams
from vllm.config.load import SharedBackboneConfig
from vllm.model_executor.model_loader.shared_backbone import (
    SharedBackboneRegistry,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROMPT = "Tell me a story about dinosaurs."
MAX_TOKENS = 80
TEMPERATURE = 0.6
BACKBONE_MODEL = "Qwen/Qwen2-0.5B"
DERIVED_MODELS: List[str] = [
    "Qwen/Qwen2-0.5B-Instruct",
    # Add more model IDs here if desired.
]
GPU_MEMORY_UTILIZATION = 0.45
SHARED_BACKBONE_ID = "shared_backbone_vllm_qwen2"
# Prefixes that are known to be identical across BACKBONE_MODEL and every entry
# in DERIVED_MODELS. Run `VERIFY_SHARED_PREFIXES=1 …` (or the helper script
# in tools/verify_shared_backbone.py) before changing this list.
SHARED_PREFIXES = [
    "model.embed_tokens.",
    "model.layers.",
    "model.norm.",
]
VERIFY_SHARED_PREFIXES = os.environ.get("VERIFY_SHARED_PREFIXES", "0") == "1"
os.environ.setdefault("VLLM_LOGGING_LEVEL", "INFO")
logging.getLogger("vllm.model_executor.model_loader.default_loader").setLevel(
    logging.INFO
)
logging.getLogger("vllm.model_executor.model_loader.gguf_loader").setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO, format="%(message)s")


def _require_single_process() -> None:
    if os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING", "1") != "0":
        print(
            "\nERROR: This demo requires single-process mode.\n"
            "Set VLLM_ENABLE_V1_MULTIPROCESSING=0 before running:\n"
            "    VLLM_ENABLE_V1_MULTIPROCESSING=0 python shared_backbone_vllm.py\n"
        )
        sys.exit(1)


def _print_section(title: str) -> None:
    print(f"\n{'=' * 80}")
    print(title)
    print(f"{'=' * 80}")


def _report_vram(label: str) -> None:
    if not torch.cuda.is_available():
        return
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated() / (1024**3)
    reserved = torch.cuda.memory_reserved() / (1024**3)
    print(f"{label}: allocated={allocated:.2f} GB, reserved={reserved:.2f} GB")


def load_backbone() -> Tuple[str, LLM]:
    _print_section(f"Loading backbone model: {BACKBONE_MODEL}")
    cfg = SharedBackboneConfig(
        model=BACKBONE_MODEL,
        tensor_prefixes=list(SHARED_PREFIXES),
        identifier=SHARED_BACKBONE_ID,
    )
    llm = LLM(
        model=BACKBONE_MODEL,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        disable_log_stats=True,
        shared_backbone=cfg,
    )
    _report_vram("After backbone load")
    return BACKBONE_MODEL, llm


def load_head(model_id: str) -> Tuple[str, LLM]:
    _print_section(f"Loading derived model: {model_id}")
    cfg = SharedBackboneConfig(
        model=BACKBONE_MODEL,
        tensor_prefixes=list(SHARED_PREFIXES),
        identifier=SHARED_BACKBONE_ID,
    )
    llm = LLM(
        model=model_id,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        disable_log_stats=True,
        shared_backbone=cfg,
    )
    _report_vram(f"After loading {model_id}")
    return model_id, llm


def report_registry_summary() -> None:
    _print_section("Shared backbone registry summary")
    stats = SharedBackboneRegistry.get_stats()
    if not stats:
        print("WARNING: registry is empty; shared tensors were not cached.")
        return

    for backbone_id, tensor_count in stats.items():
        tensors = SharedBackboneRegistry.tensors_for(backbone_id)
        total_mb = sum(
            tensor.numel() * tensor.element_size() / (1024**2)
            for tensor in tensors.values()
        )
        print(f"Backbone ID: {backbone_id}")
        print(f"  Cached tensors: {tensor_count}")
        print(f"  Total shared size: {total_mb:.2f} MB ({total_mb / 1024:.2f} GB)")


def generate_from_models(models: List[Tuple[str, LLM]]) -> None:
    sampling = SamplingParams(temperature=TEMPERATURE, max_tokens=MAX_TOKENS)
    for name, llm in models:
        _print_section(f"Generating with {name}")
        outputs = llm.generate(PROMPT, sampling)
        print(outputs[0].outputs[0].text.strip())


def main() -> None:
    if VERIFY_SHARED_PREFIXES:
        from tools.verify_shared_backbone import verify_shared_prefixes

        ok = verify_shared_prefixes(
            backbone_model=BACKBONE_MODEL,
            derived_models=DERIVED_MODELS,
            prefixes=SHARED_PREFIXES,
            verbose=True,
        )
        if not ok:
            raise SystemExit(
                "Shared-backbone verification failed. "
                "Adjust SHARED_PREFIXES before rerunning."
            )

    _require_single_process()
    SharedBackboneRegistry.clear()

    loaded_models: List[Tuple[str, LLM]] = []
    loaded_models.append(load_backbone())

    for model_id in DERIVED_MODELS:
        loaded_models.append(load_head(model_id))

    report_registry_summary()
    generate_from_models(loaded_models)


if __name__ == "__main__":
    main()
