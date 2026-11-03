# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Publish one checkpoint through vLLM's NCCL weight-transfer engine."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from hf_checkpoint_nccl_publisher import (
    CheckpointManifest,
    NcclCheckpointPublisher,
)
from update_timing import format_timing_table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--checkpoint-path")
    parser.add_argument("--draft-model")
    parser.add_argument("--draft-revision")
    parser.add_argument("--draft-checkpoint-path")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--update-bucket-size-mb", type=int, default=512)
    parser.add_argument(
        "--expert-tensor-order",
        choices=("natural", "lexical"),
        default="natural",
        help=(
            "Order tensors within each complete expert layer. 'lexical' "
            "matches the physical order of checkpoints written by "
            "safetensors with lexically sorted names."
        ),
    )
    parser.add_argument(
        "--direct-file-expert-h2d",
        action="store_true",
        help=(
            "Copy a physically contiguous expert-layer safetensors payload "
            "directly from its mmap into one GPU allocation."
        ),
    )
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument(
        "--enable-mtp",
        action="store_true",
        help="Send the same canonical checkpoint to the draft model after main.",
    )
    parser.add_argument(
        "--enable-draft-update",
        action="store_true",
        help="Send an independent checkpoint to the speculative draft model.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.enable_mtp and args.enable_draft_update:
        parser.error("--enable-mtp and --enable-draft-update are mutually exclusive")
    if args.enable_draft_update and not (
        args.draft_model and args.draft_revision and args.draft_checkpoint_path
    ):
        parser.error(
            "--enable-draft-update requires --draft-model, --draft-revision, "
            "and --draft-checkpoint-path"
        )

    manifest_started = time.perf_counter()
    manifest = CheckpointManifest.load(
        model=args.model,
        revision=args.revision,
        checkpoint_path=args.checkpoint_path,
        expert_tensor_order=args.expert_tensor_order,
    )
    manifest_load_seconds = time.perf_counter() - manifest_started
    draft_manifest = None
    if args.enable_draft_update:
        draft_manifest = CheckpointManifest.load(
            model=args.draft_model,
            revision=args.draft_revision,
            checkpoint_path=args.draft_checkpoint_path,
            expert_tensor_order=args.expert_tensor_order,
        )
    publisher = NcclCheckpointPublisher(
        base_url=args.base_url,
        manifest=manifest,
        device=args.device,
        update_bucket_size_bytes=args.update_bucket_size_mb * 1024**2,
        direct_file_expert_h2d=args.direct_file_expert_h2d,
        timeout_seconds=args.timeout,
    )
    try:
        initialize_started = time.perf_counter()
        initialization = publisher.initialize()
        initialize_seconds = time.perf_counter() - initialize_started
        updates = [publisher.publish_update(target="main")]
        if args.enable_mtp:
            updates.append(publisher.publish_update(target="draft"))
        elif draft_manifest is not None:
            updates.append(
                publisher.publish_update(
                    target="draft",
                    manifest=draft_manifest,
                )
            )
    finally:
        publisher.shutdown()

    print(
        f"manifest load {manifest_load_seconds:.2f} s, "
        f"initialize (incl. NCCL rendezvous) {initialize_seconds:.2f} s",
        file=sys.stderr,
    )
    for update in updates:
        print(
            format_timing_table(update["timing"], title=f"{update['target']} update"),
            file=sys.stderr,
        )
    result = {
        "checkpoint": manifest.summary(),
        "manifest_load_seconds": manifest_load_seconds,
        "initialize_seconds": initialize_seconds,
        "initialization": initialization,
        "mtp_enabled": args.enable_mtp,
        "independent_draft_enabled": draft_manifest is not None,
        "draft_checkpoint": (
            draft_manifest.summary() if draft_manifest is not None else None
        ),
        "update": updates[0],
        "updates": updates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"send_weights_completed": True}))


if __name__ == "__main__":
    main()
