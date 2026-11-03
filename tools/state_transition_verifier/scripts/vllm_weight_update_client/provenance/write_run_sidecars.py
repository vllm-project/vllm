#!/usr/bin/env python3
"""Write one provenance sidecar for every artifact in a run directory."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("model_path")
    parser.add_argument("revision")
    parser.add_argument("nodes")
    parser.add_argument("head")
    parser.add_argument(
        "--cmd", default="run_qwen36_gsm8k_identity_update_tp1_container.sh"
    )
    parser.add_argument(
        "--image", default="<registry>/dev:vllm-modelb-arm64-cu13-0f0fd91"
    )
    parser.add_argument(
        "--image-digest",
        default="sha256:26ceb82f9891e573dc5a7c754431f137c24b868c106dc3c6e1515b6ff4991069",
    )
    parser.add_argument(
        "--source-commit",
        default="1d09155fe31e05c0e08c98e4d08c37aaab6abccd",
    )
    parser.add_argument("--slurm-job", default=os.environ.get("SLURM_JOB_ID"))
    parser.add_argument("--runtime-version", default="0.17.2rc1.dev3992+g0f0fd9186")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--backend", default="resolved by serving log")
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--dp", type=int, default=1)
    parser.add_argument("--ep", type=int, default=1)
    parser.add_argument("--transfer-engine", default="NCCLTrainerWeightTransferEngine")
    args = parser.parse_args()
    meta = {
        "agent": "codex",
        "cmd": args.cmd,
        "git_commit": args.source_commit,
        "env_pins": {
            "model_path": args.model_path,
            "model_revision": args.revision,
            "image": args.image,
            "image_digest": args.image_digest,
            "source_commit": args.source_commit,
            "vllm_runtime_version": args.runtime_version,
            "nodes": args.nodes.split(","),
            "head": args.head,
            "slurm_job": args.slurm_job,
            "dtype": args.dtype,
            "backend": args.backend,
            "tp": args.tp,
            "dp": args.dp,
            "ep": args.ep,
            "transfer_engine": args.transfer_engine,
        },
        "ts": time.time(),
    }
    for path in args.run.rglob("*"):
        if path.is_file() and not path.name.endswith(".meta.json"):
            sidecar = path.with_name(path.name + ".meta.json")
            sidecar.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
