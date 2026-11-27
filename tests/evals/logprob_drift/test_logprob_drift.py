# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Logprob drift of a candidate build against a reference build.

Usage, on the reference build and then on the candidate:
pytest -s -v tests/evals/logprob_drift/test_logprob_drift.py --drift-out=runs/main
pytest -s -v tests/evals/logprob_drift/test_logprob_drift.py \
    --drift-out=runs/pr --drift-reference=runs/main

When the candidate's first runs differ from the reference's, the test asks for more
reference runs, which --drift-runs records, then for more of its own.
"""

import asyncio
import json
import secrets
import shlex
import tempfile
from pathlib import Path

import pytest
import yaml

from tests.utils import RemoteOpenAIServer

from .logprob_drift_eval import (
    STAGES,
    compare,
    fresh_caches,
    jsonable,
    load,
    record,
    render,
    save,
)

DEFAULT_STARTUP_MAX_WAIT_SECONDS = 1200


def test_logprob_drift(config_filename: Path, pytestconfig):
    cfg = yaml.safe_load(config_filename.read_text())
    out = Path(pytestconfig.getoption("--drift-out")) / config_filename.stem
    server_args = shlex.split(cfg.get("server_args", ""))
    env = {k: str(v) for k, v in cfg.get("env", {}).items()}

    def recorded(n: int) -> list[dict]:
        """The first n runs in out, recording the ones missing. Each run has its
        own server, compiling and autotuning from empty caches: a run's numerics
        can depend on its launch, and the comparison needs independent runs."""
        while len(paths := sorted(out.glob("*.npz"))) < n:
            seed = secrets.randbelow(2**31)
            with (
                tempfile.TemporaryDirectory() as cache,
                RemoteOpenAIServer(
                    cfg["model_name"],
                    [*server_args, "--disable-uvicorn-access-log"],
                    env_dict=env | fresh_caches(cache),
                    max_wait_seconds=cfg.get(
                        "startup_max_wait_seconds", DEFAULT_STARTUP_MAX_WAIT_SECONDS
                    ),
                ) as server,
            ):
                setup = {"server_args": server_args, "env": env}
                run = asyncio.run(record(server.url_root, seed, setup))
            save(run, out / f"{seed}.npz")
            print(f"Recorded {out / f'{seed}.npz'}")
        return [load(p) for p in paths[:n]]

    reference = pytestconfig.getoption("--drift-reference")
    if reference is None:
        recorded(pytestconfig.getoption("--drift-runs"))
        print(f"{out} holds the reference; pass --drift-reference to compare with it")
        return

    ref_dir = Path(reference) / config_filename.stem
    refs = [load(p) for p in sorted(ref_dir.glob("*.npz"))[: STAGES[-1][0]]]
    for n_refs, n_cands in STAGES:
        if len(refs) < n_refs:
            pytest.fail(
                f"{ref_dir} holds {len(refs)} runs and this stage takes {n_refs}: "
                f"record them on the reference build with --drift-out={reference} "
                f"--drift-runs={n_refs}, then run this again"
            )
        result = compare(refs, recorded(n_cands))
        print(render(result))
        if result["verdict"] != "more":
            break
    (out / "result.json").write_text(json.dumps(jsonable(result), indent=2) + "\n")
    assert result["verdict"] == "same", "the candidate differs beyond rerun noise"
