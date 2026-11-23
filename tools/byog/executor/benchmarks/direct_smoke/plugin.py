# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Phase 0 direct-mode benchmark plugin executing via `ctx.run_command`."""

from __future__ import annotations

import json
import sys
from typing import Any

from tools.byog.executor.interfaces import (
    BenchmarkRunner,
    DirectRunContext,
    PointResult,
    ServerRunContext,
    WorkloadPoint,
)
from tools.byog.schemas.result import ExecutionStatus

_DIRECT_PROBE_CODE = (
    "import json, os, time; "
    "t0 = time.monotonic(); "
    "iters = int(os.environ.get('BYOG_DIRECT_ITERS', '2')); "
    "acc = sum(i * i for i in range(1000 * iters)); "
    "dt = max(1e-6, time.monotonic() - t0); "
    "print(json.dumps({'iterations': iters, 'elapsed_s': dt, "
    "'checksum': acc, 'hf_token_leaked': 'HF_TOKEN' in os.environ}))"
)


class DirectSmokeBenchmark(BenchmarkRunner):
    """Direct-mode micro-benchmark proving server-less execution via PlatformAdapter."""

    def points(self, params: dict[str, Any]) -> list[WorkloadPoint]:
        cases = params.get("cases")
        if isinstance(cases, list) and cases:
            return [
                WorkloadPoint(
                    point_id=str(case),
                    params={**params, "case": str(case)},
                )
                for case in cases
            ]
        return [WorkloadPoint(point_id="default", params=dict(params))]

    def run_point(
        self,
        point: WorkloadPoint,
        ctx: ServerRunContext | DirectRunContext,
    ) -> PointResult:
        if not isinstance(ctx, DirectRunContext):
            raise TypeError("DirectSmokeBenchmark requires DirectRunContext.")

        iterations = int(point.params.get("iterations", 2))
        cmd = [sys.executable, "-c", _DIRECT_PROBE_CODE]
        cmd_res = ctx.run_command(cmd, ctx.timeout_s)

        if cmd_res.timed_out:
            return PointResult(
                status=ExecutionStatus.TIMED_OUT,
                attempted=iterations,
                completed=0,
                infra_errors=1,
                error_message="Direct smoke subprocess timed out.",
            )
        if cmd_res.returncode != 0:
            return PointResult(
                status=ExecutionStatus.FAILED,
                attempted=iterations,
                completed=0,
                infra_errors=1,
                error_message=(
                    f"Direct smoke exited with code {cmd_res.returncode}: "
                    f"{cmd_res.stderr.strip()}"
                ),
            )

        try:
            parsed = json.loads(cmd_res.stdout.strip().splitlines()[-1])
        except (IndexError, ValueError) as exc:
            return PointResult(
                status=ExecutionStatus.FAILED,
                attempted=iterations,
                completed=0,
                infra_errors=1,
                error_message=f"Invalid JSON from direct smoke subprocess: {exc}",
            )

        out_path = ctx.output_dir / f"direct_smoke_{point.point_id}.json"
        out_path.write_text(
            json.dumps(parsed, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return PointResult(
            status=ExecutionStatus.COMPLETED,
            metrics={
                "elapsed_s": float(parsed["elapsed_s"]),
                "wall_s": cmd_res.elapsed_s,
                "hf_token_leaked": int(bool(parsed.get("hf_token_leaked"))),
            },
            attempted=iterations,
            completed=iterations,
            artifacts=[out_path],
        )
