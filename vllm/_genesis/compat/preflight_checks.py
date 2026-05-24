# SPDX-License-Identifier: Apache-2.0
"""Genesis preflight checks — PN60 quant validator + club#34/#43 community rules.

Three operator-facing diagnostic rules surfaced via `genesis doctor` and
optionally callable from start scripts BEFORE `vllm serve` boots:

1. **PN60 quant arg validator** — cross-checks `--quantization` CLI vs
   `config.json:quantization_config.quant_method`. Catches the apnar
   club-3090#51 NVFP4 boot failure (`auto_round` vs `compressed-tensors`
   mismatch yields 30-line pydantic ValidationError) with a one-line
   remediation hint BEFORE vLLM loads.

2. **club#43 grammar rejection rule** — scans last N container log lines
   for repeated `RejectionSampler` failures + low acceptance rate that
   indicate the OpenClaw-class "very quick failure to provide response"
   bug (noonghunna club-3090#43). Emits actionable hint.

3. **club#34 spec-decode token-loop rule** — detects the noonghunna
   club-3090#34 "stuck on reasoning loops" pattern: spec-decode metrics
   show normal acceptance rate but generation throughput drops to 0 for
   sustained windows.

All three are read-only diagnostics — no monkey-patching, no runtime
behavior change. Outputs route through `compat/doctor.py` recommendation
section + can be called from the `genesis doctor` CLI.

Author: Sandermage 2026-05-05.
Sources:
- apnar club-3090#51 (PN60)
- noonghunna club-3090#43 (grammar rejection)
- noonghunna club-3090#34 (token-loop)
"""
from __future__ import annotations

import json
import logging
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

log = logging.getLogger("genesis.compat.preflight_checks")


# ─── Shared types ─────────────────────────────────────────────────────────


@dataclass
class CheckResult:
    """Outcome of a single preflight check.

    severity ∈ {"OK", "INFO", "WARN", "ERROR"} — ERROR means refuse to boot,
    WARN means proceed-with-caution, INFO is purely informational.
    """
    name: str
    severity: str
    message: str
    remediation: Optional[str] = None

    def __str__(self) -> str:
        out = f"[{self.severity}] {self.name}: {self.message}"
        if self.remediation:
            out += f"\n  → {self.remediation}"
        return out


# ─── PN60 — quant arg validator ───────────────────────────────────────────


_QUANT_ALIASES = {
    # CLI flag → set of valid config.json quant_method values
    "compressed-tensors": {"compressed-tensors", "compressed_tensors"},
    "compressed_tensors": {"compressed-tensors", "compressed_tensors"},
    "auto_round": {"auto_round", "auto-round", "autoround"},
    "auto-round": {"auto_round", "auto-round", "autoround"},
    "awq": {"awq"},
    "gptq": {"gptq"},
    "fp8": {"fp8", "compressed-tensors"},  # FP8 often delivered as compressed-tensors
    "marlin": {"marlin", "compressed-tensors"},
}


def check_quant_arg(
    cli_quantization: Optional[str],
    model_dir: str,
) -> CheckResult:
    """PN60: validate CLI --quantization vs config.json quant_method.

    Inputs
    ------
    cli_quantization : str or None
        Value passed to `vllm serve --quantization`. None means CLI did
        not provide one (vLLM will infer from config).
    model_dir : str
        Local path or HF model id of the loaded model. Must contain
        `config.json` for the validator to fire.

    Returns
    -------
    CheckResult — OK / INFO / ERROR with remediation hint.
    """
    config_path = Path(model_dir) / "config.json"
    if not config_path.is_file():
        return CheckResult(
            name="PN60 quant validator",
            severity="INFO",
            message=(
                f"config.json not found at {config_path} — skip "
                "(model may be HF-cached path or remote)"
            ),
        )

    try:
        with open(config_path) as f:
            config = json.load(f)
    except Exception as e:
        return CheckResult(
            name="PN60 quant validator",
            severity="WARN",
            message=f"failed to read {config_path}: {e}",
        )

    quant_cfg = config.get("quantization_config", {}) or {}
    config_method = (
        quant_cfg.get("quant_method")
        or quant_cfg.get("method")
        or ""
    ).strip()

    if not config_method:
        return CheckResult(
            name="PN60 quant validator",
            severity="INFO",
            message="model config has no quantization_config.quant_method "
                    "— vLLM will infer from weight format",
        )

    if cli_quantization is None:
        return CheckResult(
            name="PN60 quant validator",
            severity="OK",
            message=f"CLI did not pass --quantization; vLLM will use "
                    f"config-detected method='{config_method}'",
        )

    cli_norm = cli_quantization.strip().lower()
    valid_set = _QUANT_ALIASES.get(cli_norm, {cli_norm})

    if config_method.lower() in {v.lower() for v in valid_set}:
        return CheckResult(
            name="PN60 quant validator",
            severity="OK",
            message=f"--quantization='{cli_quantization}' matches "
                    f"config method='{config_method}'",
        )

    return CheckResult(
        name="PN60 quant validator",
        severity="ERROR",
        message=(
            f"--quantization='{cli_quantization}' does NOT match "
            f"config.json quant_method='{config_method}'"
        ),
        remediation=(
            f"Replace `--quantization {cli_quantization}` with "
            f"`--quantization {config_method}` (or remove --quantization "
            "and let vLLM infer)"
        ),
    )


# ─── club#43 — grammar rejection / OpenClaw quick-failure rule ────────────


_GRAMMAR_REJECT_PATTERNS = [
    # vLLM emits these when grammar acceptance fails:
    re.compile(r"GrammarRejection|grammar.+reject", re.I),
    re.compile(r"all (\d+) candidate tokens were rejected", re.I),
    re.compile(r"acceptance.{0,20}rate.{0,20}(0\.|0\b)", re.I),
    # OpenClaw-style early-failure signature:
    re.compile(r"finish_reason.*length.*completion_tokens.*(0|1|2|3|4|5)\b", re.I),
]


def check_grammar_rejection_pattern(
    log_text: str,
    sample_lines: int = 200,
) -> CheckResult:
    """club#43: scan recent log lines for grammar-rejection storms.

    Heuristic: if 3+ different rejection-pattern hits OR a single
    "all candidates rejected" event in the recent log window, emit WARN
    with hint to (a) check tool schema strictness, (b) try
    `--guided-decoding-backend outlines` instead of default, (c) upgrade
    `--max-tokens` floor (some OpenClaw clients send max_tokens=1 by
    accident).

    Returns INFO if no pattern observed.
    """
    lines = log_text.splitlines()[-sample_lines:]
    text = "\n".join(lines)

    hits: list[str] = []
    for pat in _GRAMMAR_REJECT_PATTERNS:
        if pat.search(text):
            hits.append(pat.pattern[:50])

    if not hits:
        return CheckResult(
            name="club#43 grammar rejection",
            severity="OK",
            message=f"no grammar-rejection pattern detected in last "
                    f"{len(lines)} log lines",
        )

    if len(hits) >= 2 or "all" in text.lower() and "candidate" in text.lower():
        return CheckResult(
            name="club#43 grammar rejection",
            severity="WARN",
            message=(
                f"detected {len(hits)} grammar-rejection signal(s) in recent logs"
            ),
            remediation=(
                "Check: (1) tool schema is not over-constrained "
                "(test `tool_choice='auto'` instead of 'required'); "
                "(2) max_tokens floor is sane (≥32); "
                "(3) try `--guided-decoding-backend outlines` if using "
                "default xgrammar. Cross-ref noonghunna club-3090#43."
            ),
        )

    return CheckResult(
        name="club#43 grammar rejection",
        severity="INFO",
        message=f"weak signal ({len(hits)} pattern hit) — monitor",
    )


# ─── club#34 — spec-decode token-loop / stuck-on-reasoning rule ───────────


_SPEC_METRICS_RE = re.compile(
    r"Mean acceptance length:\s*([\d.]+).+?"
    r"Accepted throughput:\s*([\d.]+).+?"
    r"Drafted throughput:\s*([\d.]+).+?"
    r"Avg Draft acceptance rate:\s*([\d.]+)%",
    re.S,
)
_GEN_THROUGHPUT_RE = re.compile(
    r"Avg generation throughput:\s*([\d.]+) tokens/s.+?Running:\s*(\d+)",
    re.S,
)


def check_spec_decode_token_loop(
    log_text: str,
    sample_lines: int = 200,
) -> CheckResult:
    """club#34: detect "stuck on reasoning loops" pattern.

    Symptom: spec-decode metrics show healthy acceptance (>50%) AND
    Running >= 1 request, BUT generation throughput stays at 0 tokens/s
    for multiple consecutive metric lines.

    Returns WARN if pattern detected; OK otherwise.
    """
    lines = log_text.splitlines()[-sample_lines:]
    text = "\n".join(lines)

    spec_metrics = _SPEC_METRICS_RE.findall(text)
    gen_metrics = _GEN_THROUGHPUT_RE.findall(text)

    if not spec_metrics or not gen_metrics:
        return CheckResult(
            name="club#34 token-loop",
            severity="OK",
            message=(
                f"insufficient metrics in last {len(lines)} lines "
                f"(spec={len(spec_metrics)}, gen={len(gen_metrics)}) — "
                "no signal yet"
            ),
        )

    # Look for: accept_rate >= 50% AND throughput = 0 AND Running >= 1
    # in CONSECUTIVE recent metric lines.
    # Audit P2 fix 2026-05-05 (genesis_deep_cross_audit): the previous
    # implementation summed all matches in the window which gave false
    # positives when normal snapshots interleaved bad ones. Track the
    # current streak and report the maximum streak observed.
    current_streak = 0
    max_streak = 0
    for i in range(min(len(spec_metrics), len(gen_metrics))):
        try:
            accept_rate = float(spec_metrics[i][3])
            gen_throughput = float(gen_metrics[i][0])
            running = int(gen_metrics[i][1])
        except (ValueError, IndexError):
            current_streak = 0
            continue
        if accept_rate >= 50.0 and gen_throughput < 0.5 and running >= 1:
            current_streak += 1
            if current_streak > max_streak:
                max_streak = current_streak
        else:
            current_streak = 0
    stuck_count = max_streak

    if stuck_count >= 2:
        return CheckResult(
            name="club#34 token-loop",
            severity="WARN",
            message=(
                f"detected {stuck_count} consecutive metric snapshot(s) with "
                "spec-decode accepting (>50%) but gen throughput=0 + "
                "Running ≥1 — classic 'stuck on reasoning loops' pattern"
            ),
            remediation=(
                "Try: (1) restart container — preempt cycle stuck on EOS; "
                "(2) check P107 MTP truncation detector is enabled "
                "(`GENESIS_ENABLE_P107_MTP_TRUNCATION_DETECTOR=1`) for "
                "retryable error visibility; (3) reduce "
                "`--max-num-seqs` if running multiple concurrent requests. "
                "Cross-ref noonghunna club-3090#34."
            ),
        )

    return CheckResult(
        name="club#34 token-loop",
        severity="OK",
        message=f"spec-decode + generation throughput in healthy regime "
                f"({stuck_count} weak hits in last {len(lines)} lines)",
    )


# ─── Compose ──────────────────────────────────────────────────────────────


def fetch_container_logs(
    container_name: str = "vllm-server-mtp-test",
    tail_lines: int = 200,
) -> str:
    """Return last N lines from the named docker container, or empty."""
    if not os.environ.get("GENESIS_DOCKER_LOGS_AVAILABLE", "1") == "1":
        return ""
    try:
        result = subprocess.run(
            ["docker", "logs", "--tail", str(tail_lines), container_name],
            capture_output=True, text=True, timeout=10,
        )
        return result.stdout + result.stderr
    except Exception as e:
        log.debug("fetch_container_logs failed: %s", e)
        return ""


def run_all_preflight_checks(
    cli_quantization: Optional[str] = None,
    model_dir: Optional[str] = None,
    container_name: str = "vllm-server-mtp-test",
) -> list[CheckResult]:
    """Run PN60 + club#43 + club#34 in sequence; return all results.

    Designed for invocation from `genesis doctor` (which passes through
    discovered config) or from a start-script preflight call:

        # bash:
        python3 -m vllm._genesis.compat.preflight_checks \\
            --quantization auto_round \\
            --model /models/Qwen3.6-27B-int4-AutoRound

    """
    results: list[CheckResult] = []

    # PN60 quant validator
    if model_dir:
        results.append(check_quant_arg(cli_quantization, model_dir))

    # club#43 + club#34 — log-driven, only if container has output
    log_text = fetch_container_logs(container_name)
    if log_text:
        results.append(check_grammar_rejection_pattern(log_text))
        results.append(check_spec_decode_token_loop(log_text))

    return results


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entrypoint for standalone preflight invocation.

    Exit codes:
      0 — all checks OK / INFO
      1 — at least one WARN
      2 — at least one ERROR (refuse-to-boot signal)
    """
    import argparse
    parser = argparse.ArgumentParser(
        description="Genesis preflight checks (PN60 + club#43 + club#34)",
    )
    parser.add_argument("--quantization", default=None,
                        help="value of --quantization CLI arg passed to vllm serve")
    parser.add_argument("--model", default=None,
                        help="model dir or HF id for config.json validation")
    parser.add_argument("--container", default="vllm-server-mtp-test",
                        help="docker container to read logs from")
    parser.add_argument("--json", action="store_true",
                        help="emit JSON instead of human-readable text")
    args = parser.parse_args(argv)

    results = run_all_preflight_checks(
        cli_quantization=args.quantization,
        model_dir=args.model,
        container_name=args.container,
    )

    if args.json:
        out = [{"name": r.name, "severity": r.severity,
                "message": r.message, "remediation": r.remediation}
               for r in results]
        print(json.dumps(out, indent=2))
    else:
        print("=" * 70)
        print("Genesis preflight check results")
        print("=" * 70)
        for r in results:
            print(str(r))
        print("=" * 70)

    if any(r.severity == "ERROR" for r in results):
        return 2
    if any(r.severity == "WARN" for r in results):
        return 1
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
