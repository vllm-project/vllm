#!/usr/bin/env bash
# Ornith XPU graphs canary ladder: A (eager) -> B (PIECEWISE) -> C (FA-in-graph
# FULL), stop-on-fail. Each arm runs full correctness smokes + perf via
# smoke_ornith.sh. On a graph-arm failure, re-verifies the node with a short
# eager smoke (never leave a broken graph server as the last state), then
# exits non-zero. Writes AB_COMPARE_<stamp>.{json,md} under results/.
set -uo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${DIR}/../.." && pwd)"
RESULTS_DIR="${RESULTS_DIR:-${DIR}/results}"
mkdir -p "${RESULTS_DIR}"
export STAMP="${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
export RESULTS_DIR
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3)}"
ARMS="${ARMS:-A B C}"

declare -A ARM_STATUS
failed_arm=""

for arm in ${ARMS}; do
  echo "=== ladder: arm ${arm} ==="
  if ARM="${arm}" PERF=1 bash "${DIR}/smoke_ornith.sh"; then
    ARM_STATUS[${arm}]="PASS"
  else
    ARM_STATUS[${arm}]="FAIL"
    failed_arm="${arm}"
    echo "=== arm ${arm} FAILED — stopping ladder ==="
    break
  fi
done

# Eager re-verify after a graph-arm failure: prove the node still serves
# correctly in the known-good config before finishing.
if [[ -n "${failed_arm}" && "${failed_arm}" != "A" ]]; then
  echo "=== eager re-verify after arm ${failed_arm} failure ==="
  if ARM=A PERF=0 STAMP="${STAMP}_reverify" bash "${DIR}/smoke_ornith.sh"; then
    echo "eager re-verify PASS — node healthy, graphs arm ${failed_arm} is the problem"
  else
    echo "eager re-verify FAIL — NODE/WEIGHTS PROBLEM, investigate before anything else"
  fi
fi

# Comparison report over whatever arms completed.
"${PYTHON_BIN}" - "${RESULTS_DIR}" "${STAMP}" <<'PY'
import difflib, json, os, sys

results_dir, stamp = sys.argv[1:3]

def load(kind, arm):
    path = os.path.join(results_dir, f"{kind}_{arm}_{stamp}.json")
    return json.load(open(path)) if os.path.exists(path) else None

arms = [a for a in ("A", "B", "C") if load("outputs", a)]
outputs = {a: load("outputs", a)["outputs"] for a in arms}
perf = {a: load("arm", a) for a in arms}

agreement = {}
for a in arms:
    if a == "A" or "A" not in outputs:
        continue
    per_smoke = {}
    for name, ref in outputs["A"].items():
        got = outputs[a].get(name, "")
        ratio = difflib.SequenceMatcher(None, ref, got).ratio()
        per_smoke[name] = {"identical": ref == got, "similarity": round(ratio, 4)}
    agreement[f"A_vs_{a}"] = per_smoke

summary = {"stamp": stamp, "arms_completed": arms,
           "perf": {a: {k: v for k, v in (perf[a] or {}).items() if k != "text"}
                    for a in arms if perf.get(a)},
           "output_agreement": agreement}
json.dump(summary, open(os.path.join(results_dir, f"AB_COMPARE_{stamp}.json"), "w"),
          indent=2)

names = {"A": "A: eager (prod config)", "B": "B: PIECEWISE",
         "C": "C: FA-in-graph FULL"}
lines = [f"# Ornith XPU graphs canary A/B — {stamp}", "",
         "| Arm | Ready s | TTFT ms mean | TTFT ms p50 | Decode tok/s mean | Decode tok/s p50 |",
         "| --- | --- | --- | --- | --- | --- |"]
for a in arms:
    p = perf.get(a)
    if not p:
        lines.append(f"| {names[a]} | - | - | - | - | - |")
        continue
    lines.append(
        f"| {names[a]} | {p.get('ready_s','-')} | {p['ttft_ms_mean']:.1f} | "
        f"{p['ttft_ms_p50']:.1f} | {p['decode_tok_s_mean']:.1f} | "
        f"{p['decode_tok_s_p50']:.1f} |")
base = perf.get("A")
for a in arms:
    if a == "A" or not perf.get(a) or not base:
        continue
    d = (perf[a]["decode_tok_s_mean"] - base["decode_tok_s_mean"]) \
        / base["decode_tok_s_mean"] * 100
    t = (perf[a]["ttft_ms_mean"] - base["ttft_ms_mean"]) \
        / base["ttft_ms_mean"] * 100
    lines.append(f"\n**{a} vs A:** decode {d:+.1f}%, TTFT {t:+.1f}%")
for pair, per in agreement.items():
    ident = sum(1 for v in per.values() if v["identical"])
    lines.append(f"\n**Output agreement {pair}:** {ident}/{len(per)} identical; "
                 + ", ".join(f"{k}={v['similarity']}" for k, v in per.items()
                             if not v["identical"]))
md = "\n".join(lines) + "\n"
open(os.path.join(results_dir, f"AB_COMPARE_{stamp}.md"), "w").write(md)
print(md)
PY

echo "=== ladder summary ==="
for arm in ${ARMS}; do
  echo "arm ${arm}: ${ARM_STATUS[${arm}]:-SKIPPED}"
done
[[ -z "${failed_arm}" ]] || exit 1
