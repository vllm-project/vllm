#!/usr/bin/env bash
#
# check_pct_status_7.sh — verify Intel Priority Core Turbo (PCT) / CLOS status
# and print CPU ID lists by CLOS using `core-power get-assoc`.
#
# Requires:
#   - intel-speed-select installed (root needed)
#   - this intel-speed-select build supports: core-power get-assoc
#
# Config via env:
#   TARGET_CLOS=0     # which CLOS to print as "PCT list" (default 0)
#   CHUNK=64          # CPUs per get-assoc call
#   DEBUG_MAP=0       # 1 = show tmp_map with invisible chars (cat -A)
#

set -euo pipefail

TARGET_CLOS="${TARGET_CLOS:-0}"
CHUNK="${CHUNK:-64}"
DEBUG_MAP="${DEBUG_MAP:-0}"
SUDO=""
if [ "$(id -u)" -ne 0 ]; then
  SUDO="sudo"
fi

# Write outputs here (host-visible via your compose volume mount)
RESULTS_DIR="/workspace/benchmarks/results"
mkdir -p "${RESULTS_DIR}"

# --- Functions -------------------------------------------------------------

print_header() {
  echo "------------------------------------------------------------"
  echo "$1"
  echo "------------------------------------------------------------"
}

# Get per-CPU CLOS assignment for a list/range via get-assoc
# Prints normalized lines: "<cpu> <clos>\n"
get_assoc_for_cpulist() {
  local cpu_list="$1"
  $SUDO intel-speed-select -c "$cpu_list" core-power get-assoc 2>&1 |
    while IFS= read -r line; do
      if [[ "$line" =~ cpu-([0-9]+) ]]; then
        cur_cpu="${BASH_REMATCH[1]}"
      fi
      if [[ "$line" =~ clos:([0-9]+) ]]; then
        printf "%s %s\n" "${cur_cpu:-?}" "${BASH_REMATCH[1]}"
      fi
    done
}

# --- 1. Basic CPU / tool check -------------------------------------------

print_header "CPU and Intel Speed Select Capability"

if ! command -v intel-speed-select &>/dev/null; then
  echo "❌ intel-speed-select not found. Please install/build it first."
  exit 1
fi

$SUDO intel-speed-select --info 2>&1 | grep -E "Intel|Executing|Supported|Features" || true
echo

# --- 2. Check Turbo Frequency (PCT) status --------------------------------

print_header "PCT (Turbo-Frequency) Feature Status"

TF_OUT="$($SUDO intel-speed-select turbo-freq info -l 1 2>&1 || true)"

if echo "$TF_OUT" | grep -qi "Invalid command: specify tdp_level"; then
  echo "⚠️  Multiple TDP levels detected. Use: $SUDO intel-speed-select turbo-freq info --tdp_level <N>"
elif echo "$TF_OUT" | grep -qi "Failed to get turbo-freq info"; then
  echo "⚠️  turbo-freq info failed at this level. This does not block Core Power / CLOS usage."
elif echo "$TF_OUT" | grep -qi "high-priority"; then
  echo "✅ PCT (Turbo-Frequency) data present."
else
  echo "⚠️  turbo-freq data not returned. PCT turbo tables may be unavailable or BIOS not configured."
fi
echo

# --- 3. Check Core Power (CLOS) status ------------------------------------

print_header "Core Power (CLOS) Feature Status"

CP_OUT="$($SUDO intel-speed-select core-power info 2>&1 || true)"

if echo "$CP_OUT" | grep -q "support-status:supported"; then
  if echo "$CP_OUT" | grep -q "enable-status:enabled"; then
    echo "✅ Core Power feature ENABLED"
  else
    echo "⚠️  Core Power supported but DISABLED in BIOS"
  fi

  if echo "$CP_OUT" | grep -q "clos-enable-status:enabled"; then
    echo "✅ CLOS ENABLED"
  else
    echo "⚠️  CLOS disabled"
  fi
else
  echo "❌ Core Power not supported on this system"
fi
echo

# --- 4. Enumerate CPU->CLOS mappings & print lists ------------------------

print_header "CPU -> CLOS Mapping via get-assoc"

MAX_CPU="$(lscpu -p=CPU | grep -v '^#' | cut -d, -f1 | sort -n | tail -n 1 || true)"
if [[ -z "${MAX_CPU:-}" ]]; then
  echo "❌ Could not determine CPU range from lscpu."
  exit 1
fi

# Check get-assoc support
if ! $SUDO intel-speed-select -c 0 core-power get-assoc >/dev/null 2>&1; then
  echo "❌ This intel-speed-select build does not support: core-power get-assoc"
  exit 1
fi

tmp_map="$(mktemp)"
trap 'rm -f "$tmp_map"' EXIT

start=0
while (( start <= MAX_CPU )); do
  end=$(( start + CHUNK - 1 ))
  if (( end > MAX_CPU )); then end="$MAX_CPU"; fi
  range="${start}-${end}"
  get_assoc_for_cpulist "$range" >> "$tmp_map"
  start=$(( end + 1 ))
done

if [[ "$DEBUG_MAP" == "1" ]]; then
  echo "DEBUG_MAP=1: Showing first 40 tmp_map lines with invisible chars:"
  cat -A "$tmp_map" | head -n 40
  echo
fi

# Distribution by CLOS (robust; normalize digits)
echo "CLOS distribution (count by clos id):"
python3 - <<'PY' "$tmp_map"
import re,sys
path=sys.argv[1]
counts={}
with open(path,'r',errors='replace') as f:
    for line in f:
        parts=line.strip().split()
        if len(parts)<2:
            continue
        clos=re.sub(r'[^0-9]','',parts[1])
        if clos=="":
            continue
        counts[clos]=counts.get(clos,0)+1
for k in sorted(counts, key=lambda x:int(x)):
    print(f"  clos:{k} -> {counts[k]} CPUs")
PY
echo

# Print target CLOS list as compressed ranges (single Python pass; no tmp_target, no piping)
print_header "CPU list for TARGET_CLOS=${TARGET_CLOS}"

CLOS_LINE="$(
python3 - <<'PY' "$tmp_map" "$TARGET_CLOS"
import re,sys
path=sys.argv[1]
target=str(sys.argv[2])

cpus=[]
with open(path,'r',errors='replace') as f:
    for line in f:
        parts=line.strip().split()
        if len(parts)<2:
            continue
        cpu=re.sub(r'[^0-9]','',parts[0])
        clos=re.sub(r'[^0-9]','',parts[1])
        if cpu=="" or clos=="":
            continue
        if clos==target:
            cpus.append(int(cpu))

cpus=sorted(set(cpus))

if not cpus:
    print(f"⚠️  No CPUs currently report clos:{target}.")
    raise SystemExit(0)

# compress
res=[]
i=0
while i<len(cpus):
    j=i
    while j+1<len(cpus) and cpus[j+1]==cpus[j]+1:
        j+=1
    res.append(str(cpus[i]) if i==j else f"{cpus[i]}-{cpus[j]}")
    i=j+1

print(f"clos:{target} CPU list: {','.join(res)}")
PY
)"

# Always print the same line as before
echo "${CLOS_LINE}"

# Write the list to a host-visible file
# - If TARGET_CLOS=0 (default), file will be: ./results/clos0_cpulist.txt
if [[ "${CLOS_LINE}" =~ ^clos:${TARGET_CLOS}[[:space:]]CPU[[:space:]]list:\ (.*)$ ]]; then
  CLOS_LIST="${BASH_REMATCH[1]}"
  OUT_FILE="${RESULTS_DIR}/clos${TARGET_CLOS}_cpulist.txt"
  echo "${CLOS_LIST}" > "${OUT_FILE}"
  echo "Wrote clos:${TARGET_CLOS} CPU list to ${OUT_FILE}"
else
  echo "WARNING: Did not write clos list file (unexpected output): ${CLOS_LINE}" >&2
fi

echo

# --- 5. Friendly summary --------------------------------------------------

print_header "Summary"

if echo "$TF_OUT" | grep -qi "high-priority"; then
  echo "✅ PCT turbo tables detected (turbo-freq reports high-priority data)"
else
  echo "⚠️  PCT turbo tables not confirmed via turbo-freq output"
fi

if echo "$CP_OUT" | grep -q "enable-status:enabled"; then
  echo "✅ Core Power enabled"
else
  echo "❌ Core Power disabled (enable Intel® Speed Select Core Power in BIOS)"
fi

if echo "$CP_OUT" | grep -q "clos-enable-status:enabled"; then
  echo "✅ CLOS enabled"
else
  echo "❌ CLOS disabled"
fi

echo "Done."

