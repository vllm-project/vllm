#!/usr/bin/env bash
# pct_map_and_set_clos_v3.sh
#
# Auto-detect HP_PER_DOMAIN from:
#   intel-speed-select perf-profile info
# by reading: speed-select-turbo-freq-properties -> bucket-X -> high-priority-cores-count
#
# Then:
#  - Build HP CPU list per NUMA node (domain approximation via numactl -H)
#  - Close HP set under HT siblings (required on your platform)
#  - Compute Non-HP = All online CPUs - HP_effective
#  - Optionally apply CLOS:
#        HP_effective -> HP_CLOS (default 0)
#        Non-HP       -> OTHER_CLOS (default 2)
#
# Output control:
#   Quiet by default: suppresses verbose intel-speed-select structural output.
#   DEBUG_VERBOSE=1 shows raw intel-speed-select outputs.
#
# Modes:
#   DEBUG_MODE=1 : compute-only (no writes, no verification)
#   DRY_RUN=1    : print commands only (no writes)
#
set -euo pipefail

HP_PER_DOMAIN="${HP_PER_DOMAIN:-}"
HP_BUCKET="${HP_BUCKET:-0}"
INCLUDE_HT="${INCLUDE_HT:-0}"

HP_CLOS="${HP_CLOS:-0}"
OTHER_CLOS="${OTHER_CLOS:-2}"

DEBUG_MODE="${DEBUG_MODE:-0}"
DRY_RUN="${DRY_RUN:-0}"
DEBUG_VERBOSE="${DEBUG_VERBOSE:-0}"
DEBUG_MAP="${DEBUG_MAP:-0}"
SHOW_VERIFY_LINES="${SHOW_VERIFY_LINES:-40}"

SUDO=""
[[ "$(id -u)" -ne 0 ]] && SUDO="sudo"

ISS="${ISS:-$SUDO intel-speed-select}"
ISS_PERF_CMD="${ISS_PERF_CMD:-$ISS perf-profile info}"

print_header() {
  echo "------------------------------------------------------------"
  echo "$1"
  echo "------------------------------------------------------------"
}

die() { echo "ERROR: $*" >&2; exit 1; }

detect_hp_per_domain() {
  local want_bucket="${1:-0}"
  local out
  out="$($ISS_PERF_CMD 2>&1 || true)"

  local pairs
  pairs="$(
    echo "$out" | tr -d '\r' | awk '
      BEGIN { in_tf=0; b="" }
      /speed-select-turbo-freq-properties/ { in_tf=1; b=""; next }
      in_tf && $1 ~ /^bucket-[0-9]+$/ { b=$1; next }
      in_tf && index($0, "high-priority-cores-count:") {
        line=$0
        sub(/.*high-priority-cores-count:[[:space:]]*/, "", line)
        sub(/[^0-9].*$/, "", line)
        if (b != "" && line != "") print b ":" line
      }
    ' | sort -V | uniq
  )"
  [[ -z "$pairs" ]] && { echo ""; return 0; }

  if [[ "$DEBUG_MAP" == "1" ]]; then
    print_header "Detected HP buckets (unique)"
    echo "$pairs" | sed 's/^/  /'
    echo
  fi

  local sel
  sel="$(
    echo "$pairs" | awk -F: -v want="bucket-${want_bucket}" '
      $1==want {print $2; found=1}
      END{ if(!found) exit 1 }
    ' 2>/dev/null | head -n1 || true
  )"
  [[ -n "$sel" ]] && { echo "$sel"; return 0; }

  echo "$pairs" | awk -F: 'BEGIN{m=999999}{v=$2+0; if(v<m)m=v}END{print m}'
}

compress_ranges_from_list() {
  python3 -c '
import sys
xs=[]
for line in sys.stdin:
  line=line.strip()
  if not line: continue
  try: xs.append(int(line.split()[0]))
  except: pass
xs=sorted(set(xs))
if not xs:
  print("")
  raise SystemExit(0)
res=[]
i=0
while i<len(xs):
  j=i
  while j+1<len(xs) and xs[j+1]==xs[j]+1: j+=1
  res.append(str(xs[i]) if i==j else f"{xs[i]}-{xs[j]}")
  i=j+1
print(",".join(res))
'
}

expand_with_ht_siblings() {
  local cpu_ranges="$1"
  [[ -n "$cpu_ranges" ]] || { echo ""; return; }

  local total
  total="$(lscpu -p=CPU | grep -v '^#' | wc -l | tr -d ' ')"
  if ! [[ "$total" =~ ^[0-9]+$ ]] || (( total < 2 )) || (( total % 2 != 0 )); then
    echo "$cpu_ranges"
    return
  fi
  local offset=$(( total / 2 ))

  HP_LIST="$cpu_ranges" OFFSET="$offset" python3 - <<'PY'
import os
hp=os.environ["HP_LIST"]
offset=int(os.environ["OFFSET"])

def expand(s):
  out=set()
  for part in s.split(","):
    part=part.strip()
    if not part: continue
    if "-" in part:
      a,b=part.split("-",1)
      out.update(range(int(a), int(b)+1))
    else:
      out.add(int(part))
  return out

hp_set=expand(hp)
all_set=set(hp_set)
for c in list(hp_set):
  all_set.add(c+offset)

xs=sorted(all_set)
res=[]
i=0
while i<len(xs):
  j=i
  while j+1<len(xs) and xs[j+1]==xs[j]+1: j+=1
  res.append(str(xs[i]) if i==j else f"{xs[i]}-{xs[j]}")
  i=j+1
print(",".join(res))
PY
}

apply_assoc() {
  # IMPORTANT: cpu_list first, clos second
  local cpu_list="$1"
  local clos="$2"

  [[ -n "$cpu_list" ]] || die "apply_assoc got empty cpu_list (clos=$clos)"
  [[ "$clos" =~ ^[0-9]+$ ]] || die "apply_assoc got non-numeric clos='$clos'"

  local out rc
  if [[ "$DEBUG_VERBOSE" == "1" ]]; then
    $ISS -c "$cpu_list" core-power assoc --clos "$clos"
    return 0
  fi
  ## IMPORTANT: need to enable CLOS feature first in OS before assoc
  $ISS core-power enable --clos
  out="$($ISS -c "$cpu_list" core-power assoc --clos "$clos" 2>&1 >/dev/null)" || rc=$?
  rc="${rc:-0}"

  # Some builds print errors but still return 0; catch both.
  if (( rc != 0 )) || echo "$out" | grep -qiE 'malformed arguments|Error:'; then
    echo "$out" >&2
    die "intel-speed-select assoc failed (clos=$clos cpu_list=$cpu_list)"
  fi
}

get_assoc_pairs() {
  local cpu_list="$1"
  [[ -n "$cpu_list" ]] || return 0
  $ISS -c "$cpu_list" core-power get-assoc 2>&1 | awk '
    /cpu-[0-9]+/{
      cpu=$0; sub(/^.*cpu-/,"",cpu); sub(/[^0-9].*$/,"",cpu); next
    }
    /clos:[0-9]+/{
      clos=$0; sub(/^.*clos:/,"",clos); sub(/[^0-9].*$/,"",clos);
      if (cpu!="") printf "cpu-%s clos:%s\n", cpu, clos
    }'
}

# ---- checks
command -v numactl >/dev/null 2>&1 || die "numactl not found"
command -v lscpu >/dev/null 2>&1 || die "lscpu not found"
command -v python3 >/dev/null 2>&1 || die "python3 not found"
command -v intel-speed-select >/dev/null 2>&1 || die "intel-speed-select not found"

# ---- NUMA
TMP_NUMA="$(mktemp)"
trap 'rm -f "$TMP_NUMA"' EXIT
numactl -H > "$TMP_NUMA"

declare -A NODE_CPUS
while IFS= read -r line; do
  [[ "$line" =~ ^node[[:space:]]+([0-9]+)[[:space:]]+cpus: ]] || continue
  node="${BASH_REMATCH[1]}"
  cpus="${line#*cpus:}"
  cpus="$(echo "$cpus" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
  NODE_CPUS["$node"]="$cpus"
done < "$TMP_NUMA"
[[ "${#NODE_CPUS[@]}" -gt 0 ]] || die "Could not parse NUMA nodes from numactl -H"

# ---- HP_PER_DOMAIN auto
if [[ -z "${HP_PER_DOMAIN}" || "${HP_PER_DOMAIN}" == "0" ]]; then
  det="$(detect_hp_per_domain "$HP_BUCKET")"
  [[ -n "$det" ]] || die "Could not auto-detect HP_PER_DOMAIN (set HP_PER_DOMAIN manually)"
  HP_PER_DOMAIN="$det"
fi

print_header "Config"
echo "HP_PER_DOMAIN=$HP_PER_DOMAIN (HP_BUCKET=$HP_BUCKET)"
echo "INCLUDE_HT=$INCLUDE_HT"
echo "HP_CLOS=$HP_CLOS  OTHER_CLOS=$OTHER_CLOS"
echo "DEBUG_MODE=$DEBUG_MODE  DRY_RUN=$DRY_RUN  DEBUG_VERBOSE=$DEBUG_VERBOSE  DEBUG_MAP=$DEBUG_MAP"
echo

print_header "HP selection per NUMA node (initial pick)"
HP_LIST=()
for node in $(printf '%s\n' "${!NODE_CPUS[@]}" | sort -n); do
  read -r -a arr <<< "${NODE_CPUS[$node]}"
  len="${#arr[@]}"
  phys=("${arr[@]}"); ht=()

  if (( len >= 2 && len % 2 == 0 )); then
    half=$((len/2))
    offset=$(( arr[half] - arr[0] ))
    ok=1
    for ((i=0;i<half;i++)); do
      if (( arr[i+half] - arr[i] != offset )); then ok=0; break; fi
    done
    if (( ok==1 )); then
      phys=( "${arr[@]:0:half}" )
      ht=( "${arr[@]:half:half}" )
    fi
  fi

  hp_phys=( "${phys[@]:0:HP_PER_DOMAIN}" )
  out=( "${hp_phys[@]}" )
  if [[ "$INCLUDE_HT" == "1" && "${#ht[@]}" -gt 0 ]]; then
    out+=( "${ht[@]:0:${#hp_phys[@]}}" )
  fi

  echo "node $node -> ${out[*]}"
  HP_LIST+=( "${out[@]}" )
done

HP_RANGES="$(printf "%s\n" "${HP_LIST[@]}" | sort -n | uniq | compress_ranges_from_list)"
[[ -n "$HP_RANGES" ]] || die "HP_RANGES is empty"
HP_EFFECTIVE="$(expand_with_ht_siblings "$HP_RANGES")"
[[ -n "$HP_EFFECTIVE" ]] || die "HP_EFFECTIVE is empty"

echo
echo "HP initial ranges      : $HP_RANGES"
echo "HP effective (with HT) : $HP_EFFECTIVE"
echo

ALL_CPUS_CSV="$(lscpu -p=CPU | grep -v '^#' | cut -d, -f1 | sort -n | uniq | paste -sd, -)"
NON_HP_RANGES="$(
  HP_LIST="$HP_EFFECTIVE" ALL_CPUS="$ALL_CPUS_CSV" python3 - <<'PY'
import os
hp=os.environ["HP_LIST"]; all_=os.environ["ALL_CPUS"]
def expand(s):
  out=set()
  for part in s.split(","):
    part=part.strip()
    if not part: continue
    if "-" in part:
      a,b=part.split("-",1); out.update(range(int(a), int(b)+1))
    else:
      out.add(int(part))
  return out
hp_set=expand(hp); all_set=expand(all_)
non=sorted(all_set-hp_set)
res=[]; i=0
while i<len(non):
  j=i
  while j+1<len(non) and non[j+1]==non[j]+1: j+=1
  res.append(str(non[i]) if i==j else f"{non[i]}-{non[j]}")
  i=j+1
print(",".join(res))
PY
)"

print_header "Computed CPU lists"
echo "HP (effective) : $HP_EFFECTIVE"
echo "Non-HP         : $NON_HP_RANGES"
echo

if [[ "$DEBUG_MODE" == "1" ]]; then
  print_header "DEBUG_MODE=1 (read-only)"
  echo "No CLOS changes applied. No verification performed."
  exit 0
fi

if [[ "$DRY_RUN" == "1" ]]; then
  print_header "DRY_RUN=1 (no changes)"
  echo "Would run:"
  echo "  $ISS -c \"$HP_EFFECTIVE\" core-power assoc --clos $HP_CLOS"
  echo "  $ISS -c \"$NON_HP_RANGES\" core-power assoc --clos $OTHER_CLOS"
  exit 0
fi

print_header "Apply CLOS assignments (quiet)"
echo "Setting HP -> CLOS${HP_CLOS}, Non-HP -> CLOS${OTHER_CLOS}"

apply_assoc "$HP_EFFECTIVE" "$HP_CLOS"
apply_assoc "$NON_HP_RANGES" "$OTHER_CLOS"

echo "Applied."
echo

print_header "Verification (concise CPU->CLOS)"
echo "HP list should be clos:$HP_CLOS"
get_assoc_pairs "$HP_EFFECTIVE" | head -n "$SHOW_VERIFY_LINES" || true
echo "… (showing first $SHOW_VERIFY_LINES lines)"
echo

echo "Non-HP list should be clos:$OTHER_CLOS"
get_assoc_pairs "$NON_HP_RANGES" | head -n "$SHOW_VERIFY_LINES" || true
echo "… (showing first $SHOW_VERIFY_LINES lines)"
echo

echo "Done."
