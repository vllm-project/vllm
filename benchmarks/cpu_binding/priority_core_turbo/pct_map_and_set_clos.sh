#!/usr/bin/env bash
# pct_map_and_set_clos.sh
#
# Combines:
#   - pct_map.sh  (derive HP CPU list from numactl -H + intel-speed-select core-power info)
#   - set_clos.sh (apply HP -> CLOS0 and non-HP -> CLOS2)
# And adds authoritative verification via:
#   sudo intel-speed-select -c <cpu-list> core-power get-assoc
#
# IMPORTANT BEHAVIOR ON YOUR PLATFORM:
#   Your get-assoc output shows that when a CPU thread is assigned to a CLOS, its HT sibling
#   (e.g., +128 on a 256-CPU system) is also effectively in the same CLOS.
#   Therefore we "close" the HP set under HT-sibling relation before computing the non-HP set.
#
# Usage:
#   chmod +x pct_map_and_set_clos.sh
#   DRY_RUN=1 ./pct_map_and_set_clos.sh
#   ./pct_map_and_set_clos.sh
#   HP_PER_DOMAIN=8 INCLUDE_HT=0 ./pct_map_and_set_clos.sh      # INCLUDE_HT influences initial HP pick
#   HP_CLOS=0 OTHER_CLOS=2 ./pct_map_and_set_clos.sh
#
set -euo pipefail

HP_PER_DOMAIN="${HP_PER_DOMAIN:-8}"   # number of "physical" CPUs to pick per domain (initial pick)
INCLUDE_HT="${INCLUDE_HT:-0}"         # 1 = include HT half in initial pick per domain (still will be closed under siblings later)
HP_CLOS="${HP_CLOS:-0}"
OTHER_CLOS="${OTHER_CLOS:-2}"
DRY_RUN="${DRY_RUN:-0}"

ISS_INFO_CMD="${ISS_INFO_CMD:-sudo intel-speed-select core-power info}"
NUMACTL_CMD="${NUMACTL_CMD:-numactl -H}"

tmp_iss="$(mktemp)"
tmp_num="$(mktemp)"
trap 'rm -f "$tmp_iss" "$tmp_num"' EXIT

# -------------------------- collect inputs (capture stderr too) --------------------------
$ISS_INFO_CMD > "$tmp_iss" 2>&1
$NUMACTL_CMD > "$tmp_num" 2>&1

# -------------------------- parse representative CPUs from intel-speed-select ------------
mapfile -t REP_CPUS < <(
  awk '
    BEGIN{FS="cpu-"}
    /^[[:space:]]*cpu-/{
      x=$2
      gsub(/[:[:space:]].*$/,"",x)     # trim trailing ":" or spaces
      if (x=="None") next
      if (x ~ /^-?[0-9]+$/ && x+0 >= 0) print x+0
    }
  ' "$tmp_iss" | sort -n | uniq
)

if [[ "${#REP_CPUS[@]}" -eq 0 ]]; then
  echo "ERROR: Could not extract any representative cpu-N entries from: $ISS_INFO_CMD" >&2
  echo "---- First 120 lines of intel-speed-select output ----" >&2
  sed -n '1,120p' "$tmp_iss" >&2
  exit 2
fi

# -------------------------- parse numactl nodes: node->cpus and cpu->node ----------------
declare -A NODE_CPUS
declare -A CPU_TO_NODE

while IFS= read -r line; do
  [[ "$line" =~ ^node[[:space:]]+([0-9]+)[[:space:]]+cpus: ]] || continue
  node="${BASH_REMATCH[1]}"
  cpus="${line#*cpus:}"
  cpus="$(echo "$cpus" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
  NODE_CPUS["$node"]="$cpus"
  for c in $cpus; do
    CPU_TO_NODE["$c"]="$node"
  done
done < "$tmp_num"

if [[ "${#NODE_CPUS[@]}" -eq 0 ]]; then
  echo "ERROR: Could not parse any NUMA node cpu lists from: $NUMACTL_CMD" >&2
  echo "---- First 120 lines of numactl -H output ----" >&2
  sed -n '1,120p' "$tmp_num" >&2
  exit 2
fi

# -------------------------- helpers ------------------------------------------------------
pick_hp_for_node() {
  local node="$1"
  local n="$2"
  local include_ht="$3"

  read -r -a arr <<< "${NODE_CPUS[$node]}"
  local len="${#arr[@]}"

  local -a phys=()
  local -a ht=()

  # Infer "physical half + HT half" if second half = first half + constant offset
  if (( len >= 2 && len % 2 == 0 )); then
    local half=$((len/2))
    local ok=1
    local offset=$(( arr[half] - arr[0] ))
    for ((i=0;i<half;i++)); do
      if (( arr[i+half] - arr[i] != offset )); then
        ok=0
        break
      fi
    done
    if (( ok==1 )); then
      phys=( "${arr[@]:0:half}" )
      ht=( "${arr[@]:half:half}" )
    else
      phys=( "${arr[@]}" )
      ht=()
    fi
  else
    phys=( "${arr[@]}" )
    ht=()
  fi

  local -a hp_phys=( "${phys[@]:0:n}" )
  local -a out=( "${hp_phys[@]}" )

  if [[ "$include_ht" == "1" && "${#ht[@]}" -gt 0 ]]; then
    local k="${#hp_phys[@]}"
    local -a hp_ht=( "${ht[@]:0:k}" )
    out+=( "${hp_ht[@]}" )
  fi

  printf '%s\n' "${out[@]}" | sort -n | tr '\n' ' ' | sed 's/[[:space:]]*$//'
}

compress_ranges() {
  awk '
    NR==1{start=$1;prev=$1;next}
    {
      if ($1==prev+1){prev=$1;next}
      print start "-" prev
      start=$1;prev=$1
    }
    END{print start "-" prev}
  ' | awk '
    {
      split($0,a,"-");
      if (a[1]==a[2]) print a[1];
      else print $0
    }
  ' | paste -sd, -
}

expand_with_ht_siblings() {
  # Input: cpu ranges, e.g. "0-7,32-39"
  # Output: includes HT siblings based on inferred offset = total_online_cpus/2 (e.g., +128)
  local cpu_ranges="$1"

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
    while j+1<len(xs) and xs[j+1]==xs[j]+1:
        j+=1
    res.append(str(xs[i]) if i==j else f"{xs[i]}-{xs[j]}")
    i=j+1
print(",".join(res))
PY
}

verify_get_assoc() {
  local cpu_list="$1"
  local expect_clos="$2"
  local label="$3"

  echo "=== Verify ($label): expect clos:$expect_clos on cpu-list [$cpu_list] ==="

  out="$(sudo intel-speed-select -c "$cpu_list" core-power get-assoc 2>&1)"

  # Show a short preview
  echo "$out" | sed -n '1,25p'

  local bad=0
  local current_cpu=""

  while IFS= read -r line; do
    if [[ "$line" =~ cpu-([0-9]+) ]]; then
      current_cpu="${BASH_REMATCH[1]}"
    fi
    if [[ "$line" =~ clos:([0-9]+) ]]; then
      found="${BASH_REMATCH[1]}"
      if [[ "$found" != "$expect_clos" ]]; then
        echo "ERROR: cpu-$current_cpu has clos:$found (expected clos:$expect_clos)"
        bad=1
      fi
    fi
  done <<< "$out"

  if [[ "$bad" -eq 1 ]]; then
    echo "Verification FAILED for $label"
    return 1
  fi

  echo "OK: All queried CPUs show clos:$expect_clos"
  echo
}

# -------------------------- map rep CPUs to NUMA nodes (domains) -------------------------
echo "=== Representative CPUs discovered from intel-speed-select core-power info ==="
printf 'rep_cpu: %s\n' "${REP_CPUS[@]}"
echo

echo "=== Domain NUMA nodes (inferred via rep_cpu membership) ==="
declare -A DOMAIN_NODES
for rep in "${REP_CPUS[@]}"; do
  node="${CPU_TO_NODE[$rep]:-}"
  [[ -n "$node" ]] && DOMAIN_NODES["$node"]=1
done

if [[ "${#DOMAIN_NODES[@]}" -eq 0 ]]; then
  echo "ERROR: None of the representative CPUs were found in numactl -H CPU lists." >&2
  exit 3
fi

for node in $(printf '%s\n' "${!DOMAIN_NODES[@]}" | sort -n); do
  rep_show=""
  for rep in "${REP_CPUS[@]}"; do
    if [[ "${CPU_TO_NODE[$rep]:-}" == "$node" ]]; then
      rep_show="$rep"
      break
    fi
  done
  echo "NUMA node $node (rep_cpu=$rep_show)"
  echo "  node_cpus: ${NODE_CPUS[$node]}"
done
echo

# -------------------------- build initial HP list across domains -------------------------
echo "=== Suggested High-Priority (HP) CPU list (initial pick) ==="
echo "HP_PER_DOMAIN=$HP_PER_DOMAIN INCLUDE_HT=$INCLUDE_HT"
hp_all=()
for node in $(printf '%s\n' "${!DOMAIN_NODES[@]}" | sort -n); do
  hp_node="$(pick_hp_for_node "$node" "$HP_PER_DOMAIN" "$INCLUDE_HT")"
  echo "node $node hp_cpus: $hp_node"
  hp_all+=( $hp_node )
done

hp_sorted_unique="$(printf '%s\n' "${hp_all[@]}" | sort -n | uniq)"
HP_RANGES="$(printf '%s\n' "$hp_sorted_unique" | compress_ranges)"

echo
echo "HP cpu-list (ranges, initial): $HP_RANGES"

# Close HP set under HT-sibling relation (required on your platform)
HP_EFFECTIVE_RANGES="$(expand_with_ht_siblings "$HP_RANGES")"
echo "HP effective cpu-list (incl HT siblings): $HP_EFFECTIVE_RANGES"
echo

# -------------------------- compute non-HP list across ALL online CPUs -------------------
ALL_CPUS_CSV="$(lscpu -p=CPU | grep -v '^#' | cut -d, -f1 | sort -n | uniq | paste -sd, -)"

NON_HP_RANGES="$(
  HP_LIST="$HP_EFFECTIVE_RANGES" ALL_CPUS="$ALL_CPUS_CSV" python3 - <<'PY'
import os
hp=os.environ["HP_LIST"]
all_=os.environ["ALL_CPUS"]
def expand(s: str):
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
all_set=expand(all_)
non=sorted(all_set - hp_set)
res=[]
i=0
while i < len(non):
    j=i
    while j+1 < len(non) and non[j+1] == non[j] + 1:
        j += 1
    res.append(str(non[i]) if i==j else f"{non[i]}-{non[j]}")
    i = j + 1
print(",".join(res))
PY
)"

echo "NON-HP cpu-list (ranges): $NON_HP_RANGES"
echo

# -------------------------- apply CLOS assignments ---------------------------------------
echo "=== Apply CLOS assignments ==="
echo "HP -> CLOS${HP_CLOS}       : $HP_EFFECTIVE_RANGES"
echo "Non-HP -> CLOS${OTHER_CLOS}: $NON_HP_RANGES"
echo

if [[ "$DRY_RUN" == "1" ]]; then
  echo "DRY_RUN=1 set; not applying changes."
  echo "Would run:"
  echo "  sudo intel-speed-select -c \"$HP_EFFECTIVE_RANGES\" core-power assoc --clos \"$HP_CLOS\""
  echo "  sudo intel-speed-select -c \"$NON_HP_RANGES\" core-power assoc --clos \"$OTHER_CLOS\""
  echo "And verify via get-assoc."
  exit 0
fi

sudo intel-speed-select -c "$HP_EFFECTIVE_RANGES" core-power assoc --clos "$HP_CLOS"
sudo intel-speed-select -c "$NON_HP_RANGES" core-power assoc --clos "$OTHER_CLOS"

echo
echo "=== Verification using get-assoc (authoritative for your build) ==="
verify_get_assoc "$HP_EFFECTIVE_RANGES" "$HP_CLOS" "HP (effective, incl siblings)"
verify_get_assoc "$NON_HP_RANGES" "$OTHER_CLOS" "Non-HP"

echo "Done."

