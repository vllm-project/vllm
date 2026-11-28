#!/usr/bin/env bash
# Model-check the shm_broadcast specs with TLC (exhaustive) and Apalache
# (bounded symbolic). Usage: check.sh [tlc|apalache|all] [jobs]
#
# Checks run in parallel (default 12 jobs; each is a JVM of up to 4 GiB), the
# slowest first; the whole suite takes about 2-3 minutes on 16 cores.
# Apalache bounds: a check expected to fail gets the length of its TLC
# counterexample for that invariant (its state count in .runs/tlc-*.txt,
# minus the initial state), so
# Apalache must find a violation of the same depth; a check expected to hold
# is checked for every behavior of up to 10 steps (6 for inference.qnt, whose
# formula grows fastest). TLC covers the full state space of every check.
# "-" skips Apalache where the counterexample is too deep for it within the
# suite's time (17 or more steps on paths.qnt, 20 or more elsewhere: over half
# an hour each); hwm1OneReader is the Apalache-sized copy of hwm1.
set -uo pipefail
cd "$(dirname "$0")"
backend=${1:-all}
max_jobs=${2:-12}
usage() {
  echo "usage: $0 [tlc|apalache|all] [jobs >= 1]" >&2
  exit 2
}
case $backend in tlc | apalache | all) ;; *) usage ;; esac
[[ $max_jobs =~ ^[1-9][0-9]*$ ]] || usage
mkdir -p .runs .tmp
# One run at a time: checks share .runs/ and their numbered server ports.
exec 9>.runs/lock
flock -n 9 || {
  echo "another check.sh run holds $PWD/.runs/lock" >&2
  exit 1
}
export TMPDIR=$PWD/.tmp JAVA_TOOL_OPTIONS="-Djava.io.tmpdir=$PWD/.tmp"

# spec module invariant expected(ok|violation) apalache-steps
# Each invariant is checked on its own, so a violation of one cannot mask an
# unexpected result for another. A configuration expected to hold everywhere
# keeps one combined check.
checks=(
  "paths pathsFixed inOrder,noLoss,progress ok 10"
  "paths pathsCancel inOrder violation -"
  "paths pathsCancel noLoss violation 11"
  "paths pathsCancel progress violation -"
  "message_queue skipIdleFenced noLostWakeup,ringSafe ok 10"
  "message_queue hwm1NoThrottle noLostWakeup,ringSafe ok 10"
  "message_queue unlimitedHandshake noLostWakeup,ringSafe ok 10"
  "inference fixedDecode noLostWakeup,ringSafe,progress ok 6"
  "inference fixedAllReply noLostWakeup,ringSafe,progress ok 6"
  "inference backpressureAtBound noLostWakeup,ringSafe,progress ok 6"
  "message_queue hwm1OneReader noLostWakeup violation 14"
  "message_queue hwm1 noLostWakeup violation -"
  "message_queue hwm4Slow noLostWakeup violation -"
  "message_queue hwm4Burst noLostWakeup violation -"
  "message_queue hwm2Slow noLostWakeup violation -"
  "message_queue pr58917 noLostWakeup violation 5"
  "message_queue unlimitedNoHandshake noLostWakeup violation 5"
  "message_queue skipIdleUnfenced noLostWakeup violation 9"
  "message_queue skipIdleFlaglessPark noLostWakeup violation 8"
  "inference hwm1Decode noLostWakeup violation 4"
  "inference hwm1Decode progress violation 6"
  "inference raceDecode noLostWakeup violation 4"
  "inference raceDecode progress violation 6"
  "inference backpressureOver progress violation -"
  "paths pathsCurrent inOrder violation -"
  "paths pathsCurrent noLoss violation 11"
  "paths pathsCurrent progress violation 3"
  "paths pathsNoDrop progress violation 2"
)
# Lossy channels never break ring safety: no dropped notification or socket
# message can make a reader read a slot out of order.
for main in hwm1 hwm4Slow hwm4Burst hwm2Slow; do
  checks+=("message_queue $main ringSafe ok -")
done
for main in hwm1OneReader pr58917 unlimitedNoHandshake skipIdleUnfenced \
  skipIdleFlaglessPark; do
  checks+=("message_queue $main ringSafe ok 10")
done
for main in hwm1Decode raceDecode; do
  checks+=("inference $main ringSafe ok 6")
done
checks+=("inference backpressureOver noLostWakeup,ringSafe ok 6")
checks+=("paths pathsNoDrop inOrder,noLoss ok 10")

# run <id> <backend> <spec> <module> <invariants> <expected> <steps>
run() {
  local id=$1 be=$2 spec=$3 main=$4 invs=$5 want=$6 steps=$7 out got text
  local -a inv
  IFS=, read -ra inv <<<"$invs"
  out=$PWD/.runs/$be-$main-${invs//,/-}.txt
  local start=$SECONDS
  # Every quint verify starts its own Apalache server (the TLC backend uses
  # it to compile), so each check needs a private port, and a private working
  # and temp directory: TLC unpacks its standard modules into java.io.tmpdir.
  local dir=$PWD/.runs/work-$id
  rm -rf "$dir" && mkdir -p "$dir/tmp"
  local -a cmd=(quint verify "../../$spec.qnt" --main "$main"
    --server-endpoint "localhost:$((8900 + id))" --invariants "${inv[@]}")
  if [[ $be == tlc ]]; then
    cmd+=(--backend tlc)
  else
    cmd+=(--max-steps "$steps")
  fi
  (cd "$dir" && TMPDIR=$dir/tmp JAVA_TOOL_OPTIONS="-Djava.io.tmpdir=$dir/tmp" \
    "${cmd[@]}") >"$out" 2>&1
  rm -rf "$dir"
  # Classify by the reported outcome, not the exit code: quint also exits 1
  # on tool errors.
  text=$(<"$out")
  if [[ $text == *"[ok]"* ]]; then
    got=ok
  elif [[ $text == *"[violation]"* ]]; then
    got=violation
  else
    got=error
  fi
  printf '%-9s %-22s %-28s %-10s want %-10s %5ss %s\n' "$be" "$main" "$invs" \
    "$got" "$want" $((SECONDS - start)) \
    "$([[ $got == "$want" ]] && echo PASS || echo FAIL)" >".runs/$id.result"
}

rm -f .runs/*.result
id=0
for c in "${checks[@]}"; do
  read -r spec main invs want steps <<<"$c"
  for be in tlc apalache; do
    [[ $backend == "$be" || $backend == all ]] || continue
    [[ $be == apalache && $steps == - ]] && continue
    id=$((id + 1))
    while (($(jobs -rp | wc -l) >= max_jobs)); do
      wait -n
    done
    run "$id" "$be" "$spec" "$main" "$invs" "$want" "$steps" &
  done
done
wait

status=0
for ((i = 1; i <= id; i++)); do
  cat ".runs/$i.result"
  [[ $(<".runs/$i.result") == *PASS ]] || status=1
done
exit $status
