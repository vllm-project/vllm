#!/usr/bin/env bash
# Regenerate the model traces that tests/distributed/test_shm_broadcast_model.py
# replays on MessageQueue, and record the hash of the model they came from.
# Usage: gen_traces.sh [kept-traces] [steps] [samples]
#
# Traces come from `quint run --mbt` on skipIdleFenced, the protocol the code
# implements. A trace is kept only if the code can follow it step by step:
# libzmq moves a notification to the reader within microseconds, so the code
# cannot hold one in flight while the reader consumes another. Each kept trace
# is one JSON line: per step the action, the reader it picked, and the state
# the test compares.
# The jq programs below are single-quoted on purpose: `$t`, `$s` and `$i`
# are jq variables, not shell ones.
# shellcheck disable=SC2016
set -euo pipefail
cd "$(dirname "$0")"
keep=${1:-12}
steps=${2:-50}
samples=${3:-600}
# Build into a staging directory; the committed traces are replaced only
# once the new set is complete.
stage=.tmp/traces-new
out=$stage/skipIdleFenced.jsonl
work=.tmp/traces
rm -rf "$work" "$stage"
mkdir -p "$work" "$stage"
export TMPDIR=$PWD/.tmp
quint run message_queue.qnt --main skipIdleFenced --mbt --seed 20261118 \
  --max-samples "$samples" --n-traces "$samples" --max-steps "$steps" \
  --invariants noLostWakeup ringSafe \
  --out-itf "$work/t_{seq}.itf.json" >/dev/null
compact='
def d:
  if type == "object" then
    if has("#bigint") then .["#bigint"] | tonumber
    elif has("#map") then .["#map"] | sort_by(.[0] | d) | map(.[1] | d)
    elif has("tag") then (if .tag == "None" then null else .value | d end)
    else . end
  else . end;
[.states[] | . as $s
  | {a: .["mbt::actionTaken"], r: (.["mbt::nondetPicks"].r | d)}
  + ([ "written", "readFlag", "parked", "conflated", "inflight", "rPc",
       "sent", "wPend" ]
     | map({key: ., value: ($s["skipIdleFenced::messageQueue::" + .] | d)})
     | from_entries)]'
replayable='
. as $t
| [range(1; length) | . as $i | $t[$i]
   | select(.a == "readerDrain" or .a == "readerPoll")
   | $t[$i - 1].inflight[.r]]
| all(. == 0)'
# A second ping reaching a reader that has not consumed the first, later
# consumed: the case where a socket holds two messages.
conflates='
. as $t
| [range(1; length) | . as $i | $t[$i]
   | select(.a == "deliver" and $t[$i - 1].conflated[.r])]
| length > 0'
: >"$out"
kept=0
# Half the kept traces must exercise the two-message case.
want_conflating=$((keep / 2))
conflating=0
while IFS= read -r f; do
  trace=$(jq -e -c "$compact" "$f")
  ok=$(jq -e "$replayable" <<<"$trace") || [[ $ok == false ]] || {
    echo "cannot classify $f" >&2
    exit 1
  }
  [[ $ok == true ]] || continue
  c=$(jq "$conflates" <<<"$trace")
  if [[ $c == true ]] && ((conflating < want_conflating)); then
    conflating=$((conflating + 1))
  elif [[ $c == true ]] || ((kept - conflating >= keep - want_conflating)); then
    continue
  fi
  echo "$trace" >>"$out"
  kept=$((kept + 1))
  ((kept < keep)) || break
done < <(find "$work" -name 't_*.itf.json' | sort -V)
((kept == keep)) || {
  echo "only $kept of $keep replayable traces in $samples samples" >&2
  exit 1
}
sha256sum message_queue.qnt >"$stage/message_queue.qnt.sha256"
rm -rf traces "$work"
mv "$stage" traces
echo "kept $kept traces in traces/skipIdleFenced.jsonl ($(wc -c <traces/skipIdleFenced.jsonl) bytes)"
