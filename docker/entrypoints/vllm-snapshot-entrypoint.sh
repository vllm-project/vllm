#!/bin/bash
# Default-on imports-snapshot for the official image (opt-out: VLLM_SNAPSHOT=0).
# Caps-only gate: no vllm import, no manifest logic in shell. Without
# CAP_CHECKPOINT_RESTORE + CAP_SYS_PTRACE (docker run --cap-add CHECKPOINT_RESTORE
# --cap-add SYS_PTRACE), criu can neither dump nor restore, so skip straight to
# a normal cold start with zero snapshot overhead (env stays unset).
# Best-effort priming: EVERY create outcome (created / already-exists / failed)
# falls through to exec'ing the server; create's exit code is never propagated.

serve() { exec vllm serve "$@"; }

if [ "${VLLM_SNAPSHOT:-}" = "0" ]; then
    serve "$@"
fi

caps=$(awk '/^CapEff:/ {print $2}' /proc/self/status 2>/dev/null)
# CAP_CHECKPOINT_RESTORE = bit 40, CAP_SYS_PTRACE = bit 19
if [ -z "$caps" ] || [ "$((16#$caps >> 40 & 1))" -ne 1 ] || [ "$((16#$caps >> 19 & 1))" -ne 1 ]; then
    serve "$@"
fi

export VLLM_SNAPSHOT=1
vllm snapshot create
status=$?
case "$status" in
    0) echo "vllm snapshot: primed; serve will restore" >&2 ;;
    3) echo "vllm snapshot: snapshot present; serve will restore" >&2 ;;
    *) echo "vllm snapshot: create failed (status $status); cold start" >&2 ;;
esac
serve "$@"
