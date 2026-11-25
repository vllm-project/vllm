#!/usr/bin/env bash
# Runs on the dev machine. Deletes every pod named $POD_NAME (and nothing
# else), then prints the remaining pods and the balance.
set -euo pipefail

POD_NAME=${POD_NAME:-inkling2-4090}

ids=$(runpodctl pod list -o json \
  | jq -r --arg n "$POD_NAME" '.[] | select(.name == $n) | .id')
if [[ -z "$ids" ]]; then
  echo "no pod named $POD_NAME"
fi
for id in $ids; do
  echo "deleting $POD_NAME ($id)"
  runpodctl pod delete "$id" >/dev/null
done

echo "remaining pods:"
runpodctl pod list -o json | jq -r '.[] | "  \(.id) \(.name) \(.desiredStatus)"'
runpodctl user -o json | jq -r '"balance: $\(.clientBalance | . * 100 | round / 100)  spend/hr: $\(.currentSpendPerHr)"'
