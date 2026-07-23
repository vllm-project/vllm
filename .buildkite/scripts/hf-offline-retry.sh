#!/usr/bin/env bash

set -uo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 COMMAND_FILE" >&2
  exit 64
fi

command_file=$1
if [[ ! -f "${command_file}" || ! -r "${command_file}" ]]; then
  echo "Command file is not readable: ${command_file}" >&2
  exit 66
fi

run_attempt() (
  if [[ "$1" == "offline" ]]; then
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    export HF_DATASETS_OFFLINE=1
  else
    unset HF_HUB_OFFLINE
    unset TRANSFORMERS_OFFLINE
    unset HF_DATASETS_OFFLINE
  fi

  bash -e -o pipefail -- "${command_file}"
)

echo "--- :package: Hugging Face offline attempt"
run_attempt offline
offline_status=$?

if [[ ${offline_status} -eq 0 ]]; then
  exit 0
fi

case "${offline_status}" in
  1 | 2 | 123)
    echo "--- :globe_with_meridians: Hugging Face offline attempt failed; retrying online"
    run_attempt online
    online_status=$?
    exit "${online_status}"
    ;;
  *)
    exit "${offline_status}"
    ;;
esac
