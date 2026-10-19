#!/bin/bash
# TEMPORARY: repeat v1/shutdown to reproduce the XPU TP=2 startup hang.
cd tests || exit 1
rc=0
for i in 1 2 3 4 5 6 7 8; do
  echo "=== shutdown iter ${i} psm=$(ls /dev/shm/psm_* 2>/dev/null | wc -l) tmp_dist=$(ls /tmp/vllm_dist_* 2>/dev/null | wc -l) ==="
  pytest -v -s v1/shutdown || rc=1
done
exit ${rc}
