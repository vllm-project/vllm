#!/bin/bash
# TEMPORARY: repeat v1/shutdown to reproduce the XPU TP=2 startup hang.
cd tests || exit 1
rc=0
for i in 1 2; do
  echo "=== shutdown iter ${i} psm=$(find /dev/shm -maxdepth 1 -name 'psm_*' | wc -l) tmp_dist=$(find /tmp -maxdepth 1 -name 'vllm_dist_*' | wc -l) ==="
  pytest -v -s v1/shutdown || rc=1
done
exit ${rc}
