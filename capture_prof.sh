#!/bin/bash
# Runs ON node 10 (where serve binds localhost:8005). Captures a torch trace
# over a short steady-state decode.
set -u
cd /home/woosuk/workspace/vllm
P=8005
curl -s -X POST http://localhost:$P/start_profile -o /dev/null -w "start=%{http_code}\n"
# steady-state decode (single request, long enough to hit the profiler schedule)
curl -s http://localhost:$P/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"GLM-5.2","prompt":"Write a detailed essay about the history of computing.","max_tokens":80,"temperature":0,"ignore_eos":true}' \
  -o /dev/null -w "gen=%{http_code}\n"
curl -s -X POST http://localhost:$P/stop_profile -o /dev/null -w "stop=%{http_code}\n"
echo "waiting for trace flush..."
sleep 8
ls -lt prof/*.json* 2>/dev/null | head
