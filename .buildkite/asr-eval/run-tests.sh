#!/bin/bash
set -x

# Start server
python3 -m vllm.entrypoints.openai.api_server --model openai/whisper-large-v3 $@ &
server_pid=$!

# Wait for server to start, timeout after 600 seconds
timeout 180 bash -c 'until curl localhost:8000/v1/models; do sleep 4; done' || exit 1
 
# NOTE: Expected WER measured with hf.transformers equivalent model on same dataset.
# Original dataset split is about 23GB in size, hence we use a pre-filtered slice.
python test_transcription_api_correctness.py -m openai/whisper-large-v3 -dr D4nt3/esb-datasets-earnings22-validation-tiny-filtered --expected-wer 12.744980

# Wait for graceful exit
kill $server_pid
