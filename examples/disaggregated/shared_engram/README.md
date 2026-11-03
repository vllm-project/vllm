# Shared Engram owner with vLLM P/D

This is the tested single-host four-GPU launch shape for
[shared host-backed Engram](../../../docs/features/engram_shared_host.md).
The owner holds one immutable DeepSeek-V4.1-Flash table generation; two TP2
workers import it. NIXL moves P/D KV separately. Use the same checkpoint
directory on all three processes.

```bash
export MODEL_DIR=/models/DeepSeek-V4.1-Flash
export TOKEN_FILE=/shared/private/engram.token
export OWNER_ENDPOINT=127.0.0.1:56000
export VLLM_PYTHON=/path/to/vllm/.venv/bin/python
umask 077
mkdir -p "$(dirname "$TOKEN_FILE")"
openssl rand -hex 32 > "$TOKEN_FILE"
```

In terminal 1, run `./launch.sh owner`. Wait for `ENGRAM_OWNER_READY` and copy
the `generation=` value. In terminals 2 and 3, export that value as
`GENERATION` along with the three variables above, then run
`./launch.sh prefill` and `./launch.sh decode`. Once both API servers are
ready, run:

```bash
"$VLLM_PYTHON" request.py
```

Stop each server by signalling its launcher PID, then stop the owner. The owner
waits for every lease to detach;
if a worker loses control without detaching, verify its GPU context is gone
before forcing owner termination. The example uses GPUs 0/1 for P and 2/3
for D. Override `P_GPUS`, `D_GPUS`, ports, and `VLLM_PYTHON` as needed. For
cross-host use, set a routable `OWNER_ENDPOINT` and `OWNER_BIND_HOST`, share
the token file securely, and keep all processes in one working IMEX/NVLink
domain. Cross-host serving has not been validated by this PR.
