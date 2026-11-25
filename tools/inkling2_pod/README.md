# Inkling 2 Mini RTX 4090 pod

Throwaway tooling for the "Inkling 2 Mini on RTX 4090 (SM89)" wayfinder map
(suyashpatel98/vllm#1). This branch is never merged.

```bash
./pod_up.sh     # create a 4090 pod, install vLLM (precompiled), download the model
# ... ssh in with the command it prints ...
./pod_down.sh   # delete the pod (never just stop it)
```

On the pod:

```bash
cd /root/work/vllm && source .venv/bin/activate && export HF_HOME=/root/hf
vllm serve thinkingmachines/Inkling-2-Mini ...
```

- Needs `runpodctl` (configured in `~/.runpod/config.toml`), `gh`, `jq`.
- Tries community cloud ($0.34/hr) then secure ($0.89/hr). Override with
  `CLOUDS=COMMUNITY`.
- `pod_up.sh` installs the branch named by `BRANCH` (default
  `inkling2-mini-sm89`), as pushed to the fork. Push before running.
- If setup fails after the pod exists, fix and rerun with `POD_ID=<id>`.
- Only `cu132` precompiled wheels exist for the base commit, so the host needs
  a CUDA 13 driver (`MIN_CUDA`, default 13.0).
