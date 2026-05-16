# OpenVLA Manual Verification

```bash
bash manual_verification/run_openvla_check.sh
```

On a fresh GPU machine this creates:

- `.venv`: vLLM editable install
- `/workspace/openvla_hf_ref_venv`: Hugging Face reference env
- `/workspace/openvla_check/result.json`: final result
- `/workspace/logs/openvla_check_*.log`: run log

The vLLM install step can take 10+ minutes.

To compare `vllm serve` responses with the saved HF reference artifacts, run:

```bash
bash manual_verification/run_openvla_server_check.sh
```

The server check starts `vllm serve`, sends the sampled cases to
`/v1/chat/completions` with `return_token_ids=true`, compares returned token IDs
with the saved HF reference outputs, and writes
`/workspace/openvla_check/server_result.json`.

Optional `/workspace/.hf_env`:

```bash
export HF_TOKEN='your_token_here'
```
