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

Optional `/workspace/.hf_env`:

```bash
export HF_TOKEN='your_token_here'
```
