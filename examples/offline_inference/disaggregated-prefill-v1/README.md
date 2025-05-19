# Disaggregated Prefill V1

This example contains scripts that demonstrate disaggregated prefill in the offline setting of vLLM.

## Files

- `run.sh` - A helper script that will run `prefill_example.py` and `decode_example.py` sequentially.
- `prefill_example.py` - A script which performs prefill only, saving the KV state to the `local_storage` directory and the prompts to `output.txt`.
- `decode_example.py` - A script which performs decode only, loading the KV state from the `local_storage` directory and the prompts from `output.txt`.
