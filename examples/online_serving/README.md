# MiniMax-M3.1-Flash on AMD

`minimax_m31_flash_rocm.sh` is the TP4 MI350/MI355 recipe:

- Official `MiniMaxAI/MiniMax-M3.1-Flash` checkpoint
- Routed experts requantized NVFP4→MXFP4 at load (`--quantization-config.moe mxfp4`)
- FP8 main KV and FP8 indexer (`--kv-cache-dtype fp8`, `indexer_kv_dtype=fp8`)
- Nested DSpark draft from `{model}/dspark`
- Block size 128
- Monokernel and dual-batch overlap left off

Published numbers must name that served precision. This is not native NVFP4.

`minimax_m31_flash_rocm_qos.sh` wraps the recipe for an `amd-agentx-4-qos` TP4
allocation. It runs:

1. model-load / generation smoke
2. long-context indexer/KV check
3. Prometheus memory snapshot
4. AgentX concurrencies `c1,c4,c8,c16,c32,c64` with fixed DSpark settings
5. optional accuracy compare when `REFERENCE_URL` points at a reference server

`minimax_m31_flash_rocm_validate.py` is the smoke / long-context / metrics helper.

```bash
# local node that already has 4x MI350/MI355
bash examples/online_serving/minimax_m31_flash_rocm_qos.sh local

# Slurm
ACCOUNT=amd-agentx QOS=amd-agentx-4-qos PARTITION=amd-spur \
  bash examples/online_serving/minimax_m31_flash_rocm_qos.sh submit
```
