# DeepSeek V4.1 shared-pool offload prototype

Experimental H100 FP8 path. Main-KV source pools are selected with
`kv_connector_extra_config.dsv41_sources`; `[2, 20]` is the measured policy.
Indexer histories and sliding-window state remain on GPU.

The [RFC](proposal/RFC.md) contains the workload and measured comparison.
`agentic.py` runs cached continuations; `offloading_control.py` runs the ordinary
CPU control. `graph_validate.py` independently checks KV and attention outputs.

The harness retains the measured deployment's model path and uses a source
manifest and isolated ops library. Adjust those paths for another deployment.
This WIP preserves the tested source; rebasing onto current main and repeating
validation, task-quality evaluation, and connector composition remain pending.

After adapting the model path and building the isolated ops library, the measured
selective run uses:

```bash
.venv/bin/python benchmarks/kernels/dsv41_hisparse_decode/agentic.py \
  --mode hisparse --sources 2,20 --host-gib 32 \
  --jobs benchmarks/kernels/dsv41_hisparse_decode/agentic-jobs.json \
  --source-manifest benchmarks/kernels/dsv41_hisparse_decode/source-manifest.json \
  --output /tmp/dsv41-selective
```

Set `VLLM_DSV41_HISPARSE_OPS_LIB` to the library built by `build_ops.py`.
Add `--validate` only for independent shadow checks, not performance timing.
