Loading model weights with fastsafetensors
===================================================================

Using fastsafetensors library enables loading model weights to GPU memory by leveraging GPU direct storage. See [their GitHub repository](https://github.com/foundation-model-stack/fastsafetensors) for more details.

To enable this feature, use the `--load-format fastsafetensors` command-line argument

To use buffered transfers without GPU Direct Storage, pass
`--model-loader-extra-config '{"fastsafetensors_nogds": true}'`. The option accepts
only a boolean and only with `--load-format fastsafetensors`. When omitted, the
existing default is unchanged: buffered transfers for process groups larger
than one, and GDS otherwise.

An explicit `false` requests the existing GDS path for a single-rank process
group. It is rejected for a multi-rank process group, before selecting a CUDA
device or starting the loader: GDS can initialize CUDA contexts on every visible
GPU and cause memory failures. Group size is taken from the initialized global
process group, not just the tensor-parallel setting. This preserves the existing
multi-rank safety policy instead of silently ignoring `false`.

The existing fallback to buffered transfers on an unsupported-GDS error before
any weight is yielded still applies. `VLLM_FASTSAFETENSORS_QUEUE_SIZE` continues
to control the existing loader's pipeline depth. This option only selects a mode
of the existing fastsafetensors library; it does not add a loader, change
steady-state inference or guarantee a startup/CPU speedup. No saving is implied
for multi-rank deployments that already use buffered transfers by default.
