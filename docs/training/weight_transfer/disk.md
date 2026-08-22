# Shared-disk weight updates

The `disk` backend reloads a safetensors checkpoint from a local directory that
is visible at the same path on every inference worker. It is useful when a
trainer writes checkpoints to a shared filesystem and the orchestration layer
wants to update a running vLLM engine without sending the tensors over NCCL or
CUDA IPC.

Configure the inference engine with the backend:

```bash
VLLM_SERVER_DEV_MODE=1 vllm serve my-model \
    --weight-transfer-config '{"backend": "disk"}'
```

Then drive one update through the weight-transfer control plane:

```python
from vllm.distributed.weight_transfer import HTTPVLLMWeightSyncClient

client = HTTPVLLMWeightSyncClient("http://localhost:8000")
client.init_weight_transfer_engine({})
client.start_weight_update()
client.update_weights({"path": "/shared/checkpoints/step-100"})
client.finish_weight_update("step-100")
```

The path must be an absolute local directory containing safetensors files, and
every worker must see the same completed checkpoint at that path before the
update starts. The backend accepts one complete checkpoint per
start/update/finish session. It loads only safetensors from that directory; it
does not reuse secondary weight sources retained from initial model loading. It
uses the standard loader pipeline, including model-specific weight mapping,
expert-parallel filtering, and missing-weight checks when the model reports
loaded-weight metadata.

This backend does not download Hugging Face Hub model IDs and does not use
`fastsafetensors` or GPU Direct Storage. A failed reload is surfaced to the
caller and the update session is cleaned up, but updates are not transactional:
weights loaded before an error may already have been applied.
