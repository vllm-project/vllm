# Rust gRPC shared-memory inputs

The native Rust gRPC `Inference.Generate` and `GenerateStream` APIs accept
preprocessed multimodal kwargs from local shared-memory files. Enable this with
`--grpc-shm-directory`; it is disabled by default. This avoids sending the kwargs
through protobuf, but **copies** them into server-owned memory before submitting
the request to EngineCore. It is not an end-to-end zero-copy transport.

```bash
mkdir -m 700 /dev/shm/vllm-inputs
VLLM_USE_RUST_FRONTEND=1 vllm serve YOUR_MULTIMODAL_MODEL \
  --host 127.0.0.1 --grpc-port 50051 \
  --grpc-shm-directory /dev/shm/vllm-inputs
```

The directory must already exist and must not be a symlink. A normal directory
also works for development. For containers, mount the same tmpfs into the client
and frontend; they must see the same files. Only the Rust frontend needs this
mount, not EngineCore or GPU workers. This is the native Rust API, distinct from
the Python `--grpc` API.

## Sending preprocessed features

Use bindings generated from this branch's `rust/proto/inference.proto`
(`vllm-proto` 0.8 for Rust). Supply `PreprocessedMediaFeatures.kwargs_shm` instead
of `kwargs`. Both contain the same version-compatible `MmKwargsItem` MessagePack
encoding, including inline tensor data. Cache-only inputs remain unsupported.
All existing feature/model/placeholder validation still applies; the request
must use `token_ids` and retain its feature identifiers and placeholder ranges.

For example, given a prepared `GenerateRequest` named `request` and its first
feature's encoded `kwargs`:

```python
import hashlib
import os
import uuid
from pathlib import Path

import inference_pb2 as pb

root = Path("/dev/shm/vllm-inputs")
name = uuid.uuid4().hex
feature = request.media[0].features
kwargs = feature.kwargs
with (root / name).open("xb") as output:
    output.write(kwargs)
feature.ClearField("kwargs")
feature.kwargs_shm.CopyFrom(pb.SharedMemoryInput(
    name=name, offset=0, length=len(kwargs),
    sha256=hashlib.sha256(kwargs).digest(),
))
try:
    response = client.Generate(request)
finally:
    os.unlink(root / name)
```

Each name must be a single component containing only ASCII letters, digits,
`_`, or `-`, at most 255 bytes. `offset` and `length` select a nonempty byte
range, and `sha256` is its SHA-256 digest as **32 raw bytes**, not a hex string.
The server rejects symlinks, non-regular files, missing/truncated ranges, and
digest mismatches. Inline kwargs retain their aggregate 16 MiB limit; shared
kwargs have a separate aggregate 256 MiB limit per request.

## Ownership and cleanup

The client creates and owns each file. Finish writing and close the writer
before submitting the RPC. Use a unique name per publication, keep the contents
immutable, and never truncate or overwrite an object still in use. The digest
detects stale names or changed contents; it is not an authentication token.

The frontend opens files relative to a pinned directory descriptor, copies the
selected ranges, verifies their digests, closes the files, and decodes the
snapshot before submitting any request to EngineCore. Successful responses
therefore have no dependency on the producer's files. No cancellation or GPU
completion acknowledgement is required to reclaim these copied inputs.

The server never unlinks client files. After a unary RPC finishes, or after a
stream ends or is cancelled, the client can **unlink** its unique publications.
An import already reading a file keeps its own descriptor until it finishes;
unlinking does not invalidate that descriptor. An import that has not opened
the file fails if it is gone. Cancellation does not guarantee background file
copying has stopped, so do not truncate or recycle an inode in place. Keep files
available separately if retrying. A crash may leave files for the producer to
clean up.

This first version does not accept Python processor-cache handles or tensor
arena slots and does not map producer memory into engine requests. Those
internal mechanisms have different ownership and authentication contracts.
Future zero-copy support would need a separate reader-completion contract.

## Deployment boundary

Only enable this on a trusted local inference endpoint. Anyone with access to
the gRPC API can name inputs in the configured directory; this does not provide
per-client or per-tenant authorization. Restrict endpoint and directory access,
use a dedicated input directory, and do not place unrelated files there. Reads
and hashing run off the async executor, but large or concurrent requests still
consume CPU and memory. No throughput improvement is claimed here.

Older servers ignore `kwargs_shm` and reject the resulting missing kwargs.
Do not also send inline kwargs as a fallback: specifying both is rejected.
