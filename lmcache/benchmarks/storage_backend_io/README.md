# Storage Backend I/O Benchmark

This microbenchmark compares multiple storage backends under high write/read concurrency:
- **LocalDiskBackend** - Local disk with optional O_DIRECT
- **RustRawBlockBackend** - Raw block device with optional O_DIRECT and io_uring
- **Hf3fsBackend** - HF3FS remote storage
- **FsBackend** - Filesystem connector backend
- The **io_uring** backend can optionally use **io_uring_cmd** (NVMe passthrough) for direct device access.

## What It Measures

- Total time to submit and complete `num_ops` write (put) or read (get) operations
- Effective ops/sec under concurrent submission
- Optional data integrity verification for read benchmarks

## Usage

```bash
# Local disk backend (write benchmark)
python benchmarks/storage_backend_io/storage_backend_io_benchmark.py \
  --num-ops 512 \
  --concurrency 32 \
  --backend local_disk \
  --local-disk-dir /tmp/lmcache_local_disk_bench \
  --max-local-disk-gb 2 \
  --local-disk-odirect \
  --output-json /tmp/storage_backend_io.json

# Rust raw block backend with io_uring (write benchmark)
python benchmarks/storage_backend_io/storage_backend_io_benchmark.py \
  --num-ops 512 \
  --concurrency 32 \
  --backend rust_raw_block \
  --raw-device /dev/nvme0n1 \
  --raw-odirect \
  --use-uring \
  --output-json /tmp/storage_backend_io.json

# Rust raw block backend (write + read benchmark with integrity check)
python benchmarks/storage_backend_io/storage_backend_io_benchmark.py \
  --num-ops 512 \
  --concurrency 32 \
  --backend rust_raw_block \
  --write_bench False \
  --raw-device /dev/nvme0n1 \
  --verify-integrity \
  --output-json /tmp/storage_backend_io.json

# Rust raw block backend with io_uring_cmd (write benchmark)
# Note: io_uring_cmd requires NVMe character device node (/dev/ngXnY)
python benchmarks/storage_backend_io/storage_backend_io_benchmark.py \
  --num-ops 1024 \
  --concurrency 4 \
  --backend rust_raw_block \
  --raw-device /dev/ng0n1 \
  --chunk-size 256 \
  --alignment 4096 \
  --use-uring \
  --use-uring-cmd \
  --output-json /tmp/rust_raw_block_uring_cmd.json

# HF3FS backend (write benchmark)
python benchmarks/storage_backend_io/storage_backend_io_benchmark.py \
  --num-ops 512 \
  --concurrency 32 \
  --backend hf3fs_backend \
  --remote-url "hf3fs:///3fs/stage/hello,/3fs/stage/world" \
  --output-json /tmp/storage_backend_io.json

# FS backend (write benchmark)
python benchmarks/storage_backend_io/storage_backend_io_benchmark.py \
  --num-ops 512 \
  --concurrency 32 \
  --backend fs_backend \
  --remote-url "/tmp/fs_backend_test" \
  --output-json /tmp/storage_backend_io.json
```

### Backend Options

| Backend | Description |
|---------|-------------|
| `local_disk` | Local disk storage with optional O_DIRECT support |
| `rust_raw_block` | Raw block device with O_DIRECT and optional io_uring |
| `hf3fs_backend` | HF3FS remote storage backend |
| `fs_backend` | Filesystem connector backend |
| `both` | Run both local_disk and rust_raw_block backends |

### Key Arguments

- `--write_bench`: Set to `True` (default) for write-only benchmark, `False` for write+read benchmark
- `--use-uring`: Enable io_uring for raw block backend (Linux 5.1+)
- `--chunk-size`: Chunk size for the backend (default: 256)
- `--verify_integrity`: Verify data integrity after reads (requires `--write_bench False`)
- `--remote-url`: Remote storage URL for hf3fs/fs backends

### Notes

- If `--raw-device` is not provided, the benchmark creates `raw_block.bin` in the same `--local-disk-dir` so both backends use the same filesystem.
- This is safe but **not** representative of true raw block performance.
- If `--raw-device` points to a real block device (`/dev/...`), the benchmark does not call `truncate()` on that path.
- `--raw-odirect` should only be used with a real block device that supports O_DIRECT.
- When `--local-disk-odirect` is enabled, the benchmark allocates **page-aligned** buffers to avoid EINVAL from O_DIRECT.
- Local disk backend uses its internal worker pool; completion is tracked via callbacks.
- Rust raw block benchmark uses a unique manifest path per run to avoid stale-index reuse between runs.
- For io_uring there is a limit on the number of fixed buffers that can be registered. For unprivileged users its 16384.
- Buffer registration and de-registration is time consuming.
- **io_uring_cmd** requires using the NVMe character device node (e.g., `/dev/ng0n1`) instead of the block device node (e.g., `/dev/nvme0n1`).
- **io_uring_cmd** requires io_uring as the underlying I/O engine.

## How To Compare On Real NVMe

Use the same physical device for both tests:
- local_disk on an ext4 mount
- rust_raw_block on the raw block device (unmounted)

Example parameters:
- `num_ops=65536`
- `concurrency=4`
- `--local-disk-odirect`
- `--raw-odirect`

### 1) Run local_disk on ext4

```bash
# WARNING: mkfs will erase the target device.
sudo mkfs.ext4 -F /dev/nvme1n1
sudo mkdir -p /mnt/local_disk_mount
sudo mount -t ext4 /dev/nvme1n1 /mnt/local_disk_mount
sudo chown "$USER":"$USER" /mnt/local_disk_mount

python benchmarks/storage_backend_io/storage_backend_io_benchmark.py \
  --num-ops 65536 \
  --concurrency 4 \
  --backend local_disk \
  --local-disk-dir /mnt/local_disk_mount/lmcache_local_disk_bench \
  --max-local-disk-gb 120 \
  --local-disk-odirect \
  --output-json /tmp/local_disk_ext4.json
```

### 2) Run rust_raw_block on raw device

```bash
sudo umount /mnt/local_disk_mount

python benchmarks/storage_backend_io/storage_backend_io_benchmark.py \
  --num-ops 65536 \
  --concurrency 4 \
  --backend rust_raw_block \
  --raw-device /dev/nvme1n1 \
  --raw-odirect \
  --output-json /tmp/rust_raw_block_raw.json
```

### 3) Compute comparison

```bash
python - <<'PY'
import json

with open("/tmp/local_disk_ext4.json") as f:
    local = json.load(f)[0]["ops_per_sec"]
with open("/tmp/rust_raw_block_raw.json") as f:
    rust = json.load(f)[0]["ops_per_sec"]

print(f"local_disk ops/sec: {local:.2f}")
print(f"rust_raw_block ops/sec: {rust:.2f}")
print(f"rust vs local: {(rust / local - 1.0) * 100.0:+.2f}%")
PY
```

## Output

The script prints a summary and optionally writes JSON results if `--output-json` is provided.

### Write Benchmark Output

```
local_disk: ops=512 concurrency=32 elapsed=1.234s ops/sec=415.23
```

### Write + Read Benchmark Output

```
read_rust_raw_block: ops=512 concurrency=32 write_elapsed=1.234s write_ops/sec=415.23 read_elapsed=0.567s read_ops/sec=902.84 total_elapsed=1.801s
  Integrity check: PASSED (errors=0)
```

### JSON Output Structure

```json
[
  {
    "backend": "local_disk",
    "num_ops": 512,
    "concurrency": 32,
    "write_elapsed_sec": 1.234,
    "write_ops_per_sec": 415.23,
    "use_odirect": true
  }
]
```

For write+read benchmarks (`--write_bench False`):

```json
[
  {
    "backend": "rust_raw_block",
    "num_ops": 512,
    "concurrency": 32,
    "write_elapsed_sec": 1.234,
    "write_ops_per_sec": 415.23,
    "read_elapsed_sec": 0.567,
    "read_ops_per_sec": 902.84,
    "total_elapsed_sec": 1.801,
    "use_odirect": true,
    "verify_integrity": true,
    "integrity_errors": 0,
    "integrity_passed": true,
    "use_uring": false,
    "use_uring_cmd": false
  }
]
```
