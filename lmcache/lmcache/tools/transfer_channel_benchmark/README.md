# Transfer Channel Throughput Benchmark

Measures read throughput (GB/s) of the LMCache **transfer channel**
(`lmcache/v1/distributed/transfer_channel/`) for batched peer-to-peer reads.

Unlike a raw-tensor microbenchmark, this tool uses LMCache's `L1MemoryManager`
to initialize the registered memory region and to **allocate the transferred
objects** on both sides, so it exercises the same memory path production uses.

## How it works

The benchmark runs as **two separate processes**:

- **server** — uses an `L1MemoryManager` to allocate a registered L1 buffer and
  a pool of source memory objects, registers the buffer with the transfer
  channel, and publishes the source object catalog (`offset`/`size` per object)
  over a small ZMQ side-channel.
- **client** — fetches the catalog, allocates its own destination objects via an
  `L1MemoryManager`, connects the transfer channel, and repeatedly reads a random
  `--num-objects` subset of the source objects, then reports throughput.

A side-channel is needed because the transfer channel handshake only exchanges
the whole-buffer registration, not per-object offsets.

> Only the `nixl` transfer channel type is registered today. The tool is generic
> over `--transfer-channel-type`; an unknown type raises a clear error.

## Usage

Run via the module or the `lmcache` CLI. Start the server first.

### `python -m`

```bash
# terminal 1 — server
python -m lmcache.tools.transfer_channel_benchmark \
  --role server --transfer-channel-type nixl \
  --url 0.0.0.0:7600 --control-url 0.0.0.0:7610 \
  --buffer-size 2GB --page-size 512KB --object-size 10MB

# terminal 2 — client
python -m lmcache.tools.transfer_channel_benchmark \
  --role client --transfer-channel-type nixl \
  --url 127.0.0.1:7600 --control-url 127.0.0.1:7610 \
  --listen-url 0.0.0.0:7601 \
  --page-size 512KB --object-size 10MB \
  --num-objects 10 --iters 3 --warmup 1
```

### `lmcache tool`

```bash
lmcache tool transfer-channel-benchmark --role server  --url 0.0.0.0:7600 \
  --control-url 0.0.0.0:7610 --buffer-size 2GB --object-size 10MB
lmcache tool transfer-channel-benchmark --role client --url 127.0.0.1:7600 \
  --control-url 127.0.0.1:7610 --object-size 10MB --num-objects 10
```

The server always writes a deterministic per-object byte pattern (object index
mod 256) into its source objects, so adding `--verify` on the **client** checks
the transferred bytes against that pattern (no `--verify` needed on the server).

## Key options

| Option | Role | Meaning |
| --- | --- | --- |
| `--role {server,client}` | both | Which side to run (required). |
| `--transfer-channel-type` | both | Implementation to benchmark (default `nixl`). |
| `--nixl-backend` | both | nixl backend, e.g. `UCX` (nixl-specific). |
| `--url` | both | Server binds its transfer-channel server here; client dials it. |
| `--listen-url` | client | Client's own (mandatory) transfer-channel server bind. |
| `--control-url` | both | Catalog side-channel: server binds, client connects. |
| `--buffer-size` | server | Registered L1 source buffer size (e.g. `8GB`). |
| `--page-size` | both | Page / alignment size; **must match** on both sides. |
| `--object-size` | both | Per-object size; multiple of `--page-size`. |
| `--num-objects` | client | Objects transferred per read. |
| `--num-source-objects` | server | Source pool size (default `5 * --num-objects`). |
| `--iters` / `--warmup` | client | Measured / warmup read iterations. |
| `--seed` | client | RNG seed for the read subset. |
| `--verify` | client | Verify transferred bytes against the server's known pattern. |
| `--use-lazy` | both | Use the lazy L1 allocator (experimental for registration). |
| `--server-timeout` | server | Seconds to serve catalog requests before exiting. |

## Notes

- `--page-size` must be identical on server and client (enforced via the catalog
  handshake) so remote page-index math lines up.
- With the default (non-lazy) allocator the whole `--buffer-size` is allocated up
  front and may be CUDA-pinned; keep it within available host memory.
- Requires a working transfer channel runtime (for `nixl`, a UCX backend).

## Performance: NUMA placement

On a multi-NUMA host with NICs spread across nodes (e.g. an 8-NIC, 2-socket
box), run **both** the server and client under `numactl --interleave=all`:

```bash
numactl --interleave=all \
python -m lmcache.tools.transfer_channel_benchmark --role server ...
```

Without it the registered buffer is allocated on a single NUMA node, so only the
rails local to that node reach full bandwidth and the rest are throttled by the
cross-socket link — roughly halving throughput. In testing on a 2-NUMA / 8-rail
host this was the difference between **~110 GB/s** (no interleave) and
**~210 GB/s** (`--interleave=all`), with everything else identical. Page size
only matters in the small regime (descriptor-bound below ~64KB); from ~128KB up
the transfer is bandwidth-bound and flat.

## Troubleshooting

- **Client hangs after "connected" or during connect, then raises
  `TimeoutError` after ~60s.** The transfer-channel handshake could not reach the
  server. Check that `--url`/`--control-url` on the client point at the *server's*
  reachable address (a common mistake is a typo such as a trailing dot:
  `10.0.0.5.:7600`), that the server is running, and that the ports are open
  between hosts:
  ```bash
  nc -vz <server-host> 7600   # transfer channel
  nc -vz <server-host> 7610   # catalog side-channel
  ```
  The handshake has a fixed 60s timeout, so a misconfiguration fails with a clear
  error instead of hanging forever.
- **`page_size`/`object_size` mismatch error.** The client validates these against
  the server's catalog; pass the same `--page-size` and `--object-size` on both.
- **Allocation `RuntimeError` on the server.** The source pool
  (`--num-source-objects` × `--object-size`) must fit in `--buffer-size`; increase
  the buffer or reduce the pool/object size.
