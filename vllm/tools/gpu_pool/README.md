# GPU Pool - Shared VRAM for vLLM

This directory contains the implementation of a shared VRAM pool for vLLM using CUDA Virtual Memory Management (VMM).

## Components

### C++ Components

1. **uds_rpc.h/cc**: Unix domain socket RPC utilities
   - Simple client/server for inter-process communication
   - Used for broker-client protocol

2. **gpu_poold.cc**: GPU memory pool broker daemon
   - Manages shared VRAM pool using CUDA VMM
   - Exports memory handle to clients
   - Handles allocation/free requests via UDS
   - Maintains free-list with coalescing

3. **vmm_pool_client.h/cc**: Client library for pool access
   - Imports shared memory handle
   - Provides allocation/free interface
   - Manages virtual address mapping

4. **vmm_pool_pybind.cc**: Python bindings
   - pybind11 wrapper for VmmPoolClient
   - PyTorch tensor integration
   - Exposes Python API for vLLM

### Build System

- **CMakeLists.txt**: CMake configuration
  - Builds gpu_poold executable
  - Builds vmm_pool_py Python extension
  - Links CUDA Driver API and PyTorch

## Building

```bash
# From this directory
mkdir -p build && cd build
cmake .. && make -j

# Output:
# - gpu_poold: Standalone broker
# - vmm_pool_py.so: Python extension (in build/)
```

### Requirements

- CMake 3.20+
- CUDA Toolkit (Driver API)
- PyTorch with CUDA support
- pybind11

## Protocol

The broker and clients communicate via Unix domain sockets using a simple binary protocol:

### Commands

1. **HELLO (cmd=1)**: Initial handshake
   - Client → Broker: Request connection
   - Broker → Client: Pool metadata (granularity, size) + FD via SCM_RIGHTS

2. **ALLOC (cmd=2)**: Allocate memory
   - Client → Broker: nbytes (uint64_t)
   - Broker → Client: status + (offset, length)

3. **FREE (cmd=3)**: Free memory
   - Client → Broker: (offset, length)
   - Broker → Client: status

4. **STATS (cmd=4)**: Get pool statistics
   - Client → Broker: Request
   - Broker → Client: (used, total, granularity)

### File Descriptor Passing

The broker passes the CUDA memory handle to clients using `SCM_RIGHTS` ancillary data over the Unix domain socket. This allows clients to import the memory into their own CUDA context.

## Usage

### Start Broker

```bash
./build/gpu_poold \
  --device 0 \
  --pool-bytes $((24*1024*1024*1024)) \
  --endpoint /tmp/gpu_pool.sock
```

### Use from Python

```python
from vmm_pool_py import VmmPoolClient, tensor_from_device_ptr

# Connect to broker
client = VmmPoolClient("/tmp/gpu_pool.sock", device_id=0)

# Allocate memory
offset, length = client.allocate(1024 * 1024)  # 1MB
ptr = client.map(offset, length)

# Create PyTorch tensor
tensor = tensor_from_device_ptr(ptr, 1024 * 1024, device_id=0)

# Use tensor...

# Cleanup
client.unmap(offset, length)
client.free(offset, length)
```

## Architecture

### Broker (gpu_poold)

```
┌─────────────────────────────────┐
│     GPU Pool Broker             │
│                                 │
│  ┌───────────────────────────┐ │
│  │ CUDA VMM Physical Memory  │ │
│  │   (cuMemCreate)           │ │
│  └───────────────────────────┘ │
│              ↓                  │
│  ┌───────────────────────────┐ │
│  │ Shareable Handle (FD)     │ │
│  │   (cuMemExportToHandle)   │ │
│  └───────────────────────────┘ │
│              ↓                  │
│  ┌───────────────────────────┐ │
│  │ Free-list Management      │ │
│  │   - Allocation            │ │
│  │   - Deallocation          │ │
│  │   - Coalescing            │ │
│  └───────────────────────────┘ │
│              ↓                  │
│     Unix Domain Socket          │
└─────────────────────────────────┘
          │         │         │
     ┌────┘    ┌────┘    └────┐
     ↓         ↓              ↓
Client 1   Client 2   ... Client N
(vLLM)     (vLLM)        (vLLM)
```

### Client Flow

```
1. Connect to broker via UDS
2. Send HELLO, receive:
   - Pool metadata (granularity, size)
   - File descriptor via SCM_RIGHTS
3. Import handle: cuMemImportFromShareableHandle()
4. Reserve VA space: cuMemAddressReserve()
5. For each allocation:
   a. Send ALLOC request
   b. Receive (offset, length)
   c. Map to VA: cuMemMap(VA + offset)
   d. Set access: cuMemSetAccess()
6. Use memory (e.g., wrap as PyTorch tensor)
7. For deallocation:
   a. Unmap: cuMemUnmap()
   b. Send FREE request
```

## Memory Management

### Granularity

CUDA VMM requires allocations aligned to a granularity (typically 2MB). The broker:
- Queries granularity at startup
- Rounds all allocations up to granularity
- Aligns the pool size to granularity

### Free-list

The broker maintains a free-list of available slices:
- **Allocation**: First-fit algorithm
- **Deallocation**: Add to free-list + coalesce adjacent slices
- **Thread-safe**: Protected by mutex

### Coalescing

Adjacent free slices are automatically merged to reduce fragmentation:
```
Before: [Free: 0-2MB] [Free: 2-4MB] [Used: 4-6MB]
After:  [Free: 0-4MB] [Used: 4-6MB]
```

## Testing

### Unit Tests

```bash
# Run Python unit tests
pytest tests/v1/test_kv_budget_unit.py
pytest tests/v1/test_external_allocator.py
```

### Integration Test

```bash
# Start broker
./build/gpu_poold --endpoint /tmp/test.sock &

# Run integration test (requires broker running)
python tests/integration/test_vmm_pool.py

# Cleanup
pkill gpu_poold
```

## Performance Considerations

### Allocation Overhead

- **RPC cost**: ~10-100μs per UDS round-trip
- **Mapping cost**: ~1-10μs per cuMemMap call
- **Amortization**: Allocate large slabs, subdivide in client

### Thread Safety

- Broker: Thread-per-connection model
- Client: CUDA primary context is thread-safe
- PyTorch: Tensor wrapper is lightweight

### Scalability

- **Clients**: Up to ~100 concurrent clients (UDS limit)
- **Pool size**: Limited by GPU memory (e.g., 24-80GB)
- **Fragmentation**: Coalescing helps but can still occur

## Debugging

### Enable Logging

```bash
# Broker: stdout/stderr logs
./build/gpu_poold --endpoint /tmp/pool.sock 2>&1 | tee broker.log
```

### Check Pool Stats

```python
client = VmmPoolClient("/tmp/gpu_pool.sock", 0)
used, total, gran = client.stats()
print(f"Pool: {used}/{total} bytes used, {gran} byte granularity")
```

### Common Issues

1. **Connection refused**: Broker not running or wrong socket path
2. **Out of memory**: Pool exhausted, increase `--pool-bytes`
3. **Import failed**: CUDA Driver API version mismatch
4. **Permissions**: Socket file permissions (chmod 666 /tmp/gpu_pool.sock)

## Future Enhancements

- [ ] Multi-GPU support with CUDA peer mappings
- [ ] Asynchronous RPC for lower latency
- [ ] Best-fit or buddy allocator (instead of first-fit)
- [ ] Client-side caching of small allocations
- [ ] Telemetry (Prometheus metrics)
- [ ] Graceful shutdown and reconnection

## References

- [CUDA Driver API](https://docs.nvidia.com/cuda/cuda-driver-api/)
- [CUDA VMM Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#virtual-memory-management)
- [Unix Domain Sockets](https://man7.org/linux/man-pages/man7/unix.7.html)
- [pybind11 Documentation](https://pybind11.readthedocs.io/)
