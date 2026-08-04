# LMCache Controller ZMQ Benchmark Tool

This tool performs load testing on LMCache Controller using ZMQ interface to measure message throughput, latency, and system performance.

## Overview

The benchmark tool simulates multiple instances and workers sending various types of messages to the LMCache Controller:

- **BatchedKVOperationMsg**: admit/evict messages via pull socket
- **BatchedP2PLookupMsg**: p2p_lookup messages via reply socket
- **RegisterMsg/DeRegisterMsg/HeartbeatMsg**: worker lifecycle messages

### Key Components

- **constants.py**: Defines ZMQ socket timeouts and other constants
- **config.py**: `ZMQBenchmarkConfig` dataclass for benchmark configuration
- **handlers/**: Operation handlers using Strategy Pattern with dynamic discovery
  - Each operation has its own file (e.g., `admit.py`, `evict.py`)
  - Automatically discovers and registers all handlers at import time
  - Add new operations by creating a new handler file - no need to modify existing code
- **benchmark.py**: `ZMQControllerBenchmark` class with core logic
- **__main__.py**: Argument parsing and main entry point

## Prerequisites

- A running LMCache Controller instance
- Python 3.10+
- Required dependencies: `zmq`, `msgspec`, `psutil`

## Quick Start

### Basic Usage
- Start the controller
```bash
python3 -m lmcache.v1.api_server --host 0.0.0.0 --port 9009 \
  --monitor-ports "{\"pull\":7555,\"reply\":7556}" \
  --lmcache-worker-timeout 100 --health-check-interval 10 
```

- Start the benchmark
```bash
python3 -m lmcache.tools.controller_benchmark \
  --monitor-ports  "{\"pull\":7555,\"reply\":7556}" \
  --num-instances 50 --num-workers 1 --num-keys 1000000 --batch-size 100 \
  --operations "admit:35,evict:29,heartbeat:1,p2p_lookup:35"
```


## Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--controller-host` | `localhost` | Controller host address |
| `--monitor-ports` | `{"pull":8100,"reply":8101}` | Monitor ports in JSON format |
| `--duration` | `60` | Benchmark duration in seconds |
| `--batch-size` | `50` | Number of KV operations per batch message |
| `--operations` | `admit:70,evict:25,heartbeat:5` | Operation distribution (name:percentage) |
| `--num-instances` | `10` | Number of instances to simulate |
| `--num-workers` | `1` | Number of workers per instance |
| `--num-locations` | `1` | Number of storage locations |
| `--num-keys` | `10000` | Number of unique keys |
| `--num-hashes` | `100` | Number of hashes for P2P lookup operations |
| `--no-register-first` | `false` | Skip pre-registering workers before benchmark |

## Operation Types

The benchmark supports the following operation types:

| Operation | Description                               |
|-----------|-------------------------------------------|
| `admit` | Simulates KV cache admission (adds entries) |
| `evict` | Simulates KV cache eviction (removes entries) |
| `p2p_lookup` | Simulates p2p batch lookup messages       |
| `register` | Simulates worker registration             |
| `deregister` | Simulates worker deregistration           |

### Adding New Operations

To add a new operation, simply create a new handler file in `handlers/` directory:

1. Create `handlers/your_operation.py` implementing `OperationHandler` base class
2. Define `operation_name` property and implement required methods
3. The handler will be automatically discovered and registered

No need to modify existing code - the system uses dynamic discovery!

## Output Metrics

The benchmark reports:

- **Overall QPS**: Total messages per second
- **Per-operation QPS**: Messages per second for each operation type
- **Latency statistics**: avg, min, max, p95 (in milliseconds)
- **Error counts**: Number of failed operations
- **Memory usage**: System memory usage during the test

### Sample Output

The following is a sample output from the benchmark ran in my macbook m4 pro.

```
================================================================================
LMCache Controller ZMQ Benchmark Results
================================================================================

Configuration:
  Controller URL: 127.0.0.1:7555
  Duration: 60 seconds
  Batch Size: 100
  Operations: {'admit': 35.0, 'evict': 29.0, 'heartbeat': 1.0, 'p2p_lookup': 35.0}
  Instances: 50, Workers: 1, Locations: 1, Keys: 1000000

Overall Performance:
  Total Requests: 270035
  Total Messages: 26736200
  Total Time: 60.00s
  Overall RPS (Requests/sec): 4500.58
  Overall QPS (Messages/sec): 445602.80

Per-Operation Performance:
  admit:
    RPS (Requests/sec): 1575.23
    QPS (Messages/sec): 157523.14
    Latency - Avg: 0.016ms, Min: 0.007ms, Max: 0.249ms, P95: 0.031ms
    Errors: 0
  evict:
    RPS (Requests/sec): 1305.13
    QPS (Messages/sec): 130513.18
    Latency - Avg: 0.016ms, Min: 0.007ms, Max: 1.201ms, P95: 0.031ms
    Errors: 0
  heartbeat:
    RPS (Requests/sec): 45.00
    QPS (Messages/sec): 45.00
    Latency - Avg: 0.010ms, Min: 0.003ms, Max: 0.138ms, P95: 0.024ms
    Errors: 0
  p2p_lookup:
    RPS (Requests/sec): 1575.21
    QPS (Messages/sec): 157521.48
    Latency - Avg: 0.440ms, Min: 0.150ms, Max: 6.291ms, P95: 0.843ms
    Errors: 0

System Metrics:
  Memory Usage - Avg: 62.3%, Max: 63.5%
================================================================================
```

## Troubleshooting

### Send Timeout Error

If you see "Send timeout - Controller may not be running", ensure:
1. The LMCache Controller is running
2. The `--controller-host` and `--monitor-ports` are correct
3. No firewall is blocking the connection

### High Error Rate

If you observe high error rates:
1. Reduce `--batch-size` to decrease message size
2. Increase controller resources
3. Check network connectivity
