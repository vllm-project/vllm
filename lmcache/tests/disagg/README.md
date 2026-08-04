# Test disaggregated prefill related components

## NIXL Pipe

```bash
# Terminal 1 (sender)
UCX_TLS=cuda_ipc,cuda_copy,tcp CUDA_VISIBLE_DEVICES=0 python3 test_nixl_pipe.py --role sender
 
# Terminal 2 (receiver)
UCX_TLS=cuda_ipc,cuda_copy,tcp CUDA_VISIBLE_DEVICES=1 python3 test_nixl_pipe.py --role receiver
```

## NIXL Channel

```bash
# Terminal 1 (Sender)
UCX_TLS=cuda_ipc,cuda_copy,tcp CUDA_VISIBLE_DEVICES=0 python3 test_nixl_channel.py --role sender --num-objs 500

# Terminal 2 (Receiver)
UCX_TLS=cuda_ipc,cuda_copy,tcp CUDA_VISIBLE_DEVICES=1 python3 test_nixl_channel.py --role receiver --num-objs 500
```

NOTE: why 500 objects? -- Because we the pipe only has 4GB buffer, but 500 objects are 16GB in total. This is to test when the data to transfer is larger than the buffer size, how the sender/receiver behaves.

## NIXL Backend

```bash
# Terminal 1 (Sender)
UCX_TLS=cuda_ipc,cuda_copy,tcp CUDA_VISIBLE_DEVICES=0 python3 test_nixl_storage_backend.py --role sender --num-objs 500

# Terminal 2 (Receiver)
UCX_TLS=cuda_ipc,cuda_copy,tcp CUDA_VISIBLE_DEVICES=1 python3 test_nixl_storage_backend.py --role receiver --num-objs 500
```

Sender side logs:
```plaintext
[2025-04-07 13:00:26,142] LMCache INFO: Generated 500 objects with total size 16000.00 MB (test_nixl_storage_backend.py:70:__main__)
Loaded plugin UCX
Loaded plugin UCX_MO
Initialized NIXL agent: NixlRole.SENDER
[2025-04-07 13:00:26,339] LMCache INFO: Received remote transfer descriptors (nixl_connector.py:103:lmcache.v1.storage_backend.connector.nixl_connector)
[2025-04-07 13:00:28,340] LMCache INFO: Sending 500 objects... (test_nixl_storage_backend.py:89:__main__)
[2025-04-07 13:00:28,343] LMCache DEBUG: Committing write with 128 transfers (nixl_connector.py:188:lmcache.v1.storage_backend.connector.nixl_connector)
[2025-04-07 13:00:28,377] LMCache DEBUG: Transfer completed in 33.7618 ms, creating the transfer: 0.0229 ms, transfer time: 18.9432 ms, wait for receiver: 14.7957 ms
Pure transfer throughput: 211.1577 GB/s (nixl_connector.py:229:lmcache.v1.storage_backend.connector.nixl_connector)
[2025-04-07 13:00:28,378] LMCache DEBUG: Committing write with 128 transfers (nixl_connector.py:188:lmcache.v1.storage_backend.connector.nixl_connector)
[2025-04-07 13:00:28,404] LMCache DEBUG: Transfer completed in 25.6334 ms, creating the transfer: 0.0167 ms, transfer time: 11.9316 ms, wait for receiver: 13.6851 ms
Pure transfer throughput: 335.2434 GB/s (nixl_connector.py:229:lmcache.v1.storage_backend.connector.nixl_connector)
[2025-04-07 13:00:28,406] LMCache DEBUG: Committing write with 128 transfers (nixl_connector.py:188:lmcache.v1.storage_backend.connector.nixl_connector)
[2025-04-07 13:00:28,432] LMCache DEBUG: Transfer completed in 26.4169 ms, creating the transfer: 0.0132 ms, transfer time: 11.6180 ms, wait for receiver: 14.7857 ms
Pure transfer throughput: 344.2922 GB/s (nixl_connector.py:229:lmcache.v1.storage_backend.connector.nixl_connector)
[2025-04-07 13:00:28,434] LMCache DEBUG: Committing write with 116 transfers (nixl_connector.py:188:lmcache.v1.storage_backend.connector.nixl_connector)
[2025-04-07 13:00:28,458] LMCache DEBUG: Transfer completed in 23.8715 ms, creating the transfer: 0.0131 ms, transfer time: 11.1843 ms, wait for receiver: 12.6741 ms
Pure transfer throughput: 324.1156 GB/s (nixl_connector.py:229:lmcache.v1.storage_backend.connector.nixl_connector)
[2025-04-07 13:00:28,458] LMCache INFO: Sent 500 objects in 0.117705 seconds (test_nixl_storage_backend.py:95:__main__)
[2025-04-07 13:00:28,458] LMCache INFO: Throughput: 132.75 GB/s (test_nixl_storage_backend.py:97:__main__)
[2025-04-07 13:00:30,461] LMCache INFO: Test completed (test_nixl_storage_backend.py:153:__main__)
```

Receiver side logs:
```
[2025-04-07 13:00:26,094] LMCache INFO: Generated 500 objects with total size 16000.00 MB (test_nixl_storage_backend.py:70:__main__)
Loaded plugin UCX
Loaded plugin UCX_MO
Initialized NIXL agent: NixlRole.RECEIVER
[2025-04-07 13:00:26,317] LMCache INFO: Sent local transfer descriptors to sender (nixl_connector.py:115:lmcache.v1.storage_backend.connector.nixl_connector)
[2025-04-07 13:00:26,317] LMCache INFO: Waiting to receive data... (test_nixl_storage_backend.py:101:__main__)
[2025-04-07 13:00:28,341] LMCache DEBUG: Received event on the side channel, processing message... (nixl_connector.py:394:lmcache.v1.storage_backend.connector.nixl_connector)
[2025-04-07 13:00:28,342] LMCache DEBUG: Received request with 500 keys and UUID: f69f0509846943eb9e478b021afe8127 (nixl_connector.py:403:lmcache.v1.storage_backend.connector.nixl_connector)
[2025-04-07 13:00:28,359] LMCache INFO: Transfer for UUID 'f69f0509846943eb9e478b021afe8127' completed on the remote side (NixlRole.SENDER) (nixl_connector.py:251:lmcache.v1.storage_backend.connector.nixl_connector)
[2025-04-07 13:00:28,359] LMCache DEBUG: Received 128 keys and 128 objects. (nixl_backend.py:51:lmcache.v1.storage_backend.nixl_backend)
[2025-04-07 13:00:28,375] LMCache DEBUG: Observers processing in 15.6546 ms (nixl_connector.py:370:lmcache.v1.storage_backend.connector.nixl_connector)
[2025-04-07 13:00:28,387] LMCache INFO: Transfer for UUID 'e4db696cc1604caaac790af47f111145' completed on the remote side (NixlRole.SENDER) (nixl_connector.py:251:lmcache.v1.storage_backend.connector.nixl_connector)
[2025-04-07 13:00:28,388] LMCache DEBUG: Received 128 keys and 128 objects. (nixl_backend.py:51:lmcache.v1.storage_backend.nixl_backend)
[2025-04-07 13:00:28,403] LMCache DEBUG: Observers processing in 15.8146 ms (nixl_connector.py:370:lmcache.v1.storage_backend.connector.nixl_connector)
[2025-04-07 13:00:28,414] LMCache INFO: Transfer for UUID 'ead0858f0f134270b260ad4f2e05e121' completed on the remote side (NixlRole.SENDER) (nixl_connector.py:251:lmcache.v1.storage_backend.connector.nixl_connector)
[2025-04-07 13:00:28,415] LMCache DEBUG: Received 128 keys and 128 objects. (nixl_backend.py:51:lmcache.v1.storage_backend.nixl_backend)
[2025-04-07 13:00:28,432] LMCache DEBUG: Observers processing in 16.5569 ms (nixl_connector.py:370:lmcache.v1.storage_backend.connector.nixl_connector)
[2025-04-07 13:00:28,441] LMCache INFO: Transfer for UUID '7cdbf69484ed4d3ca2ad9df35d00a402' completed on the remote side (NixlRole.SENDER) (nixl_connector.py:251:lmcache.v1.storage_backend.connector.nixl_connector)
[2025-04-07 13:00:28,442] LMCache DEBUG: Received 116 keys and 116 objects. (nixl_backend.py:51:lmcache.v1.storage_backend.nixl_backend)
[2025-04-07 13:00:28,457] LMCache DEBUG: Observers processing in 15.1075 ms (nixl_connector.py:370:lmcache.v1.storage_backend.connector.nixl_connector)
[2025-04-07 13:00:28,522] LMCache INFO: Received all 500 objects (test_nixl_storage_backend.py:123:__main__)
[2025-04-07 13:00:28,556] LMCache INFO: All data verified successfully! (test_nixl_storage_backend.py:146:__main__)
```

**Measured performance:** 132.75 GB/s

## CacheEngine

```bash
# Terminal 1 (Sender)
UCX_TLS=cuda_ipc,cuda_copy,tcp CUDA_VISIBLE_DEVICES=0 python3 test_nixl_cache_engine.py --role sender --num-chunks 500 --num-rounds 5

# Terminal 2 (Receiver)
UCX_TLS=cuda_ipc,cuda_copy,tcp CUDA_VISIBLE_DEVICES=1 python3 test_nixl_cache_engine.py --role receiver --num-chunks 500 --num-rounds 5
```

Measured performance: 70.97 ± 7.66 GB/s 

## NIXL Pipe V2

Added new `--simulate-work` flag to simulate the LLM work on both sender and receiver sides. On sender side: 50ms per 10 objects, and on receiver side: 20ms per 10 objects.

```bash
# Terminal 1 (Sender)
UCX_TLS=cuda_ipc,cuda_copy,tcp CUDA_VISIBLE_DEVICES=0 python3 test_nixl_pipe_v2.py --role sender --num-rounds 5 --num-objs 500 --simulate-work

# Terminal 2 (Receiver)
UCX_TLS=cuda_ipc,cuda_copy,tcp CUDA_VISIBLE_DEVICES=1 python3 test_nixl_pipe_v2.py --role receiver --num-rounds 5 --num-objs 500 --simulate-work
```

## NIXL Channel v2 testing

Introduced a new option: `--batch-size` to control the number of objects in each batch when calling send. 

```bash
# Terminal 1 (Sender)
UCX_TLS=cuda_ipc,cuda_copy,tcp CUDA_VISIBLE_DEVICES=0 python3 test_nixl_channel_v2.py --role sender --num-objs 1000 --batch-size 30 --simulate-workload

# Terminal 2 (Receiver)
UCX_TLS=cuda_ipc,cuda_copy,tcp CUDA_VISIBLE_DEVICES=1 python3 test_nixl_channel_v2.py --role receiver --num-objs 1000 --batch-size 30 --simulate-workload
```
## NIXL Channel v2 multiplexing testing

Introduced a new option: `--num-expected-sender` to control the number of senders.

```bash
# Terminal 1 (Receiver)
UCX_TLS=cuda_ipc,cuda_copy,tcp CUDA_VISIBLE_DEVICES=7 python3 test_nixl_channel_v2.py --role receiver --num-objs 500 --batch-size 30 --simulate-workload --num-expected-senders 2

# Terminal 2 (Sender)
UCX_TLS=cuda_ipc,cuda_copy,tcp CUDA_VISIBLE_DEVICES=6 python3 test_nixl_channel_v2.py --role sender --num-objs 500 --batch-size 30 --simulate-workload

# Terminal 3 (Sender)
UCX_TLS=cuda_ipc,cuda_copy,tcp CUDA_VISIBLE_DEVICES=3 python3 test_nixl_channel_v2.py --role sender --num-objs 500 --batch-size 30 --simulate-workload
```
