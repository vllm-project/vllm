
from vllm.distributed.kv_transfer.kv_connector.v1.nixl_cpu_utils import (
    NixlCPUReceiver, NixlKVSender, RingBufferAllocator)

sender = NixlKVSender(1024 * 1024 * 1024)

sender.close()
