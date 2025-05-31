# SPDX-License-Identifier: Apache-2.0
from vllm.distributed.kv_transfer.kv_connector.v1.nixl_cpu_utils import (
    NixlKVSender)

sender = NixlKVSender(1024 * 1024 * 1024)

sender.close()
