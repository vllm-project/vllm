# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from mooncake.store import MooncakeDistributedStore

# 1. Create store instance
store = MooncakeDistributedStore()

# 2. Setup with all required parameters
store.setup(
    "localhost",  # Your node's address
    "http://localhost:8080/metadata",  # HTTP metadata server
    512 * 1024 * 1024,  # 512MB segment size
    128 * 1024 * 1024,  # 128MB local buffer
    "tcp",  # Use TCP (RDMA for high performance)
    "",  # Leave empty; Mooncake auto-picks RDMA devices when needed
    "localhost:50051",  # Master service
)

# 3. Store data
store.put("hello_key", b"Hello, Mooncake Store!")

# 4. Retrieve data
data = store.get("hello_key")
print(data.decode())  # Output: Hello, Mooncake Store!

# 5. Clean up
store.close()
