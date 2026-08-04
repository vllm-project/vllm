lmcache could use [infinistore](https://github.com/bd-iaas-us/InfiniStore) as a backend storage.

Infinistore is a memory storage which support RDMA and NVLINK. lmcache's infinistore connector is using RDMA transport for now.

This is a simple instruction how to use lmcache with infinistore

# install infistore

1. make sure RDMA driver is installed.

use ibverbs-utils to find active NIC. In this example we assume NIC:mlx5_0 is the active NIC.

```
ibv_devinfo
```

2. install infinistore 

```
pip install infinistore
```

now, infinistore support python3.10, python3.11, python3.12

# start infinistore and lmcache

1. start infinistore

mlx5_0 is an active RDMA NIC

```
python -m infinistore.server --service-port 12345 --dev-name mlx5_0 --link-type Ethernet  --manage-port 8080
```

2. start lmcache
```
LMCACHE_CONFIG_FILE=backend_type.yaml python -m lmcache_vllm.vllm.entrypoints.openai.api_server --model mistralai/Mistral-7B-Instruct-v0.2  --max-model-len 8192
```


# RDMA/infinistore troubleshooting

1. infinistore's log could reveal many details. if lmcache read/write cache, you will find corresponding PUT/GET requests.
2. you could use [perftest](https://github.com/linux-rdma/perftes) utils to check connectivity and bandwidth.
3. infinistore itself has client [examples](https://github.com/bd-iaas-us/InfiniStore/tree/main/infinistore/example) to check if RDMA connection works.
