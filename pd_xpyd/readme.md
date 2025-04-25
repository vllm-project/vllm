# Mooncake Installation and Prefill/Decode Disaggregation Usage Guide
> **Note**: This document does not cover MLA data parallel setup.

## Mooncake Installation

### Install via pip
```bash
apt install etcd -y
pip3 install mooncake-transfer-engine==0.3.0b3
```

### Install from Source
```bash
# Install required packages
apt install git wget curl net-tools sudo iputils-ping etcd -y

# Clone the Mooncake repository
git clone https://github.com/kvcache-ai/Mooncake.git -b v0.3.0-beta
cd Mooncake

# Install dependencies
bash dependencies.sh

# Build and install Mooncake
mkdir build
cd build
cmake ..
make -j
make install
```

## Prefill/Decode Disaggregation Usage

0. prepare and modify mooncake.json

modify `metadata_server` as etcd address
modify `master_server_address` as mooncake store master address, better use high speed network for this since this will influence kv cache data transer performance.

modify `local_hostname` as node ip

1. start etcd server on master node
```
etcd --listen-client-urls http://0.0.0.0:2379 --advertise-client-urls http://localhost:2379  >etcd.log 2>&1 &
```

2. start Mooncake Store on master node
```
mooncake_master -enable_gc true -port 50001
```

3. start Prefill and Decode instance
refer `start_prefill.sh`, `start_decode.sh`
nencessary env/paras are:
```
MOONCAKE_CONFIG_PATH=./pd_xpyd/mooncake.json

# for prefill
--kv-transfer-config '{"kv_connector":"MooncakeStoreConnector","kv_role":"kv_producer"}' 
--port 8100

# for decode
--kv-transfer-config '{"kv_connector":"MooncakeStoreConnector","kv_role":"kv_consumer"}'
--port 8200
```

4. start proxy server
refer `start_proxy.sh`, modify parameters accordingly.
