Note: this doc didn't cover mla data parallel!!!

# setup docker and install mooncake

Please refer Chaojun's https://github.com/chaojun-zhang/vllm/blob/c9154592820bf1375a030e46e0334b83ac36287b/pd_distributed/setup.md 

0. make sure proxy works in intel env!
1. install mooncake [link](#https://github.com/kvcache-ai/Mooncake/?tab=readme-ov-file#-quick-start)

```
apt install git wget curl net-tools sudo iputils-ping etcd  -y

git clone https://github.com/kvcache-ai/Mooncake.git
cd Mooncake

bash dependencies.sh

mkdir build
cd build
cmake ..
make -j
make install
```

# How to Run

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
# some mooncake components will install to here
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib

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
