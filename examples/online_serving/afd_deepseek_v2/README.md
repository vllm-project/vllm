# P2P Connector
P2P connector is used for testing the afd implementation for deepseek-v2-lite models. It uses torch.distributed to send/recv intermediate tensors between attn and ffn instances. 

When the --enable-dbo flag is currently enabled, the num_stage parameter becomes ineffective, and the actual number of microbatches is 2.

Currently, the P2PConnector only supports scenarios where the number of dies of A is an integer multiple of that of F. Asymmetric configurations will be supported in future updates.
Currently, only scenarios where A uses DP (Data Parallelism) with TP=1 and F uses EP (Expert Parallelism) with TP=1 are supported. If there is a need for TP support, it may be considered in future updates.

# PD Mix

In the current PD mixed mode, AFD does not support CUDA Graphs for now; you need to run in eager mode. Also, afd_host must be set to the first FFN node.

1. Attn

```
vllm serve "/path/to/DeepSeek-V2-Lite"  --data_parallel_size=2 --enable_expert_parallel --enforce_eager --enable-dbo --dbo-prefill-token-threshold 12 --dbo-decode-token-threshold 2 --afd-config '{"afd_connector":"p2pconnector", "afd_role": "attention", "afd_host":"127.0.0.1", "afd_port":"29500","num_afd_stages":"2","afd_extra_config":{"afd_size":"2A2F"}}'

```

2. FFN

```
vllm serve "/path/to/DeepSeek-V2-Lite" --data_parallel_size=2 --enable_expert_parallel --enforce_eager --afd-config '{"afd_connector":"p2pconnector", "num_afd_stages":"2", "afd_role": "ffn", "afd_host":"127.0.0.1", "afd_port":"29500", "afd_extra_config":{"afd_size":"2A2F"}}'
```

# Decode Instances in PD Separation

In the D instances of PD separation, we now support the FULL_DECODE_ONLY mode of cudagraph. Please note the following:

* When using RDMA across machines, cudagraph replay may get stuck. The reason is currently unknown. Please run the following command:

  ```bash
  # When using RDMA across machines, cudagraph replay may get stuck. Please run:
  export NCCL_IB_DISABLE=1
  ```

* afd_host must be set to the first FFN node.
* Delete the torch inductor cache (if not deleted, it may cause a hang for reasons currently unknown):
  ```bash
  rm -rf /tmp/torchinductor_root/ 
  ```

Here is an example: for 2A2F, each DP in the decode instance can handle up to 64 requests, and the number of tokens captured by cudagraph is also 64:
1. Attn

```
vllm serve "/home/fq9hpsac/fq9hpsacuser03/deepseek-v2-lite" \
    --max-num-batched-tokens 64 \
    --data-parallel-size=2 \
    --enable_expert_parallel \
    --enable-dbo \
    --dbo-prefill-token-threshold 12 \
    --dbo-decode-token-threshold 2 \
    --port 8022 \
    --no-enable-prefix-caching \
    --compilation-config '{
		"cudagraph_mode": "FULL_DECODE_ONLY",
		"cudagraph_capture_sizes": [64]
	}' \
    --kv-transfer-config '{
        "kv_connector": "MooncakeConnector",
        "kv_role": "kv_consumer"
    }' \
    --afd-config '{
        "afd_connector":"p2pconnector",
        "afd_role": "attention",
        "afd_host":"10.248.12.142",
        "afd_port":"29531",
        "num_afd_stages":"2",
        "afd_extra_config":{
            "afd_size":"2A2F"
        }
    }'
```

2. FFN

```
vllm serve "/home/fq9hpsac/fq9hpsacuser03/deepseek-v2-lite" \
    -dp=2 \
    --enable_expert_parallel \
    --compilation-config '{
		"cudagraph_mode": "FULL_DECODE_ONLY",
		"cudagraph_capture_sizes": [64]
	}' \
    --enable-dbo \
    --dbo-prefill-token-threshold 12 \
    --dbo-decode-token-threshold 2 \
    --port 8021 \
    --afd-config '{
        "afd_connector":"p2pconnector",
        "num_afd_stages":"2",
        "afd_role": "ffn",
        "afd_host":"10.248.12.142",
        "afd_port":"29531",
        "afd_extra_config":{
            "afd_size":"4A2F"
        }
    }'
```