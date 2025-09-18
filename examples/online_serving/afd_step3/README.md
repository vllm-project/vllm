# Dummy Connector
Dummy connector is used for testing basic functions. Attn and FFN server would not be connected as dummy connector would intermediately return input tensors.

1. Attn

```
vllm fserver /path/step3v -dp 8 --afd-config '{"afd_connector": "dummy", "afd_role": "attention", "afd_host": "127.0.0.0"}' --max-num-batched-tokens 384 --max-num-seqs 384 --compilation-config '{"cudagraph_capture_sizes": [1, 8]}'
```

2. FFN

```
vllm fserver /path/step3v -tp 8 --afd-config '{"afd_connector": "dummy", "afd_role": "ffn", "afd_host": "127.0.0.0"}' --max-num-batched-tokens 384 --max-num-seqs 384 --compilation-config '{"cudagraph_capture_sizes": [1, 8]}'
```

# StepMesh Connector
StepMesh connector is used for production deployment. Make sure [stepmesh](https://github.com/stepfun-ai/StepMesh) is installed and `afd_host` and `afd_port` are correctly set.

1. Attn

```
vllm fserver /path/step3v -dp 8 --afd-config '{"afd_connector": "stepmesh", "afd_role": "attention", "afd_host": "127.0.0.0"}' --max-num-batched-tokens 384 --max-num-seqs 384 --compilation-config '{"cudagraph_capture_sizes": [1, 8]}'
```

2. FFN

```
vllm fserver /path/step3v -tp 8 --afd-config '{"afd_connector": "stepmesh", "afd_role": "ffn", "afd_host": "127.0.0.0"}' --max-num-batched-tokens 384 --max-num-seqs 384 --compilation-config '{"cudagraph_capture_sizes": [1, 8]}'
```