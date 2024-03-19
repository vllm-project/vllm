# how to run
single card:

```
python benchmarks/benchmark_throughput.py --backend=vllm --dataset=/home/kunshang/kunshang/ShareGPT_V3_unfiltered_cleaned_split.json --model=/home/kunshang/kunshang/llama-7b-hf/ --n=1 --num-prompts=1000 --dtype=half --trust-remote-code --device=xpu  --enforce-eager 
```

dual card:

```
CCL_WORKER_COUNT=2 FI_PROVIDER=psm3 CCL_ATL_TRANSPORT=ofi CCL_ZE_IPC_EXCHANGE=sockets python benchmarks/benchmark_throughput.py --backend=vllm --dataset=/home/kunshang/kunshang/ShareGPT_V3_unfiltered_cleaned_split.json --model=/home/kunshang/kunshang/llama-7b-hf/ --n=1 --num-prompts=1000 --dtype=half --trust-remote-code --device=xpu  --enforce-eager --tensor-parallel-size=2
```

# performance

a quick performance on my env:

|model| tp_size| performance: token/s|
|--|--|--|
|llama-2-7b| 1 | 2768.63| 
|llama-2-7b| 2 | 1402.40| 
|llama-2-7b| 4 | 1204.37| 
|llama-2-13b| 1 | 1451.74| 
|llama-2-13b| 2 | 1138.24| 

