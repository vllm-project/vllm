# install

```
docker run -d -it --runtime=habana --name deepseek-vllm-1.20  -v `pwd`:/workspace/vllm/  -v /data:/data -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice --ipc=host --net=host -e HF_HOME=/data/huggingface vault.habana.ai/gaudi-docker/1.20.0/ubuntu22.04/habanalabs/pytorch-installer-2.6.0:latest /bin/bash
```

or use 1.21 engineering build => better performance

```
docker run -d -it --runtime=habana --name deepseek-vllm-1.20  -v `pwd`:/workspace/vllm/  -v /data:/data -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice --ipc=host --net=host -e HF_HOME=/data/huggingface artifactory-kfs.habana-labs.com/docker-local/1.21.0/ubuntu22.04/habanalabs/pytorch-installer-2.6.0:1.21.0-275 /bin/bash
```

```
git clone https://github.com/HabanaAI/vllm-fork.git; git checkout deepseek_r1
cd vllm;  pip install -r requirements-hpu.txt; VLLM_TARGET_DEVICE=hpu pip install -e .  --no-build-isolation;
```

# prepare model

```
huggingface-cli download --local-dir ${YOUR_PATH}/DeepSeek-R1 deepseek-ai/DeepSeek-R1
```

# Option 1. run with runtime dequantize with block-based scale
> expect new DynamicMOE kernel ready in few weeks.
> Current Performance is worse than static quantization due to lack of dynamic MOE support.
## step 1. run example

```
VLLM_ENABLE_RUNTIME_DEQUANT=1 python scripts/run_example_tp.py --model ${YOUR_PATH}/DeepSeek-R1
```

## step 2. run lm_eval

```
VLLM_ENABLE_RUNTIME_DEQUANT=1 python scripts/run_lm_eval.py -l 64 --batch_size 1 --ep_size 1
{"gsm8k": {"alias": "gsm8k", "exact_match,strict-match": 0.96875, "exact_match_stderr,strict-match": 0.021921011700381302, "exact_match,flexible-extract": 0.96875, "exact_match_stderr,flexible-extract": 0.021921011700381302}}{"e2e time(secs)": 938.2986768169999}
```

# Option 2. run with dynamic quantization
> expect new DynamicMOE kernel ready in few weeks.
> Current Performance is worse than static quantization due to lack of dynamic MOE support.
## step 1. run example

```
# if you're testing with patched kernel
# use VLLM_DMOE_DYNAMIC_SCALE=1 to enable dynamic scaling supported DynamicMOE
VLLM_DMOE_DYNAMIC_SCALE=1 python scripts/run_example_tp.py --model ${YOUR_PATH}/DeepSeek-R1
```

## step 2. run lm_eval

```
VLLM_DMOE_DYNAMIC_SCALE=1 python scripts/run_lm_eval.py -l 64 --batch_size 1
{"gsm8k": {"alias": "gsm8k", "exact_match,strict-match": 0.96875, "exact_match_stderr,strict-match": 0.021921011700381302, "exact_match,flexible-extract": 0.96875, "exact_match_stderr,flexible-extract": 0.021921011700381302}}{"e2e time(secs)": 938.2986768169999}
```

## step 3. run benchmark

```
VLLM_DMOE_DYNAMIC_SCALE=1 bash scripts/benchmark-dynamicfp8-i1k-o1k-ep8-bestperf.sh
```

# Option 3. run with static quantization

## 3.1 convert model offline
> current best performance
## step 1. Prepare static quantization model

```
python scripts/convert_block_fp8_to_channel_fp8.py --model_path ${YOUR_PATH}/DeepSeek-R1 --qmodel_path ${YOUR_PATH}/DeepSeek-R1-static --input_scales_path scripts/DeepSeek-R1-BF16-w8afp8-static-no-ste_input_scale_inv.pkl.gz
```

## step 2. run example

```
python scripts/run_example_tp.py --model ${YOUR_PATH}/DeepSeek-R1-static
```

## step 3. run benchmark

```
bash scripts/benchmark-staticfp8-i1k-o1k-ep8-bestperf.sh
bash scripts/benchmark-staticfp8-i3200-o800-ep8-bestperf.sh
```

## 3.2 convert model online

## step 1. install INC

```bash
pip install git+https://github.com/intel/neural-compressor.git@r1-woq
```

## step 2. Get calibration file

```bash
# download from huggingface
huggingface-cli download Yi30/inc-woq-full-pile-512-1024-331  --local-dir ./scripts/nc_workspace_measure_kvache
```

or

```bash
# calibration by your own. Takes about 1 hour
export OFFICIAL_FP8_MODEL=deepseek-ai/DeepSeek-R1
cd ./scripts
VLLM_REQUANT_FP8_INC=1 QUANT_CONFIG=inc_measure_with_fp8kv_config.json VLLM_ENABLE_RUNTIME_DEQUANT=1 python run_example_tp.py --model ${OFFICIAL_FP8_MODEL} --tokenizer ${OFFICIAL_FP8_MODEL} --osl 32 --max_num_seqs 1 --nprompts 512 --dataset pile
```

## step 3. Benchmark

```bash
bash scripts/benchmark-inc-staticfp8-i1k-o1k-ep8-bestperf.sh
bash scripts/benchmark-inc-staticfp8-i3200-o800-ep8-bestperf.sh
```

# Option 4. run with Data Parallel

## step 1. deploy
> Use 1.20 docker is necessary, 1.21 will sometimes trigger kernel failing

```
docker run -d -it --runtime=habana --name deepseek-vllm-1.20  -v `pwd`:/workspace/vllm/  -v /data:/data -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice --ipc=host --net=host -e HF_HOME=/data/huggingface vault.habana.ai/gaudi-docker/1.20.0/ubuntu22.04/habanalabs/pytorch-installer-2.6.0:latest /bin/bash

docker exec -it deepseek-vllm-1.20 /bin/bash

git clone https://github.com/HabanaAI/vllm-fork.git; git checkout deepseek_r1
cd vllm;  pip install -r requirements-hpu.txt; VLLM_TARGET_DEVICE=hpu pip install -e .  --no-build-isolation;

```

## step 2. benchmark

``` bash
# need to skip test run in benchmark_serving.py to avoid server hang issue
cp dp_only/benchmark_serving.py benchmarks/benchmark_serving.py

bash dp_only/benchmark-inc-staticfp8-i1k-o1k-ep8-bestperf-nowarmup.sh
```

# Others. run with multi nodes

```

# head node
HABANA_VISIBLE_MODULES='0,1,2,3,4,5,6,7'  \
PT_HPU_WEIGHT_SHARING=0 \
PT_HPUGRAPH_DISABLE_TENSOR_CACHE=1 \
PT_HPU_ENABLE_LAZY_COLLECTIVES="true" \
VLLM_RAY_DISABLE_LOG_TO_DRIVER="1" \
RAY_IGNORE_UNHANDLED_ERRORS="1" \
ray start --head --resources='{"HPU": 8, "TPU": 0}'

```

```

# worker node
HABANA_VISIBLE_MODULES='0,1,2,3,4,5,6,7'  \
PT_HPU_WEIGHT_SHARING=0 \
PT_HPUGRAPH_DISABLE_TENSOR_CACHE=1 \
PT_HPU_ENABLE_LAZY_COLLECTIVES="true" \
VLLM_RAY_DISABLE_LOG_TO_DRIVER="1" \
RAY_IGNORE_UNHANDLED_ERRORS="1" \
ray start --address='${head_ip}:6379' --resources='{"HPU": 8, "TPU": 0}'

```

```

python scripts/run_example_tp_2nodes.py --model ${YOUR_PATH}/DeepSeek-R1-static

```
