## 0. Prerequisites

Please use the firmware and software stack mentioned [here](https://github.com/HabanaAI/vllm-fork/tree/deepseek_r1/scripts/quickstart).

## 1. Installation

- vLLM

```bash
git clone -b deepseek_r1 https://github.com/HabanaAI/vllm-fork.git
cd vllm-fork
pip install -r requirements-hpu.txt
VLLM_TARGET_DEVICE=hpu pip install -e .  --no-build-isolation
```

- INC

```bash
pip install git+https://github.com/intel/neural-compressor.git@r1-woq
```

## 2. Convert the model files

```bash
cd vllm-fork
python ./scripts/convert_for_g2.py -i /path/to/official/model -o /path/to/converted/model/
```

This script 1) converts official model weights from `torch.float8_e4m3fn` format to `torch.float8_e4m3fnuz` format, and 2) copies other JSON and Python files into the target path.

## 3. Benchmark

> [!NOTE]
> For INC WoQ requantization, make sure to:
> 1) Specify the path to the measurement files in the quantization configuration JSON file.
> 
> 2) Set the `QUANT_CONFIG` environment variable to point to this configuration file.
> 
>For more details, refer to the `INC WOQ ReQuant` section in the `single_16k_len_inc.sh` script.

> [!CAUTION]
> Before running the benchmark, make sure to update the `model_path` in the `single_16k_len_inc.sh` script.

### 3.1 BF16 KV + Per-Channel Quantization

- Get calibration files

```bash
cd vllm-fork
huggingface-cli download Yi30/inc-woq-default-pile-one-cache-412-g2  --local-dir ./scripts/nc_workspace_measure_kvache
```

- Running the Benchmark

```bash
cd vllm-fork
bash ./scripts/single_16k_len_inc.sh
```


### 3.2 FP8 KV + Per-Channel Quantization

- Get calibration files

```bash
cd vllm-fork
huggingface-cli download Yi30/inc-woq-default-pile-one-cache-412-g2  --local-dir ./scripts/nc_workspace_measure_kvache
```

- Running the Benchmark

```bash
cd vllm-fork
bash scripts/single_16k_len.sh --fp8_kv
```