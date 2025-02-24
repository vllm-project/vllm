# Note for quantize vLLM DeepSeek V3/R1 using INC

## Perquisites

- Hardware: ~~2xG3~~  ~~2x8XG3 or 2x8XG2~~ 8XG2 or 8XG3
- Docker: 1.20.0-521

- INC https://github.com/intel/neural-compressor/tree/dev/yi/quant_vllm-patch-19

```bash
git clone https://github.com/intel/neural-compressor.git inc
cd inc
git checkout dev/yi/quant_vllm-patch-19
pip install -r requirements.txt
pip install -r requirements_pt.txt
python setup.py pt develop
```
- vLLM  https://github.com/yiliu30/vllm-fork/pull/13

```
cd vllm;  pip install -r requirements-hpu.txt; VLLM_TARGET_DEVICE=hpu pip install -e .  --no-build-isolation;
```
- Model
    - ~~Reduced DeepSeek V3 model (4 layers with random weights)~~
    -  ~~Reduced DeepSeek V3 model (4 layers with real weights)~~
    - DeepSeek R1 (BF16)

## Example
- Quantize the BF16 model using the unified measurement results on 2x8XG2.


```bash
# vllm root
cd vllm
cd scripts
# Download the unified measurement results
# Make sure that the `nc_workspace_tmp` is under the `scripts` folder.
git clone https://huggingface.co/Yi30/nc_workspace_tmp
# Run example
python n2_ep8_tp8.py --mode q
```

> [!CAUTION]
> - The `QUANT_CONFIG` was hard-coded in [1](https://github.com/yiliu30/vllm-fork/blob/bc3a26c3d6143b6405ef9af7e06f6eddcbcbdad0/scripts/g4_multi_nodes_source.sh#L34C8-L34C20) and [2](https://github.com/yiliu30/vllm-fork/blob/bc3a26c3d6143b6405ef9af7e06f6eddcbcbdad0/scripts/g5_multi_nodes_source.sh#L38).
> - `VLLMKVCache`, `KVCache` and `lm-head` were skipped to quantize, will add them back.
> - ~~FAKE `EP` was hard-coded as 16. Please check `TEMP_EP` in vllm and `DEEPSEEK_EP` in INC.~~


## Others
- 1. Measured on 2x8G2 w/ 513 samples https://huggingface.co/Yi30/nc_workspace_tmp_pile_512_backup
- 2. 4 layers smoke on 8G2 test https://huggingface.co/Yi30/nc_workspace_tmp_4l_ep8_tp8
- 3. Merged result of 1) https://huggingface.co/Yi30/nc_workspace_tmp
- 4. 4 layers on 2x8G2 https://huggingface.co/Yi30/nc_workspace_tmp_4l_smoke