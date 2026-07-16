export VLLM_ENABLE_V1_MULTIPROCESSING=0
python dump_layer_outputs.py \
    --model  /models/MiniCPM5-2.6B \
    --backend metax_flagplugin \
    --prompt "hello?" \
    --max-tokens 3 \
    --output-dir ./layer_dumps
