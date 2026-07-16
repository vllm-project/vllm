export GEMS_VENDOR=metax   # 指定厂商，FlagGems 加载 MetaX 后端
export VLLM_PLUGINS=fl     # 加载 vllm-plugin-FL 插件
export VLLM_ENABLE_V1_MULTIPROCESSING=0
python dump_layer_outputs.py \
    --model  /models/MiniCPM5-2.6B \
    --backend metax_flagplugin \
    --prompt "hello?" \
    --max-tokens 3 \
    --dump-mode fine \
    --output-dir ./layer_dumps
