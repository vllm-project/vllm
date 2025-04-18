export MODEL_PATH=/data/models/DeepSeek-R1-static/

# --prefill is a dummy instance, not used actually.
python3 examples/online_serving/disagg_examples/disagg_proxy_demo.py \
    --model $MODEL_PATH \
    --decode 127.0.0.1:8801 127.0.0.1:8802 127.0.0.1:8803 127.0.0.1:8804 127.0.0.1:8805 127.0.0.1:8806 127.0.0.1:8807 127.0.0.1:8808\
    --port 8123