CUDA_VISIBLE_DEVICES=7 nsys profile \
        --trace=cuda,nvtx,osrt \
        --output=prefiller \
        --force-overwrite=true \
        python3 toy_example.py
