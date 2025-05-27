CUDA_VISIBLE_DEVICES=7 nsys profile \
        --trace=cuda,nvtx,osrt,ucx \
	--gpu-metrics-devices=cuda-visible \
	--python-sampling=true \
	--trace-fork-before-exec=true \
        --output=prefiller \
        --force-overwrite=true \
        python3 toy_decode.py
