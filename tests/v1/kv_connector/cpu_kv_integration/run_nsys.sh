#!/bin/bash

if [[ $1 == "decoder" ]]; then
echo "Running decoder"
CUDA_VISIBLE_DEVICES=7 nsys profile \
        --trace=cuda,nvtx,osrt \
	--gpu-metrics-devices=cuda-visible \
	--python-sampling=true \
	--trace-fork-before-exec=true \
        --output=decoder \
        --force-overwrite=true \
        python3 toy_decode.py

else
echo "Running prefiller"
CUDA_VISIBLE_DEVICES=6 nsys profile \
        --trace=cuda,nvtx,osrt \
	--gpu-metrics-devices=cuda-visible \
	--python-sampling=true \
	--trace-fork-before-exec=true \
        --output=prefiller \
        --force-overwrite=true \
        python3 toy_example.py
fi
