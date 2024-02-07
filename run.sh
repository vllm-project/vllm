docker run --gpus all -it --rm --ipc=host \
           -p 12301:12301 \
           -p 12302:12302 \
	   -p 12303:12303 \
	   -p 12304:12304 \
	   -v /home/xianw/workspace/vllm/:/vllm \
           -v /home/xianw/.cache:/root/.cache \
           wx_vllm:v1 \
           /bin/bash
