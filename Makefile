.PHONY: clean build

install_dnnl:
	git clone -b rls-v3.5 https://github.com/oneapi-src/oneDNN.git
	cd oneDNN && mkdir build && \
	cmake -B build -G Ninja -DONEDNN_LIBRARY_TYPE=STATIC -DONEDNN_BUILD_DOC=OFF -DONEDNN_BUILD_EXAMPLES=OFF -DONEDNN_BUILD_TESTS=OFF -DONEDNN_BUILD_GRAPH=OFF -DONEDNN_ENABLE_WORKLOAD=INFERENCE -DONEDNN_ENABLE_PRIMITIVE=MATMUL && \
	cmake --build build --target install --config Release

install_deps:
	pip install wheel packaging ninja setuptools>=49.4.0 numpy
	pip install -v -r requirements-cpu.txt --extra-index-url https://download.pytorch.org/whl/cpu

install:
	VLLM_TARGET_DEVICE=cpu pip install --no-build-isolation  -v -e .

VLLM_TP_2S_bench:
	cd benchmarks && VLLM_CPU_OMP_THREADS_BIND="0-23|24-47" VLLM_CPU_KVCACHE_SPACE=40 LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4:/usr/local/lib/libiomp5.so python3 benchmark_throughput.py --backend=vllm --dataset=./ShareGPT_V3_unfiltered_cleaned_split.json --model=lmsys/vicuna-7b-v1.5 --n=1 --num-prompts=100 --dtype=bfloat16 --trust-remote-code --device=cpu -tp=2

VLLM_2S_offline:
	ray stop
	OMP_DISPLAY_ENV=VERBOSE VLLM_CPU_KVCACHE_SPACE=40 OMP_PROC_BIND=close numactl --physcpubind=32-63 --membind=1 ray start --head --num-cpus=32 --num-gpus=0
	cd examples && OMP_DISPLAY_ENV=VERBOSE VLLM_CPU_KVCACHE_SPACE=40 OMP_PROC_BIND=close numactl --physcpubind=0-31 --membind=0 python3 offline_inference.py

VLLM_TP_4S_bench:
	cd benchmarks &&  VLLM_CPU_OMP_THREADS_BIND="0-31|32-63|64-95|96-127" VLLM_CPU_KVCACHE_SPACE=40 LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4:/usr/local/lib/libiomp5.so python3 benchmark_throughput.py --backend=vllm --dataset=./ShareGPT_V3_unfiltered_cleaned_split.json --model=lmsys/vicuna-7b-v1.5 --n=1 --num-prompts=1000 --dtype=bfloat16 --trust-remote-code --device=cpu -tp=4

VLLM_4S_offline:
	ray stop
	OMP_DISPLAY_ENV=VERBOSE VLLM_CPU_KVCACHE_SPACE=40 OMP_PROC_BIND=close numactl --physcpubind=32-63 --membind=1 ray start --head --num-cpus=32 --num-gpus=0
	OMP_DISPLAY_ENV=VERBOSE VLLM_CPU_KVCACHE_SPACE=40 OMP_PROC_BIND=close numactl --physcpubind=64-95 --membind=2 ray start --address=auto --num-cpus=32 --num-gpus=0
	OMP_DISPLAY_ENV=VERBOSE VLLM_CPU_KVCACHE_SPACE=40 OMP_PROC_BIND=close numactl --physcpubind=96-127 --membind=3 ray start --address=auto --num-cpus=32 --num-gpus=0
	cd examples && OMP_DISPLAY_ENV=VERBOSE VLLM_CPU_KVCACHE_SPACE=40 OMP_PROC_BIND=close numactl --physcpubind=0-31 --membind=0 python3 offline_inference.py

HF_TP_bench:
	cd benchmarks && python benchmark_throughput.py --backend=hf --dataset=../ShareGPT_V3_unfiltered_cleaned_split.json --model=/root/frameworks.bigdata.dev-ops/vicuna-7b-v1.5/ --n=1 --num-prompts=1 --hf-max-batch-size=1 --trust-remote-code --device=cpu

VLLM_TP_bench:
	cd benchmarks && \
	 VLLM_CPU_OMP_THREADS_BIND="0-47" \
	 VLLM_CPU_KVCACHE_SPACE=100 \
	 TORCH_LOGS="recompiles" \
	 LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4 \
	 python3 benchmark_throughput.py --backend=vllm --dataset=./ShareGPT_V3_unfiltered_cleaned_split.json --model=lmsys/vicuna-7b-v1.5 --n=1 --num-prompts=1000 --dtype=bfloat16 --trust-remote-code --device=cpu

VLLM_TP_bench_slm:
	cd benchmarks && \
	 VLLM_CPU_OMP_THREADS_BIND="0-47" \
	 VLLM_CPU_KVCACHE_SPACE=100 \
	 TORCH_LOGS="recompiles" \
	 LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4 \
	 python3 benchmark_throughput.py --backend=vllm --model=facebook/opt-125m --n=1 --num-prompts=1000 --input-len=128 --output-len=128 --dtype=bfloat16 --trust-remote-code --device=cpu

VLLM_LT_bench:
	cd benchmarks && \
	VLLM_CPU_OMP_THREADS_BIND="0-47" \
	 VLLM_CPU_KVCACHE_SPACE=100 \
	 TORCH_LOGS="recompiles" \
	 LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4 \
	 python3 benchmark_latency.py --model=facebook/opt-125m --n=1 --batch-size=32 --input-len=1024 --output-len=1024 --num-iters-warmup=1 --num-iters=3 --dtype=bfloat16 --trust-remote-code --device=cpu

VLLM_SERVE_bench:
	cd benchmarks && python -m vllm.entrypoints.api_server \
        --model /root/HF_models/vicuna-7b-v1.5/ --swap-space 40 \
        --disable-log-requests --dtype=bfloat16 --device cpu & \
	cd benchmarks && sleep 30 && python benchmark_serving.py \
        --backend vllm \
        --tokenizer /root/HF_models/vicuna-7b-v1.5/ --dataset /root/HF_models/ShareGPT_V3_unfiltered_cleaned_split.json \
        --request-rate 10

VLLM_Serve:
	cd benchmarks && VLLM_CPU_KVCACHE_SPACE=40 VLLM_CPU_OMP_THREADS_BIND="0-47" LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4 python3 -m vllm.entrypoints.openai.api_server --model lmsys/vicuna-7b-v1.5 --dtype=bfloat16 --device cpu

VLLM_2S_Serve:
	cd benchmarks && VLLM_CPU_KVCACHE_SPACE=40 VLLM_CPU_OMP_THREADS_BIND="0-23|24-47" LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4 python3 -m vllm.entrypoints.openai.api_server --model lmsys/vicuna-7b-v1.5 --dtype=bfloat16 --device cpu -tp=2

VLLM_bench_client:
	cd benchmarks && python3 benchmark_serving.py --backend vllm --model lmsys/vicuna-7b-v1.5 --tokenizer lmsys/vicuna-7b-v1.5 --dataset ./ShareGPT_V3_unfiltered_cleaned_split.json --request-rate 4 --num-prompts 1000