.PHONY: clean build

install_deps:
	pip install wheel packaging ninja setuptools>=49.4.0 numpy
	pip install -v -r requirements-cpu.txt --extra-index-url https://download.pytorch.org/whl/cpu

install:
	VLLM_TARGET_DEVICE=cpu pip install --no-build-isolation  -v -e .

VLLM_TP_2S_bench:
	ray stop
	OMP_DISPLAY_ENV=VERBOSE VLLM_CPU_KVCACHE_SPACE=40 OMP_PROC_BIND=close numactl --physcpubind=32-63 --membind=1 ray start --head --num-cpus=32 --num-gpus=0
	cd benchmarks && OMP_DISPLAY_ENV=VERBOSE VLLM_CPU_KVCACHE_SPACE=40 OMP_PROC_BIND=close numactl --physcpubind=0-31 --membind=0 python3 benchmark_throughput.py --backend=vllm --dataset=./ShareGPT_V3_unfiltered_cleaned_split.json --model=lmsys/vicuna-7b-v1.5 --n=1 --num-prompts=1000 --dtype=bfloat16 --trust-remote-code --device=cpu -tp=2

VLLM_2S_offline:
	ray stop
	OMP_DISPLAY_ENV=VERBOSE VLLM_CPU_KVCACHE_SPACE=40 OMP_PROC_BIND=close numactl --physcpubind=32-63 --membind=1 ray start --head --num-cpus=32 --num-gpus=0
	cd examples && OMP_DISPLAY_ENV=VERBOSE VLLM_CPU_KVCACHE_SPACE=40 OMP_PROC_BIND=close numactl --physcpubind=0-31 --membind=0 python3 offline_inference.py

VLLM_TP_4S_bench:
	export RAY_worker_niceness=0
	ray stop
	OMP_DISPLAY_ENV=VERBOSE VLLM_CPU_KVCACHE_SPACE=40 OMP_PROC_BIND=close numactl --physcpubind=192 --membind=1 ray start --head --num-cpus=0 --num-gpus=0 --disable-usage-stats --include-dashboard=false
	OMP_DISPLAY_ENV=VERBOSE VLLM_CPU_KVCACHE_SPACE=40 OMP_PROC_BIND=close OMP_NUM_THREADS=32 numactl --physcpubind=32-63 --membind=1 ray start --address=auto --num-cpus=32 --num-gpus=0
	OMP_DISPLAY_ENV=VERBOSE VLLM_CPU_KVCACHE_SPACE=40 OMP_PROC_BIND=close OMP_NUM_THREADS=32 numactl --physcpubind=64-95 --membind=2 ray start --address=auto --num-cpus=32 --num-gpus=0
	OMP_DISPLAY_ENV=VERBOSE VLLM_CPU_KVCACHE_SPACE=40 OMP_PROC_BIND=close OMP_NUM_THREADS=32 numactl --physcpubind=96-127 --membind=3 ray start --address=auto --num-cpus=32 --num-gpus=0
	cd benchmarks && OMP_DISPLAY_ENV=VERBOSE VLLM_CPU_KVCACHE_SPACE=40 OMP_PROC_BIND=close numactl --physcpubind=0-15 --membind=0 python3 benchmark_throughput.py --backend=vllm --dataset=/root/ShareGPT_V3_unfiltered_cleaned_split.json --model=meta-llama/Llama-2-7b-chat-hf --n=1 --num-prompts=1000 --dtype=bfloat16 --trust-remote-code --device=cpu -tp=4

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
	 OMP_DISPLAY_ENV=verbose \
	 VLLM_CPU_KVCACHE_SPACE=40 \
	 OMP_PROC_BIND=close \
	 python3 benchmark_throughput.py --backend=vllm --dataset=./ShareGPT_V3_unfiltered_cleaned_split.json --model=lmsys/vicuna-7b-v1.5 --n=1 --num-prompts=1000 --dtype=bfloat16 --trust-remote-code --device=cpu

VLLM_LT_bench:
	cd benchmarks && python benchmark_latency.py --model=/root/HF_models/vicuna-7b-v1.5/ --n=1 --batch-size=8 --input-len=128 --output-len=512 --num-iters=4 --dtype=bfloat16 --trust-remote-code --device=cpu --swap-space=40

VLLM_SERVE_bench:
	cd benchmarks && python -m vllm.entrypoints.api_server \
        --model /root/HF_models/vicuna-7b-v1.5/ --swap-space 40 \
        --disable-log-requests --dtype=bfloat16 --device cpu & \
	cd benchmarks && sleep 30 && python benchmark_serving.py \
        --backend vllm \
        --tokenizer /root/HF_models/vicuna-7b-v1.5/ --dataset /root/HF_models/ShareGPT_V3_unfiltered_cleaned_split.json \
        --request-rate 10

VLLM_2S_Serve:
	ray stop
	OMP_DISPLAY_ENV=VERBOSE VLLM_CPU_KVCACHE_SPACE=40 OMP_PROC_BIND=close numactl --physcpubind=32-63 --membind=1 ray start --head --num-cpus=32 --num-gpus=0
	cd benchmarks && OMP_DISPLAY_ENV=VERBOSE VLLM_CPU_KVCACHE_SPACE=40 OMP_PROC_BIND=close numactl --physcpubind=0-31 --membind=0 python3 -m vllm.entrypoints.openai.api_server --model lmsys/vicuna-7b-v1.5 --dtype=bfloat16 --device cpu -tp=2 

VLLM_4S_Serve_Ray:
	export RAY_worker_niceness=0
	ray stop
	OMP_DISPLAY_ENV=VERBOSE OMP_NUM_THREADS=32 VLLM_CPU_KVCACHE_SPACE=40 OMP_PROC_BIND=close OMP_WAIT_POLICY=active numactl --physcpubind=0 --membind=0 ray start --head --num-cpus=0 --num-gpus=0 --disable-usage-stats --include-dashboard=false
	OMP_DISPLAY_ENV=VERBOSE OMP_NUM_THREADS=32 VLLM_CPU_KVCACHE_SPACE=40 OMP_PROC_BIND=close OMP_NUM_THREADS=32 OMP_WAIT_POLICY=active numactl --physcpubind=0-31 --membind=0 ray start --address=auto --num-cpus=32 --num-gpus=0
	OMP_DISPLAY_ENV=VERBOSE OMP_NUM_THREADS=32 VLLM_CPU_KVCACHE_SPACE=40 OMP_PROC_BIND=close OMP_NUM_THREADS=32 OMP_WAIT_POLICY=active numactl --physcpubind=32-63 --membind=1 ray start --address=auto --num-cpus=32 --num-gpus=0
	OMP_DISPLAY_ENV=VERBOSE OMP_NUM_THREADS=32 VLLM_CPU_KVCACHE_SPACE=40 OMP_PROC_BIND=close OMP_NUM_THREADS=32 OMP_WAIT_POLICY=active numactl --physcpubind=64-95 --membind=2 ray start --address=auto --num-cpus=32 --num-gpus=0
	OMP_DISPLAY_ENV=VERBOSE OMP_NUM_THREADS=32 VLLM_CPU_KVCACHE_SPACE=40 OMP_PROC_BIND=close OMP_NUM_THREADS=32 OMP_WAIT_POLICY=active numactl --physcpubind=96-127 --membind=3 ray start --address=auto --num-cpus=32 --num-gpus=0
	cd benchmarks && OMP_NUM_THREADS=32 numactl --physcpubind=32 --membind=0 python3 -m vllm.entrypoints.openai.api_server --model meta-llama/Llama-2-7b-chat-hf --dtype=bfloat16 --device cpu --engine-use-ray --disable-log-stats -tp=4

VLLM_bench_client:
	cd benchmarks && python3 benchmark_serving.py --backend vllm --model meta-llama/Llama-2-7b-chat-hf --dataset /root//ShareGPT_V3_unfiltered_cleaned_split.json --num-prompts 1000
