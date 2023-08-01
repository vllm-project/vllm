JOBS?=$(bash getconf _NPROCESSORS_CONF)

.PHONY: clean build

clean:
	@ls | grep '^build-\(Debug\|Release\)' | xargs -r rm -r

build:
	@mkdir -p build-$(BUILD_TYPE) && \
	cmake -B build-$(BUILD_TYPE) -GNinja -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) 
	cmake --build build-$(BUILD_TYPE) -j $(JOBS) 

debug:
	$(MAKE) build BUILD_TYPE=Debug ENABLE_SANITIZER=OFF

debug-asn:
	$(MAKE) build BUILD_TYPE=Debug ENABLE_SANITIZER=ON

release:
	$(MAKE) build BUILD_TYPE=Release ENABLE_SANITIZER=OFF

release-debug:
	$(MAKE) build BUILD_TYPE=RelWithDebInfo ENABLE_SANITIZER=OFF

sanitizer:
	echo 1 > /proc/sys/vm/overcommit_memory

py_install:
	VLLM_BUILD_CPU_OPS=1 MAX_JOBS=JOBS pip install --no-build-isolation  -v -e .

package:
	VLLM_BUILD_CPU_OPS=1 MAX_JOBS=JOBS python setup.py bdist_wheel
	echo "Wheel package is saved in ./dist/"

HF_TP_bench:
	cd benchmarks && python benchmark_throughput.py --backend=hf --dataset=../ShareGPT_V3_unfiltered_cleaned_split.json --model=/root/frameworks.bigdata.dev-ops/vicuna-7b-v1.5/ --n=1 --num-prompts=1 --hf-max-batch-size=1 --trust-remote-code --device=cpu

VLLM_TP_bench:
	cd benchmarks && python benchmark_throughput.py --backend=vllm --dataset=/root/HF_models/ShareGPT_V3_unfiltered_cleaned_split.json --model=/root/HF_models/vicuna-7b-v1.5/ --n=1 --num-prompts=1 --dtype=float32 --trust-remote-code --device=cpu --swap-space=4

VLLM_LT_bench:
	cd benchmarks && python benchmark_latency.py --model=/root/frameworks.bigdata.dev-ops/vicuna-7b-v1.5/ --n=1 --batch-size=48 --input-len=128 --output-len=128 --num-iters=8 --dtype=bfloat16 --trust-remote-code --device=cpu

VLLM_SERVE_bench:
	cd benchmarks && python -m vllm.entrypoints.api_server \
        --model /root/HF_models/vicuna-7b-v1.5/ --swap-space 40 \
        --disable-log-requests --dtype=bfloat16 --device cpu & \
	cd benchmarks && sleep 30 && python benchmark_serving.py \
        --backend vllm \
        --tokenizer /root/HF_models/vicuna-7b-v1.5/ --dataset /root/HF_models/ShareGPT_V3_unfiltered_cleaned_split.json \
        --request-rate 10