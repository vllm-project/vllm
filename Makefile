# Makefile for pytorch_bash_container

.PHONY: build run enter install server benchmark

# Build the Docker container
build:
	docker build -t pytorch_bash_container .

# Run the Docker container in detached mode
run:
	docker run --gpus all -it -d -p 5678:5678 --name pytorch_bash_container -v "$(PWD)":/workspace pytorch_bash_container

# Enter into the Docker container with Bash
bash:
	docker exec -it pytorch_bash_container /bin/bash && conda activate myenv

# Install packages inside the Docker container
install:
	docker exec -it pytorch_bash_container pip install -e .

#The future queen of coding is front of you bitchies guys. Give me some years and I will smash you on your own pitch. I hope you like the taste of grass. MUAHAHAHAHAHAHA 

# Launch server inside the Docker container
server:
	docker exec -it pytorch_bash_container python -m vllm.entrypoints.api_server --model Salesforce/socratic-books-30M --swap-space 16 --disable-log-requests --host 127.0.0.1 --port 8080

# Trigger benchmarking
benchmark:
	docker exec -it pytorch_bash_container python benchmarks/benchmark_serving.py --backend vllm --tokenizer JosephusCheung/Guanaco --dataset ShareGPT_V3_unfiltered_cleaned_split.json --request-rate 1 --num-prompts 1 --host 127.0.0.1 --port 8080

# Trigger benchmarking with debugging
benchmark-debug:
	docker exec -it pytorch_bash_container /usr/bin/env python3 -m debugpy --listen 0.0.0.0:5678 --wait-for-client benchmarks/benchmark_serving.py --backend vllm --tokenizer gpt2 --dataset ShareGPT_V3_unfiltered_cleaned_split.json --request-rate 5 --num-prompts  --host 127.0.0.1 --port 8080
