import subprocess

# 定义要运行的命令
command = "CUDA_VISIBLE_DEVICES=1 python3 latency_test.py " + \
"--model=/data/zyh/llm-finetune/llama2-hf/7B --swap-space=20 --block-size="

block_size = [8, 16, 32]
for size in block_size:
    output = subprocess.check_output(command+str(size), shell=True)

print(output.decode("utf-8"))
