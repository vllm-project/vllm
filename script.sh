sudo lsof -i :8000

sudo kill -9 86546

ps -C python -o pid=|xargs kill -9
ps -C python3 -o pid=|xargs kill -9
ps -C composer -o pid=|xargs kill -9
ps -C torchrun -o pid=|xargs kill -9
sudo lsof -t /dev/accel* | xargs kill
lsof -t /dev/accel* | xargs kill
sudo lsof -t /dev/vfio/*  | xargs kill
sudo killall python
sudo killall -9 python

clear

python -m vllm.entrypoints.api_server --model "mistralai/Mixtral-8x7B-Instruct-v0.1" --tensor-parallel-size 4  --max-model-len 4096 --gpu-memory-utilization 0.85 --swap-space 16 --disable-log-requests --num-scheduler-steps 4 --download_dir /dev/shm  --tokenizer mistralai/Mixtral-8x7B-Instruct-v0.1



vllm serve "mistralai/Mixtral-8x7B-Instruct-v0.1" --download_dir /dev/shm --swap-space 16 --disable-log-requests --tensor_parallel_size=4 --max-model-len=4096 --num-scheduler-steps=4 --tokenizer mistralai/Mixtral-8x7B-Instruct-v0.1
