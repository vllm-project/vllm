IP=$(hostname -I)
echo $IP
python3 start_ray.py --is_head --address ${IP} --ray_port 6379 --num_gpus 8 --num_cpus 20 --nnodes 4
