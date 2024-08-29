.. _nginxloadbalancer:

Nginx Loadbalancer
========================

This document shows how to launch multiple vLLM serving containers and use Nginx to act as a load balancer between the servers. 

.. _nginxloadbalancer_nginx_build:

Build Nginx Container
------------

Create a file named ``Dockerfile.nginx``:

.. code-block:: console

    # Copyright (C) 2024 Intel Corporation
    # SPDX-License-Identifier: Apache-2.0

    FROM nginx:latest
    RUN rm /etc/nginx/conf.d/default.conf
    EXPOSE 80
    CMD ["nginx", "-g", "daemon off;"]

Build the container:

.. code-block:: console

    docker build . -f Dockerfile.nginx --tag nginx-lb   

Create Simple Nginx Config file
------------

Create a file named ``nginx.conf``. Note that you can add as many servers as you'd like. In the below example we'll start with two. To add more, add another ``server vllmN:8000 max_fails=3 fail_timeout=10000s;`` entry to ``upstream backend``.

.. code-block:: console

    upstream backend {
        least_conn;
        server vllm0:8000 max_fails=3 fail_timeout=10000s;
        server vllm1:8000 max_fails=3 fail_timeout=10000s;
    }     
    server {
        listen 80;
        location / {
            proxy_pass http://backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
    }

Build vLLM Container
------------

.. code-block:: console

    cd $vllm_root
    docker build -f Dockerfile.cpu . --tag vllm 

Create vLLM Launch Script For Use Inside Each Container
------------

Call the script ``vllm_start_script/vllm_start.sh``

.. code-block:: console

    export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4:/usr/local/lib/libiomp5.so:$LD_PRELOAD 
    export RAY_worker_niceness=0
    export KMP_BLOCKTIME=1
    export KMP_TPAUSE=0
    export KMP_SETTINGS=0
    export KMP_FORKJOIN_BARRIER_PATTERN=dist,dist
    export KMP_PLAIN_BARRIER_PATTERN=dist,dist
    export KMP_REDUCTION_BARRIER_PATTERN=dist,dist
    svrcmd="cd benchmarks && VLLM_CPU_KVCACHE_SPACE=40 VLLM_CPU_OMP_THREADS_BIND=\"$CPU_BIND\" python3 -m vllm.entrypoints.openai.api_server --model $MODEL  --dtype=bfloat16 --device cpu --disable-log-stats"
    eval $svrcmd

Launch vLLM Containers
------------

.. code-block:: console

    model=meta-llama/Llama-2-7b-hf
    docker run -itd --privileged --ipc host --cap-add=SYS_ADMIN --shm-size=10.24gb -e CPU_BIND=0-47  -e MODEL=$model -v ./vllm_start_script/:/workspace/vllm_start_script/ -p 8081:8000 --name vllm0 vllm bash /workspace/vllm_start_script/vllm_start.sh
    docker run -itd --privileged --ipc host --cap-add=SYS_ADMIN --shm-size=10.24gb -e CPU_BIND=48-95 -e MODEL=$model -v ./vllm_start_script/:/workspace/vllm_start_script/ -p 8082:8000 --name vllm1 vllm bash /workspace/vllm_start_script/vllm_start.sh

Launch Nginx
------------

.. code-block:: console

    docker run -itd -p 8000:80 -v ./nginx_conf/:/etc/nginx/conf.d/ --name nginx-lb nginx-lb:latest 
    