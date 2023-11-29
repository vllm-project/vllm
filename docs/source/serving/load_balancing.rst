.. _load_balancing:

Load Balancing
===============

In scenarios where multiple GPUs or nodes are necessary to handle the model serving workload, it is advisable to initiate several vLLM server instances and distribute incoming requests amongst them using an HTTP load balancer.

For instances where vLLM is deployed via Docker, the `Traefik HTTP load balancer <https://doc.traefik.io/traefik/>`_ can be employed to evenly distribute the load across multiple vLLM server instances.

Below is a sample configuration:

.. code-block:: yaml

   services:
     ### Traefik load balancer
     http-load-balancer:
       image: traefik:v2.10
       command:
         # static config
         --providers.docker

         --entrypoints.web.address=:80

       volumes:
        - /var/run/docker.sock:/var/run/docker.sock:ro

       restart: unless-stopped

     ### vLLM servers
     # 0
     vllm-server-0:
       # ... Configuration of vLLM server
       image: vllm/vllm-openai:latest
       command: --model mistralai/Mistral-7B-v0.1
       volumes:
         - ./huggingface_cache:/root/.cache/huggingface:rw

       # Allocate GPU 0
       deploy:
         resources:
           reservations:
             devices:
               - driver: nvidia
                 capabilities: [gpu]
                 device_ids: ['0']

       # Traefik load balancer config
       labels:
         traefik.http.routers.vllm.rule: Host(`your_domain_name.com`)  # Your domain name
         traefik.http.services.vllm-service.loadbalancer.server.port: 8000  # Port of vLLM server

     # 1
     vllm-server-1:
       # ... Configuration of vLLM server
       image: vllm/vllm-openai:latest
       command: --model mistralai/Mistral-7B-v0.1
       volumes:
         - ./huggingface_cache:/root/.cache/huggingface:rw

       # Allocate GPU 1
       deploy:
         resources:
           reservations:
             devices:
               - driver: nvidia
                 capabilities: [gpu]
                 device_ids: ['1']

       # Traefik load balancer config
       labels:
         traefik.http.routers.vllm.rule: Host(`your_domain_name.com`)  # Your domain name
         traefik.http.services.vllm-service.loadbalancer.server.port: 8000  # Port of vLLM server

     # ...

For production deployments, enabling the HTTPS feature on the load balancer is **strongly recommended** for secure communication. An example of this setup is the `Traefik HTTPS documentation <https://doc.traefik.io/traefik/https/overview/>`_.

An illustration of this in a production context is the `OpenChat API and web service <https://openchat.team>`_, which utilizes several `OpenChat servers <https://github.com/imoneoi/openchat>`_ (enhanced vLLM OpenAI API-compatible servers with API key authentication) behind a Traefik load balancer with HTTPS enabled.
