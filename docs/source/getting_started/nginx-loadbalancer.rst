.. _nginxloadbalancer:

Nginx Loadbalancer
========================

.. _nginxloadbalancer_nginx_build:

Build Nginx Container
------------

Create a file named `Dockerfile.nginx`:

.. code-block:: console

    # Copyright (C) 2024 Intel Corporation
    # SPDX-License-Identifier: Apache-2.0

    # FROM nginx

    # RUN rm /etc/nginx/conf.d/default.conf
    # COPY nginx.conf /etc/nginx/conf.d/default.conf


    FROM nginx:latest
    RUN rm /etc/nginx/conf.d/default.conf
    EXPOSE 80
    CMD ["nginx", "-g", "daemon off;"]

Build the container:

.. code-block:: console

    docker build . -f Dockerfile.nginx --tag nginx-lb   

Build vLLM Container
------------

.. code-block:: console

    cd $vllm_root
    podman build -f Dockerfile.cpu . --tag vllm 

