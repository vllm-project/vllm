.. _on_cloud:

Deploying and scaling up with SkyPilot
================================================

.. raw:: html

  <p align="center">
    <img src="https://imgur.com/yxtzPEu.png" alt="vLLM"/>
  </p>

vLLM can be **run and scaled to multiple service replicas on clouds and Kubernetes** with `SkyPilot <https://github.com/skypilot-org/skypilot>`__, an open-source framework for running LLMs on any cloud. More examples for various open models, such as Llama-3, Mixtral, etc, can be found in `SkyPilot AI gallery <https://skypilot.readthedocs.io/en/latest/gallery/index.html>`__.


Prerequisites
-------------

- Go to the `HuggingFace model page <https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct>`__ and request access to the model :code:`meta-llama/Meta-Llama-3-8B-Instruct`.
- Check that you have installed SkyPilot (`docs <https://skypilot.readthedocs.io/en/latest/getting-started/installation.html>`__).
- Check that :code:`sky check` shows clouds or Kubernetes are enabled.

.. code-block:: console

  pip install skypilot-nightly
  sky check


Run on a single instance
------------------------

See the vLLM SkyPilot YAML for serving, `serving.yaml <https://github.com/skypilot-org/skypilot/blob/master/llm/vllm/serve.yaml>`__.

.. code-block:: yaml

  resources:
    accelerators: {L4, A10g, A10, L40, A40, A100, A100-80GB} # We can use cheaper accelerators for 8B model.
    use_spot: True
    disk_size: 512  # Ensure model checkpoints can fit.
    disk_tier: best
    ports: 8081  # Expose to internet traffic.

  envs:
    MODEL_NAME: meta-llama/Meta-Llama-3-8B-Instruct
    HF_TOKEN: <your-huggingface-token>  # Change to your own huggingface token, or use --env to pass.

  setup: |
    conda create -n vllm python=3.10 -y
    conda activate vllm

    pip install vllm==0.4.0.post1
    # Install Gradio for web UI.
    pip install gradio openai
    pip install flash-attn==2.5.7

  run: |
    conda activate vllm
    echo 'Starting vllm api server...'
    python -u -m vllm.entrypoints.openai.api_server \
      --port 8081 \
      --model $MODEL_NAME \
      --trust-remote-code \
      --tensor-parallel-size $SKYPILOT_NUM_GPUS_PER_NODE \
      2>&1 | tee api_server.log &
    
    echo 'Waiting for vllm api server to start...'
    while ! `cat api_server.log | grep -q 'Uvicorn running on'`; do sleep 1; done

    echo 'Starting gradio server...'
    git clone https://github.com/vllm-project/vllm.git || true
    python vllm/examples/gradio_openai_chatbot_webserver.py \
      -m $MODEL_NAME \
      --port 8811 \
      --model-url http://localhost:8081/v1 \
      --stop-token-ids 128009,128001

Start the serving the Llama-3 8B model on any of the candidate GPUs listed (L4, A10g, ...): 

.. code-block:: console

  HF_TOKEN="your-huggingface-token" sky launch serving.yaml --env HF_TOKEN

Check the output of the command. There will be a shareable gradio link (like the last line of the following). Open it in your browser to use the LLaMA model to do the text completion.

.. code-block:: console

  (task, pid=7431) Running on public URL: https://<gradio-hash>.gradio.live

**Optional**: Serve the 70B model instead of the default 8B and use more GPU:

.. code-block:: console

  HF_TOKEN="your-huggingface-token" sky launch serving.yaml --gpus A100:8 --env HF_TOKEN --env MODEL_NAME=meta-llama/Meta-Llama-3-70B-Instruct


Scale up to multiple replicas
-----------------------------

SkyPilot can scale up the service to multiple service replicas with built-in autoscaling, load-balancing and fault-tolerance. You can do it by adding a services section to the YAML file.

.. code-block:: yaml

  service:
    replicas: 2
    # An actual request for readiness probe.
    readiness_probe:
      path: /v1/chat/completions
      post_data:
      model: $MODEL_NAME
      messages:
        - role: user
          content: Hello! What is your name?
    max_completion_tokens: 1
    
.. raw:: html

  <details>
  <summary>Click to see the full recipe YAML</summary>


.. code-block:: yaml

  service:
    replicas: 2
    # An actual request for readiness probe.
    readiness_probe:
      path: /v1/chat/completions
      post_data:
        model: $MODEL_NAME
        messages:
          - role: user
            content: Hello! What is your name?
        max_completion_tokens: 1

  resources:
    accelerators: {L4, A10g, A10, L40, A40, A100, A100-80GB} # We can use cheaper accelerators for 8B model.
    use_spot: True
    disk_size: 512  # Ensure model checkpoints can fit.
    disk_tier: best
    ports: 8081  # Expose to internet traffic.

  envs:
    MODEL_NAME: meta-llama/Meta-Llama-3-8B-Instruct
    HF_TOKEN: <your-huggingface-token>  # Change to your own huggingface token, or use --env to pass.

  setup: |
    conda create -n vllm python=3.10 -y
    conda activate vllm

    pip install vllm==0.4.0.post1
    # Install Gradio for web UI.
    pip install gradio openai
    pip install flash-attn==2.5.7

  run: |
    conda activate vllm
    echo 'Starting vllm api server...'
    python -u -m vllm.entrypoints.openai.api_server \
      --port 8081 \
      --model $MODEL_NAME \
      --trust-remote-code \
      --tensor-parallel-size $SKYPILOT_NUM_GPUS_PER_NODE \
      2>&1 | tee api_server.log

.. raw:: html

  </details>

Start the serving the Llama-3 8B model on multiple replicas:

.. code-block:: console

  HF_TOKEN="your-huggingface-token" sky serve up -n vllm serving.yaml --env HF_TOKEN


Wait until the service is ready:

.. code-block:: console

  watch -n10 sky serve status vllm


.. raw:: html

  <details>
  <summary>Example outputs:</summary>

.. code-block:: console

  Services
  NAME  VERSION  UPTIME  STATUS  REPLICAS  ENDPOINT
  vllm  1        35s     READY   2/2       xx.yy.zz.100:30001

  Service Replicas
  SERVICE_NAME  ID  VERSION  IP            LAUNCHED     RESOURCES                STATUS  REGION
  vllm          1   1        xx.yy.zz.121  18 mins ago  1x GCP([Spot]{'L4': 1})  READY   us-east4
  vllm          2   1        xx.yy.zz.245  18 mins ago  1x GCP([Spot]{'L4': 1})  READY   us-east4

.. raw:: html
  
  </details>

After the service is READY, you can find a single endpoint for the service and access the service with the endpoint:

.. code-block:: console

  ENDPOINT=$(sky serve status --endpoint 8081 vllm)
  curl -L http://$ENDPOINT/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
      "model": "meta-llama/Meta-Llama-3-8B-Instruct",
      "messages": [
      {
        "role": "system",
        "content": "You are a helpful assistant."
      },
      {
        "role": "user",
        "content": "Who are you?"
      }
      ],
      "stop_token_ids": [128009,  128001]
    }'

To enable autoscaling, you could replace the `replicas` with the following configs in `service`:

.. code-block:: yaml

  service:
    replica_policy:
      min_replicas: 2
      max_replicas: 4
      target_qps_per_replica: 2

This will scale the service up to when the QPS exceeds 2 for each replica.

    
.. raw:: html

  <details>
  <summary>Click to see the full recipe YAML</summary>


.. code-block:: yaml

  service:
    replica_policy:
      min_replicas: 2
      max_replicas: 4
      target_qps_per_replica: 2
    # An actual request for readiness probe.
    readiness_probe:
      path: /v1/chat/completions
      post_data:
        model: $MODEL_NAME
        messages:
          - role: user
            content: Hello! What is your name?
        max_completion_tokens: 1

  resources:
    accelerators: {L4, A10g, A10, L40, A40, A100, A100-80GB} # We can use cheaper accelerators for 8B model.
    use_spot: True
    disk_size: 512  # Ensure model checkpoints can fit.
    disk_tier: best
    ports: 8081  # Expose to internet traffic.

  envs:
    MODEL_NAME: meta-llama/Meta-Llama-3-8B-Instruct
    HF_TOKEN: <your-huggingface-token>  # Change to your own huggingface token, or use --env to pass.

  setup: |
    conda create -n vllm python=3.10 -y
    conda activate vllm

    pip install vllm==0.4.0.post1
    # Install Gradio for web UI.
    pip install gradio openai
    pip install flash-attn==2.5.7

  run: |
    conda activate vllm
    echo 'Starting vllm api server...'
    python -u -m vllm.entrypoints.openai.api_server \
      --port 8081 \
      --model $MODEL_NAME \
      --trust-remote-code \
      --tensor-parallel-size $SKYPILOT_NUM_GPUS_PER_NODE \
      2>&1 | tee api_server.log


.. raw:: html
  
  </details>

To update the service with the new config:

.. code-block:: console

  HF_TOKEN="your-huggingface-token" sky serve update vllm serving.yaml --env HF_TOKEN


To stop the service:

.. code-block:: console

  sky serve down vllm


**Optional**: Connect a GUI to the endpoint
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


It is also possible to access the Llama-3 service with a separate GUI frontend, so the user requests send to the GUI will be load-balanced across replicas.

.. raw:: html

  <details>
  <summary>Click to see the full GUI YAML</summary>

.. code-block:: yaml

  envs:
    MODEL_NAME: meta-llama/Meta-Llama-3-8B-Instruct
    ENDPOINT: x.x.x.x:3031 # Address of the API server running vllm. 

  resources:
    cpus: 2

  setup: |
    conda create -n vllm python=3.10 -y
    conda activate vllm

    # Install Gradio for web UI.
    pip install gradio openai

  run: |
    conda activate vllm
    export PATH=$PATH:/sbin

    echo 'Starting gradio server...'
    git clone https://github.com/vllm-project/vllm.git || true
    python vllm/examples/gradio_openai_chatbot_webserver.py \
      -m $MODEL_NAME \
      --port 8811 \
      --model-url http://$ENDPOINT/v1 \
      --stop-token-ids 128009,128001 | tee ~/gradio.log


.. raw:: html
  
  </details>

1. Start the chat web UI:

.. code-block:: console

  sky launch -c gui ./gui.yaml --env ENDPOINT=$(sky serve status --endpoint vllm)


2. Then, we can access the GUI at the returned gradio link:

.. code-block:: console

  | INFO | stdout | Running on public URL: https://6141e84201ce0bb4ed.gradio.live


