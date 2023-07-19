.. _on_cloud:

Running on clouds with SkyPilot
===============================

.. raw:: html

    <p align="center">
        <img src="https://imgur.com/yxtzPEu.png" alt="vLLM"/>
    </p>

vLLM can be run on the cloud to scale to multiple GPUs with `SkyPilot <https://github.com/skypilot-org/skypilot>`__, an open-source framework for running LLMs on any cloud.

To install SkyPilot and setup your cloud credentials, run:

.. code-block:: console

    $ pip install skypilot
    $ sky check

See the vLLM SkyPilot YAML for serving, `serving.yaml <https://github.com/skypilot-org/skypilot/blob/master/llm/vllm/serve.yaml>`__.

.. code-block:: yaml

    resources:
        accelerators: A100

    envs:
        MODEL_NAME: decapoda-research/llama-13b-hf
        TOKENIZER: hf-internal-testing/llama-tokenizer

    setup: |
        conda create -n vllm python=3.9 -y
        conda activate vllm
        git clone https://github.com/vllm-project/vllm.git
        cd vllm
        pip install .
        pip install gradio

    run: |
        conda activate vllm
        echo 'Starting vllm api server...'
        python -u -m vllm.entrypoints.api_server \
                        --model $MODEL_NAME \
                        --tensor-parallel-size $SKYPILOT_NUM_GPUS_PER_NODE \
                        --tokenizer $TOKENIZER 2>&1 | tee api_server.log &
        echo 'Waiting for vllm api server to start...'
        while ! `cat api_server.log | grep -q 'Uvicorn running on'`; do sleep 1; done
        echo 'Starting gradio server...'
        python vllm/examples/gradio_webserver.py

Start the serving the LLaMA-13B model on an A100 GPU:

.. code-block:: console

    $ sky launch serving.yaml

Check the output of the command. There will be a sharable gradio link (like the last line of the following). Open it in your browser to use the LLaMA model to do the text completion.

.. code-block:: console

    (task, pid=7431) Running on public URL: https://<gradio-hash>.gradio.live

**Optional**: Serve the 65B model instead of the default 13B and use more GPU:

.. code-block:: console

    sky launch -c vllm-serve-new -s serve.yaml --gpus A100:8 --env MODEL_NAME=decapoda-research/llama-65b-hf

