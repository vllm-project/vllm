Environment Variables
========================

vLLM uses the following environment variables to configure the system:

.. warning::
    Please note that ``VLLM_PORT`` and ``VLLM_HOST_IP`` set the port and ip for vLLM's **internal usage**. It is not the port and ip for the API server. If you use ``--host $VLLM_HOST_IP`` and ``--port $VLLM_PORT`` to start the API server, it will not work.

.. literalinclude:: ../../../vllm/envs.py
    :language: python
    :start-after: begin-env-vars-definition
    :end-before: end-env-vars-definition
