.. _run_on_llamastack:

Serving with Llama Stack
============================

vLLM is also available via `Llama Stack <https://github.com/meta-llama/llama-stack>`_ .

To install Llama Stack, run

.. code-block:: console

    $ pip install llama-stack -q

Then start Llama Stack server pointing to your vLLM server with the following configuration:

.. code-block:: yaml

    inference:
      - provider_id: vllm0
        provider_type: remote::vllm
        config:
          url: http://127.0.0.1:8000

Please refer to `this guide <https://github.com/meta-llama/llama-stack/blob/main/docs/source/getting_started/distributions/self_hosted_distro/remote_vllm.md>`_ for more details.
