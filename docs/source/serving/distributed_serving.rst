.. _distributed_serving:

Distributed Inference and Serving
=================================

vLLM supports distributed tensor-parallel inference and serving. Currently, we support `Megatron-LM's tensor parallel algorithm <https://arxiv.org/pdf/1909.08053.pdf>`_. We manage the distributed runtime with `Ray <https://github.com/ray-project/ray>`_. To run distributed inference, install Ray with:

.. code-block:: console

    $ pip install ray

To run multi-GPU inference with the :code:`LLM` class, set the :code:`tensor_parallel_size` argument to the number of GPUs you want to use. For example, to run inference on 4 GPUs:

.. code-block:: python

    from vllm import LLM
    llm = LLM("facebook/opt-13b", tensor_parallel_size=4)
    output = llm.generate("San Franciso is a")

To run multi-GPU serving, pass in the :code:`--tensor-parallel-size` argument when starting the server. For example, to run API server on 4 GPUs:

.. code-block:: console

    $ python -m vllm.entrypoints.api_server \
    $     --model facebook/opt-13b \
    $     --tensor-parallel-size 4

To scale vLLM beyond a single machine, start a `Ray runtime <https://docs.ray.io/en/latest/ray-core/starting-ray.html>`_ via CLI before running vLLM:

.. code-block:: console

    $ # On head node
    $ ray start --head

    $ # On worker nodes
    $ ray start --address=<ray-head-address>

After that, you can run inference and serving on multiple machines by launching the vLLM process on the head node by setting :code:`tensor_parallel_size` to the number of GPUs to be the total number of GPUs across all machines.