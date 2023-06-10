Quickstart
==========

CacheFlow is a high-throughput and memory-efficient inference server for large language models (LLM). This quickstart guide will show you how to use CacheFlow for different LLM application scenarios. To run the examples in this guide, you will need to install CacheFlow following the `installation guide <installation.html>`_.

Offline Batched Inference
-------------------------

We first show an example to use CacheFlow for offline batched inference. In this example, we will use CacheFlow to generate texts for a list of input prompts. This can be used to apply LLM on a large dataset. The code example can also be found in `examples/offline_inference.py <https://github.com/WoosukKwon/cacheflow/blob/main/examples/offline_inference.py>`_.

First, we import the ``LLM`` and ``SamplingParams`` classes from CacheFlow. ``LLM`` class is the main class for running offline inference jobs with CacheFlow server. ``SamplingParams`` class is used to specify the parameters for the sampling process.

.. code-block:: python

    from cacheflow import LLM, SamplingParams

We perform inference on the following list of input prompts. We use a ``SamplingParams`` object to specify the sampling parameters for the generation process. In this example, we set the sampling temperature to 0.8 and nucleus sampling probability to 0.95. For more information about the sampling parameters, please refer to the `class definition <https://github.com/WoosukKwon/cacheflow/blob/main/cacheflow/sampling_params.py>`_.

.. code-block:: python

    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

Next, we initialize the LLM server for offline inference with the ``LLM`` class. We use the `OPT model <https://arxiv.org/abs/2205.01068>`_ with 125 million parameters as an example. The list of supported models can be found in the `supported models page </models/supported_models.html>`_.

.. code-block:: python

    llm = LLM(model="facebook/opt-125m")

We call ``llm.generate`` to generate the outputs with the given input prompts and sampling parameters. It adds the input prompts to CacheFlow server's waiting queue. Then, the CacheFlow server generates the outputs for the input prompts and returns the outputs with high throughput. The outputs are returned as a list of ``RequestOutput`` objects, which includes the output tokens and log probabilites of each output.

.. code-block:: python

    outputs = llm.generate(prompts, sampling_params)

    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


Simple FastAPI Server
---------------------


OpenAI-Compatible Server
------------------------