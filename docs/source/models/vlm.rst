.. _vlm:

Using VLMs
==========

This document shows you how to run and serve Vision Language Models (VLMs) using vLLM.

Engine Arguments
----------------

The following :ref:`engine arguments <engine_args>` are specific to VLMs:

.. argparse::
    :module: vllm.engine.arg_utils
    :func: _vlm_engine_args_parser
    :prog: -m vllm.entrypoints.openai.api_server
    :nodefaultconst:

Offline Batched Inference
-------------------------

To initialize a VLM, the aforementioned arguments must be passed to the ``LLM`` class for instantiating the engine.

.. code-block:: python

    llm = LLM(
        model="llava-hf/llava-1.5-7b-hf",
        image_input_type="pixel_values",
        image_token_id=32000,
        image_input_shape="1,3,336,336",
        image_feature_size=576,
    )

For now, we only support a single image per text prompt when calling ``llm.generate``. To pass an image to the model, note the following parameters:

* ``prompt``: The prompt should have a number of ``<image>`` tokens equal to ``image_feature_size``.
* ``multi_modal_datas``: This should be an instance of ``ImagePixelData``.

.. code-block:: python

    prompt = "<image>" * 576 + (
        "\nUSER: What is the content of this image?\nASSISTANT:")

    # Load the image using PIL.Image
    image = ...

    outputs = llm.generate(prompt, multi_modal_datas=ImagePixelData(image))

    for o in outputs:
        generated_text = o.outputs[0].text
        print(generated_text)

A code example can be found in `examples/llava_example.py <https://github.com/vllm-project/vllm/blob/main/examples/llava_example.py>`_.
