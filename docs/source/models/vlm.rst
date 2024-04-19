.. _vlm:

Using VLMs
==========

This document shows you how to run and serve Vision Language Models (VLMs) using vLLM.

Additional Engine Arguments
---------------------------

Apart from the :ref:`basic engine arguments <engine_args>`, VLMs additionally require the following engine arguments for vLLM.

.. option:: --image-input-type {pixel_values,image_features}

    The image input type passed into vLLM. Should be one of "pixel_values" or "image_features".

.. option:: --image-token-id <id>

    Input ID for image token.

.. option:: --image-input-shape <tuple>

    The biggest image input shape (worst for memory footprint) given an input type. Only used for vLLM's profile_run.

    For example, if the image tensor has shape :code:`(1, 3, 336, 336)`, then you should pass :code:`--image-input-shape 1,3,336,336`.

.. option:: --image-feature-size <size>

    The image feature size along the context dimension.

.. option:: --image-processor <size>

    Name or path of the huggingface image processor to use.

.. option:: --image-processor-revision <revision>

    The specific image processor version to use. It can be a branch name, a tag name, or a commit id. If unspecified, will use the default version.

.. option:: --no-image-processor

    Disables the use of image processor, even if one is defined for the model on huggingface.

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
