.. _multimodal_inputs:

Multimodal Inputs
=================

This page teaches you how to pass multi-modal inputs to :ref:`multi-modal models <supported_mm_models>` in vLLM.

.. note::
    We are actively iterating on multi-modal support. See `this RFC <https://github.com/vllm-project/vllm/issues/4194>`_ for upcoming changes,
    and `open an issue on GitHub <https://github.com/vllm-project/vllm/issues/new/choose>`_ if you have any feedback or feature requests.

Offline Inference
-----------------

To input multi-modal data, follow this schema in :class:`vllm.inputs.PromptType`:

* ``prompt``: The prompt should follow the format that is documented on HuggingFace.
* ``multi_modal_data``: This is a dictionary that follows the schema defined in :class:`vllm.multimodal.MultiModalDataDict`.

Image
^^^^^

You can pass a single image to the :code:`'image'` field of the multi-modal dictionary, as shown in the following examples:

.. code-block:: python

    llm = LLM(model="llava-hf/llava-1.5-7b-hf")

    # Refer to the HuggingFace repo for the correct format to use
    prompt = "USER: <image>\nWhat is the content of this image?\nASSISTANT:"

    # Load the image using PIL.Image
    image = PIL.Image.open(...)

    # Single prompt inference
    outputs = llm.generate({
        "prompt": prompt,
        "multi_modal_data": {"image": image},
    })

    for o in outputs:
        generated_text = o.outputs[0].text
        print(generated_text)

    # Batch inference
    image_1 = PIL.Image.open(...)
    image_2 = PIL.Image.open(...)
    outputs = llm.generate(
        [
            {
                "prompt": "USER: <image>\nWhat is the content of this image?\nASSISTANT:",
                "multi_modal_data": {"image": image_1},
            },
            {
                "prompt": "USER: <image>\nWhat's the color of this image?\nASSISTANT:",
                "multi_modal_data": {"image": image_2},
            }
        ]
    )

    for o in outputs:
        generated_text = o.outputs[0].text
        print(generated_text)

A code example can be found in `examples/offline_inference_vision_language.py <https://github.com/vllm-project/vllm/blob/main/examples/offline_inference_vision_language.py>`_.

To substitute multiple images inside the same text prompt, you can pass in a list of images instead:

.. code-block:: python

    llm = LLM(
        model="microsoft/Phi-3.5-vision-instruct",
        trust_remote_code=True,  # Required to load Phi-3.5-vision
        max_model_len=4096,  # Otherwise, it may not fit in smaller GPUs
        limit_mm_per_prompt={"image": 2},  # The maximum number to accept
    )

    # Refer to the HuggingFace repo for the correct format to use
    prompt = "<|user|>\n<|image_1|>\n<|image_2|>\nWhat is the content of each image?<|end|>\n<|assistant|>\n"

    # Load the images using PIL.Image
    image1 = PIL.Image.open(...)
    image2 = PIL.Image.open(...)

    outputs = llm.generate({
        "prompt": prompt,
        "multi_modal_data": {
            "image": [image1, image2]
        },
    })

    for o in outputs:
        generated_text = o.outputs[0].text
        print(generated_text)

A code example can be found in `examples/offline_inference_vision_language_multi_image.py <https://github.com/vllm-project/vllm/blob/main/examples/offline_inference_vision_language_multi_image.py>`_.

Multi-image input can be extended to perform video captioning. We show this with `Qwen2-VL <https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct>`_ as it supports videos:

.. code-block:: python

    # Specify the maximum number of frames per video to be 4. This can be changed.
    llm = LLM("Qwen/Qwen2-VL-2B-Instruct", limit_mm_per_prompt={"image": 4})

    # Create the request payload.
    video_frames = ... # load your video making sure it only has the number of frames specified earlier.
    message = {
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe this set of frames. Consider the frames to be a part of the same video."},
        ],
    }
    for i in range(len(video_frames)):
        base64_image = encode_image(video_frames[i]) # base64 encoding.
        new_image = {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
        message["content"].append(new_image)

    # Perform inference and log output.
    outputs = llm.chat([message])

    for o in outputs:
        generated_text = o.outputs[0].text
        print(generated_text)

Video
^^^^^

You can pass a list of NumPy arrays directly to the :code:`'video'` field of the multi-modal dictionary
instead of using multi-image input.

Please refer to `examples/offline_inference_vision_language.py <https://github.com/vllm-project/vllm/blob/main/examples/offline_inference_vision_language.py>`_ for more details.

Audio
^^^^^

You can pass a tuple :code:`(array, sampling_rate)` to the :code:`'audio'` field of the multi-modal dictionary.

Please refer to `examples/offline_inference_audio_language.py <https://github.com/vllm-project/vllm/blob/main/examples/offline_inference_audio_language.py>`_ for more details.

Embedding
^^^^^^^^^

To input pre-computed embeddings belonging to a data type (i.e. image, video, or audio) directly to the language model,
pass a tensor of shape :code:`(num_items, feature_size, hidden_size of LM)` to the corresponding field of the multi-modal dictionary.

.. code-block:: python

    # Inference with image embeddings as input
    llm = LLM(model="llava-hf/llava-1.5-7b-hf")

    # Refer to the HuggingFace repo for the correct format to use
    prompt = "USER: <image>\nWhat is the content of this image?\nASSISTANT:"

    # Embeddings for single image
    # torch.Tensor of shape (1, image_feature_size, hidden_size of LM)
    image_embeds = torch.load(...)

    outputs = llm.generate({
        "prompt": prompt,
        "multi_modal_data": {"image": image_embeds},
    })

    for o in outputs:
        generated_text = o.outputs[0].text
        print(generated_text)

For Qwen2-VL and MiniCPM-V, we accept additional parameters alongside the embeddings:

.. code-block:: python

    # Construct the prompt based on your model
    prompt = ...

    # Embeddings for multiple images
    # torch.Tensor of shape (num_images, image_feature_size, hidden_size of LM)
    image_embeds = torch.load(...)

    # Qwen2-VL
    llm = LLM("Qwen/Qwen2-VL-2B-Instruct", limit_mm_per_prompt={"image": 4})
    mm_data = {
        "image": {
            "image_embeds": image_embeds,
            # image_grid_thw is needed to calculate positional encoding.
            "image_grid_thw": torch.load(...),  # torch.Tensor of shape (1, 3),
        }
    }

    # MiniCPM-V
    llm = LLM("openbmb/MiniCPM-V-2_6", trust_remote_code=True, limit_mm_per_prompt={"image": 4})
    mm_data = {
        "image": {
            "image_embeds": image_embeds,
            # image_size_list is needed to calculate details of the sliced image.
            "image_size_list": [image.size for image in images],  # list of image sizes
        }
    }

    outputs = llm.generate({
        "prompt": prompt,
        "multi_modal_data": mm_data,
    })

    for o in outputs:
        generated_text = o.outputs[0].text
        print(generated_text)

Online Inference
----------------

Our OpenAI-compatible server accepts multi-modal data via the `Chat Completions API <https://platform.openai.com/docs/api-reference/chat>`_.

.. important::
    A chat template is **required** to use Chat Completions API.

    Although most models come with a chat template, for others you have to define one yourself.
    The chat template can be inferred based on the documentation on the model's HuggingFace repo.
    For example, LLaVA-1.5 (``llava-hf/llava-1.5-7b-hf``) requires a chat template that can be found `here <https://github.com/vllm-project/vllm/blob/main/examples/template_llava.jinja>`__.

Image
^^^^^

Image input is supported according to `OpenAI Vision API <https://platform.openai.com/docs/guides/vision>`_.
Here is a simple example using Phi-3.5-Vision.

First, launch the OpenAI-compatible server:

.. code-block:: bash

    vllm serve microsoft/Phi-3.5-vision-instruct --task generate \
      --trust-remote-code --max-model-len 4096 --limit-mm-per-prompt image=2

Then, you can use the OpenAI client as follows:

.. code-block:: python

    from openai import OpenAI

    openai_api_key = "EMPTY"
    openai_api_base = "http://localhost:8000/v1"

    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    # Single-image input inference
    image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"

    chat_response = client.chat.completions.create(
        model="microsoft/Phi-3.5-vision-instruct",
        messages=[{
            "role": "user",
            "content": [
                # NOTE: The prompt formatting with the image token `<image>` is not needed
                # since the prompt will be processed automatically by the API server.
                {"type": "text", "text": "What’s in this image?"},
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
        }],
    )
    print("Chat completion output:", chat_response.choices[0].message.content)

    # Multi-image input inference
    image_url_duck = "https://upload.wikimedia.org/wikipedia/commons/d/da/2015_Kaczka_krzy%C5%BCowka_w_wodzie_%28samiec%29.jpg"
    image_url_lion = "https://upload.wikimedia.org/wikipedia/commons/7/77/002_The_lion_king_Snyggve_in_the_Serengeti_National_Park_Photo_by_Giles_Laurent.jpg"

    chat_response = client.chat.completions.create(
        model="microsoft/Phi-3.5-vision-instruct",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": "What are the animals in these images?"},
                {"type": "image_url", "image_url": {"url": image_url_duck}},
                {"type": "image_url", "image_url": {"url": image_url_lion}},
            ],
        }],
    )
    print("Chat completion output:", chat_response.choices[0].message.content)

A full code example can be found in `examples/openai_chat_completion_client_for_multimodal.py <https://github.com/vllm-project/vllm/blob/main/examples/openai_chat_completion_client_for_multimodal.py>`_.

.. tip::
    Loading from local file paths is also supported on vLLM: You can specify the allowed local media path via ``--allowed-local-media-path`` when launching the API server/engine,
    and pass the file path as ``url`` in the API request.

.. tip::
    There is no need to place image placeholders in the text content of the API request - they are already represented by the image content.
    In fact, you can place image placeholders in the middle of the text by interleaving text and image content.

.. note::

    By default, the timeout for fetching images through HTTP URL is ``5`` seconds.
    You can override this by setting the environment variable:

    .. code-block:: console

        $ export VLLM_IMAGE_FETCH_TIMEOUT=<timeout>

Video
^^^^^

Instead of :code:`image_url`, you can pass a video file via :code:`video_url`.

You can use `these tests <https://github.com/vllm-project/vllm/blob/main/tests/entrypoints/openai/test_video.py>`_ as reference.

.. note::

    By default, the timeout for fetching videos through HTTP URL url is ``30`` seconds.
    You can override this by setting the environment variable:

    .. code-block:: console

        $ export VLLM_VIDEO_FETCH_TIMEOUT=<timeout>

Audio
^^^^^

Instead of :code:`image_url`, you can pass an audio file via :code:`audio_url`.

A full code example can be found in `examples/openai_chat_completion_client_for_multimodal.py <https://github.com/vllm-project/vllm/blob/main/examples/openai_chat_completion_client_for_multimodal.py>`_.

.. note::

    By default, the timeout for fetching audios through HTTP URL is ``10`` seconds.
    You can override this by setting the environment variable:

    .. code-block:: console

        $ export VLLM_AUDIO_FETCH_TIMEOUT=<timeout>

Embedding
^^^^^^^^^

vLLM's Embeddings API is a superset of OpenAI's `Embeddings API <https://platform.openai.com/docs/api-reference/embeddings>`_,
where a list of chat ``messages`` can be passed instead of batched ``inputs``. This enables multi-modal inputs to be passed to embedding models.

.. tip::
    The schema of ``messages`` is exactly the same as in Chat Completions API.
    You can refer to the above tutorials for more details on how to pass each type of multi-modal data.

Usually, embedding models do not expect chat-based input, so we need to use a custom chat template to format the text and images.
Refer to the examples below for illustration.

Here is an end-to-end example using VLM2Vec. To serve the model:

.. code-block:: bash

    vllm serve TIGER-Lab/VLM2Vec-Full --task embedding \
      --trust-remote-code --max-model-len 4096 --chat-template examples/template_vlm2vec.jinja

.. important::

    Since VLM2Vec has the same model architecture as Phi-3.5-Vision, we have to explicitly pass ``--task embedding``
    to run this model in embedding mode instead of text generation mode.

    The custom chat template is completely different from the original one for this model,
    and can be found `here <https://github.com/vllm-project/vllm/blob/main/examples/template_vlm2vec.jinja>`__.

Since the request schema is not defined by OpenAI client, we post a request to the server using the lower-level ``requests`` library:

.. code-block:: python

    import requests

    image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"

    response = requests.post(
        "http://localhost:8000/v1/embeddings",
        json={
            "model": "TIGER-Lab/VLM2Vec-Full",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": "Represent the given image."},
                ],
            }],
            "encoding_format": "float",
        },
    )
    response.raise_for_status()
    response_json = response.json()
    print("Embedding output:", response_json["data"][0]["embedding"])

Below is another example, this time using the ``MrLight/dse-qwen2-2b-mrl-v1`` model.

.. code-block:: bash

    vllm serve MrLight/dse-qwen2-2b-mrl-v1 --task embedding \
      --trust-remote-code --max-model-len 8192 --chat-template examples/template_dse_qwen2_vl.jinja

.. important::

    Like with VLM2Vec, we have to explicitly pass ``--task embedding``.
    
    Additionally, ``MrLight/dse-qwen2-2b-mrl-v1`` requires an EOS token for embeddings, which is handled
    by `this custom chat template <https://github.com/vllm-project/vllm/blob/main/examples/template_dse_qwen2_vl.jinja>`__.

.. important::

    Also important, ``MrLight/dse-qwen2-2b-mrl-v1`` requires a placeholder image of the minimum image size for text query embeddings. See the full code 
    example below for details.

A full code example can be found in `examples/openai_chat_embedding_client_for_multimodal.py <https://github.com/vllm-project/vllm/blob/main/examples/openai_chat_embedding_client_for_multimodal.py>`_.
