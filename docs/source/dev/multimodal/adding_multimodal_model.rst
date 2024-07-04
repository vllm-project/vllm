.. _adding_a_new_multimodal_model:

Adding a New Multimodal Model
=============================

This document provides a high-level guide on integrating a :ref:`multi-modal model <multi_modality>` into vLLM.

.. note::
    The complexity of adding a new model depends heavily on the model's architecture.
    The process is considerably straightforward if the model shares a similar architecture with an existing model in vLLM.
    However, for models that include new operators (e.g., a new attention mechanism), the process can be a bit more complex.

.. tip::
    If you are encountering issues while integrating your model into vLLM, feel free to open an issue on our `GitHub <https://github.com/vllm-project/vllm/issues>`_ repository.
    We will be happy to help you out!


1. Set up the base vLLM model
-----------------------------

As usual, follow :ref:`these steps <adding_a_new_model>` to implement the model in vLLM, but note the following:

- You should additionally implement the :class:`~vllm.model_executor.models.interfaces.SupportsVision` interface.

  .. code-block:: diff

      + from vllm.model_executor.models.interfaces import SupportsVision

      - class YourModelForImage2Seq(nn.Module):
      + class YourModelForImage2Seq(nn.Module, SupportsVision):

  .. note::
      The model class does not have to be named :code:`*ForCausalLM`.
      Check out `the HuggingFace Transformers documentation <https://huggingface.co/docs/transformers/model_doc/auto#multimodal>`__ for some examples.

- While implementing the :meth:`~torch.nn.Module.forward` method, reserve a keyword parameter
  for each input tensor that corresponds to a multi-modal input, as shown in the following example:

  .. code-block:: diff

      def forward(
          self,
          input_ids: torch.Tensor,
          positions: torch.Tensor,
          kv_caches: List[torch.Tensor],
          attn_metadata: AttentionMetadata,
      +   pixel_values: torch.Tensor,
      ) -> SamplerOutput:


2. Register input mappers
-------------------------

For each modality type that the model accepts as input, decorate the model class with :meth:`MULTIMODAL_REGISTRY.register_input_mapper <vllm.multimodal.MultiModalRegistry.register_input_mapper>`.
This decorator accepts a function that maps multi-modal inputs to the keyword arguments you have previously defined in :meth:`~torch.nn.Module.forward`.

.. code-block:: diff

      from vllm.model_executor.models.interfaces import SupportsVision
    + from vllm.multimodal import MULTIMODAL_REGISTRY

    + @MULTIMODAL_REGISTRY.register_image_input_mapper()
      class YourModelForImage2Seq(nn.Module, SupportsVision):

A default mapper is available for each modality in the core vLLM library. This input mapper will be used if you do not provide your own function.

.. seealso::
    :ref:`input_processing_pipeline`


3. Register maximum number of multimodal tokens
----------------------------------------------------------

For each modality type that the model accepts as input, calculate the maximum possible number of tokens
and register it via :meth:`INPUT_REGISTRY.register_dummy_data <vllm.inputs.registry.InputRegistry.register_max_multimodal_tokens>`.

.. code-block:: diff

      from vllm.inputs import INPUT_REGISTRY
      from vllm.model_executor.models.interfaces import SupportsVision
      from vllm.multimodal import MULTIMODAL_REGISTRY

      @MULTIMODAL_REGISTRY.register_image_input_mapper()
    + @MULTIMODAL_REGISTRY.register_max_image_tokens(<your_calculation>)
      @INPUT_REGISTRY.register_dummy_data(<your_dummy_data_factory>)
      class YourModelForImage2Seq(nn.Module, SupportsVision):

Here are some examples:

- Image inputs (static feature size): `LLaVA-1.5 Model <https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/llava.py>`__
- Image inputs (dynamic feature size): `LLaVA-NeXT Model <https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/llava_next.py>`__

.. seealso::
    :ref:`input_processing_pipeline`


4. (Optional) Register dummy data
---------------------------------

During startup, dummy data is passed to the vLLM model to allocate memory. This only consists of text input by default, which may not be applicable to multi-modal models.
In such cases, you can define your own dummy data by registering a factory method via :meth:`INPUT_REGISTRY.register_dummy_data <vllm.inputs.registry.InputRegistry.register_dummy_data>`.

.. code-block:: diff

      from vllm.inputs import INPUT_REGISTRY
      from vllm.model_executor.models.interfaces import SupportsVision
      from vllm.multimodal import MULTIMODAL_REGISTRY

      @MULTIMODAL_REGISTRY.register_image_input_mapper()
      @MULTIMODAL_REGISTRY.register_max_image_tokens(<your_calculation>)
    + @INPUT_REGISTRY.register_dummy_data(<your_dummy_data_factory>)
      class YourModelForImage2Seq(nn.Module, SupportsVision):

.. note::
    The dummy data should have the maximum possible number of multi-modal tokens, as described in the previous step.

Here are some examples:

- Image inputs (static feature size): `LLaVA-1.5 Model <https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/llava.py>`__
- Image inputs (dynamic feature size): `LLaVA-NeXT Model <https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/llava_next.py>`__

.. seealso::
    :ref:`input_processing_pipeline`


5. (Optional) Register input processor
--------------------------------------

Sometimes, there is a need to process inputs at the :class:`~vllm.LLMEngine` level before they are passed to the model executor. 
This is often due to the fact that unlike implementations in HuggingFace Transformers, the reshaping and/or expansion of multi-modal embeddings needs to take place outside model's :meth:`~torch.nn.Module.forward` call.
You can register input processors via :meth:`INPUT_REGISTRY.register_input_processor <vllm.inputs.registry.InputRegistry.register_input_processor>`.

.. code-block:: diff

      from vllm.inputs import INPUT_REGISTRY
      from vllm.model_executor.models.interfaces import SupportsVision
      from vllm.multimodal import MULTIMODAL_REGISTRY

      @MULTIMODAL_REGISTRY.register_image_input_mapper()
      @MULTIMODAL_REGISTRY.register_max_image_tokens(<your_calculation>)
      @INPUT_REGISTRY.register_dummy_data(<your_dummy_data_factory>)
    + @INPUT_REGISTRY.register_input_processor(<your_input_processor>)
      class YourModelForImage2Seq(nn.Module, SupportsVision):

A common use case of input processors is inserting placeholder tokens to leverage the vLLM framework for attention mask generation.
Here are some examples:

- Insert static number of image tokens: `LLaVA-1.5 Model <https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/llava.py>`__
- Insert dynamic number of image tokens: `LLaVA-NeXT Model <https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/llava_next.py>`__

.. seealso::
    :ref:`input_processing_pipeline`
