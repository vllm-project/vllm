.. _adding_a_new_model:

Adding a New Model
==================

This document provides a high-level guide on the process of adding a new model into CacheFlow.

.. note::
    The complexity of adding a new model varies based on the model's architecture.
    If the model shares the similar architecture with an already existing one in CacheFlow, the process is more straightforward.
    However, if the model architecture includes new operators (e.g., a new attention mechanism), the process can be more challenging.

.. tip::
    If you are having trouble integrating your model into CacheFlow, we encourage you to open an issue on our `GitHub <https://github.com/WoosukKwon/cacheflow/issues>`_ repository.
    We will be happy to help you out!


0. Fork the CacheFlow repository
--------------------------------

The first step is to fork our `GitHub <https://github.com/WoosukKwon/cacheflow/issues>`_ repository and :ref:`build it from source <build_from_source>`.
This will allow you to make changes to the codebase and test your model.


1. Bring your model code
------------------------

Copy the PyTorch model code from the `HuggingFace Transformers <https://github.com/huggingface/transformers>`_ repository and put it into the `cacheflow/model_executor/models <https://github.com/WoosukKwon/cacheflow/tree/main/cacheflow/model_executor/models>`_ directory.
For example, you can use the code from the HuggingFace's `modeling_llama.py <https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py>`_ file for LLaMA models.

.. warning::
    In copying the model code, make sure to review and adhere to the code's copyright and licensing terms.


2. Rewrite the :code:`forward` methods
--------------------------------------

The next step is to rewrite the :code:`forward` methods of your model by following the steps below:

1. Prune out unnecessary code. For example, you can remove the code only used for training.
2. Change the input parameters:

.. code-block:: diff

    def forward(
        self,
        input_ids: torch.Tensor,
    -    attention_mask: Optional[torch.Tensor] = None,
    -    position_ids: Optional[torch.LongTensor] = None,
    -    past_key_values: Optional[List[torch.FloatTensor]] = None,
    -    inputs_embeds: Optional[torch.FloatTensor] = None,
    -    labels: Optional[torch.LongTensor] = None,
    -    use_cache: Optional[bool] = None,
    -    output_attentions: Optional[bool] = None,
    -    output_hidden_states: Optional[bool] = None,
    -    return_dict: Optional[bool] = None,
    -) -> Union[Tuple, CausalLMOutputWithPast]:
    +    positions: torch.Tensor,
    +    kv_caches: List[KVCache],
    +    input_metadata: InputMetadata,
    +    cache_events: Optional[List[torch.cuda.Event]],
    +) -> Dict[int, SequenceOutputs]:

3. Fix the code by considering that :code:`input_ids` and :code:`positions` are now flattened tensors.
4. Replace the attention operation with either :code:`GPTCacheFlowAttention` or :code:`GPTNeoXCacheFlowAttention` depending on the model's architecture.

.. note::
    As of now, CacheFlow supports the vanilla multi-head attention mechanism and its variant with rotary positional embeddings.
    If your model uses a different attention mechanism, you need to implement a new attention layer in CacheFlow.


3. (Optional) Add tensor parallelism support
--------------------------------------------

If your model is too large to fit into a single GPU, you can add tensor parallelism support to your model.
To do so, you need to replace your model's linear and embedding layers with their tensor-parallel counterparts.
For the embedding layer, you can simply replace :code:`nn.Embedding` with :code:`VocabParallelEmbedding`.
For the linear layers, you need to replace them with either :code:`RowParallelLinear` or :code:`ColumnParallelLinear`.
Typically, we use :code:`ColumnParallelLinear` for QKV linear layers and the first linear layers of the MLP blocks.
We use :code:`RowParallelLinear` for the other linear layers.

4. Implement the weight loading logic
-------------------------------------

The next step is to implement :code:`load_weights` method in your :code:`*ForCausalLM` class.
This method should load the weights from the HuggingFace's checkpoint file and set them to the corresponding layers in your model.
While the process is straightforward for most layers, the tensor-parallel layers require some additional care as you need to split the weights across multiple GPUs.

5. Register your model
----------------------

Finally, add your :code:`*ForCausalLM` class to `cacheflow/model_executor/models/__init__.py <https://github.com/WoosukKwon/cacheflow/blob/main/cacheflow/model_executor/models/__init__.py>`_ and register it in :code:`_MODEL_REGISTRY` in the `cacheflow/model_executor/model_loader.py <https://github.com/WoosukKwon/cacheflow/blob/main/cacheflow/model_executor/model_loader.py>`_ file.
