.. _adding_a_new_model:

Adding a New Model
==================

This document provides a high-level guide on integrating a `HuggingFace Transformers <https://github.com/huggingface/transformers>`_ model into vLLM.

.. note::
    The complexity of adding a new model depends heavily on the model's architecture.
    The process is considerably straightforward if the model shares a similar architecture with an existing model in vLLM.
    However, for models that include new operators (e.g., a new attention mechanism), the process can be a bit more complex.

.. tip::
    If you are encountering issues while integrating your model into vLLM, feel free to open an issue on our `GitHub <https://github.com/vllm-project/vllm/issues>`_ repository.
    We will be happy to help you out!


0. Fork the vLLM repository
--------------------------------

Start by forking our `GitHub`_ repository and then :ref:`build it from source <build_from_source>`.
This gives you the ability to modify the codebase and test your model.


1. Bring your model code
------------------------

Clone the PyTorch model code from the HuggingFace Transformers repository and put it into the `vllm/model_executor/models <https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/models>`_ directory.
For instance, vLLM's `OPT model <https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/opt.py>`_ was adapted from the HuggingFace's `modeling_opt.py <https://github.com/huggingface/transformers/blob/main/src/transformers/models/opt/modeling_opt.py>`_ file.

.. warning::
    When copying the model code, make sure to review and adhere to the code's copyright and licensing terms.


2. Rewrite the :code:`forward` methods
--------------------------------------

Next, you need to rewrite the :code:`forward` methods of your model by following these steps:

1. Remove any unnecessary code, such as the code only used for training.
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
    +) -> SamplerOutput:

3. Update the code by considering that :code:`input_ids` and :code:`positions` are now flattened tensors.
4. Replace the attention operation with either :code:`PagedAttention`, :code:`PagedAttentionWithRoPE`, or :code:`PagedAttentionWithALiBi` depending on the model's architecture.

.. note::
    Currently, vLLM supports the basic multi-head attention mechanism and its variant with rotary positional embeddings.
    If your model employs a different attention mechanism, you will need to implement a new attention layer in vLLM.


3. (Optional) Implement tensor parallelism and quantization support
-------------------------------------------------------------------

If your model is too large to fit into a single GPU, you can use tensor parallelism to manage it.
To do this, substitute your model's linear and embedding layers with their tensor-parallel versions.
For the embedding layer, you can simply replace :code:`nn.Embedding` with :code:`VocabParallelEmbedding`. For the output LM head, you can use :code:`ParallelLMHead`.
When it comes to the linear layers, we provide the following options to parallelize them:

* :code:`ReplicatedLinear`: Replicates the inputs and weights across multiple GPUs. No memory saving.
* :code:`RowParallelLinear`: The input tensor is partitioned along the hidden dimension. The weight matrix is partitioned along the rows (input dimension). An *all-reduce* operation is performed after the matrix multiplication to reduce the results. Typically used for the second FFN layer and the output linear transformation of the attention layer.
* :code:`ColumnParallelLinear`: The input tensor is replicated. The weight matrix is partitioned along the columns (output dimension). The result is partitioned along the column dimension. Typically used for the first FFN layer and the separated QKV transformation of the attention layer in the original Transformer.
* :code:`MergedColumnParallelLinear`: Column-parallel linear that merges multiple `ColumnParallelLinear` operators. Typically used for the first FFN layer with weighted activation functions (e.g., SiLU). This class handles the sharded weight loading logic of multiple weight matrices.
* :code:`QKVParallelLinear`: Parallel linear layer for the query, key, and value projections of the multi-head and grouped-query attention mechanisms. When number of key/value heads are less than the world size, this class replicates the key/value heads properly. This class handles the weight loading and replication of the weight matrices.

Note that all the linear layers above take `linear_method` as an input. vLLM will set this parameter according to different quantization schemes to support weight quantization.

4. Implement the weight loading logic
-------------------------------------

You now need to implement the :code:`load_weights` method in your :code:`*ForCausalLM` class.
This method should load the weights from the HuggingFace's checkpoint file and assign them to the corresponding layers in your model. Specifically, for `MergedColumnParallelLinear` and `QKVParallelLinear` layers, if the original model has separated weight matrices, you need to load the different parts separately.

5. Register your model
----------------------

Finally, include your :code:`*ForCausalLM` class in `vllm/model_executor/models/__init__.py <https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/__init__.py>`_ and register it to the :code:`_MODEL_REGISTRY` in `vllm/model_executor/model_loader.py <https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/model_loader.py>`_.
