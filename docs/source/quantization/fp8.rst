.. _fp8:

FP8
==================

vLLM supports FP8 (8-bit floating point) computation using hardware acceleration on GPUs such as Nvidia H100 and AMD MI300x. Currently, only Hopper and Ada Lovelace GPUs are supported. Quantization of models with FP8 allows for a 2x reduction in model memory requirements and up to a 1.6x improvement in throughput with minimal impact on accuracy.

Please visit the HF collection of `quantized FP8 checkpoints of popular LLMs ready to use with vLLM <https://huggingface.co/collections/neuralmagic/fp8-llms-for-vllm-666742ed2b78b7ac8df13127>`_.

The FP8 types typically supported in hardware have two distinct representations, each useful in different scenarios:

- **E4M3**: Consists of 1 sign bit, 4 exponent bits, and 3 bits of mantissa. It can store values up to +/-448 and ``nan``.
- **E5M2**: Consists of 1 sign bit, 5 exponent bits, and 2 bits of mantissa. It can store values up to +/-57344, +/- ``inf``, and ``nan``. The tradeoff for the increased dynamic range is lower precision of the stored values.

Quick Start with Online Dynamic Quantization
--------------------------------------------

Dynamic quantization of an original precision BF16/FP16 model to FP8 can be achieved with vLLM without any calibration data required. You can enable the feature by specifying ``--quantization="fp8"`` in the command line or setting ``quantization="fp8"`` in the LLM constructor.

In this mode, all Linear modules (except for the final ``lm_head``) have their weights quantized down to FP8_E4M3 precision with a per-tensor scale. Activations have their minimum and maximum values calculated during each forward pass to provide a dynamic per-tensor scale for high accuracy. As a result, latency improvements are limited in this mode.

.. code-block:: python

    from vllm import LLM
    model = LLM("facebook/opt-125m", quantization="fp8")
    # INFO 06-10 17:55:42 model_runner.py:157] Loading model weights took 0.1550 GB
    result = model.generate("Hello, my name is")

.. warning::

    Currently, we load the model at original precision before quantizing down to 8-bits, so you need enough memory to load the whole model.

Offline Quantization
--------------------

For offline quantization to FP8, please install the `AutoFP8 library <https://github.com/neuralmagic/autofp8>`_.

.. code-block:: bash

    git clone https://github.com/neuralmagic/AutoFP8.git
    pip install -e AutoFP8

This package introduces the ``AutoFP8ForCausalLM`` and ``BaseQuantizeConfig`` objects for managing how your model will be compressed.

Offline Quantization with Dynamic Activation Scaling Factors
------------------------------------------------------------

You can use AutoFP8 to produce checkpoints with their weights quantized to FP8 ahead of time and let vLLM handle calculating dynamic scales for the activations at runtime for maximum accuracy. You can enable this with the ``activation_scheme="dynamic"`` argument.

.. warning::

    Please note that although this mode doesn't give you better performance, it reduces memory footprint compared to online quantization.

.. code-block:: python

    from auto_fp8 import AutoFP8ForCausalLM, BaseQuantizeConfig

    pretrained_model_dir = "meta-llama/Meta-Llama-3-8B-Instruct"
    quantized_model_dir = "Meta-Llama-3-8B-Instruct-FP8-Dynamic"

    # Define quantization config with static activation scales
    quantize_config = BaseQuantizeConfig(quant_method="fp8", activation_scheme="dynamic")
    # For dynamic activation scales, there is no need for calbration examples
    examples = []

    # Load the model, quantize, and save checkpoint
    model = AutoFP8ForCausalLM.from_pretrained(pretrained_model_dir, quantize_config)
    model.quantize(examples)
    model.save_quantized(quantized_model_dir)

In the output of the above script, you should be able to see the quantized Linear modules (FP8DynamicLinear) replaced in the model definition. 
Note that the ``lm_head`` Linear module at the end is currently skipped by default.

.. code-block:: text

    LlamaForCausalLM(
      (model): LlamaModel(
        (embed_tokens): Embedding(128256, 4096)
        (layers): ModuleList(
          (0-31): 32 x LlamaDecoderLayer(
            (self_attn): LlamaSdpaAttention(
              (q_proj): FP8DynamicLinear()
              (k_proj): FP8DynamicLinear()
              (v_proj): FP8DynamicLinear()
              (o_proj): FP8DynamicLinear()
              (rotary_emb): LlamaRotaryEmbedding()
            )
            (mlp): LlamaMLP(
              (gate_proj): FP8DynamicLinear()
              (up_proj): FP8DynamicLinear()
              (down_proj): FP8DynamicLinear()
              (act_fn): SiLU()
            )
            (input_layernorm): LlamaRMSNorm()
            (post_attention_layernorm): LlamaRMSNorm()
          )
        )
        (norm): LlamaRMSNorm()
      )
      (lm_head): Linear(in_features=4096, out_features=128256, bias=False)
    )
    Saving the model to Meta-Llama-3-8B-Instruct-FP8-Dynamic

Your model checkpoint with quantized weights should be available at ``Meta-Llama-3-8B-Instruct-FP8/``.
We can see that the weights are smaller than the original BF16 precision.

.. code-block:: bash

    ls -lh Meta-Llama-3-8B-Instruct-FP8-Dynamic/
    total 8.5G
    -rw-rw-r-- 1 user user  869 Jun  7 14:43 config.json
    -rw-rw-r-- 1 user user  194 Jun  7 14:43 generation_config.json
    -rw-rw-r-- 1 user user 4.7G Jun  7 14:43 model-00001-of-00002.safetensors
    -rw-rw-r-- 1 user user 3.9G Jun  7 14:43 model-00002-of-00002.safetensors
    -rw-rw-r-- 1 user user  43K Jun  7 14:43 model.safetensors.index.json
    -rw-rw-r-- 1 user user  296 Jun  7 14:43 special_tokens_map.json
    -rw-rw-r-- 1 user user  50K Jun  7 14:43 tokenizer_config.json
    -rw-rw-r-- 1 user user 8.7M Jun  7 14:43 tokenizer.json

Finally, you can load the quantized model checkpoint directly in vLLM.

.. code-block:: python

    from vllm import LLM
    model = LLM(model="Meta-Llama-3-8B-Instruct-FP8-Dynamic/")
    # INFO 06-10 21:15:41 model_runner.py:159] Loading model weights took 8.4596 GB
    result = model.generate("Hello, my name is")

Offline Quantization with Static Activation Scaling Factors
-----------------------------------------------------------

For the best inference performance, you can use AutoFP8 with calibration data to produce per-tensor static scales for both the weights and activations by enabling the ``activation_scheme="static"`` argument.

.. code-block:: python

    from datasets import load_dataset
    from transformers import AutoTokenizer
    from auto_fp8 import AutoFP8ForCausalLM, BaseQuantizeConfig

    pretrained_model_dir = "meta-llama/Meta-Llama-3-8B-Instruct"
    quantized_model_dir = "Meta-Llama-3-8B-Instruct-FP8"

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dir, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Load and tokenize 512 dataset samples for calibration of activation scales
    ds = load_dataset("mgoin/ultrachat_2k", split="train_sft").select(range(512))
    examples = [tokenizer.apply_chat_template(batch["messages"], tokenize=False) for batch in ds]
    examples = tokenizer(examples, padding=True, truncation=True, return_tensors="pt").to("cuda")

    # Define quantization config with static activation scales
    quantize_config = BaseQuantizeConfig(quant_method="fp8", activation_scheme="static")

    # Load the model, quantize, and save checkpoint
    model = AutoFP8ForCausalLM.from_pretrained(pretrained_model_dir, quantize_config)
    model.quantize(examples)
    model.save_quantized(quantized_model_dir)

Your model checkpoint with quantized weights and activations should be available at ``Meta-Llama-3-8B-Instruct-FP8/``.
Finally, you can load the quantized model checkpoint directly in vLLM.

.. code-block:: python

    from vllm import LLM
    model = LLM(model="Meta-Llama-3-8B-Instruct-FP8/")
    # INFO 06-10 21:15:41 model_runner.py:159] Loading model weights took 8.4596 GB
    result = model.generate("Hello, my name is")

FP8 checkpoint structure explanation
-----------------------------------------------------------

Here we detail the structure for the FP8 checkpoints.

The following is necessary to be present in the model's ``config.json``:

.. code-block:: text

    "quantization_config": {
        "quant_method": "fp8",
        "activation_scheme": "static" or "dynamic"
    }


Each quantized layer in the state_dict will have these tensors:

* If the config has ``"activation_scheme": "static"``:

.. code-block:: text

    model.layers.0.mlp.down_proj.weight              < F8_E4M3
    model.layers.0.mlp.down_proj.input_scale         < F32
    model.layers.0.mlp.down_proj.weight_scale        < F32

* If the config has ``"activation_scheme": "dynamic"``:

.. code-block:: text

    model.layers.0.mlp.down_proj.weight              < F8_E4M3
    model.layers.0.mlp.down_proj.weight_scale        < F32


Additionally, there can be `FP8 kv-cache scaling factors <https://github.com/vllm-project/vllm/pull/4893>`_ contained within quantized checkpoints specified through the ``.kv_scale`` parameter present on the Attention Module, such as:

.. code-block:: text

    model.layers.0.self_attn.kv_scale	             < F32
