.. _fp8:

FP8 W8A8
==================

vLLM supports FP8 (8-bit floating point) weight and activation quantization using hardware acceleration on GPUs such as Nvidia H100 and AMD MI300x. 
Currently, only Hopper and Ada Lovelace GPUs are officially supported for W8A8. 
Ampere GPUs are supported for W8A16 (weight-only FP8) utilizing Marlin kernels.
Quantization of models with FP8 allows for a 2x reduction in model memory requirements and up to a 1.6x improvement in throughput with minimal impact on accuracy.

Please visit the HF collection of `quantized FP8 checkpoints of popular LLMs ready to use with vLLM <https://huggingface.co/collections/neuralmagic/fp8-llms-for-vllm-666742ed2b78b7ac8df13127>`_.

The FP8 types typically supported in hardware have two distinct representations, each useful in different scenarios:

- **E4M3**: Consists of 1 sign bit, 4 exponent bits, and 3 bits of mantissa. It can store values up to +/-448 and ``nan``.
- **E5M2**: Consists of 1 sign bit, 5 exponent bits, and 2 bits of mantissa. It can store values up to +/-57344, +/- ``inf``, and ``nan``. The tradeoff for the increased dynamic range is lower precision of the stored values.

.. note::

   FP8 computation is supported on NVIDIA GPUs with compute capability > 8.9 (Ada Lovelace, Hopper).
   FP8 models will run on compute capability > 8.0 (Ampere) as weight-only W8A16, utilizing FP8 Marlin.

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

Installation
------------

To produce performant FP8 quantized models with vLLM, you'll need to install the `llm-compressor <https://github.com/vllm-project/llm-compressor/>`_ library:

.. code-block:: console

   $ pip install llmcompressor==0.1.0

Quantization Process
--------------------

The quantization process involves three main steps:

1. Loading the model
2. Applying quantization
3. Evaluating accuracy in vLLM

1. Loading the Model
^^^^^^^^^^^^^^^^^^^^

Use ``SparseAutoModelForCausalLM``, which wraps ``AutoModelForCausalLM``, for saving and loading quantized models:

.. code-block:: python

   from llmcompressor.transformers import SparseAutoModelForCausalLM
   from transformers import AutoTokenizer

   MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"

   model = SparseAutoModelForCausalLM.from_pretrained(
     MODEL_ID, device_map="auto", torch_dtype="auto")
   tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

2. Applying Quantization
^^^^^^^^^^^^^^^^^^^^^^^^

For FP8 quantization, we can recover accuracy with simple RTN quantization. We recommend targeting all ``Linear`` layers using the ``FP8_DYNAMIC`` scheme, which uses:

- Static, per-channel quantization on the weights
- Dynamic, per-token quantization on the activations

Since simple RTN does not require data for weight quantization and the activations are quantized dynamically, we do not need any calibration data for this quantization flow.

.. code-block:: python

   from llmcompressor.transformers import oneshot
   from llmcompressor.modifiers.quantization import QuantizationModifier

   # Configure the simple PTQ quantization
   recipe = QuantizationModifier(
     targets="Linear", scheme="FP8_DYNAMIC", ignore=["lm_head"])

   # Apply the quantization algorithm.
   oneshot(model=model, recipe=recipe)

   # Save the model.
   SAVE_DIR = MODEL_ID.split("/")[1] + "-FP8-Dynamic"
   model.save_pretrained(SAVE_DIR)
   tokenizer.save_pretrained(SAVE_DIR)

3. Evaluating Accuracy
^^^^^^^^^^^^^^^^^^^^^^

Install ``vllm`` and ``lm-evaluation-harness``:

.. code-block:: console

   $ pip install vllm lm_eval==0.4.3

Load and run the model in ``vllm``:

.. code-block:: python

   from vllm import LLM
   model = LLM("./Meta-Llama-3-8B-Instruct-FP8-Dynamic")
   model.generate("Hello my name is")

Evaluate accuracy with ``lm_eval`` (for example on 250 samples of ``gsm8k``):

.. note::

   Quantized models can be sensitive to the presence of the ``bos`` token. ``lm_eval`` does not add a ``bos`` token by default, so make sure to include the ``add_bos_token=True`` argument when running your evaluations.

.. code-block:: console

   $ MODEL=$PWD/Meta-Llama-3-8B-Instruct-FP8-Dynamic 
   $ lm_eval \
     --model vllm \
     --model_args pretrained=$MODEL,add_bos_token=True \
     --tasks gsm8k  --num_fewshot 5 --batch_size auto --limit 250

Here's an example of the resulting scores:

.. code-block:: text

   |Tasks|Version|     Filter     |n-shot|  Metric   |   |Value|   |Stderr|
   |-----|------:|----------------|-----:|-----------|---|----:|---|-----:|
   |gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.768|±  |0.0268|
   |     |       |strict-match    |     5|exact_match|↑  |0.768|±  |0.0268|

Troubleshooting and Support
---------------------------

If you encounter any issues or have feature requests, please open an issue on the ``vllm-project/llm-compressor`` GitHub repository.


Deprecated Flow
------------------

.. note::

   The following information is preserved for reference and search purposes.
   The quantization method described below is deprecated in favor of the ``llmcompressor`` method described above.

For static per-tensor offline quantization to FP8, please install the `AutoFP8 library <https://github.com/neuralmagic/autofp8>`_.

.. code-block:: bash

    git clone https://github.com/neuralmagic/AutoFP8.git
    pip install -e AutoFP8

This package introduces the ``AutoFP8ForCausalLM`` and ``BaseQuantizeConfig`` objects for managing how your model will be compressed.

Offline Quantization with Static Activation Scaling Factors
-----------------------------------------------------------

You can use AutoFP8 with calibration data to produce per-tensor static scales for both the weights and activations by enabling the ``activation_scheme="static"`` argument.

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

