.. _engine_args:

Engine Arguments
================

Below, you can find an explanation of every engine argument for vLLM:

.. option:: --model <model_name_or_path>

    Name or path of the huggingface model to use.

.. option:: --tokenizer <tokenizer_name_or_path>

    Name or path of the huggingface tokenizer to use.

.. option:: --revision <revision>

    The specific model version to use. It can be a branch name, a tag name, or a commit id. If unspecified, will use the default version.

.. option:: --tokenizer-revision <revision>

    The specific tokenizer version to use. It can be a branch name, a tag name, or a commit id. If unspecified, will use the default version.

.. option:: --tokenizer-mode {auto,slow}

    The tokenizer mode.
    
    * "auto" will use the fast tokenizer if available.
    * "slow" will always use the slow tokenizer.

.. option:: --trust-remote-code

    Trust remote code from huggingface.

.. option:: --download-dir <directory>

    Directory to download and load the weights, default to the default cache dir of huggingface.

.. option:: --load-format {auto,pt,safetensors,npcache,dummy}

    The format of the model weights to load.

    * "auto" will try to load the weights in the safetensors format and fall back to the pytorch bin format if safetensors format is not available.
    * "pt" will load the weights in the pytorch bin format.
    * "safetensors" will load the weights in the safetensors format.
    * "npcache" will load the weights in pytorch format and store a numpy cache to speed up the loading.
    * "dummy" will initialize the weights with random values, mainly for profiling.

.. option:: --dtype {auto,half,float16,bfloat16,float,float32}

    Data type for model weights and activations.

    * "auto" will use FP16 precision for FP32 and FP16 models, and BF16 precision for BF16 models.
    * "half" for FP16. Recommended for AWQ quantization.
    * "float16" is the same as "half".
    * "bfloat16" for a balance between precision and range.
    * "float" is shorthand for FP32 precision.
    * "float32" for FP32 precision.

.. option:: --max-model-len <length>

    Model context length. If unspecified, will be automatically derived from the model config.

.. option:: --worker-use-ray

    Use Ray for distributed serving, will be automatically set when using more than 1 GPU.

.. option:: --pipeline-parallel-size (-pp) <size>

    Number of pipeline stages.

.. option:: --tensor-parallel-size (-tp) <size>

    Number of tensor parallel replicas.

.. option:: --max-parallel-loading-workers <workers>

    Load model sequentially in multiple batches, to avoid RAM OOM when using tensor parallel and large models.

.. option:: --block-size {8,16,32}

    Token block size for contiguous chunks of tokens.

.. option:: --seed <seed>

    Random seed for operations.

.. option:: --swap-space <size>

    CPU swap space size (GiB) per GPU.

.. option:: --gpu-memory-utilization <fraction>

    The fraction of GPU memory to be used for the model executor, which can range from 0 to 1. 
    For example, a value of 0.5 would imply 50% GPU memory utilization.
    If unspecified, will use the default value of 0.9.

.. option:: --max-num-batched-tokens <tokens>

    Maximum number of batched tokens per iteration.

.. option:: --max-num-seqs <sequences>

    Maximum number of sequences per iteration.

.. option:: --max-paddings <paddings>

    Maximum number of paddings in a batch.

.. option:: --disable-log-stats

    Disable logging statistics.

.. option:: --quantization (-q) {awq,squeezellm,None}

    Method used to quantize the weights.
