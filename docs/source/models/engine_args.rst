.. _engine_args:

Engine Arguments
==================

Below, you can find an explanation of every engine argument for vLLM:

* --model
    Name or path of the huggingface model to use.
* --tokenizer
    Name or path of the huggingface tokenizer to use.
* --revision
    The specific model version to use. It can be a branch name, a tag name, or a commit id. If unspecified, will use the default version.
* --tokenizer-revision
    The specific tokenizer version to use. It can be a branch name, a tag name, or a commit id. If unspecified, will use the default version.
* --tokenizer-mode
    {auto,slow} The tokenizer mode.
    
    * -- "auto" will use the fast tokenizer if available
    * -- "slow" will always use the slow tokenizer.
* --trust-remote-code
    Trust remote code from huggingface.
* --download-dir
    Directory to download and load the weights, default to the default cache dir of huggingface
* --load-format
    {auto,pt,safetensors,npcache,dummy} The format of the model weights to load. 

    * -- "auto" will try to load the weights in the safetensors format and fall back to the pytorch bin format if safetensors format is not available. 

    * -- "pt" will load the weights in the pytorch bin format. "safetensors" will load the weights in the safetensors format. 

    * -- "npcache" will load the weights in pytorch format and store a numpy cache to speed up the loading. 

    * -- "dummy" will initialize the weights with random values, which is mainly for profiling.
* --dtype
    {auto,half,float16,bfloat16,float,float32} Data type for model weights and activations. 

    * -- The "auto" option will use FP16 precision for FP32 and FP16 models, and BF16 precision for BF16 models.
    * -- Use "half" for AWQ quantization.

* --max-model-len
    Model context length. If unspecified, will be automatically derived from the model config.
* --worker-use-ray
    Use Ray for distributed serving, will be automatically set when using more than 1 GPU.
* --pipeline-parallel-size
    (-pp) Number of pipeline stages.
* --tensor-parallel-size
    (-tp) Number of tensor parallel replicas.
* --max-parallel-loading-workers
    Load model sequentially in multiple batches, to avoid RAM OOM when using tensor parallel and large models.
* --block-size 
    {8,16,32} Token block size.
* --seed
    Random seed for operations.
* --swap-space
    CPU swap space size (GiB) per GPU.
* --gpu-memory-utilization
    The percentage of GPU memory to be used forthe model executor.
* --max-num-batched-tokens
    Maximum number of batched tokens per iteration.
* --max-num-seqs
    Maximum number of sequences per iteration.
* --max-paddings
    Maximum number of paddings in a batch.
* --disable-log-stats
    Disable logging statistics.
* --quantization
    (-q) {awq,squeezellm,None} Method used to quantize the weights.