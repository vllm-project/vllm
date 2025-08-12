.. _tensorizer:

Loading Models with CoreWeave's Tensorizer
==========================================
vLLM supports loading models with `CoreWeave's Tensorizer <https://docs.coreweave.com/coreweave-machine-learning-and-ai/inference/tensorizer>`_.
vLLM model tensors that have been serialized to disk, an HTTP/HTTPS endpoint, or S3 endpoint can be deserialized
at runtime extremely quickly directly to the GPU, resulting in significantly
shorter Pod startup times and CPU memory usage. Tensor encryption is also supported.

For more information on CoreWeave's Tensorizer, please refer to
`CoreWeave's Tensorizer documentation <https://github.com/coreweave/tensorizer>`_. For more information on serializing a vLLM model, as well a general usage guide to using Tensorizer with vLLM, see
the `vLLM example script <https://docs.vllm.ai/en/stable/getting_started/examples/tensorize_vllm_model.html>`_.