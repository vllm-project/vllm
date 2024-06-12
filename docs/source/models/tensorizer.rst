.. _tensorizer:

Loading Models with CoreWeave's Tensorizer
================
vLLM supports loading models with `CoreWeave's Tensorizer <https://github.com/coreweave/tensorizer>`_.
vLLM model tensors serialized to a HTTP/HTTPS endpoint, a S3 endpoint, or disk, can be deserialized
at runtime extremely quickly and directly to the GPU, allowing for significantly
shorter pod startup time and CPU memory usage. Tensor encryption is also supported.

For more information on how to use CoreWeave's Tensorizer, please refer to
`CoreWeave's Tensorizer documentation <https://github.com/coreweave/tensorizer>`_ and
the `example script here <https://docs.vllm.ai/en/stable/getting_started/examples/tensorize_vllm_model.html>`_ for
how to serialize a vLLM model as well a general usage guide to using Tensorizer with vLLM.