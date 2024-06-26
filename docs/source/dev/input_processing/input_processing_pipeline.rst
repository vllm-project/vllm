.. _input_processing_pipeline:

Input Processing Pipeline
=========================

1. Input data is passed to :class:`~vllm.LLMEngine` (or :class:`~vllm.AsyncLLMEngine`).
2. Tokenize the data if necessary.
3. Process the inputs using :meth:`~vllm.inputs.registry.InputRegistry.process_input`.
4. Send the processed inputs to :class:`~vllm.executor.executor_base.ExecutorBase`.
5. Distribute the inputs via :class:`~vllm.worker.worker_base.WorkerBase` to :class:`~vllm.worker.model_runner_base.ModelRunner`.
6. If the data contains multi-modal data, convert it into keyword arguments using :meth:`~vllm.multimodal.MultiModalRegistry.map_input`.
