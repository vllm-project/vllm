Without CUDAGraph capturing we have:

root@vllm-vm:/app# VLLM_USE_V1=1 vllm serve meta-llama/Llama-3.2-1B
INFO 08-01 12:38:45 [__init__.py:241] Automatically detected platform cuda.
(APIServer pid=6828) INFO 08-01 12:38:48 [api_server.py:1774] vLLM API server version 0.1.dev8168+g475c1a0
(APIServer pid=6828) INFO 08-01 12:38:48 [utils.py:326] non-default args: {'model_tag': 'meta-llama/Llama-3.2-1B', 'model': 'meta-llama/Llama-3.2-1B'}
(APIServer pid=6828) INFO 08-01 12:38:54 [config.py:713] Resolved architecture: LlamaForCausalLM
(APIServer pid=6828) INFO 08-01 12:38:54 [config.py:1716] Using max model len 131072
(APIServer pid=6828) INFO 08-01 12:38:54 [config.py:2542] Chunked prefill is enabled with max_num_batched_tokens=2048.
INFO 08-01 12:38:59 [__init__.py:241] Automatically detected platform cuda.
INFO 08-01 12:39:01 [core.py:591] Waiting for init message from front-end.
INFO 08-01 12:39:01 [core.py:73] Initializing a V1 LLM engine (v0.1.dev8168+g475c1a0) with config: model='meta-llama/Llama-3.2-1B', speculative_config=None, tokenizer='meta-llama/Llama-3.2-1B', skip_tokenizer_init=False, tokenizer_mode=auto, revision=None, override_neuron_config={}, tokenizer_revision=None, trust_remote_code=False, dtype=torch.bfloat16, max_seq_len=131072, download_dir=None, load_format=auto, tensor_parallel_size=1, pipeline_parallel_size=1, disable_custom_all_reduce=False, quantization=None, enforce_eager=False, kv_cache_dtype=auto, device_config=cuda, decoding_config=DecodingConfig(backend='auto', disable_fallback=False, disable_any_whitespace=False, disable_additional_properties=False, reasoning_backend=''), observability_config=ObservabilityConfig(show_hidden_metrics_for_version=None, otlp_traces_endpoint=None, collect_detailed_traces=None), seed=0, served_model_name=meta-llama/Llama-3.2-1B, num_scheduler_steps=1, multi_step_stream_outputs=True, enable_prefix_caching=True, chunked_prefill_enabled=True, use_async_output_proc=True, pooler_config=None, compilation_config={"level":3,"debug_dump_path":"","cache_dir":"","backend":"","custom_ops":[],"splitting_ops":["vllm.unified_attention","vllm.unified_attention_with_output","vllm.mamba_mixer2"],"use_inductor":true,"compile_sizes":[],"inductor_compile_config":{"enable_auto_functionalized_v2":false},"inductor_passes":{},"use_cudagraph":true,"cudagraph_num_of_warmups":1,"cudagraph_capture_sizes":[512,504,496,488,480,472,464,456,448,440,432,424,416,408,400,392,384,376,368,360,352,344,336,328,320,312,304,296,288,280,272,264,256,248,240,232,224,216,208,200,192,184,176,168,160,152,144,136,128,120,112,104,96,88,80,72,64,56,48,40,32,24,16,8,4,2,1],"cudagraph_copy_inputs":false,"full_cuda_graph":false,"max_capture_size":512,"local_cache_dir":null}
INFO 08-01 12:39:02 [parallel_state.py:1102] rank 0 in world size 1 is assigned as DP rank 0, PP rank 0, TP rank 0, EP rank 0
INFO 08-01 12:39:02 [topk_topp_sampler.py:49] Using FlashInfer for top-p & top-k sampling.
INFO 08-01 12:39:02 [gpu_model_runner.py:1921] Starting to load model meta-llama/Llama-3.2-1B...
INFO 08-01 12:39:02 [gpu_model_runner.py:1953] Loading model from scratch...
INFO 08-01 12:39:03 [cuda.py:305] Using Flash Attention backend on V1 engine.
INFO 08-01 12:39:03 [weight_utils.py:296] Using model weights format ['*.safetensors']
INFO 08-01 12:39:03 [weight_utils.py:349] No model.safetensors.index.json found in remote.
Loading safetensors checkpoint shards:   0% Completed | 0/1 [00:00<?, ?it/s]
Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  2.09it/s]
Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  2.09it/s]

INFO 08-01 12:39:03 [default_loader.py:262] Loading weights took 0.56 seconds
INFO 08-01 12:39:04 [gpu_model_runner.py:1970] Model loading took 2.3185 GiB and 0.822538 seconds
INFO 08-01 12:39:07 [backends.py:534] Using cache directory: /root/.cache/vllm/torch_compile_cache/25d4010308/rank_0_0/backbone for vLLM's torch.compile
INFO 08-01 12:39:07 [backends.py:545] Dynamo bytecode transform time: 3.55 s
INFO 08-01 12:39:10 [backends.py:165] Directly load the compiled graph(s) for dynamic shape from the cache, took 2.497 s
INFO 08-01 12:39:11 [monitor.py:34] torch.compile takes 3.55 s in total
/usr/local/lib/python3.12/dist-packages/torch/utils/cpp_extension.py:2356: UserWarning: TORCH_CUDA_ARCH_LIST is not set, all archs for visible cards are included for compilation.
If this is not desired, please set os.environ['TORCH_CUDA_ARCH_LIST'].
  warnings.warn(
INFO 08-01 12:39:11 [gpu_worker.py:276] Available KV cache memory: 37.10 GiB
INFO 08-01 12:39:12 [kv_cache_utils.py:831] GPU KV cache size: 1,215,552 tokens
INFO 08-01 12:39:12 [kv_cache_utils.py:835] Maximum concurrency for 131,072 tokens per request: 9.27x
INFO 08-01 12:39:12 [core.py:201] init engine (profile, create kv cache, warmup model) took 7.99 seconds
(APIServer pid=6828) INFO 08-01 12:39:12 [loggers.py:142] Engine 000: vllm cache_config_info with initialization after num_gpu_blocks is: 75972
(APIServer pid=6828) INFO 08-01 12:39:12 [api_server.py:1595] Supported_tasks: ['generate']
(APIServer pid=6828) WARNING 08-01 12:39:12 [config.py:1616] Default sampling parameters have been overridden by the model's Hugging Face generation config recommended from the model creator. If this is not intended, please relaunch vLLM instance with `--generation-config vllm`.
(APIServer pid=6828) INFO 08-01 12:39:12 [serving_responses.py:89] Using default chat sampling params from model: {'temperature': 0.6, 'top_p': 0.9}
(APIServer pid=6828) INFO 08-01 12:39:12 [serving_chat.py:125] Using default chat sampling params from model: {'temperature': 0.6, 'top_p': 0.9}
(APIServer pid=6828) INFO 08-01 12:39:12 [serving_completion.py:77] Using default completion sampling params from model: {'temperature': 0.6, 'top_p': 0.9}
(APIServer pid=6828) INFO 08-01 12:39:12 [api_server.py:1847] Starting vLLM API server 0 on <http://0.0.0.0:8000>
(APIServer pid=6828) INFO 08-01 12:39:12 [launcher.py:29] Available routes are:
(APIServer pid=6828) INFO 08-01 12:39:12 [launcher.py:37] Route: /openapi.json, Methods: GET, HEAD
(APIServer pid=6828) INFO 08-01 12:39:12 [launcher.py:37] Route: /docs, Methods: GET, HEAD
(APIServer pid=6828) INFO 08-01 12:39:12 [launcher.py:37] Route: /docs/oauth2-redirect, Methods: GET, HEAD
(APIServer pid=6828) INFO 08-01 12:39:12 [launcher.py:37] Route: /redoc, Methods: GET, HEAD
(APIServer pid=6828) INFO 08-01 12:39:12 [launcher.py:37] Route: /health, Methods: GET
(APIServer pid=6828) INFO 08-01 12:39:12 [launcher.py:37] Route: /load, Methods: GET
(APIServer pid=6828) INFO 08-01 12:39:12 [launcher.py:37] Route: /ping, Methods: POST
(APIServer pid=6828) INFO 08-01 12:39:12 [launcher.py:37] Route: /ping, Methods: GET
(APIServer pid=6828) INFO 08-01 12:39:12 [launcher.py:37] Route: /tokenize, Methods: POST
(APIServer pid=6828) INFO 08-01 12:39:12 [launcher.py:37] Route: /detokenize, Methods: POST
(APIServer pid=6828) INFO 08-01 12:39:12 [launcher.py:37] Route: /v1/models, Methods: GET
(APIServer pid=6828) INFO 08-01 12:39:12 [launcher.py:37] Route: /version, Methods: GET
(APIServer pid=6828) INFO 08-01 12:39:12 [launcher.py:37] Route: /v1/responses, Methods: POST
(APIServer pid=6828) INFO 08-01 12:39:12 [launcher.py:37] Route: /v1/responses/{response_id}, Methods: GET
(APIServer pid=6828) INFO 08-01 12:39:12 [launcher.py:37] Route: /v1/responses/{response_id}/cancel, Methods: POST
(APIServer pid=6828) INFO 08-01 12:39:12 [launcher.py:37] Route: /v1/chat/completions, Methods: POST
(APIServer pid=6828) INFO 08-01 12:39:12 [launcher.py:37] Route: /v1/completions, Methods: POST
(APIServer pid=6828) INFO 08-01 12:39:12 [launcher.py:37] Route: /v1/embeddings, Methods: POST
(APIServer pid=6828) INFO 08-01 12:39:12 [launcher.py:37] Route: /pooling, Methods: POST
(APIServer pid=6828) INFO 08-01 12:39:12 [launcher.py:37] Route: /classify, Methods: POST
(APIServer pid=6828) INFO 08-01 12:39:12 [launcher.py:37] Route: /score, Methods: POST
(APIServer pid=6828) INFO 08-01 12:39:12 [launcher.py:37] Route: /v1/score, Methods: POST
(APIServer pid=6828) INFO 08-01 12:39:12 [launcher.py:37] Route: /v1/audio/transcriptions, Methods: POST
(APIServer pid=6828) INFO 08-01 12:39:12 [launcher.py:37] Route: /v1/audio/translations, Methods: POST
(APIServer pid=6828) INFO 08-01 12:39:12 [launcher.py:37] Route: /rerank, Methods: POST
(APIServer pid=6828) INFO 08-01 12:39:12 [launcher.py:37] Route: /v1/rerank, Methods: POST
(APIServer pid=6828) INFO 08-01 12:39:12 [launcher.py:37] Route: /v2/rerank, Methods: POST
(APIServer pid=6828) INFO 08-01 12:39:12 [launcher.py:37] Route: /scale_elastic_ep, Methods: POST
(APIServer pid=6828) INFO 08-01 12:39:12 [launcher.py:37] Route: /is_scaling_elastic_ep, Methods: POST
(APIServer pid=6828) INFO 08-01 12:39:12 [launcher.py:37] Route: /invocations, Methods: POST
(APIServer pid=6828) INFO 08-01 12:39:12 [launcher.py:37] Route: /metrics, Methods: GET
(APIServer pid=6828) INFO:     Started server process [6828]
(APIServer pid=6828) INFO:     Waiting for application startup.
(APIServer pid=6828) INFO:     Application startup complete.

With CUDAGraph capturing:
root@vllm-vm:/app# VLLM_USE_V1=1 vllm serve meta-llama/Llama-3.2-1B
INFO 08-01 12:51:00 [__init__.py:241] Automatically detected platform cuda.
(APIServer pid=7224) INFO 08-01 12:51:03 [api_server.py:1774] vLLM API server version 0.1.dev8168+g475c1a0
(APIServer pid=7224) INFO 08-01 12:51:03 [utils.py:326] non-default args: {'model_tag': 'meta-llama/Llama-3.2-1B', 'model': 'meta-llama/Llama-3.2-1B'}
(APIServer pid=7224) INFO 08-01 12:51:09 [config.py:713] Resolved architecture: LlamaForCausalLM
(APIServer pid=7224) INFO 08-01 12:51:09 [config.py:1716] Using max model len 131072
(APIServer pid=7224) INFO 08-01 12:51:09 [config.py:2542] Chunked prefill is enabled with max_num_batched_tokens=2048.
INFO 08-01 12:51:13 [__init__.py:241] Automatically detected platform cuda.
INFO 08-01 12:51:16 [core.py:591] Waiting for init message from front-end.
INFO 08-01 12:51:16 [core.py:73] Initializing a V1 LLM engine (v0.1.dev8168+g475c1a0) with config: model='meta-llama/Llama-3.2-1B', speculative_config=None, tokenizer='meta-llama/Llama-3.2-1B', skip_tokenizer_init=False, tokenizer_mode=auto, revision=None, override_neuron_config={}, tokenizer_revision=None, trust_remote_code=False, dtype=torch.bfloat16, max_seq_len=131072, download_dir=None, load_format=auto, tensor_parallel_size=1, pipeline_parallel_size=1, disable_custom_all_reduce=False, quantization=None, enforce_eager=False, kv_cache_dtype=auto, device_config=cuda, decoding_config=DecodingConfig(backend='auto', disable_fallback=False, disable_any_whitespace=False, disable_additional_properties=False, reasoning_backend=''), observability_config=ObservabilityConfig(show_hidden_metrics_for_version=None, otlp_traces_endpoint=None, collect_detailed_traces=None), seed=0, served_model_name=meta-llama/Llama-3.2-1B, num_scheduler_steps=1, multi_step_stream_outputs=True, enable_prefix_caching=True, chunked_prefill_enabled=True, use_async_output_proc=True, pooler_config=None, compilation_config={"level":3,"debug_dump_path":"","cache_dir":"","backend":"","custom_ops":[],"splitting_ops":["vllm.unified_attention","vllm.unified_attention_with_output","vllm.mamba_mixer2"],"use_inductor":true,"compile_sizes":[],"inductor_compile_config":{"enable_auto_functionalized_v2":false},"inductor_passes":{},"use_cudagraph":true,"cudagraph_num_of_warmups":1,"cudagraph_capture_sizes":[512,504,496,488,480,472,464,456,448,440,432,424,416,408,400,392,384,376,368,360,352,344,336,328,320,312,304,296,288,280,272,264,256,248,240,232,224,216,208,200,192,184,176,168,160,152,144,136,128,120,112,104,96,88,80,72,64,56,48,40,32,24,16,8,4,2,1],"cudagraph_copy_inputs":false,"full_cuda_graph":false,"max_capture_size":512,"local_cache_dir":null}
INFO 08-01 12:51:17 [parallel_state.py:1102] rank 0 in world size 1 is assigned as DP rank 0, PP rank 0, TP rank 0, EP rank 0
INFO 08-01 12:51:17 [topk_topp_sampler.py:49] Using FlashInfer for top-p & top-k sampling.
INFO 08-01 12:51:17 [gpu_model_runner.py:1921] Starting to load model meta-llama/Llama-3.2-1B...
INFO 08-01 12:51:17 [gpu_model_runner.py:1953] Loading model from scratch...
INFO 08-01 12:51:17 [cuda.py:305] Using Flash Attention backend on V1 engine.
INFO 08-01 12:51:17 [weight_utils.py:296] Using model weights format ['*.safetensors']
INFO 08-01 12:51:17 [weight_utils.py:349] No model.safetensors.index.json found in remote.
Loading safetensors checkpoint shards:   0% Completed | 0/1 [00:00<?, ?it/s]
Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  2.18it/s]
Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  2.17it/s]

INFO 08-01 12:51:18 [default_loader.py:262] Loading weights took 0.54 seconds
INFO 08-01 12:51:18 [gpu_model_runner.py:1970] Model loading took 2.3185 GiB and 0.903764 seconds
INFO 08-01 12:51:22 [backends.py:534] Using cache directory: /root/.cache/vllm/torch_compile_cache/25d4010308/rank_0_0/backbone for vLLM's torch.compile
INFO 08-01 12:51:22 [backends.py:545] Dynamo bytecode transform time: 3.55 s
INFO 08-01 12:51:25 [backends.py:165] Directly load the compiled graph(s) for dynamic shape from the cache, took 2.826 s
INFO 08-01 12:51:26 [monitor.py:34] torch.compile takes 3.55 s in total
/usr/local/lib/python3.12/dist-packages/torch/utils/cpp_extension.py:2356: UserWarning: TORCH_CUDA_ARCH_LIST is not set, all archs for visible cards are included for compilation.
If this is not desired, please set os.environ['TORCH_CUDA_ARCH_LIST'].
  warnings.warn(
INFO 08-01 12:51:26 [gpu_worker.py:276] Available KV cache memory: 37.10 GiB
INFO 08-01 12:51:26 [kv_cache_utils.py:831] GPU KV cache size: 1,215,552 tokens
INFO 08-01 12:51:26 [kv_cache_utils.py:835] Maximum concurrency for 131,072 tokens per request: 9.27x
Capturing CUDA graph shapes:   0%|                                                                     | 0/67 [00:00<?, ?it/s]INFO 08-01 12:51:27 [gpu_model_runner.py:2561] DIEGO: compilation for number of tokens 512
INFO 08-01 12:51:27 [gpu_model_runner.py:2561] DIEGO: compilation for number of tokens 504
INFO 08-01 12:51:27 [gpu_model_runner.py:2561] DIEGO: compilation for number of tokens 496
INFO 08-01 12:51:27 [gpu_model_runner.py:2561] DIEGO: compilation for number of tokens 488
INFO 08-01 12:51:27 [gpu_model_runner.py:2561] DIEGO: compilation for number of tokens 480
INFO 08-01 12:51:27 [gpu_model_runner.py:2561] DIEGO: compilation for number of tokens 472
Capturing CUDA graph shapes:   9%|█████▍                                                       | 6/67 [00:00<00:01, 58.92it/s]INFO 08-01 12:51:27 [gpu_model_runner.py:2561] DIEGO: compilation for number of tokens 464
INFO 08-01 12:51:27 [gpu_model_runner.py:2561] DIEGO: compilation for number of tokens 456
INFO 08-01 12:51:27 [gpu_model_runner.py:2561] DIEGO: compilation for number of tokens 448
INFO 08-01 12:51:27 [gpu_model_runner.py:2561] DIEGO: compilation for number of tokens 440
INFO 08-01 12:51:27 [gpu_model_runner.py:2561] DIEGO: compilation for number of tokens 432
INFO 08-01 12:51:27 [gpu_model_runner.py:2561] DIEGO: compilation for number of tokens 424
INFO 08-01 12:51:27 [gpu_model_runner.py:2561] DIEGO: compilation for number of tokens 416
Capturing CUDA graph shapes:  19%|███████████▋                                                | 13/67 [00:00<00:00, 60.49it/s]INFO 08-01 12:51:27 [gpu_model_runner.py:2561] DIEGO: compilation for number of tokens 408
INFO 08-01 12:51:27 [gpu_model_runner.py:2561] DIEGO: compilation for number of tokens 400
INFO 08-01 12:51:27 [gpu_model_runner.py:2561] DIEGO: compilation for number of tokens 392
INFO 08-01 12:51:27 [gpu_model_runner.py:2561] DIEGO: compilation for number of tokens 384
INFO 08-01 12:51:27 [gpu_model_runner.py:2561] DIEGO: compilation for number of tokens 376
INFO 08-01 12:51:27 [gpu_model_runner.py:2561] DIEGO: compilation for number of tokens 368
INFO 08-01 12:51:27 [gpu_model_runner.py:2561] DIEGO: compilation for number of tokens 360
Capturing CUDA graph shapes:  30%|█████████████████▉                                          | 20/67 [00:00<00:00, 62.08it/s]INFO 08-01 12:51:27 [gpu_model_runner.py:2561] DIEGO: compilation for number of tokens 352
INFO 08-01 12:51:27 [gpu_model_runner.py:2561] DIEGO: compilation for number of tokens 344
INFO 08-01 12:51:27 [gpu_model_runner.py:2561] DIEGO: compilation for number of tokens 336
INFO 08-01 12:51:27 [gpu_model_runner.py:2561] DIEGO: compilation for number of tokens 328
INFO 08-01 12:51:27 [gpu_model_runner.py:2561] DIEGO: compilation for number of tokens 320
INFO 08-01 12:51:27 [gpu_model_runner.py:2561] DIEGO: compilation for number of tokens 312
INFO 08-01 12:51:27 [gpu_model_runner.py:2561] DIEGO: compilation for number of tokens 304
Capturing CUDA graph shapes:  40%|████████████████████████▏                                   | 27/67 [00:00<00:00, 62.62it/s]INFO 08-01 12:51:27 [gpu_model_runner.py:2561] DIEGO: compilation for number of tokens 296
INFO 08-01 12:51:27 [gpu_model_runner.py:2561] DIEGO: compilation for number of tokens 288
INFO 08-01 12:51:27 [gpu_model_runner.py:2561] DIEGO: compilation for number of tokens 280
INFO 08-01 12:51:27 [gpu_model_runner.py:2561] DIEGO: compilation for number of tokens 272
INFO 08-01 12:51:27 [gpu_model_runner.py:2561] DIEGO: compilation for number of tokens 264
INFO 08-01 12:51:27 [gpu_model_runner.py:2561] DIEGO: compilation for number of tokens 256
INFO 08-01 12:51:27 [gpu_model_runner.py:2561] DIEGO: compilation for number of tokens 248
Capturing CUDA graph shapes:  51%|██████████████████████████████▍                             | 34/67 [00:00<00:00, 61.52it/s]INFO 08-01 12:51:27 [gpu_model_runner.py:2561] DIEGO: compilation for number of tokens 240
INFO 08-01 12:51:27 [gpu_model_runner.py:2561] DIEGO: compilation for number of tokens 232
INFO 08-01 12:51:27 [gpu_model_runner.py:2561] DIEGO: compilation for number of tokens 224
INFO 08-01 12:51:27 [gpu_model_runner.py:2561] DIEGO: compilation for number of tokens 216
INFO 08-01 12:51:27 [gpu_model_runner.py:2561] DIEGO: compilation for number of tokens 208
INFO 08-01 12:51:27 [gpu_model_runner.py:2561] DIEGO: compilation for number of tokens 200
INFO 08-01 12:51:27 [gpu_model_runner.py:2561] DIEGO: compilation for number of tokens 192
Capturing CUDA graph shapes:  61%|████████████████████████████████████▋                       | 41/67 [00:00<00:00, 62.24it/s]INFO 08-01 12:51:27 [gpu_model_runner.py:2561] DIEGO: compilation for number of tokens 184
INFO 08-01 12:51:27 [gpu_model_runner.py:2561] DIEGO: compilation for number of tokens 176
INFO 08-01 12:51:27 [gpu_model_runner.py:2561] DIEGO: compilation for number of tokens 168
INFO 08-01 12:51:27 [gpu_model_runner.py:2561] DIEGO: compilation for number of tokens 160
INFO 08-01 12:51:27 [gpu_model_runner.py:2561] DIEGO: compilation for number of tokens 152
INFO 08-01 12:51:27 [gpu_model_runner.py:2561] DIEGO: compilation for number of tokens 144
INFO 08-01 12:51:27 [gpu_model_runner.py:2561] DIEGO: compilation for number of tokens 136
Capturing CUDA graph shapes:  72%|██████████████████████████████████████████▉                 | 48/67 [00:00<00:00, 60.79it/s]INFO 08-01 12:51:27 [gpu_model_runner.py:2561] DIEGO: compilation for number of tokens 128
INFO 08-01 12:51:27 [gpu_model_runner.py:2561] DIEGO: compilation for number of tokens 120
INFO 08-01 12:51:27 [gpu_model_runner.py:2561] DIEGO: compilation for number of tokens 112
INFO 08-01 12:51:27 [gpu_model_runner.py:2561] DIEGO: compilation for number of tokens 104
INFO 08-01 12:51:27 [gpu_model_runner.py:2561] DIEGO: compilation for number of tokens 96
INFO 08-01 12:51:27 [gpu_model_runner.py:2561] DIEGO: compilation for number of tokens 88
INFO 08-01 12:51:27 [gpu_model_runner.py:2561] DIEGO: compilation for number of tokens 80
Capturing CUDA graph shapes:  82%|█████████████████████████████████████████████████▎          | 55/67 [00:00<00:00, 62.53it/s]INFO 08-01 12:51:27 [gpu_model_runner.py:2561] DIEGO: compilation for number of tokens 72
INFO 08-01 12:51:28 [gpu_model_runner.py:2561] DIEGO: compilation for number of tokens 64
INFO 08-01 12:51:28 [gpu_model_runner.py:2561] DIEGO: compilation for number of tokens 56
INFO 08-01 12:51:28 [gpu_model_runner.py:2561] DIEGO: compilation for number of tokens 48
INFO 08-01 12:51:28 [gpu_model_runner.py:2561] DIEGO: compilation for number of tokens 40
INFO 08-01 12:51:28 [gpu_model_runner.py:2561] DIEGO: compilation for number of tokens 32
INFO 08-01 12:51:28 [gpu_model_runner.py:2561] DIEGO: compilation for number of tokens 24
Capturing CUDA graph shapes:  93%|███████████████████████████████████████████████████████▌    | 62/67 [00:00<00:00, 63.12it/s]INFO 08-01 12:51:28 [gpu_model_runner.py:2561] DIEGO: compilation for number of tokens 16
INFO 08-01 12:51:28 [gpu_model_runner.py:2561] DIEGO: compilation for number of tokens 8
INFO 08-01 12:51:28 [gpu_model_runner.py:2561] DIEGO: compilation for number of tokens 4
INFO 08-01 12:51:28 [gpu_model_runner.py:2561] DIEGO: compilation for number of tokens 2
INFO 08-01 12:51:28 [gpu_model_runner.py:2561] DIEGO: compilation for number of tokens 1
Capturing CUDA graph shapes: 100%|████████████████████████████████████████████████████████████| 67/67 [00:01<00:00, 62.26it/s]
INFO 08-01 12:51:28 [gpu_model_runner.py:2577] Graph capturing finished in 1 secs, took 0.31 GiB
INFO 08-01 12:51:28 [core.py:201] init engine (profile, create kv cache, warmup model) took 9.48 seconds
(APIServer pid=7224) INFO 08-01 12:51:28 [loggers.py:142] Engine 000: vllm cache_config_info with initialization after num_gpu_blocks is: 75972
(APIServer pid=7224) INFO 08-01 12:51:28 [api_server.py:1595] Supported_tasks: ['generate']
(APIServer pid=7224) WARNING 08-01 12:51:28 [config.py:1616] Default sampling parameters have been overridden by the model's Hugging Face generation config recommended from the model creator. If this is not intended, please relaunch vLLM instance with `--generation-config vllm`.
(APIServer pid=7224) INFO 08-01 12:51:28 [serving_responses.py:89] Using default chat sampling params from model: {'temperature': 0.6, 'top_p': 0.9}
(APIServer pid=7224) INFO 08-01 12:51:28 [serving_chat.py:125] Using default chat sampling params from model: {'temperature': 0.6, 'top_p': 0.9}
(APIServer pid=7224) INFO 08-01 12:51:28 [serving_completion.py:77] Using default completion sampling params from model: {'temperature': 0.6, 'top_p': 0.9}
(APIServer pid=7224) INFO 08-01 12:51:28 [api_server.py:1847] Starting vLLM API server 0 on <http://0.0.0.0:8000>
(APIServer pid=7224) INFO 08-01 12:51:28 [launcher.py:29] Available routes are:
(APIServer pid=7224) INFO 08-01 12:51:28 [launcher.py:37] Route: /openapi.json, Methods: HEAD, GET
(APIServer pid=7224) INFO 08-01 12:51:28 [launcher.py:37] Route: /docs, Methods: HEAD, GET
(APIServer pid=7224) INFO 08-01 12:51:28 [launcher.py:37] Route: /docs/oauth2-redirect, Methods: HEAD, GET
(APIServer pid=7224) INFO 08-01 12:51:28 [launcher.py:37] Route: /redoc, Methods: HEAD, GET
(APIServer pid=7224) INFO 08-01 12:51:28 [launcher.py:37] Route: /health, Methods: GET
(APIServer pid=7224) INFO 08-01 12:51:28 [launcher.py:37] Route: /load, Methods: GET
(APIServer pid=7224) INFO 08-01 12:51:28 [launcher.py:37] Route: /ping, Methods: POST
(APIServer pid=7224) INFO 08-01 12:51:28 [launcher.py:37] Route: /ping, Methods: GET
(APIServer pid=7224) INFO 08-01 12:51:28 [launcher.py:37] Route: /tokenize, Methods: POST
(APIServer pid=7224) INFO 08-01 12:51:28 [launcher.py:37] Route: /detokenize, Methods: POST
(APIServer pid=7224) INFO 08-01 12:51:28 [launcher.py:37] Route: /v1/models, Methods: GET
(APIServer pid=7224) INFO 08-01 12:51:28 [launcher.py:37] Route: /version, Methods: GET
(APIServer pid=7224) INFO 08-01 12:51:28 [launcher.py:37] Route: /v1/responses, Methods: POST
(APIServer pid=7224) INFO 08-01 12:51:28 [launcher.py:37] Route: /v1/responses/{response_id}, Methods: GET
(APIServer pid=7224) INFO 08-01 12:51:28 [launcher.py:37] Route: /v1/responses/{response_id}/cancel, Methods: POST
(APIServer pid=7224) INFO 08-01 12:51:28 [launcher.py:37] Route: /v1/chat/completions, Methods: POST
(APIServer pid=7224) INFO 08-01 12:51:28 [launcher.py:37] Route: /v1/completions, Methods: POST
(APIServer pid=7224) INFO 08-01 12:51:28 [launcher.py:37] Route: /v1/embeddings, Methods: POST
(APIServer pid=7224) INFO 08-01 12:51:28 [launcher.py:37] Route: /pooling, Methods: POST
(APIServer pid=7224) INFO 08-01 12:51:28 [launcher.py:37] Route: /classify, Methods: POST
(APIServer pid=7224) INFO 08-01 12:51:28 [launcher.py:37] Route: /score, Methods: POST
(APIServer pid=7224) INFO 08-01 12:51:28 [launcher.py:37] Route: /v1/score, Methods: POST
(APIServer pid=7224) INFO 08-01 12:51:28 [launcher.py:37] Route: /v1/audio/transcriptions, Methods: POST
(APIServer pid=7224) INFO 08-01 12:51:28 [launcher.py:37] Route: /v1/audio/translations, Methods: POST
(APIServer pid=7224) INFO 08-01 12:51:28 [launcher.py:37] Route: /rerank, Methods: POST
(APIServer pid=7224) INFO 08-01 12:51:28 [launcher.py:37] Route: /v1/rerank, Methods: POST
(APIServer pid=7224) INFO 08-01 12:51:28 [launcher.py:37] Route: /v2/rerank, Methods: POST
(APIServer pid=7224) INFO 08-01 12:51:28 [launcher.py:37] Route: /scale_elastic_ep, Methods: POST
(APIServer pid=7224) INFO 08-01 12:51:28 [launcher.py:37] Route: /is_scaling_elastic_ep, Methods: POST
(APIServer pid=7224) INFO 08-01 12:51:28 [launcher.py:37] Route: /invocations, Methods: POST
(APIServer pid=7224) INFO 08-01 12:51:28 [launcher.py:37] Route: /metrics, Methods: GET
(APIServer pid=7224) INFO:     Started server process [7224]
(APIServer pid=7224) INFO:     Waiting for application startup.
(APIServer pid=7224) INFO:     Application startup complete.

RESULTS 10 prompts WITHOUT CUDAGRAPH:
============ Serving Benchmark Result ============
Successful requests:                     10
Benchmark duration (s):                  4.70
Total input tokens:                      1369
Total generated tokens:                  1238
Request throughput (req/s):              2.13
Output token throughput (tok/s):         263.58
Total Token throughput (tok/s):          555.05
---------------Time to First Token----------------
Mean TTFT (ms):                          325.27
Median TTFT (ms):                        326.54
P99 TTFT (ms):                           328.43
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          19.37
Median TPOT (ms):                        9.53
P99 TPOT (ms):                           55.94
---------------Inter-token Latency----------------
Mean ITL (ms):                           7.67
Median ITL (ms):                         4.99
P99 ITL (ms):                            174.35
==================================================

RESULTS 10 prompts WITH CUDAGraph:
============ Serving Benchmark Result ============
Successful requests:                     10
Benchmark duration (s):                  3.99
Total input tokens:                      1369
Total generated tokens:                  1746
Request throughput (req/s):              2.51
Output token throughput (tok/s):         437.61
Total Token throughput (tok/s):          780.72
---------------Time to First Token----------------
Mean TTFT (ms):                          37.32
Median TTFT (ms):                        38.79
P99 TTFT (ms):                           40.16
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          5.20
Median TPOT (ms):                        5.23
P99 TPOT (ms):                           5.29
---------------Inter-token Latency----------------
Mean ITL (ms):                           5.16
Median ITL (ms):                         5.12
P99 ITL (ms):                            6.08
==================================================
