

######################################

#### LAZY CUDA GRAPH:
After some code modifications and using:
>> VLLM_LOGGING_LEVEL=DEBUG vllm serve meta-llama/Llama-3.2-1B

Initialization:
root@vllm-vm:/app/vllm# VLLM_LOGGING_LEVEL=DEBUG vllm serve meta-llama/Llama-3.2-1B
DEBUG 08-06 10:17:02 [__init__.py:30] No plugins for group vllm.platform_plugins found.
DEBUG 08-06 10:17:02 [__init__.py:34] Checking if TPU platform is available.
DEBUG 08-06 10:17:02 [__init__.py:52] TPU platform is not available because: No module named 'libtpu'
DEBUG 08-06 10:17:02 [__init__.py:58] Checking if CUDA platform is available.
DEBUG 08-06 10:17:02 [__init__.py:78] Confirmed CUDA platform is available.
DEBUG 08-06 10:17:02 [__init__.py:106] Checking if ROCm platform is available.
DEBUG 08-06 10:17:02 [__init__.py:120] ROCm platform is not available because: No module named 'amdsmi'
DEBUG 08-06 10:17:02 [__init__.py:127] Checking if XPU platform is available.
DEBUG 08-06 10:17:02 [__init__.py:146] XPU platform is not available because: No module named 'intel_extension_for_pytorch'
DEBUG 08-06 10:17:02 [__init__.py:153] Checking if CPU platform is available.
DEBUG 08-06 10:17:02 [__init__.py:175] Checking if Neuron platform is available.
DEBUG 08-06 10:17:02 [__init__.py:58] Checking if CUDA platform is available.
DEBUG 08-06 10:17:02 [__init__.py:78] Confirmed CUDA platform is available.
INFO 08-06 10:17:02 [__init__.py:241] Automatically detected platform cuda.
DEBUG 08-06 10:17:04 [utils.py:168] Setting VLLM_WORKER_MULTIPROC_METHOD to 'spawn'
DEBUG 08-06 10:17:04 [__init__.py:38] Available plugins for group vllm.general_plugins:
DEBUG 08-06 10:17:04 [__init__.py:40] - lora_filesystem_resolver -> vllm.plugins.lora_resolvers.filesystem_resolver:register_filesystem_resolver
DEBUG 08-06 10:17:04 [__init__.py:43] All plugins in this group will be loaded. Set `VLLM_PLUGINS` to control which plugins to load.
(APIServer pid=190319) INFO 08-06 10:17:05 [api_server.py:1774] vLLM API server version 0.1.dev8168+g475c1a0
(APIServer pid=190319) INFO 08-06 10:17:05 [utils.py:326] non-default args: {'model_tag': 'meta-llama/Llama-3.2-1B', 'model': 'meta-llama/Llama-3.2-1B'}
(APIServer pid=190319) INFO 08-06 10:17:11 [config.py:713] Resolved architecture: LlamaForCausalLM
(APIServer pid=190319) INFO 08-06 10:17:11 [config.py:1716] Using max model len 131072
(APIServer pid=190319) DEBUG 08-06 10:17:11 [arg_utils.py:1657] Setting max_num_batched_tokens to 2048 for OPENAI_API_SERVER usage context.
(APIServer pid=190319) DEBUG 08-06 10:17:11 [arg_utils.py:1666] Setting max_num_seqs to 256 for OPENAI_API_SERVER usage context.
(APIServer pid=190319) INFO 08-06 10:17:11 [config.py:2542] Chunked prefill is enabled with max_num_batched_tokens=2048.
DEBUG 08-06 10:17:15 [__init__.py:30] No plugins for group vllm.platform_plugins found.
DEBUG 08-06 10:17:15 [__init__.py:34] Checking if TPU platform is available.
DEBUG 08-06 10:17:15 [__init__.py:52] TPU platform is not available because: No module named 'libtpu'
DEBUG 08-06 10:17:15 [__init__.py:58] Checking if CUDA platform is available.
DEBUG 08-06 10:17:15 [__init__.py:78] Confirmed CUDA platform is available.
DEBUG 08-06 10:17:15 [__init__.py:106] Checking if ROCm platform is available.
DEBUG 08-06 10:17:15 [__init__.py:120] ROCm platform is not available because: No module named 'amdsmi'
DEBUG 08-06 10:17:15 [__init__.py:127] Checking if XPU platform is available.
DEBUG 08-06 10:17:15 [__init__.py:146] XPU platform is not available because: No module named 'intel_extension_for_pytorch'
DEBUG 08-06 10:17:15 [__init__.py:153] Checking if CPU platform is available.
DEBUG 08-06 10:17:15 [__init__.py:175] Checking if Neuron platform is available.
DEBUG 08-06 10:17:15 [__init__.py:58] Checking if CUDA platform is available.
DEBUG 08-06 10:17:15 [__init__.py:78] Confirmed CUDA platform is available.
INFO 08-06 10:17:15 [__init__.py:241] Automatically detected platform cuda.
INFO 08-06 10:17:18 [core.py:591] Waiting for init message from front-end.
(APIServer pid=190319) DEBUG 08-06 10:17:18 [utils.py:822] HELLO from local core engine process 0.
DEBUG 08-06 10:17:18 [core.py:599] Received init message: EngineHandshakeMetadata(addresses=EngineZmqAddresses(inputs=['ipc:///tmp/23a10ef2-1834-4a1b-9f68-a8b568d0c194'], outputs=['ipc:///tmp/8acb1cfe-3b3a-47f8-9b99-3910e0f33b19'], coordinator_input=None, coordinator_output=None, frontend_stats_publish_address=None), parallel_config={'data_parallel_master_ip': '127.0.0.1', 'data_parallel_master_port': 0, 'data_parallel_size': 1})
DEBUG 08-06 10:17:18 [__init__.py:38] Available plugins for group vllm.general_plugins:
DEBUG 08-06 10:17:18 [__init__.py:40] - lora_filesystem_resolver -> vllm.plugins.lora_resolvers.filesystem_resolver:register_filesystem_resolver
DEBUG 08-06 10:17:18 [__init__.py:43] All plugins in this group will be loaded. Set `VLLM_PLUGINS` to control which plugins to load.
INFO 08-06 10:17:18 [core.py:73] Initializing a V1 LLM engine (v0.1.dev8168+g475c1a0) with config: model='meta-llama/Llama-3.2-1B', speculative_config=None, tokenizer='meta-llama/Llama-3.2-1B', skip_tokenizer_init=False, tokenizer_mode=auto, revision=None, override_neuron_config={}, tokenizer_revision=None, trust_remote_code=False, dtype=torch.bfloat16, max_seq_len=131072, download_dir=None, load_format=auto, tensor_parallel_size=1, pipeline_parallel_size=1, disable_custom_all_reduce=False, quantization=None, enforce_eager=False, kv_cache_dtype=auto, device_config=cuda, decoding_config=DecodingConfig(backend='auto', disable_fallback=False, disable_any_whitespace=False, disable_additional_properties=False, reasoning_backend=''), observability_config=ObservabilityConfig(show_hidden_metrics_for_version=None, otlp_traces_endpoint=None, collect_detailed_traces=None), seed=0, served_model_name=meta-llama/Llama-3.2-1B, num_scheduler_steps=1, multi_step_stream_outputs=True, enable_prefix_caching=True, chunked_prefill_enabled=True, use_async_output_proc=True, pooler_config=None, compilation_config={"level":3,"debug_dump_path":"","cache_dir":"","backend":"","custom_ops":[],"splitting_ops":["vllm.unified_attention","vllm.unified_attention_with_output","vllm.mamba_mixer2"],"use_inductor":true,"compile_sizes":[],"inductor_compile_config":{"enable_auto_functionalized_v2":false},"inductor_passes":{},"use_cudagraph":true,"cudagraph_num_of_warmups":1,"cudagraph_capture_sizes":[512,504,496,488,480,472,464,456,448,440,432,424,416,408,400,392,384,376,368,360,352,344,336,328,320,312,304,296,288,280,272,264,256,248,240,232,224,216,208,200,192,184,176,168,160,152,144,136,128,120,112,104,96,88,80,72,64,56,48,40,32,24,16,8,4,2,1],"cudagraph_copy_inputs":false,"full_cuda_graph":false,"max_capture_size":512,"local_cache_dir":null}
DEBUG 08-06 10:17:18 [decorators.py:139] Inferred dynamic dimensions for forward method of <class 'vllm.model_executor.models.llama.LlamaModel'>: ['input_ids', 'positions', 'intermediate_tensors', 'inputs_embeds']
DEBUG 08-06 10:17:18 [decorators.py:139] Inferred dynamic dimensions for forward method of <class 'vllm.model_executor.models.llama_eagle3.LlamaModel'>: ['input_ids', 'positions', 'hidden_states']
DEBUG 08-06 10:17:19 [__init__.py:3053] Methods determine_num_available_blocks,device_config,get_cache_block_size_bytes not implemented in <vllm.v1.worker.gpu_worker.Worker object at 0x7f77bc2decc0>
DEBUG 08-06 10:17:19 [config.py:4998] enabled custom ops: Counter()
DEBUG 08-06 10:17:19 [config.py:5000] disabled custom ops: Counter()
DEBUG 08-06 10:17:19 [parallel_state.py:945] world_size=1 rank=0 local_rank=0 distributed_init_method=tcp://10.129.4.27:47761 backend=nccl
DEBUG 08-06 10:17:19 [parallel_state.py:996] Detected 1 nodes in the distributed environment
INFO 08-06 10:17:19 [parallel_state.py:1102] rank 0 in world size 1 is assigned as DP rank 0, PP rank 0, TP rank 0, EP rank 0
INFO 08-06 10:17:19 [topk_topp_sampler.py:49] Using FlashInfer for top-p & top-k sampling.
DEBUG 08-06 10:17:19 [config.py:4998] enabled custom ops: Counter()
DEBUG 08-06 10:17:19 [config.py:5000] disabled custom ops: Counter()
INFO 08-06 10:17:19 [gpu_model_runner.py:1924] Starting to load model meta-llama/Llama-3.2-1B...
INFO 08-06 10:17:19 [gpu_model_runner.py:1956] Loading model from scratch...
INFO 08-06 10:17:19 [cuda.py:305] Using Flash Attention backend on V1 engine.
DEBUG 08-06 10:17:19 [backends.py:39] Using InductorAdaptor
DEBUG 08-06 10:17:19 [config.py:4998] enabled custom ops: Counter()
DEBUG 08-06 10:17:19 [config.py:5000] disabled custom ops: Counter({'rms_norm': 33, 'silu_and_mul': 16, 'rotary_embedding': 1})
DEBUG 08-06 10:17:19 [base_loader.py:47] Loading weights on cuda ...
INFO 08-06 10:17:19 [weight_utils.py:296] Using model weights format ['*.safetensors']
INFO 08-06 10:17:20 [weight_utils.py:349] No model.safetensors.index.json found in remote.
Loading safetensors checkpoint shards:   0% Completed | 0/1 [00:00<?, ?it/s]
Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  2.49it/s]
Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  2.48it/s]

INFO 08-06 10:17:20 [default_loader.py:262] Loading weights took 0.47 seconds
INFO 08-06 10:17:21 [gpu_model_runner.py:1973] Model loading took 2.3185 GiB and 1.022598 seconds
DEBUG 08-06 10:17:21 [decorators.py:237] Start compiling function <code object forward at 0xdc166b0, file "/app/vllm/vllm/model_executor/models/llama.py", line 368>
DEBUG 08-06 10:17:24 [backends.py:487] Traced files (to be considered for compilation cache):
DEBUG 08-06 10:17:24 [backends.py:487] /app/vllm/vllm/attention/layer.py
DEBUG 08-06 10:17:24 [backends.py:487] /app/vllm/vllm/distributed/communication_op.py
DEBUG 08-06 10:17:24 [backends.py:487] /app/vllm/vllm/distributed/parallel_state.py
DEBUG 08-06 10:17:24 [backends.py:487] /app/vllm/vllm/model_executor/custom_op.py
DEBUG 08-06 10:17:24 [backends.py:487] /app/vllm/vllm/model_executor/layers/activation.py
DEBUG 08-06 10:17:24 [backends.py:487] /app/vllm/vllm/model_executor/layers/layernorm.py
DEBUG 08-06 10:17:24 [backends.py:487] /app/vllm/vllm/model_executor/layers/linear.py
DEBUG 08-06 10:17:24 [backends.py:487] /app/vllm/vllm/model_executor/layers/rotary_embedding.py
DEBUG 08-06 10:17:24 [backends.py:487] /app/vllm/vllm/model_executor/layers/utils.py
DEBUG 08-06 10:17:24 [backends.py:487] /app/vllm/vllm/model_executor/layers/vocab_parallel_embedding.py
DEBUG 08-06 10:17:24 [backends.py:487] /app/vllm/vllm/model_executor/models/llama.py
DEBUG 08-06 10:17:24 [backends.py:487] /app/vllm/vllm/platforms/interface.py
DEBUG 08-06 10:17:24 [backends.py:487] /usr/local/lib/python3.12/dist-packages/torch/_dynamo/polyfills/__init__.py
DEBUG 08-06 10:17:24 [backends.py:487] /usr/local/lib/python3.12/dist-packages/torch/nn/modules/container.py
DEBUG 08-06 10:17:24 [backends.py:487] /usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py
INFO 08-06 10:17:24 [backends.py:534] Using cache directory: /root/.cache/vllm/torch_compile_cache/25d4010308/rank_0_0/backbone for vLLM's torch.compile
INFO 08-06 10:17:24 [backends.py:545] Dynamo bytecode transform time: 3.60 s
DEBUG 08-06 10:17:25 [backends.py:125] Directly load the 0-th graph for dynamic shape from inductor via handle ('foww4arjrmo5ntlgmdcjpr7xst6lgk62vi5tjijgfsyto7r2mfpr', '/root/.cache/vllm/torch_compile_cache/25d4010308/rank_0_0/inductor_cache/3i/c3ifc7om5773max53s4uxbx3idf2b3yt2edem25nule7wkt3ttgy.py')
DEBUG 08-06 10:17:25 [backends.py:157] TOTAL LOADING TIME: 0.050582 s
DEBUG 08-06 10:17:25 [backends.py:125] Directly load the 1-th graph for dynamic shape from inductor via handle ('fiuf3tlstrqbuky24wigbejkm2cnwqxxn5nouiwmwuyprblrvtoa', '/root/.cache/vllm/torch_compile_cache/25d4010308/rank_0_0/inductor_cache/ky/cky4e3zslhbt7nnarafnykqxtqf6tjofdtfo5yqj5mc7aqemgqwr.py')
DEBUG 08-06 10:17:25 [backends.py:157] TOTAL LOADING TIME: 0.053213 s
DEBUG 08-06 10:17:25 [backends.py:125] Directly load the 2-th graph for dynamic shape from inductor via handle ('fiuf3tlstrqbuky24wigbejkm2cnwqxxn5nouiwmwuyprblrvtoa', '/root/.cache/vllm/torch_compile_cache/25d4010308/rank_0_0/inductor_cache/ky/cky4e3zslhbt7nnarafnykqxtqf6tjofdtfo5yqj5mc7aqemgqwr.py')
DEBUG 08-06 10:17:25 [backends.py:157] TOTAL LOADING TIME: 0.050275 s
DEBUG 08-06 10:17:25 [backends.py:125] Directly load the 3-th graph for dynamic shape from inductor via handle ('fiuf3tlstrqbuky24wigbejkm2cnwqxxn5nouiwmwuyprblrvtoa', '/root/.cache/vllm/torch_compile_cache/25d4010308/rank_0_0/inductor_cache/ky/cky4e3zslhbt7nnarafnykqxtqf6tjofdtfo5yqj5mc7aqemgqwr.py')
DEBUG 08-06 10:17:25 [backends.py:157] TOTAL LOADING TIME: 0.051407 s
DEBUG 08-06 10:17:25 [backends.py:125] Directly load the 4-th graph for dynamic shape from inductor via handle ('fiuf3tlstrqbuky24wigbejkm2cnwqxxn5nouiwmwuyprblrvtoa', '/root/.cache/vllm/torch_compile_cache/25d4010308/rank_0_0/inductor_cache/ky/cky4e3zslhbt7nnarafnykqxtqf6tjofdtfo5yqj5mc7aqemgqwr.py')
DEBUG 08-06 10:17:25 [backends.py:157] TOTAL LOADING TIME: 0.054703 s
DEBUG 08-06 10:17:25 [backends.py:125] Directly load the 5-th graph for dynamic shape from inductor via handle ('fiuf3tlstrqbuky24wigbejkm2cnwqxxn5nouiwmwuyprblrvtoa', '/root/.cache/vllm/torch_compile_cache/25d4010308/rank_0_0/inductor_cache/ky/cky4e3zslhbt7nnarafnykqxtqf6tjofdtfo5yqj5mc7aqemgqwr.py')
DEBUG 08-06 10:17:25 [backends.py:157] TOTAL LOADING TIME: 0.049369 s
DEBUG 08-06 10:17:26 [backends.py:125] Directly load the 6-th graph for dynamic shape from inductor via handle ('fiuf3tlstrqbuky24wigbejkm2cnwqxxn5nouiwmwuyprblrvtoa', '/root/.cache/vllm/torch_compile_cache/25d4010308/rank_0_0/inductor_cache/ky/cky4e3zslhbt7nnarafnykqxtqf6tjofdtfo5yqj5mc7aqemgqwr.py')
DEBUG 08-06 10:17:26 [backends.py:157] TOTAL LOADING TIME: 0.052774 s
DEBUG 08-06 10:17:26 [backends.py:125] Directly load the 7-th graph for dynamic shape from inductor via handle ('fiuf3tlstrqbuky24wigbejkm2cnwqxxn5nouiwmwuyprblrvtoa', '/root/.cache/vllm/torch_compile_cache/25d4010308/rank_0_0/inductor_cache/ky/cky4e3zslhbt7nnarafnykqxtqf6tjofdtfo5yqj5mc7aqemgqwr.py')
DEBUG 08-06 10:17:26 [backends.py:157] TOTAL LOADING TIME: 0.051285 s
DEBUG 08-06 10:17:26 [backends.py:125] Directly load the 8-th graph for dynamic shape from inductor via handle ('fiuf3tlstrqbuky24wigbejkm2cnwqxxn5nouiwmwuyprblrvtoa', '/root/.cache/vllm/torch_compile_cache/25d4010308/rank_0_0/inductor_cache/ky/cky4e3zslhbt7nnarafnykqxtqf6tjofdtfo5yqj5mc7aqemgqwr.py')
DEBUG 08-06 10:17:26 [backends.py:157] TOTAL LOADING TIME: 0.052169 s
DEBUG 08-06 10:17:26 [backends.py:125] Directly load the 9-th graph for dynamic shape from inductor via handle ('fiuf3tlstrqbuky24wigbejkm2cnwqxxn5nouiwmwuyprblrvtoa', '/root/.cache/vllm/torch_compile_cache/25d4010308/rank_0_0/inductor_cache/ky/cky4e3zslhbt7nnarafnykqxtqf6tjofdtfo5yqj5mc7aqemgqwr.py')
DEBUG 08-06 10:17:26 [backends.py:157] TOTAL LOADING TIME: 0.050370 s
DEBUG 08-06 10:17:26 [backends.py:125] Directly load the 10-th graph for dynamic shape from inductor via handle ('fiuf3tlstrqbuky24wigbejkm2cnwqxxn5nouiwmwuyprblrvtoa', '/root/.cache/vllm/torch_compile_cache/25d4010308/rank_0_0/inductor_cache/ky/cky4e3zslhbt7nnarafnykqxtqf6tjofdtfo5yqj5mc7aqemgqwr.py')
DEBUG 08-06 10:17:26 [backends.py:157] TOTAL LOADING TIME: 0.051270 s
DEBUG 08-06 10:17:26 [backends.py:125] Directly load the 11-th graph for dynamic shape from inductor via handle ('fiuf3tlstrqbuky24wigbejkm2cnwqxxn5nouiwmwuyprblrvtoa', '/root/.cache/vllm/torch_compile_cache/25d4010308/rank_0_0/inductor_cache/ky/cky4e3zslhbt7nnarafnykqxtqf6tjofdtfo5yqj5mc7aqemgqwr.py')
DEBUG 08-06 10:17:26 [backends.py:157] TOTAL LOADING TIME: 0.053634 s
DEBUG 08-06 10:17:27 [backends.py:125] Directly load the 12-th graph for dynamic shape from inductor via handle ('fiuf3tlstrqbuky24wigbejkm2cnwqxxn5nouiwmwuyprblrvtoa', '/root/.cache/vllm/torch_compile_cache/25d4010308/rank_0_0/inductor_cache/ky/cky4e3zslhbt7nnarafnykqxtqf6tjofdtfo5yqj5mc7aqemgqwr.py')
DEBUG 08-06 10:17:27 [backends.py:157] TOTAL LOADING TIME: 0.052154 s
DEBUG 08-06 10:17:27 [backends.py:125] Directly load the 13-th graph for dynamic shape from inductor via handle ('fiuf3tlstrqbuky24wigbejkm2cnwqxxn5nouiwmwuyprblrvtoa', '/root/.cache/vllm/torch_compile_cache/25d4010308/rank_0_0/inductor_cache/ky/cky4e3zslhbt7nnarafnykqxtqf6tjofdtfo5yqj5mc7aqemgqwr.py')
DEBUG 08-06 10:17:27 [backends.py:157] TOTAL LOADING TIME: 0.051517 s
DEBUG 08-06 10:17:27 [backends.py:125] Directly load the 14-th graph for dynamic shape from inductor via handle ('fiuf3tlstrqbuky24wigbejkm2cnwqxxn5nouiwmwuyprblrvtoa', '/root/.cache/vllm/torch_compile_cache/25d4010308/rank_0_0/inductor_cache/ky/cky4e3zslhbt7nnarafnykqxtqf6tjofdtfo5yqj5mc7aqemgqwr.py')
DEBUG 08-06 10:17:27 [backends.py:157] TOTAL LOADING TIME: 0.049324 s
DEBUG 08-06 10:17:27 [backends.py:125] Directly load the 15-th graph for dynamic shape from inductor via handle ('fiuf3tlstrqbuky24wigbejkm2cnwqxxn5nouiwmwuyprblrvtoa', '/root/.cache/vllm/torch_compile_cache/25d4010308/rank_0_0/inductor_cache/ky/cky4e3zslhbt7nnarafnykqxtqf6tjofdtfo5yqj5mc7aqemgqwr.py')
DEBUG 08-06 10:17:27 [backends.py:157] TOTAL LOADING TIME: 0.055930 s
DEBUG 08-06 10:17:27 [backends.py:125] Directly load the 16-th graph for dynamic shape from inductor via handle ('fvbvyhtr37kusmqij6uiinukna7ifz6inpaixj2ehzbq2ipe7nms', '/root/.cache/vllm/torch_compile_cache/25d4010308/rank_0_0/inductor_cache/ts/ctsi5px5m6j4xfpnuarwb5hejxunelgdyavbmz52ohrcimfwoaig.py')
DEBUG 08-06 10:17:27 [backends.py:157] TOTAL LOADING TIME: 0.026462 s
INFO 08-06 10:17:27 [backends.py:165] Directly load the compiled graph(s) for dynamic shape from the cache, took 2.517 s
INFO 08-06 10:17:28 [monitor.py:34] torch.compile takes 3.60 s in total
/usr/local/lib/python3.12/dist-packages/torch/utils/cpp_extension.py:2356: UserWarning: TORCH_CUDA_ARCH_LIST is not set, all archs for visible cards are included for compilation. 
If this is not desired, please set os.environ['TORCH_CUDA_ARCH_LIST'].
  warnings.warn(
(APIServer pid=190319) DEBUG 08-06 10:17:28 [utils.py:741] Waiting for 1 local, 0 remote core engine proc(s) to start.
DEBUG 08-06 10:17:28 [gpu_worker.py:265] Initial free memory: 43.82 GiB; Requested memory: 0.90 (util), 39.88 GiB
DEBUG 08-06 10:17:28 [gpu_worker.py:272] Free memory after profiling: 41.43 GiB (total), 37.49 GiB (within requested)
DEBUG 08-06 10:17:28 [gpu_worker.py:278] Memory profiling takes 7.26 seconds. Total non KV cache memory: 2.78GiB; torch peak memory increase: 0.45GiB; non-torch forward increase memory: 0.02GiB; weights memory: 2.32GiB.
INFO 08-06 10:17:28 [gpu_worker.py:279] Available KV cache memory: 37.10 GiB
INFO 08-06 10:17:28 [kv_cache_utils.py:831] GPU KV cache size: 1,215,552 tokens
INFO 08-06 10:17:28 [kv_cache_utils.py:835] Maximum concurrency for 131,072 tokens per request: 9.27x
DEBUG 08-06 10:17:28 [config.py:4998] enabled custom ops: Counter()
DEBUG 08-06 10:17:28 [config.py:5000] disabled custom ops: Counter({'rms_norm': 33, 'silu_and_mul': 16, 'rotary_embedding': 1})
DEBUG 08-06 10:17:28 [cuda_piecewise_backend.py:154] Warming up 1/1 for shape 256
INFO 08-06 10:17:28 [core.py:201] init engine (profile, create kv cache, warmup model) took 7.70 seconds
(APIServer pid=190319) DEBUG 08-06 10:17:29 [utils.py:822] READY from local core engine process 0.
DEBUG 08-06 10:17:29 [core.py:681] EngineCore waiting for work.
(APIServer pid=190319) INFO 08-06 10:17:29 [loggers.py:142] Engine 000: vllm cache_config_info with initialization after num_gpu_blocks is: 75972
DEBUG 08-06 10:17:29 [core.py:681] EngineCore waiting for work.
DEBUG 08-06 10:17:29 [core.py:681] EngineCore waiting for work.
(APIServer pid=190319) INFO 08-06 10:17:29 [api_server.py:1595] Supported_tasks: ['generate']
(APIServer pid=190319) WARNING 08-06 10:17:29 [config.py:1616] Default sampling parameters have been overridden by the model's Hugging Face generation config recommended from the model creator. If this is not intended, please relaunch vLLM instance with `--generation-config vllm`.
(APIServer pid=190319) INFO 08-06 10:17:29 [serving_responses.py:89] Using default chat sampling params from model: {'temperature': 0.6, 'top_p': 0.9}
(APIServer pid=190319) INFO 08-06 10:17:29 [serving_chat.py:125] Using default chat sampling params from model: {'temperature': 0.6, 'top_p': 0.9}
(APIServer pid=190319) INFO 08-06 10:17:29 [serving_completion.py:77] Using default completion sampling params from model: {'temperature': 0.6, 'top_p': 0.9}
(APIServer pid=190319) INFO 08-06 10:17:29 [api_server.py:1847] Starting vLLM API server 0 on http://0.0.0.0:8000
(APIServer pid=190319) INFO 08-06 10:17:29 [launcher.py:29] Available routes are:
(APIServer pid=190319) INFO 08-06 10:17:29 [launcher.py:37] Route: /openapi.json, Methods: GET, HEAD
(APIServer pid=190319) INFO 08-06 10:17:29 [launcher.py:37] Route: /docs, Methods: GET, HEAD
(APIServer pid=190319) INFO 08-06 10:17:29 [launcher.py:37] Route: /docs/oauth2-redirect, Methods: GET, HEAD
(APIServer pid=190319) INFO 08-06 10:17:29 [launcher.py:37] Route: /redoc, Methods: GET, HEAD
(APIServer pid=190319) INFO 08-06 10:17:29 [launcher.py:37] Route: /health, Methods: GET
(APIServer pid=190319) INFO 08-06 10:17:29 [launcher.py:37] Route: /load, Methods: GET
(APIServer pid=190319) INFO 08-06 10:17:29 [launcher.py:37] Route: /ping, Methods: POST
(APIServer pid=190319) INFO 08-06 10:17:29 [launcher.py:37] Route: /ping, Methods: GET
(APIServer pid=190319) INFO 08-06 10:17:29 [launcher.py:37] Route: /tokenize, Methods: POST
(APIServer pid=190319) INFO 08-06 10:17:29 [launcher.py:37] Route: /detokenize, Methods: POST
(APIServer pid=190319) INFO 08-06 10:17:29 [launcher.py:37] Route: /v1/models, Methods: GET
(APIServer pid=190319) INFO 08-06 10:17:29 [launcher.py:37] Route: /version, Methods: GET
(APIServer pid=190319) INFO 08-06 10:17:29 [launcher.py:37] Route: /v1/responses, Methods: POST
(APIServer pid=190319) INFO 08-06 10:17:29 [launcher.py:37] Route: /v1/responses/{response_id}, Methods: GET
(APIServer pid=190319) INFO 08-06 10:17:29 [launcher.py:37] Route: /v1/responses/{response_id}/cancel, Methods: POST
(APIServer pid=190319) INFO 08-06 10:17:29 [launcher.py:37] Route: /v1/chat/completions, Methods: POST
(APIServer pid=190319) INFO 08-06 10:17:29 [launcher.py:37] Route: /v1/completions, Methods: POST
(APIServer pid=190319) INFO 08-06 10:17:29 [launcher.py:37] Route: /v1/embeddings, Methods: POST
(APIServer pid=190319) INFO 08-06 10:17:29 [launcher.py:37] Route: /pooling, Methods: POST
(APIServer pid=190319) INFO 08-06 10:17:29 [launcher.py:37] Route: /classify, Methods: POST
(APIServer pid=190319) INFO 08-06 10:17:29 [launcher.py:37] Route: /score, Methods: POST
(APIServer pid=190319) INFO 08-06 10:17:29 [launcher.py:37] Route: /v1/score, Methods: POST
(APIServer pid=190319) INFO 08-06 10:17:29 [launcher.py:37] Route: /v1/audio/transcriptions, Methods: POST
(APIServer pid=190319) INFO 08-06 10:17:29 [launcher.py:37] Route: /v1/audio/translations, Methods: POST
(APIServer pid=190319) INFO 08-06 10:17:29 [launcher.py:37] Route: /rerank, Methods: POST
(APIServer pid=190319) INFO 08-06 10:17:29 [launcher.py:37] Route: /v1/rerank, Methods: POST
(APIServer pid=190319) INFO 08-06 10:17:29 [launcher.py:37] Route: /v2/rerank, Methods: POST
(APIServer pid=190319) INFO 08-06 10:17:29 [launcher.py:37] Route: /scale_elastic_ep, Methods: POST
(APIServer pid=190319) INFO 08-06 10:17:29 [launcher.py:37] Route: /is_scaling_elastic_ep, Methods: POST
(APIServer pid=190319) INFO 08-06 10:17:29 [launcher.py:37] Route: /invocations, Methods: POST
(APIServer pid=190319) INFO 08-06 10:17:29 [launcher.py:37] Route: /metrics, Methods: GET
(APIServer pid=190319) INFO:     Started server process [190319]
(APIServer pid=190319) INFO:     Waiting for application startup.
(APIServer pid=190319) INFO:     Application startup complete.

And running:
root@vllm-vm:/app# python3 vllm/benchmarks/benchmark_serving.py --backend vllm --model meta-llama/Llama-3.2-1B --endpoint /v1/completions --dataset-name sharegpt --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json --num-prompts 10
INFO 08-06 10:19:31 [__init__.py:241] Automatically detected platform cuda.
/app/vllm/benchmarks/benchmark_serving.py:1299: DeprecationWarning: benchmark_serving.py is deprecated and will be removed in a future version. Please use 'vllm bench serve' instead.
  main(args)
Namespace(backend='vllm', base_url=None, host='127.0.0.1', port=8000, endpoint='/v1/completions', dataset_name='sharegpt', dataset_path='ShareGPT_V3_unfiltered_cleaned_split.json', no_stream=False, max_concurrency=None, model='meta-llama/Llama-3.2-1B', tokenizer=None, use_beam_search=False, num_prompts=10, logprobs=None, request_rate=inf, burstiness=1.0, seed=0, trust_remote_code=False, disable_tqdm=False, profile=False, save_result=False, save_detailed=False, append_result=False, metadata=None, result_dir=None, result_filename=None, ignore_eos=False, percentile_metrics='ttft,tpot,itl', metric_percentiles='99', goodput=None, custom_output_len=256, custom_skip_chat_template=False, sonnet_input_len=550, sonnet_output_len=150, sonnet_prefix_len=200, sharegpt_output_len=None, random_input_len=1024, random_output_len=128, random_range_ratio=0.0, random_prefix_len=0, hf_subset=None, hf_split=None, hf_output_len=None, top_p=None, top_k=None, min_p=None, temperature=None, tokenizer_mode='auto', served_model_name=None, lora_modules=None, ramp_up_strategy=None, ramp_up_start_rps=None, ramp_up_end_rps=None)
Starting initial single prompt test run...
Initial test run completed. Starting main benchmark run...
Traffic request rate: inf RPS.
Burstiness factor: 1.0 (Poisson process)
Maximum request concurrency: None
100%|████████████████████████████████████████████████████████████████████████████████████████| 10/10 [00:04<00:00,  2.06it/s]
============ Serving Benchmark Result ============
Successful requests:                     10        
Benchmark duration (s):                  4.85      
Total input tokens:                      1369      
Total generated tokens:                  1196      
Request throughput (req/s):              2.06      
Output token throughput (tok/s):         246.74    
Total Token throughput (tok/s):          529.17    
---------------Time to First Token----------------
Mean TTFT (ms):                          317.01    
Median TTFT (ms):                        319.78    
P99 TTFT (ms):                           321.26    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          20.59     
Median TPOT (ms):                        11.19     
P99 TPOT (ms):                           50.65     
---------------Inter-token Latency----------------
Mean ITL (ms):                           7.67      
Median ITL (ms):                         5.01      
P99 ITL (ms):                            233.46    
==================================================

The server shows:
(APIServer pid=190319) INFO 08-06 10:19:36 [logger.py:41] Received request cmpl-6f57010575f74a80b7b30cd80570714c-0: prompt: 'Do you know the book Traction by Gino Wickman', params: SamplingParams(n=1, presence_penalty=0.0, frequency_penalty=0.0, repetition_penalty=1.0, temperature=0.0, top_p=1.0, top_k=0, min_p=0.0, seed=None, stop=[], stop_token_ids=[], bad_words=[], include_stop_str_in_output=False, ignore_eos=False, max_tokens=120, min_tokens=0, logprobs=None, prompt_logprobs=None, skip_special_tokens=True, spaces_between_special_tokens=True, truncate_prompt_tokens=None, guided_decoding=None, extra_args=None), prompt_token_ids: [128000, 5519, 499, 1440, 279, 2363, 350, 16597, 555, 480, 3394, 75206, 1543], prompt_embeds shape: None, lora_request: None.
(APIServer pid=190319) INFO:     127.0.0.1:49854 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=190319) INFO 08-06 10:19:36 [async_llm.py:273] Added request cmpl-6f57010575f74a80b7b30cd80570714c-0.
DEBUG 08-06 10:19:36 [core.py:687] EngineCore loop active.
INFO 08-06 10:19:36 [gpu_worker.py:370] DIEGO: CUDAgraph in execution time for 13 input tokens
INFO 08-06 10:19:36 [gpu_worker.py:376] Graph capturing finished in 0.005 secs
DEBUG 08-06 10:19:36 [cuda_piecewise_backend.py:154] Warming up 1/1 for shape 16
INFO 08-06 10:19:36 [gpu_worker.py:370] DIEGO: CUDAgraph in execution time for 1 input tokens
DEBUG 08-06 10:19:36 [cuda_piecewise_backend.py:154] Warming up 1/1 for shape 1
INFO 08-06 10:19:36 [gpu_worker.py:376] Graph capturing finished in 0.004 secs
DEBUG 08-06 10:19:36 [cuda_piecewise_backend.py:165] Capturing a cudagraph for shape 1
DEBUG 08-06 10:19:36 [core.py:681] EngineCore waiting for work.
(APIServer pid=190319) INFO 08-06 10:19:36 [logger.py:41] Received request cmpl-ef3e576cc1bd4f6eb302c80de3e1d08f-0: prompt: 'Do you know the book Traction by Gino Wickman', params: SamplingParams(n=1, presence_penalty=0.0, frequency_penalty=0.0, repetition_penalty=1.0, temperature=0.0, top_p=1.0, top_k=0, min_p=0.0, seed=None, stop=[], stop_token_ids=[], bad_words=[], include_stop_str_in_output=False, ignore_eos=False, max_tokens=120, min_tokens=0, logprobs=None, prompt_logprobs=None, skip_special_tokens=True, spaces_between_special_tokens=True, truncate_prompt_tokens=None, guided_decoding=None, extra_args=None), prompt_token_ids: [128000, 5519, 499, 1440, 279, 2363, 350, 16597, 555, 480, 3394, 75206, 1543], prompt_embeds shape: None, lora_request: None.
(APIServer pid=190319) INFO 08-06 10:19:36 [logger.py:41] Received request cmpl-cb68b955c98846ac8de69a26b47928cb-0: prompt: 'help me create a rust app that supports the elevenlabs.io api and that can read the contents of clipboard aloud using tts', params: SamplingParams(n=1, presence_penalty=0.0, frequency_penalty=0.0, repetition_penalty=1.0, temperature=0.0, top_p=1.0, top_k=0, min_p=0.0, seed=None, stop=[], stop_token_ids=[], bad_words=[], include_stop_str_in_output=False, ignore_eos=False, max_tokens=770, min_tokens=0, logprobs=None, prompt_logprobs=None, skip_special_tokens=True, spaces_between_special_tokens=True, truncate_prompt_tokens=None, guided_decoding=None, extra_args=None), prompt_token_ids: [128000, 8823, 757, 1893, 264, 23941, 917, 430, 11815, 279, 45314, 71371, 4340, 6464, 323, 430, 649, 1373, 279, 8970, 315, 47134, 71511, 1701, 99640], prompt_embeds shape: None, lora_request: None.
(APIServer pid=190319) INFO 08-06 10:19:36 [logger.py:41] Received request cmpl-f76a1d9a10384fde9646d3dd8319084e-0: prompt: 'create new version. we will call it: "second draft". You need to reformat Filters part to be more ease to read', params: SamplingParams(n=1, presence_penalty=0.0, frequency_penalty=0.0, repetition_penalty=1.0, temperature=0.0, top_p=1.0, top_k=0, min_p=0.0, seed=None, stop=[], stop_token_ids=[], bad_words=[], include_stop_str_in_output=False, ignore_eos=False, max_tokens=233, min_tokens=0, logprobs=None, prompt_logprobs=None, skip_special_tokens=True, spaces_between_special_tokens=True, truncate_prompt_tokens=None, guided_decoding=None, extra_args=None), prompt_token_ids: [128000, 3261, 502, 2373, 13, 584, 690, 1650, 433, 25, 330, 5686, 10165, 3343, 1472, 1205, 311, 312, 2293, 46112, 961, 311, 387, 810, 14553, 311, 1373], prompt_embeds shape: None, lora_request: None.
(APIServer pid=190319) INFO 08-06 10:19:36 [logger.py:41] Received request cmpl-a57cd123d1c542db8318c11954541f1f-0: prompt: 'in the jtbd context whats a push?', params: SamplingParams(n=1, presence_penalty=0.0, frequency_penalty=0.0, repetition_penalty=1.0, temperature=0.0, top_p=1.0, top_k=0, min_p=0.0, seed=None, stop=[], stop_token_ids=[], bad_words=[], include_stop_str_in_output=False, ignore_eos=False, max_tokens=194, min_tokens=0, logprobs=None, prompt_logprobs=None, skip_special_tokens=True, spaces_between_special_tokens=True, truncate_prompt_tokens=None, guided_decoding=None, extra_args=None), prompt_token_ids: [128000, 258, 279, 92176, 9117, 2317, 41209, 264, 4585, 30], prompt_embeds shape: None, lora_request: None.
(APIServer pid=190319) INFO 08-06 10:19:36 [logger.py:41] Received request cmpl-da2b11c536274ed590c5d3df0c8fdc50-0: prompt: "| Project Charter |  |\n| --- | --- |\n|  | 2. Users may not be satisfied with the functionality or usability of the application, which could affect user adoption. <br> 3. Security breaches or data loss could occur, which could compromise user data and trust. <br> 4. The project budget may exceed expectations due to unforeseen issues or scope changes. |\n| **Approvals:** | The following approvals are required for this project: <br> - Project Charter: [Project Sponsor's Name] <br> - Finalized Design: [Project Sponsor's Name] <br> - User Acceptance Testing: [Project Sponsor's Name] |\n| **Project Success Criteria:** | The success of the project will be measured by the following criteria: <br> 1. Completion of the project on time and within budget. <br> 2. User satisfaction with the application and its features. <br> 3. Reduction in the time and effort required to generate appraisal reports. <br> 4. Improvement in the accuracy and quality of appraisal reports. <br> 5. Increased efficiency in the appraisal process. |\n| **Conclusion:** | This project charter outlines the scope, objectives, deliverables, timeline, budget, project team, assumptions and risks, and approvals required for the development of a web-based commercial appraisal report writing application. The success of the project will be measured by completion on time and within budget, user satisfaction, reduction in time and effort required for appraisal reports, improved accuracy and quality of appraisal reports, and increased efficiency in the appraisal process. |", params: SamplingParams(n=1, presence_penalty=0.0, frequency_penalty=0.0, repetition_penalty=1.0, temperature=0.0, top_p=1.0, top_k=0, min_p=0.0, seed=None, stop=[], stop_token_ids=[], bad_words=[], include_stop_str_in_output=False, ignore_eos=False, max_tokens=101, min_tokens=0, logprobs=None, prompt_logprobs=None, skip_special_tokens=True, spaces_between_special_tokens=True, truncate_prompt_tokens=None, guided_decoding=None, extra_args=None), prompt_token_ids: [128000, 91, 5907, 49705, 765, 220, 9432, 91, 12730, 765, 12730, 9432, 91, 220, 765, 220, 17, 13, 14969, 1253, 539, 387, 20097, 449, 279, 15293, 477, 76160, 315, 279, 3851, 11, 902, 1436, 7958, 1217, 25375, 13, 366, 1347, 29, 220, 18, 13, 8398, 69140, 477, 828, 4814, 1436, 12446, 11, 902, 1436, 30485, 1217, 828, 323, 7095, 13, 366, 1347, 29, 220, 19, 13, 578, 2447, 8199, 1253, 12771, 17078, 4245, 311, 96691, 29412, 4819, 477, 7036, 4442, 13, 9432, 91, 3146, 29688, 26678, 68063, 765, 578, 2768, 83923, 527, 2631, 369, 420, 2447, 25, 366, 1347, 29, 482, 5907, 49705, 25, 510, 8006, 48661, 596, 4076, 60, 366, 1347, 29, 482, 13321, 1534, 7127, 25, 510, 8006, 48661, 596, 4076, 60, 366, 1347, 29, 482, 2724, 21496, 685, 27866, 25, 510, 8006, 48661, 596, 4076, 60, 9432, 91, 3146, 8006, 13346, 14577, 68063, 765, 578, 2450, 315, 279, 2447, 690, 387, 17303, 555, 279, 2768, 13186, 25, 366, 1347, 29, 220, 16, 13, 57350, 315, 279, 2447, 389, 892, 323, 2949, 8199, 13, 366, 1347, 29, 220, 17, 13, 2724, 24617, 449, 279, 3851, 323, 1202, 4519, 13, 366, 1347, 29, 220, 18, 13, 59200, 304, 279, 892, 323, 5149, 2631, 311, 7068, 79392, 6821, 13, 366, 1347, 29, 220, 19, 13, 53751, 304, 279, 13708, 323, 4367, 315, 79392, 6821, 13, 366, 1347, 29, 220, 20, 13, 62697, 15374, 304, 279, 79392, 1920, 13, 9432, 91, 3146, 44534, 68063, 765, 1115, 2447, 38124, 50729, 279, 7036, 11, 26470, 11, 6493, 4893, 11, 25845, 11, 8199, 11, 2447, 2128, 11, 32946, 323, 15635, 11, 323, 83923, 2631, 369, 279, 4500, 315, 264, 3566, 6108, 8518, 79392, 1934, 4477, 3851, 13, 578, 2450, 315, 279, 2447, 690, 387, 17303, 555, 9954, 389, 892, 323, 2949, 8199, 11, 1217, 24617, 11, 14278, 304, 892, 323, 5149, 2631, 369, 79392, 6821, 11, 13241, 13708, 323, 4367, 315, 79392, 6821, 11, 323, 7319, 15374, 304, 279, 79392, 1920, 13, 765], prompt_embeds shape: None, lora_request: None.
(APIServer pid=190319) INFO 08-06 10:19:36 [logger.py:41] Received request cmpl-c80c1072660349f2898dd6f3b3e1c9dd-0: prompt: 'create react and node and express js web app for creating or add dummy data and show and How I can deploy the code after create build.', params: SamplingParams(n=1, presence_penalty=0.0, frequency_penalty=0.0, repetition_penalty=1.0, temperature=0.0, top_p=1.0, top_k=0, min_p=0.0, seed=None, stop=[], stop_token_ids=[], bad_words=[], include_stop_str_in_output=False, ignore_eos=False, max_tokens=741, min_tokens=0, logprobs=None, prompt_logprobs=None, skip_special_tokens=True, spaces_between_special_tokens=True, truncate_prompt_tokens=None, guided_decoding=None, extra_args=None), prompt_token_ids: [128000, 3261, 14085, 323, 2494, 323, 3237, 7139, 3566, 917, 369, 6968, 477, 923, 17741, 828, 323, 1501, 323, 2650, 358, 649, 10739, 279, 2082, 1306, 1893, 1977, 13], prompt_embeds shape: None, lora_request: None.
(APIServer pid=190319) INFO 08-06 10:19:36 [logger.py:41] Received request cmpl-98efd9f8a11243c9910565e17bb21e54-0: prompt: "You can use Django's built-in task scheduling framework, `django-background-tasks`, to schedule the training of your model every `n` number of days.\n\nHere's a high-level overview of how you can implement this:\n\n1. Install the `django-background-tasks` library:\n```css\npip install django-background-tasks\n```\n2. Add `background_tasks` to your `INSTALLED_APPS` in the `settings.py` file:\n```python\nINSTALLED_APPS = [\n    # ...\n    'background_tasks',\n    # ...\n]\n```\n3. Define a task function to train your model:\n```python\nimport pickle\nimport numpy as np\nfrom .models import ModelPath\n\ndef train_model():\n    # Code to train your model\n    model = ...\n    path = ...\n\n    # Save the model to disk\n    pickle.dump(model, open(path, 'wb'))\n\n    # Update the database with the new model path\n    model_path = ModelPath.objects.last()\n    model_path.path = path\n    model_path.save()\n```\n4. Register the task in the `tasks.py` file of your app:\n```python\nfrom background_tasks import background\n\n@background(schedule=60 * 60 * 24 * n)  # Schedule the task to run every n days\ndef run_train_model_task():\n    train_model()\n```\n5. Run the background task worker:\n```\npython manage.py process_tasks\n```\nIn this example, the `train_model` function trains your model, saves it to disk, and updates the database with the new model path. The `run_train_model_task` function is a background task that is scheduled to run every `n` days and calls the `train_model` function. The `process_tasks` command must be run to start the background task worker.\n\nNote: This is just one way to schedule the training of your model. The exact implementation will depend on your specific requirements and constraints.", params: SamplingParams(n=1, presence_penalty=0.0, frequency_penalty=0.0, repetition_penalty=1.0, temperature=0.0, top_p=1.0, top_k=0, min_p=0.0, seed=None, stop=[], stop_token_ids=[], bad_words=[], include_stop_str_in_output=False, ignore_eos=False, max_tokens=9, min_tokens=0, logprobs=None, prompt_logprobs=None, skip_special_tokens=True, spaces_between_special_tokens=True, truncate_prompt_tokens=None, guided_decoding=None, extra_args=None), prompt_token_ids: [128000, 2675, 649, 1005, 53704, 596, 5918, 3502, 3465, 38952, 12914, 11, 1595, 13887, 43034, 2442, 4707, 7964, 311, 9899, 279, 4967, 315, 701, 1646, 1475, 1595, 77, 63, 1396, 315, 2919, 382, 8586, 596, 264, 1579, 11852, 24131, 315, 1268, 499, 649, 4305, 420, 1473, 16, 13, 19796, 279, 1595, 13887, 43034, 2442, 4707, 63, 6875, 512, 74694, 5254, 198, 52601, 4685, 8426, 43034, 2442, 4707, 198, 14196, 4077, 17, 13, 2758, 1595, 6884, 33923, 63, 311, 701, 1595, 65562, 78340, 90854, 63, 304, 279, 1595, 6648, 7345, 63, 1052, 512, 74694, 12958, 198, 65562, 78340, 90854, 284, 2330, 262, 674, 12515, 262, 364, 6884, 33923, 756, 262, 674, 12515, 933, 14196, 4077, 18, 13, 19127, 264, 3465, 734, 311, 5542, 701, 1646, 512, 74694, 12958, 198, 475, 22975, 198, 475, 8760, 439, 2660, 198, 1527, 662, 6644, 1179, 5008, 1858, 271, 755, 5542, 5156, 4019, 262, 674, 6247, 311, 5542, 701, 1646, 198, 262, 1646, 284, 12515, 262, 1853, 284, 5585, 262, 674, 10467, 279, 1646, 311, 13668, 198, 262, 22975, 28026, 7790, 11, 1825, 5698, 11, 364, 20824, 25863, 262, 674, 5666, 279, 4729, 449, 279, 502, 1646, 1853, 198, 262, 1646, 2703, 284, 5008, 1858, 8549, 9288, 746, 262, 1646, 2703, 3960, 284, 1853, 198, 262, 1646, 2703, 5799, 746, 14196, 4077, 19, 13, 8618, 279, 3465, 304, 279, 1595, 25792, 7345, 63, 1052, 315, 701, 917, 512, 74694, 12958, 198, 1527, 4092, 33923, 1179, 4092, 271, 31, 6884, 88812, 28, 1399, 353, 220, 1399, 353, 220, 1187, 353, 308, 8, 220, 674, 24416, 279, 3465, 311, 1629, 1475, 308, 2919, 198, 755, 1629, 7745, 5156, 12461, 4019, 262, 5542, 5156, 746, 14196, 4077, 20, 13, 6588, 279, 4092, 3465, 12128, 512, 14196, 4077, 12958, 10299, 7345, 1920, 33923, 198, 14196, 4077, 644, 420, 3187, 11, 279, 1595, 10613, 5156, 63, 734, 28788, 701, 1646, 11, 27024, 433, 311, 13668, 11, 323, 9013, 279, 4729, 449, 279, 502, 1646, 1853, 13, 578, 1595, 6236, 7745, 5156, 12461, 63, 734, 374, 264, 4092, 3465, 430, 374, 13847, 311, 1629, 1475, 1595, 77, 63, 2919, 323, 6880, 279, 1595, 10613, 5156, 63, 734, 13, 578, 1595, 4734, 33923, 63, 3290, 2011, 387, 1629, 311, 1212, 279, 4092, 3465, 12128, 382, 9290, 25, 1115, 374, 1120, 832, 1648, 311, 9899, 279, 4967, 315, 701, 1646, 13, 578, 4839, 8292, 690, 6904, 389, 701, 3230, 8670, 323, 17413, 13], prompt_embeds shape: None, lora_request: None.
(APIServer pid=190319) INFO 08-06 10:19:36 [logger.py:41] Received request cmpl-db618535327d49c1b8ea1fed3b36cd7e-0: prompt: 'Lila, who sat on the deck, her arms wrapped protectively around the children she had saved. Her eyes were filled with tears, but her expression was resolute.\n\nRoran approached her, offering a handkerchief. "You did what you could," he told her gently. "You saved these children. They\'re alive because of you."\n\nLila took the handkerchief, dabbing at her eyes. "Thank you, Captain. I just wish I could\'ve done more."\n\nAs the ship sailed away from the ruins of the Salakor Shard, Roran gathered his crew, as well as the survivors. Their faces were a mix of shock, sorrow, and determination. Together, they would face the uncertain future and forge a new path for themselves and those they had saved.\n\nThe Falcon\'s Fury became a symbol of hope amidst the chaos, and the story of the Salakor Shard\'s collapse was etched into the hearts and minds of those who survived. The journey ahead would be filled with challenges, but the bonds forged in the face of tragedy would be unbreakable.\n\nAs they sailed toward the Dawn Coast, the survivors of Salakor Shard stared out at the vast expanse of the Aire Sea, their hearts heavy with loss, but also filled with a newfound sense of purpose. In the days and weeks to come, they would work together to rebuild their lives and create a new home on the resilient Dawn Coast. And while the memories of that fateful day would never fade, the resilience of the human spirit would ensure that they continued to endure, adapt, and ultimately, thrive.', params: SamplingParams(n=1, presence_penalty=0.0, frequency_penalty=0.0, repetition_penalty=1.0, temperature=0.0, top_p=1.0, top_k=0, min_p=0.0, seed=None, stop=[], stop_token_ids=[], bad_words=[], include_stop_str_in_output=False, ignore_eos=False, max_tokens=24, min_tokens=0, logprobs=None, prompt_logprobs=None, skip_special_tokens=True, spaces_between_special_tokens=True, truncate_prompt_tokens=None, guided_decoding=None, extra_args=None), prompt_token_ids: [128000, 43, 10746, 11, 889, 7731, 389, 279, 9722, 11, 1077, 11977, 20037, 6144, 3210, 2212, 279, 2911, 1364, 1047, 6924, 13, 6385, 6548, 1051, 10409, 449, 24014, 11, 719, 1077, 7645, 574, 594, 6402, 382, 49, 55504, 25735, 1077, 11, 10209, 264, 1450, 7197, 62626, 13, 330, 2675, 1550, 1148, 499, 1436, 1359, 568, 3309, 1077, 30373, 13, 330, 2675, 6924, 1521, 2911, 13, 2435, 2351, 13989, 1606, 315, 499, 2266, 43, 10746, 3952, 279, 1450, 7197, 62626, 11, 83868, 7278, 520, 1077, 6548, 13, 330, 13359, 499, 11, 22022, 13, 358, 1120, 6562, 358, 1436, 3077, 2884, 810, 2266, 2170, 279, 8448, 76844, 3201, 505, 279, 46762, 315, 279, 8375, 587, 269, 96466, 11, 432, 55504, 20802, 813, 13941, 11, 439, 1664, 439, 279, 32696, 13, 11205, 12580, 1051, 264, 6651, 315, 10988, 11, 58596, 11, 323, 26314, 13, 32255, 11, 814, 1053, 3663, 279, 36218, 3938, 323, 57728, 264, 502, 1853, 369, 5694, 323, 1884, 814, 1047, 6924, 382, 791, 43961, 596, 50479, 6244, 264, 7891, 315, 3987, 65904, 279, 28013, 11, 323, 279, 3446, 315, 279, 8375, 587, 269, 96466, 596, 18678, 574, 1880, 2454, 1139, 279, 23492, 323, 20663, 315, 1884, 889, 26968, 13, 578, 11879, 8469, 1053, 387, 10409, 449, 11774, 11, 719, 279, 27460, 54299, 304, 279, 3663, 315, 31926, 1053, 387, 653, 9137, 481, 382, 2170, 814, 76844, 9017, 279, 35607, 16377, 11, 279, 32696, 315, 8375, 587, 269, 96466, 45135, 704, 520, 279, 13057, 506, 95519, 315, 279, 362, 556, 15379, 11, 872, 23492, 8987, 449, 4814, 11, 719, 1101, 10409, 449, 264, 94621, 5647, 315, 7580, 13, 763, 279, 2919, 323, 5672, 311, 2586, 11, 814, 1053, 990, 3871, 311, 32593, 872, 6439, 323, 1893, 264, 502, 2162, 389, 279, 59780, 35607, 16377, 13, 1628, 1418, 279, 19459, 315, 430, 282, 21508, 1938, 1053, 2646, 15366, 11, 279, 56062, 315, 279, 3823, 9090, 1053, 6106, 430, 814, 8738, 311, 46753, 11, 10737, 11, 323, 13967, 11, 41972, 13], prompt_embeds shape: None, lora_request: None.
(APIServer pid=190319) INFO 08-06 10:19:36 [logger.py:41] Received request cmpl-3e1015da9e77470796dd4f94ff44f7a1-0: prompt: '**Assistant**', params: SamplingParams(n=1, presence_penalty=0.0, frequency_penalty=0.0, repetition_penalty=1.0, temperature=0.0, top_p=1.0, top_k=0, min_p=0.0, seed=None, stop=[], stop_token_ids=[], bad_words=[], include_stop_str_in_output=False, ignore_eos=False, max_tokens=6, min_tokens=0, logprobs=None, prompt_logprobs=None, skip_special_tokens=True, spaces_between_special_tokens=True, truncate_prompt_tokens=None, guided_decoding=None, extra_args=None), prompt_token_ids: [128000, 334, 72803, 334], prompt_embeds shape: None, lora_request: None.
(APIServer pid=190319) INFO 08-06 10:19:36 [logger.py:41] Received request cmpl-b8a0fd4d7b6249c380926c244534b0d0-0: prompt: '"test: [noun] a means of testing: such as. something (such as a series of questions or exercises) for measuring the skill, knowledge, intelligence, capacities, or aptitudes of an individual or group. a procedure, reaction, or reagent used to identify or characterize a substance or constituent. a positive result in such a test."\nSource: https://www.merriam-webster.com/dictionary/test\n\n"Define test. test synonyms, test pronunciation, test translation, English dictionary definition of test. n. 1. A procedure for critical evaluation; a means of determining the presence, quality, or truth of something; a trial: a test of ones eyesight;..."\nSource: https://www.thefreedictionary.com/test\n\n"Synonyms for TEST: essay, experiment, experimentation, trial, exam, examination, quiz, sample"\nSource: https://www.merriam-webster.com/thesaurus/test\n\nGiven these web results, answer the following question: test', params: SamplingParams(n=1, presence_penalty=0.0, frequency_penalty=0.0, repetition_penalty=1.0, temperature=0.0, top_p=1.0, top_k=0, min_p=0.0, seed=None, stop=[], stop_token_ids=[], bad_words=[], include_stop_str_in_output=False, ignore_eos=False, max_tokens=80, min_tokens=0, logprobs=None, prompt_logprobs=None, skip_special_tokens=True, spaces_between_special_tokens=True, truncate_prompt_tokens=None, guided_decoding=None, extra_args=None), prompt_token_ids: [128000, 1, 1985, 25, 510, 91209, 60, 264, 3445, 315, 7649, 25, 1778, 439, 13, 2555, 320, 21470, 439, 264, 4101, 315, 4860, 477, 23783, 8, 369, 30090, 279, 10151, 11, 6677, 11, 11478, 11, 59539, 11, 477, 20697, 21237, 315, 459, 3927, 477, 1912, 13, 264, 10537, 11, 13010, 11, 477, 312, 8252, 1511, 311, 10765, 477, 70755, 264, 20278, 477, 75164, 13, 264, 6928, 1121, 304, 1778, 264, 1296, 10246, 3692, 25, 3788, 1129, 2185, 749, 261, 462, 309, 30531, 3751, 916, 3529, 4003, 12986, 271, 1, 36438, 1296, 13, 1296, 86506, 11, 1296, 71722, 11, 1296, 14807, 11, 6498, 11240, 7419, 315, 1296, 13, 308, 13, 220, 16, 13, 362, 10537, 369, 9200, 16865, 26, 264, 3445, 315, 26679, 279, 9546, 11, 4367, 11, 477, 8206, 315, 2555, 26, 264, 9269, 25, 264, 1296, 315, 6305, 6548, 492, 26, 31538, 3692, 25, 3788, 1129, 2185, 13991, 830, 29616, 4003, 916, 12986, 271, 1, 38234, 46703, 369, 13916, 25, 9071, 11, 9526, 11, 66196, 11, 9269, 11, 7151, 11, 24481, 11, 28223, 11, 6205, 702, 3692, 25, 3788, 1129, 2185, 749, 261, 462, 309, 30531, 3751, 916, 14, 6509, 43613, 12986, 271, 22818, 1521, 3566, 3135, 11, 4320, 279, 2768, 3488, 25, 1296], prompt_embeds shape: None, lora_request: None.
(APIServer pid=190319) INFO:     127.0.0.1:49870 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=190319) INFO 08-06 10:19:36 [async_llm.py:273] Added request cmpl-ef3e576cc1bd4f6eb302c80de3e1d08f-0.
(APIServer pid=190319) INFO:     127.0.0.1:49882 - "POST /v1/completions HTTP/1.1" 200 OK
DEBUG 08-06 10:19:36 [core.py:687] EngineCore loop active.
(APIServer pid=190319) INFO 08-06 10:19:36 [async_llm.py:273] Added request cmpl-cb68b955c98846ac8de69a26b47928cb-0.
(APIServer pid=190319) INFO:     127.0.0.1:49898 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=190319) INFO 08-06 10:19:36 [async_llm.py:273] Added request cmpl-f76a1d9a10384fde9646d3dd8319084e-0.
(APIServer pid=190319) INFO:     127.0.0.1:49912 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=190319) INFO 08-06 10:19:36 [async_llm.py:273] Added request cmpl-a57cd123d1c542db8318c11954541f1f-0.
(APIServer pid=190319) INFO:     127.0.0.1:49924 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=190319) INFO 08-06 10:19:36 [async_llm.py:273] Added request cmpl-da2b11c536274ed590c5d3df0c8fdc50-0.
(APIServer pid=190319) INFO:     127.0.0.1:49930 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=190319) INFO 08-06 10:19:36 [async_llm.py:273] Added request cmpl-c80c1072660349f2898dd6f3b3e1c9dd-0.
(APIServer pid=190319) INFO:     127.0.0.1:49942 - "POST /v1/completions HTTP/1.1" 200 OK
DEBUG 08-06 10:19:36 [cuda_piecewise_backend.py:165] Capturing a cudagraph for shape 16
(APIServer pid=190319) INFO 08-06 10:19:36 [async_llm.py:273] Added request cmpl-98efd9f8a11243c9910565e17bb21e54-0.
(APIServer pid=190319) INFO:     127.0.0.1:49958 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=190319) INFO 08-06 10:19:36 [async_llm.py:273] Added request cmpl-db618535327d49c1b8ea1fed3b36cd7e-0.
(APIServer pid=190319) INFO:     127.0.0.1:49974 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=190319) INFO 08-06 10:19:36 [async_llm.py:273] Added request cmpl-3e1015da9e77470796dd4f94ff44f7a1-0.
(APIServer pid=190319) INFO:     127.0.0.1:49978 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=190319) INFO 08-06 10:19:36 [async_llm.py:273] Added request cmpl-b8a0fd4d7b6249c380926c244534b0d0-0.
INFO 08-06 10:19:37 [gpu_worker.py:370] DIEGO: CUDAgraph in execution time for 1357 input tokens
INFO 08-06 10:19:37 [gpu_worker.py:376] Graph capturing finished in 0.015 secs
INFO 08-06 10:19:37 [gpu_worker.py:370] DIEGO: CUDAgraph in execution time for 7 input tokens
INFO 08-06 10:19:37 [gpu_worker.py:376] Graph capturing finished in 0.004 secs
DEBUG 08-06 10:19:37 [cuda_piecewise_backend.py:154] Warming up 1/1 for shape 8
DEBUG 08-06 10:19:37 [cuda_piecewise_backend.py:165] Capturing a cudagraph for shape 8
INFO 08-06 10:19:37 [gpu_worker.py:370] DIEGO: CUDAgraph in execution time for 6 input tokens
INFO 08-06 10:19:37 [gpu_worker.py:376] Graph capturing finished in 0.004 secs
INFO 08-06 10:19:37 [gpu_worker.py:370] DIEGO: CUDAgraph in execution time for 5 input tokens
INFO 08-06 10:19:37 [gpu_worker.py:376] Graph capturing finished in 0.004 secs
INFO 08-06 10:19:37 [gpu_worker.py:370] DIEGO: CUDAgraph in execution time for 4 input tokens
DEBUG 08-06 10:19:37 [cuda_piecewise_backend.py:154] Warming up 1/1 for shape 4
INFO 08-06 10:19:37 [gpu_worker.py:376] Graph capturing finished in 0.004 secs
DEBUG 08-06 10:19:37 [cuda_piecewise_backend.py:165] Capturing a cudagraph for shape 4
INFO 08-06 10:19:38 [gpu_worker.py:370] DIEGO: CUDAgraph in execution time for 3 input tokens
INFO 08-06 10:19:38 [gpu_worker.py:376] Graph capturing finished in 0.004 secs
INFO 08-06 10:19:38 [gpu_worker.py:370] DIEGO: CUDAgraph in execution time for 2 input tokens
DEBUG 08-06 10:19:38 [cuda_piecewise_backend.py:154] Warming up 1/1 for shape 2
INFO 08-06 10:19:38 [gpu_worker.py:376] Graph capturing finished in 0.004 secs
DEBUG 08-06 10:19:38 [cuda_piecewise_backend.py:165] Capturing a cudagraph for shape 2
(APIServer pid=190319) INFO 08-06 10:19:39 [loggers.py:123] Engine 000: Avg prompt throughput: 138.2 tokens/s, Avg generation throughput: 87.5 tokens/s, Running: 1 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 0.0%
DEBUG 08-06 10:19:41 [core.py:681] EngineCore waiting for work.

#### MAIN BRANCH

>> Switched to branch 'main'
Your branch is up to date with 'origin/main'.
root@vllm-vm:/app/vllm# VLLM_LOGGING_LEVEL=DEBUG vllm serve meta-llama/Llama-3.2-1B
DEBUG 08-06 10:25:17 [__init__.py:30] No plugins for group vllm.platform_plugins found.
DEBUG 08-06 10:25:17 [__init__.py:34] Checking if TPU platform is available.
DEBUG 08-06 10:25:17 [__init__.py:52] TPU platform is not available because: No module named 'libtpu'
DEBUG 08-06 10:25:17 [__init__.py:58] Checking if CUDA platform is available.
DEBUG 08-06 10:25:17 [__init__.py:78] Confirmed CUDA platform is available.
DEBUG 08-06 10:25:17 [__init__.py:106] Checking if ROCm platform is available.
DEBUG 08-06 10:25:17 [__init__.py:120] ROCm platform is not available because: No module named 'amdsmi'
DEBUG 08-06 10:25:17 [__init__.py:127] Checking if XPU platform is available.
DEBUG 08-06 10:25:17 [__init__.py:146] XPU platform is not available because: No module named 'intel_extension_for_pytorch'
DEBUG 08-06 10:25:17 [__init__.py:153] Checking if CPU platform is available.
DEBUG 08-06 10:25:17 [__init__.py:175] Checking if Neuron platform is available.
DEBUG 08-06 10:25:17 [__init__.py:58] Checking if CUDA platform is available.
DEBUG 08-06 10:25:17 [__init__.py:78] Confirmed CUDA platform is available.
INFO 08-06 10:25:17 [__init__.py:241] Automatically detected platform cuda.
DEBUG 08-06 10:25:20 [utils.py:168] Setting VLLM_WORKER_MULTIPROC_METHOD to 'spawn'
DEBUG 08-06 10:25:20 [__init__.py:38] Available plugins for group vllm.general_plugins:
DEBUG 08-06 10:25:20 [__init__.py:40] - lora_filesystem_resolver -> vllm.plugins.lora_resolvers.filesystem_resolver:register_filesystem_resolver
DEBUG 08-06 10:25:20 [__init__.py:43] All plugins in this group will be loaded. Set `VLLM_PLUGINS` to control which plugins to load.
(APIServer pid=192421) INFO 08-06 10:25:20 [api_server.py:1774] vLLM API server version 0.1.dev8168+g475c1a0
(APIServer pid=192421) INFO 08-06 10:25:20 [utils.py:326] non-default args: {'model_tag': 'meta-llama/Llama-3.2-1B', 'model': 'meta-llama/Llama-3.2-1B'}
(APIServer pid=192421) INFO 08-06 10:25:26 [config.py:713] Resolved architecture: LlamaForCausalLM
(APIServer pid=192421) INFO 08-06 10:25:26 [config.py:1716] Using max model len 131072
(APIServer pid=192421) DEBUG 08-06 10:25:26 [arg_utils.py:1657] Setting max_num_batched_tokens to 2048 for OPENAI_API_SERVER usage context.
(APIServer pid=192421) DEBUG 08-06 10:25:26 [arg_utils.py:1666] Setting max_num_seqs to 256 for OPENAI_API_SERVER usage context.
(APIServer pid=192421) INFO 08-06 10:25:27 [config.py:2542] Chunked prefill is enabled with max_num_batched_tokens=2048.
DEBUG 08-06 10:25:31 [__init__.py:30] No plugins for group vllm.platform_plugins found.
DEBUG 08-06 10:25:31 [__init__.py:34] Checking if TPU platform is available.
DEBUG 08-06 10:25:31 [__init__.py:52] TPU platform is not available because: No module named 'libtpu'
DEBUG 08-06 10:25:31 [__init__.py:58] Checking if CUDA platform is available.
DEBUG 08-06 10:25:31 [__init__.py:78] Confirmed CUDA platform is available.
DEBUG 08-06 10:25:31 [__init__.py:106] Checking if ROCm platform is available.
DEBUG 08-06 10:25:31 [__init__.py:120] ROCm platform is not available because: No module named 'amdsmi'
DEBUG 08-06 10:25:31 [__init__.py:127] Checking if XPU platform is available.
DEBUG 08-06 10:25:31 [__init__.py:146] XPU platform is not available because: No module named 'intel_extension_for_pytorch'
DEBUG 08-06 10:25:31 [__init__.py:153] Checking if CPU platform is available.
DEBUG 08-06 10:25:31 [__init__.py:175] Checking if Neuron platform is available.
DEBUG 08-06 10:25:31 [__init__.py:58] Checking if CUDA platform is available.
DEBUG 08-06 10:25:31 [__init__.py:78] Confirmed CUDA platform is available.
INFO 08-06 10:25:31 [__init__.py:241] Automatically detected platform cuda.
INFO 08-06 10:25:33 [core.py:591] Waiting for init message from front-end.
(APIServer pid=192421) DEBUG 08-06 10:25:33 [utils.py:822] HELLO from local core engine process 0.
DEBUG 08-06 10:25:33 [core.py:599] Received init message: EngineHandshakeMetadata(addresses=EngineZmqAddresses(inputs=['ipc:///tmp/29bd6f8f-44d7-4b27-8f8d-b6fe2f785790'], outputs=['ipc:///tmp/0aba8cfa-e1a6-412c-b592-d76ab08c83b7'], coordinator_input=None, coordinator_output=None, frontend_stats_publish_address=None), parallel_config={'data_parallel_master_ip': '127.0.0.1', 'data_parallel_master_port': 0, 'data_parallel_size': 1})
DEBUG 08-06 10:25:33 [__init__.py:38] Available plugins for group vllm.general_plugins:
DEBUG 08-06 10:25:33 [__init__.py:40] - lora_filesystem_resolver -> vllm.plugins.lora_resolvers.filesystem_resolver:register_filesystem_resolver
DEBUG 08-06 10:25:33 [__init__.py:43] All plugins in this group will be loaded. Set `VLLM_PLUGINS` to control which plugins to load.
INFO 08-06 10:25:33 [core.py:73] Initializing a V1 LLM engine (v0.1.dev8168+g475c1a0) with config: model='meta-llama/Llama-3.2-1B', speculative_config=None, tokenizer='meta-llama/Llama-3.2-1B', skip_tokenizer_init=False, tokenizer_mode=auto, revision=None, override_neuron_config={}, tokenizer_revision=None, trust_remote_code=False, dtype=torch.bfloat16, max_seq_len=131072, download_dir=None, load_format=auto, tensor_parallel_size=1, pipeline_parallel_size=1, disable_custom_all_reduce=False, quantization=None, enforce_eager=False, kv_cache_dtype=auto, device_config=cuda, decoding_config=DecodingConfig(backend='auto', disable_fallback=False, disable_any_whitespace=False, disable_additional_properties=False, reasoning_backend=''), observability_config=ObservabilityConfig(show_hidden_metrics_for_version=None, otlp_traces_endpoint=None, collect_detailed_traces=None), seed=0, served_model_name=meta-llama/Llama-3.2-1B, num_scheduler_steps=1, multi_step_stream_outputs=True, enable_prefix_caching=True, chunked_prefill_enabled=True, use_async_output_proc=True, pooler_config=None, compilation_config={"level":3,"debug_dump_path":"","cache_dir":"","backend":"","custom_ops":[],"splitting_ops":["vllm.unified_attention","vllm.unified_attention_with_output","vllm.mamba_mixer2"],"use_inductor":true,"compile_sizes":[],"inductor_compile_config":{"enable_auto_functionalized_v2":false},"inductor_passes":{},"use_cudagraph":true,"cudagraph_num_of_warmups":1,"cudagraph_capture_sizes":[512,504,496,488,480,472,464,456,448,440,432,424,416,408,400,392,384,376,368,360,352,344,336,328,320,312,304,296,288,280,272,264,256,248,240,232,224,216,208,200,192,184,176,168,160,152,144,136,128,120,112,104,96,88,80,72,64,56,48,40,32,24,16,8,4,2,1],"cudagraph_copy_inputs":false,"full_cuda_graph":false,"max_capture_size":512,"local_cache_dir":null}
DEBUG 08-06 10:25:34 [decorators.py:139] Inferred dynamic dimensions for forward method of <class 'vllm.model_executor.models.llama.LlamaModel'>: ['input_ids', 'positions', 'intermediate_tensors', 'inputs_embeds']
DEBUG 08-06 10:25:34 [decorators.py:139] Inferred dynamic dimensions for forward method of <class 'vllm.model_executor.models.llama_eagle3.LlamaModel'>: ['input_ids', 'positions', 'hidden_states']
DEBUG 08-06 10:25:34 [__init__.py:3053] Methods determine_num_available_blocks,device_config,get_cache_block_size_bytes not implemented in <vllm.v1.worker.gpu_worker.Worker object at 0x7efa9ccf4f80>
DEBUG 08-06 10:25:34 [config.py:4998] enabled custom ops: Counter()
DEBUG 08-06 10:25:34 [config.py:5000] disabled custom ops: Counter()
DEBUG 08-06 10:25:35 [parallel_state.py:945] world_size=1 rank=0 local_rank=0 distributed_init_method=tcp://10.129.4.27:39137 backend=nccl
DEBUG 08-06 10:25:35 [parallel_state.py:996] Detected 1 nodes in the distributed environment
INFO 08-06 10:25:35 [parallel_state.py:1102] rank 0 in world size 1 is assigned as DP rank 0, PP rank 0, TP rank 0, EP rank 0
INFO 08-06 10:25:35 [topk_topp_sampler.py:49] Using FlashInfer for top-p & top-k sampling.
DEBUG 08-06 10:25:35 [config.py:4998] enabled custom ops: Counter()
DEBUG 08-06 10:25:35 [config.py:5000] disabled custom ops: Counter()
INFO 08-06 10:25:35 [gpu_model_runner.py:1921] Starting to load model meta-llama/Llama-3.2-1B...
INFO 08-06 10:25:35 [gpu_model_runner.py:1953] Loading model from scratch...
INFO 08-06 10:25:35 [cuda.py:305] Using Flash Attention backend on V1 engine.
DEBUG 08-06 10:25:35 [backends.py:39] Using InductorAdaptor
DEBUG 08-06 10:25:35 [config.py:4998] enabled custom ops: Counter()
DEBUG 08-06 10:25:35 [config.py:5000] disabled custom ops: Counter({'rms_norm': 33, 'silu_and_mul': 16, 'rotary_embedding': 1})
DEBUG 08-06 10:25:35 [base_loader.py:47] Loading weights on cuda ...
INFO 08-06 10:25:35 [weight_utils.py:296] Using model weights format ['*.safetensors']
INFO 08-06 10:25:35 [weight_utils.py:349] No model.safetensors.index.json found in remote.
Loading safetensors checkpoint shards:   0% Completed | 0/1 [00:00<?, ?it/s]
Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  2.46it/s]
Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  2.45it/s]

INFO 08-06 10:25:36 [default_loader.py:262] Loading weights took 0.47 seconds
INFO 08-06 10:25:36 [gpu_model_runner.py:1970] Model loading took 2.3185 GiB and 0.741368 seconds
DEBUG 08-06 10:25:36 [decorators.py:237] Start compiling function <code object forward at 0xdd838c0, file "/app/vllm/vllm/model_executor/models/llama.py", line 368>
DEBUG 08-06 10:25:40 [backends.py:487] Traced files (to be considered for compilation cache):
DEBUG 08-06 10:25:40 [backends.py:487] /app/vllm/vllm/attention/layer.py
DEBUG 08-06 10:25:40 [backends.py:487] /app/vllm/vllm/distributed/communication_op.py
DEBUG 08-06 10:25:40 [backends.py:487] /app/vllm/vllm/distributed/parallel_state.py
DEBUG 08-06 10:25:40 [backends.py:487] /app/vllm/vllm/model_executor/custom_op.py
DEBUG 08-06 10:25:40 [backends.py:487] /app/vllm/vllm/model_executor/layers/activation.py
DEBUG 08-06 10:25:40 [backends.py:487] /app/vllm/vllm/model_executor/layers/layernorm.py
DEBUG 08-06 10:25:40 [backends.py:487] /app/vllm/vllm/model_executor/layers/linear.py
DEBUG 08-06 10:25:40 [backends.py:487] /app/vllm/vllm/model_executor/layers/rotary_embedding.py
DEBUG 08-06 10:25:40 [backends.py:487] /app/vllm/vllm/model_executor/layers/utils.py
DEBUG 08-06 10:25:40 [backends.py:487] /app/vllm/vllm/model_executor/layers/vocab_parallel_embedding.py
DEBUG 08-06 10:25:40 [backends.py:487] /app/vllm/vllm/model_executor/models/llama.py
DEBUG 08-06 10:25:40 [backends.py:487] /app/vllm/vllm/platforms/interface.py
DEBUG 08-06 10:25:40 [backends.py:487] /usr/local/lib/python3.12/dist-packages/torch/_dynamo/polyfills/__init__.py
DEBUG 08-06 10:25:40 [backends.py:487] /usr/local/lib/python3.12/dist-packages/torch/nn/modules/container.py
DEBUG 08-06 10:25:40 [backends.py:487] /usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py
INFO 08-06 10:25:40 [backends.py:534] Using cache directory: /root/.cache/vllm/torch_compile_cache/25d4010308/rank_0_0/backbone for vLLM's torch.compile
INFO 08-06 10:25:40 [backends.py:545] Dynamo bytecode transform time: 3.65 s
DEBUG 08-06 10:25:40 [backends.py:125] Directly load the 0-th graph for dynamic shape from inductor via handle ('foww4arjrmo5ntlgmdcjpr7xst6lgk62vi5tjijgfsyto7r2mfpr', '/root/.cache/vllm/torch_compile_cache/25d4010308/rank_0_0/inductor_cache/3i/c3ifc7om5773max53s4uxbx3idf2b3yt2edem25nule7wkt3ttgy.py')
DEBUG 08-06 10:25:40 [backends.py:157] TOTAL LOADING TIME: 0.050496 s
DEBUG 08-06 10:25:40 [backends.py:125] Directly load the 1-th graph for dynamic shape from inductor via handle ('fiuf3tlstrqbuky24wigbejkm2cnwqxxn5nouiwmwuyprblrvtoa', '/root/.cache/vllm/torch_compile_cache/25d4010308/rank_0_0/inductor_cache/ky/cky4e3zslhbt7nnarafnykqxtqf6tjofdtfo5yqj5mc7aqemgqwr.py')
DEBUG 08-06 10:25:40 [backends.py:157] TOTAL LOADING TIME: 0.055504 s
DEBUG 08-06 10:25:41 [backends.py:125] Directly load the 2-th graph for dynamic shape from inductor via handle ('fiuf3tlstrqbuky24wigbejkm2cnwqxxn5nouiwmwuyprblrvtoa', '/root/.cache/vllm/torch_compile_cache/25d4010308/rank_0_0/inductor_cache/ky/cky4e3zslhbt7nnarafnykqxtqf6tjofdtfo5yqj5mc7aqemgqwr.py')
DEBUG 08-06 10:25:41 [backends.py:157] TOTAL LOADING TIME: 0.052288 s
DEBUG 08-06 10:25:41 [backends.py:125] Directly load the 3-th graph for dynamic shape from inductor via handle ('fiuf3tlstrqbuky24wigbejkm2cnwqxxn5nouiwmwuyprblrvtoa', '/root/.cache/vllm/torch_compile_cache/25d4010308/rank_0_0/inductor_cache/ky/cky4e3zslhbt7nnarafnykqxtqf6tjofdtfo5yqj5mc7aqemgqwr.py')
DEBUG 08-06 10:25:41 [backends.py:157] TOTAL LOADING TIME: 0.055855 s
DEBUG 08-06 10:25:41 [backends.py:125] Directly load the 4-th graph for dynamic shape from inductor via handle ('fiuf3tlstrqbuky24wigbejkm2cnwqxxn5nouiwmwuyprblrvtoa', '/root/.cache/vllm/torch_compile_cache/25d4010308/rank_0_0/inductor_cache/ky/cky4e3zslhbt7nnarafnykqxtqf6tjofdtfo5yqj5mc7aqemgqwr.py')
DEBUG 08-06 10:25:41 [backends.py:157] TOTAL LOADING TIME: 0.050149 s
DEBUG 08-06 10:25:41 [backends.py:125] Directly load the 5-th graph for dynamic shape from inductor via handle ('fiuf3tlstrqbuky24wigbejkm2cnwqxxn5nouiwmwuyprblrvtoa', '/root/.cache/vllm/torch_compile_cache/25d4010308/rank_0_0/inductor_cache/ky/cky4e3zslhbt7nnarafnykqxtqf6tjofdtfo5yqj5mc7aqemgqwr.py')
DEBUG 08-06 10:25:41 [backends.py:157] TOTAL LOADING TIME: 0.058388 s
DEBUG 08-06 10:25:41 [backends.py:125] Directly load the 6-th graph for dynamic shape from inductor via handle ('fiuf3tlstrqbuky24wigbejkm2cnwqxxn5nouiwmwuyprblrvtoa', '/root/.cache/vllm/torch_compile_cache/25d4010308/rank_0_0/inductor_cache/ky/cky4e3zslhbt7nnarafnykqxtqf6tjofdtfo5yqj5mc7aqemgqwr.py')
DEBUG 08-06 10:25:41 [backends.py:157] TOTAL LOADING TIME: 0.049910 s
DEBUG 08-06 10:25:41 [backends.py:125] Directly load the 7-th graph for dynamic shape from inductor via handle ('fiuf3tlstrqbuky24wigbejkm2cnwqxxn5nouiwmwuyprblrvtoa', '/root/.cache/vllm/torch_compile_cache/25d4010308/rank_0_0/inductor_cache/ky/cky4e3zslhbt7nnarafnykqxtqf6tjofdtfo5yqj5mc7aqemgqwr.py')
DEBUG 08-06 10:25:41 [backends.py:157] TOTAL LOADING TIME: 0.050207 s
DEBUG 08-06 10:25:42 [backends.py:125] Directly load the 8-th graph for dynamic shape from inductor via handle ('fiuf3tlstrqbuky24wigbejkm2cnwqxxn5nouiwmwuyprblrvtoa', '/root/.cache/vllm/torch_compile_cache/25d4010308/rank_0_0/inductor_cache/ky/cky4e3zslhbt7nnarafnykqxtqf6tjofdtfo5yqj5mc7aqemgqwr.py')
DEBUG 08-06 10:25:42 [backends.py:157] TOTAL LOADING TIME: 0.055021 s
DEBUG 08-06 10:25:42 [backends.py:125] Directly load the 9-th graph for dynamic shape from inductor via handle ('fiuf3tlstrqbuky24wigbejkm2cnwqxxn5nouiwmwuyprblrvtoa', '/root/.cache/vllm/torch_compile_cache/25d4010308/rank_0_0/inductor_cache/ky/cky4e3zslhbt7nnarafnykqxtqf6tjofdtfo5yqj5mc7aqemgqwr.py')
DEBUG 08-06 10:25:42 [backends.py:157] TOTAL LOADING TIME: 0.050637 s
DEBUG 08-06 10:25:42 [backends.py:125] Directly load the 10-th graph for dynamic shape from inductor via handle ('fiuf3tlstrqbuky24wigbejkm2cnwqxxn5nouiwmwuyprblrvtoa', '/root/.cache/vllm/torch_compile_cache/25d4010308/rank_0_0/inductor_cache/ky/cky4e3zslhbt7nnarafnykqxtqf6tjofdtfo5yqj5mc7aqemgqwr.py')
DEBUG 08-06 10:25:42 [backends.py:157] TOTAL LOADING TIME: 0.056030 s
DEBUG 08-06 10:25:42 [backends.py:125] Directly load the 11-th graph for dynamic shape from inductor via handle ('fiuf3tlstrqbuky24wigbejkm2cnwqxxn5nouiwmwuyprblrvtoa', '/root/.cache/vllm/torch_compile_cache/25d4010308/rank_0_0/inductor_cache/ky/cky4e3zslhbt7nnarafnykqxtqf6tjofdtfo5yqj5mc7aqemgqwr.py')
DEBUG 08-06 10:25:42 [backends.py:157] TOTAL LOADING TIME: 0.050674 s
DEBUG 08-06 10:25:42 [backends.py:125] Directly load the 12-th graph for dynamic shape from inductor via handle ('fiuf3tlstrqbuky24wigbejkm2cnwqxxn5nouiwmwuyprblrvtoa', '/root/.cache/vllm/torch_compile_cache/25d4010308/rank_0_0/inductor_cache/ky/cky4e3zslhbt7nnarafnykqxtqf6tjofdtfo5yqj5mc7aqemgqwr.py')
DEBUG 08-06 10:25:42 [backends.py:157] TOTAL LOADING TIME: 0.053627 s
DEBUG 08-06 10:25:42 [backends.py:125] Directly load the 13-th graph for dynamic shape from inductor via handle ('fiuf3tlstrqbuky24wigbejkm2cnwqxxn5nouiwmwuyprblrvtoa', '/root/.cache/vllm/torch_compile_cache/25d4010308/rank_0_0/inductor_cache/ky/cky4e3zslhbt7nnarafnykqxtqf6tjofdtfo5yqj5mc7aqemgqwr.py')
DEBUG 08-06 10:25:42 [backends.py:157] TOTAL LOADING TIME: 0.055087 s
DEBUG 08-06 10:25:42 [backends.py:125] Directly load the 14-th graph for dynamic shape from inductor via handle ('fiuf3tlstrqbuky24wigbejkm2cnwqxxn5nouiwmwuyprblrvtoa', '/root/.cache/vllm/torch_compile_cache/25d4010308/rank_0_0/inductor_cache/ky/cky4e3zslhbt7nnarafnykqxtqf6tjofdtfo5yqj5mc7aqemgqwr.py')
DEBUG 08-06 10:25:42 [backends.py:157] TOTAL LOADING TIME: 0.049484 s
DEBUG 08-06 10:25:43 [backends.py:125] Directly load the 15-th graph for dynamic shape from inductor via handle ('fiuf3tlstrqbuky24wigbejkm2cnwqxxn5nouiwmwuyprblrvtoa', '/root/.cache/vllm/torch_compile_cache/25d4010308/rank_0_0/inductor_cache/ky/cky4e3zslhbt7nnarafnykqxtqf6tjofdtfo5yqj5mc7aqemgqwr.py')
DEBUG 08-06 10:25:43 [backends.py:157] TOTAL LOADING TIME: 0.053783 s
DEBUG 08-06 10:25:43 [backends.py:125] Directly load the 16-th graph for dynamic shape from inductor via handle ('fvbvyhtr37kusmqij6uiinukna7ifz6inpaixj2ehzbq2ipe7nms', '/root/.cache/vllm/torch_compile_cache/25d4010308/rank_0_0/inductor_cache/ts/ctsi5px5m6j4xfpnuarwb5hejxunelgdyavbmz52ohrcimfwoaig.py')
DEBUG 08-06 10:25:43 [backends.py:157] TOTAL LOADING TIME: 0.028928 s
INFO 08-06 10:25:43 [backends.py:165] Directly load the compiled graph(s) for dynamic shape from the cache, took 2.550 s
(APIServer pid=192421) DEBUG 08-06 10:25:43 [utils.py:741] Waiting for 1 local, 0 remote core engine proc(s) to start.
INFO 08-06 10:25:43 [monitor.py:34] torch.compile takes 3.65 s in total
/usr/local/lib/python3.12/dist-packages/torch/utils/cpp_extension.py:2356: UserWarning: TORCH_CUDA_ARCH_LIST is not set, all archs for visible cards are included for compilation. 
If this is not desired, please set os.environ['TORCH_CUDA_ARCH_LIST'].
  warnings.warn(
DEBUG 08-06 10:25:44 [gpu_worker.py:262] Initial free memory: 43.82 GiB; Requested memory: 0.90 (util), 39.88 GiB
DEBUG 08-06 10:25:44 [gpu_worker.py:269] Free memory after profiling: 41.43 GiB (total), 37.49 GiB (within requested)
DEBUG 08-06 10:25:44 [gpu_worker.py:275] Memory profiling takes 7.68 seconds. Total non KV cache memory: 2.78GiB; torch peak memory increase: 0.45GiB; non-torch forward increase memory: 0.02GiB; weights memory: 2.32GiB.
INFO 08-06 10:25:44 [gpu_worker.py:276] Available KV cache memory: 37.10 GiB
INFO 08-06 10:25:44 [kv_cache_utils.py:831] GPU KV cache size: 1,215,552 tokens
INFO 08-06 10:25:44 [kv_cache_utils.py:835] Maximum concurrency for 131,072 tokens per request: 9.27x
DEBUG 08-06 10:25:44 [config.py:4998] enabled custom ops: Counter()
DEBUG 08-06 10:25:44 [config.py:5000] disabled custom ops: Counter({'rms_norm': 33, 'silu_and_mul': 16, 'rotary_embedding': 1})
Capturing CUDA graph shapes:   0%|                                                                    | 0/67 [00:00<?, ?it/s]DEBUG 08-06 10:25:44 [cuda_piecewise_backend.py:151] Warming up 1/1 for shape 512
DEBUG 08-06 10:25:44 [cuda_piecewise_backend.py:162] Capturing a cudagraph for shape 512
DEBUG 08-06 10:25:44 [cuda_piecewise_backend.py:151] Warming up 1/1 for shape 504
DEBUG 08-06 10:25:44 [cuda_piecewise_backend.py:162] Capturing a cudagraph for shape 504
DEBUG 08-06 10:25:44 [cuda_piecewise_backend.py:151] Warming up 1/1 for shape 496
DEBUG 08-06 10:25:44 [cuda_piecewise_backend.py:162] Capturing a cudagraph for shape 496
DEBUG 08-06 10:25:44 [cuda_piecewise_backend.py:151] Warming up 1/1 for shape 488
DEBUG 08-06 10:25:44 [cuda_piecewise_backend.py:162] Capturing a cudagraph for shape 488
DEBUG 08-06 10:25:45 [cuda_piecewise_backend.py:151] Warming up 1/1 for shape 480
DEBUG 08-06 10:25:45 [cuda_piecewise_backend.py:162] Capturing a cudagraph for shape 480
DEBUG 08-06 10:25:45 [cuda_piecewise_backend.py:151] Warming up 1/1 for shape 472
DEBUG 08-06 10:25:45 [cuda_piecewise_backend.py:162] Capturing a cudagraph for shape 472
Capturing CUDA graph shapes:   9%|█████▎                                                      | 6/67 [00:00<00:01, 58.55it/s]DEBUG 08-06 10:25:45 [cuda_piecewise_backend.py:151] Warming up 1/1 for shape 464
DEBUG 08-06 10:25:45 [cuda_piecewise_backend.py:162] Capturing a cudagraph for shape 464
DEBUG 08-06 10:25:45 [cuda_piecewise_backend.py:151] Warming up 1/1 for shape 456
DEBUG 08-06 10:25:45 [cuda_piecewise_backend.py:162] Capturing a cudagraph for shape 456
DEBUG 08-06 10:25:45 [cuda_piecewise_backend.py:151] Warming up 1/1 for shape 448
DEBUG 08-06 10:25:45 [cuda_piecewise_backend.py:162] Capturing a cudagraph for shape 448
DEBUG 08-06 10:25:45 [cuda_piecewise_backend.py:151] Warming up 1/1 for shape 440
DEBUG 08-06 10:25:45 [cuda_piecewise_backend.py:162] Capturing a cudagraph for shape 440
DEBUG 08-06 10:25:45 [cuda_piecewise_backend.py:151] Warming up 1/1 for shape 432
DEBUG 08-06 10:25:45 [cuda_piecewise_backend.py:162] Capturing a cudagraph for shape 432
DEBUG 08-06 10:25:45 [cuda_piecewise_backend.py:151] Warming up 1/1 for shape 424
DEBUG 08-06 10:25:45 [cuda_piecewise_backend.py:162] Capturing a cudagraph for shape 424
Capturing CUDA graph shapes:  18%|██████████▌                                                | 12/67 [00:00<00:00, 57.00it/s]DEBUG 08-06 10:25:45 [cuda_piecewise_backend.py:151] Warming up 1/1 for shape 416
DEBUG 08-06 10:25:45 [cuda_piecewise_backend.py:162] Capturing a cudagraph for shape 416
DEBUG 08-06 10:25:45 [cuda_piecewise_backend.py:151] Warming up 1/1 for shape 408
DEBUG 08-06 10:25:45 [cuda_piecewise_backend.py:162] Capturing a cudagraph for shape 408
DEBUG 08-06 10:25:45 [cuda_piecewise_backend.py:151] Warming up 1/1 for shape 400
DEBUG 08-06 10:25:45 [cuda_piecewise_backend.py:162] Capturing a cudagraph for shape 400
DEBUG 08-06 10:25:45 [cuda_piecewise_backend.py:151] Warming up 1/1 for shape 392
DEBUG 08-06 10:25:45 [cuda_piecewise_backend.py:162] Capturing a cudagraph for shape 392
DEBUG 08-06 10:25:45 [cuda_piecewise_backend.py:151] Warming up 1/1 for shape 384
DEBUG 08-06 10:25:45 [cuda_piecewise_backend.py:162] Capturing a cudagraph for shape 384
DEBUG 08-06 10:25:45 [cuda_piecewise_backend.py:151] Warming up 1/1 for shape 376
DEBUG 08-06 10:25:45 [cuda_piecewise_backend.py:162] Capturing a cudagraph for shape 376
DEBUG 08-06 10:25:45 [cuda_piecewise_backend.py:151] Warming up 1/1 for shape 368
DEBUG 08-06 10:25:45 [cuda_piecewise_backend.py:162] Capturing a cudagraph for shape 368
Capturing CUDA graph shapes:  28%|████████████████▋                                          | 19/67 [00:00<00:00, 59.98it/s]DEBUG 08-06 10:25:45 [cuda_piecewise_backend.py:151] Warming up 1/1 for shape 360
DEBUG 08-06 10:25:45 [cuda_piecewise_backend.py:162] Capturing a cudagraph for shape 360
DEBUG 08-06 10:25:45 [cuda_piecewise_backend.py:151] Warming up 1/1 for shape 352
DEBUG 08-06 10:25:45 [cuda_piecewise_backend.py:162] Capturing a cudagraph for shape 352
DEBUG 08-06 10:25:45 [cuda_piecewise_backend.py:151] Warming up 1/1 for shape 344
DEBUG 08-06 10:25:45 [cuda_piecewise_backend.py:162] Capturing a cudagraph for shape 344
DEBUG 08-06 10:25:45 [cuda_piecewise_backend.py:151] Warming up 1/1 for shape 336
DEBUG 08-06 10:25:45 [cuda_piecewise_backend.py:162] Capturing a cudagraph for shape 336
DEBUG 08-06 10:25:45 [cuda_piecewise_backend.py:151] Warming up 1/1 for shape 328
DEBUG 08-06 10:25:45 [cuda_piecewise_backend.py:162] Capturing a cudagraph for shape 328
DEBUG 08-06 10:25:45 [cuda_piecewise_backend.py:151] Warming up 1/1 for shape 320
DEBUG 08-06 10:25:45 [cuda_piecewise_backend.py:162] Capturing a cudagraph for shape 320
DEBUG 08-06 10:25:45 [cuda_piecewise_backend.py:151] Warming up 1/1 for shape 312
DEBUG 08-06 10:25:45 [cuda_piecewise_backend.py:162] Capturing a cudagraph for shape 312
Capturing CUDA graph shapes:  39%|██████████████████████▉                                    | 26/67 [00:00<00:00, 61.35it/s]DEBUG 08-06 10:25:45 [cuda_piecewise_backend.py:151] Warming up 1/1 for shape 304
DEBUG 08-06 10:25:45 [cuda_piecewise_backend.py:162] Capturing a cudagraph for shape 304
DEBUG 08-06 10:25:45 [cuda_piecewise_backend.py:151] Warming up 1/1 for shape 296
DEBUG 08-06 10:25:45 [cuda_piecewise_backend.py:162] Capturing a cudagraph for shape 296
DEBUG 08-06 10:25:45 [cuda_piecewise_backend.py:151] Warming up 1/1 for shape 288
DEBUG 08-06 10:25:45 [cuda_piecewise_backend.py:162] Capturing a cudagraph for shape 288
DEBUG 08-06 10:25:45 [cuda_piecewise_backend.py:151] Warming up 1/1 for shape 280
DEBUG 08-06 10:25:45 [cuda_piecewise_backend.py:162] Capturing a cudagraph for shape 280
DEBUG 08-06 10:25:45 [cuda_piecewise_backend.py:151] Warming up 1/1 for shape 272
DEBUG 08-06 10:25:45 [cuda_piecewise_backend.py:162] Capturing a cudagraph for shape 272
DEBUG 08-06 10:25:45 [cuda_piecewise_backend.py:151] Warming up 1/1 for shape 264
DEBUG 08-06 10:25:45 [cuda_piecewise_backend.py:162] Capturing a cudagraph for shape 264
DEBUG 08-06 10:25:45 [cuda_piecewise_backend.py:151] Warming up 1/1 for shape 256
DEBUG 08-06 10:25:45 [cuda_piecewise_backend.py:162] Capturing a cudagraph for shape 256
Capturing CUDA graph shapes:  49%|█████████████████████████████                              | 33/67 [00:00<00:00, 61.79it/s]DEBUG 08-06 10:25:45 [cuda_piecewise_backend.py:151] Warming up 1/1 for shape 248
DEBUG 08-06 10:25:45 [cuda_piecewise_backend.py:162] Capturing a cudagraph for shape 248
DEBUG 08-06 10:25:45 [cuda_piecewise_backend.py:151] Warming up 1/1 for shape 240
DEBUG 08-06 10:25:45 [cuda_piecewise_backend.py:162] Capturing a cudagraph for shape 240
DEBUG 08-06 10:25:45 [cuda_piecewise_backend.py:151] Warming up 1/1 for shape 232
DEBUG 08-06 10:25:45 [cuda_piecewise_backend.py:162] Capturing a cudagraph for shape 232
DEBUG 08-06 10:25:45 [cuda_piecewise_backend.py:151] Warming up 1/1 for shape 224
DEBUG 08-06 10:25:45 [cuda_piecewise_backend.py:162] Capturing a cudagraph for shape 224
DEBUG 08-06 10:25:45 [cuda_piecewise_backend.py:151] Warming up 1/1 for shape 216
DEBUG 08-06 10:25:45 [cuda_piecewise_backend.py:162] Capturing a cudagraph for shape 216
DEBUG 08-06 10:25:45 [cuda_piecewise_backend.py:151] Warming up 1/1 for shape 208
DEBUG 08-06 10:25:45 [cuda_piecewise_backend.py:162] Capturing a cudagraph for shape 208
DEBUG 08-06 10:25:45 [cuda_piecewise_backend.py:151] Warming up 1/1 for shape 200
DEBUG 08-06 10:25:45 [cuda_piecewise_backend.py:162] Capturing a cudagraph for shape 200
Capturing CUDA graph shapes:  60%|███████████████████████████████████▏                       | 40/67 [00:00<00:00, 59.86it/s]DEBUG 08-06 10:25:45 [cuda_piecewise_backend.py:151] Warming up 1/1 for shape 192
DEBUG 08-06 10:25:45 [cuda_piecewise_backend.py:162] Capturing a cudagraph for shape 192
DEBUG 08-06 10:25:45 [cuda_piecewise_backend.py:151] Warming up 1/1 for shape 184
DEBUG 08-06 10:25:45 [cuda_piecewise_backend.py:162] Capturing a cudagraph for shape 184
DEBUG 08-06 10:25:45 [cuda_piecewise_backend.py:151] Warming up 1/1 for shape 176
DEBUG 08-06 10:25:45 [cuda_piecewise_backend.py:162] Capturing a cudagraph for shape 176
DEBUG 08-06 10:25:45 [cuda_piecewise_backend.py:151] Warming up 1/1 for shape 168
DEBUG 08-06 10:25:45 [cuda_piecewise_backend.py:162] Capturing a cudagraph for shape 168
DEBUG 08-06 10:25:45 [cuda_piecewise_backend.py:151] Warming up 1/1 for shape 160
DEBUG 08-06 10:25:45 [cuda_piecewise_backend.py:162] Capturing a cudagraph for shape 160
DEBUG 08-06 10:25:45 [cuda_piecewise_backend.py:151] Warming up 1/1 for shape 152
DEBUG 08-06 10:25:45 [cuda_piecewise_backend.py:162] Capturing a cudagraph for shape 152
Capturing CUDA graph shapes:  69%|████████████████████████████████████████▌                  | 46/67 [00:00<00:00, 58.89it/s]DEBUG 08-06 10:25:45 [cuda_piecewise_backend.py:151] Warming up 1/1 for shape 144
DEBUG 08-06 10:25:45 [cuda_piecewise_backend.py:162] Capturing a cudagraph for shape 144
DEBUG 08-06 10:25:45 [cuda_piecewise_backend.py:151] Warming up 1/1 for shape 136
DEBUG 08-06 10:25:45 [cuda_piecewise_backend.py:162] Capturing a cudagraph for shape 136
DEBUG 08-06 10:25:45 [cuda_piecewise_backend.py:151] Warming up 1/1 for shape 128
DEBUG 08-06 10:25:45 [cuda_piecewise_backend.py:162] Capturing a cudagraph for shape 128
DEBUG 08-06 10:25:45 [cuda_piecewise_backend.py:151] Warming up 1/1 for shape 120
DEBUG 08-06 10:25:45 [cuda_piecewise_backend.py:162] Capturing a cudagraph for shape 120
DEBUG 08-06 10:25:45 [cuda_piecewise_backend.py:151] Warming up 1/1 for shape 112
DEBUG 08-06 10:25:45 [cuda_piecewise_backend.py:162] Capturing a cudagraph for shape 112
DEBUG 08-06 10:25:45 [cuda_piecewise_backend.py:151] Warming up 1/1 for shape 104
DEBUG 08-06 10:25:45 [cuda_piecewise_backend.py:162] Capturing a cudagraph for shape 104
DEBUG 08-06 10:25:45 [cuda_piecewise_backend.py:151] Warming up 1/1 for shape 96
DEBUG 08-06 10:25:45 [cuda_piecewise_backend.py:162] Capturing a cudagraph for shape 96
Capturing CUDA graph shapes:  79%|██████████████████████████████████████████████▋            | 53/67 [00:00<00:00, 60.65it/s]DEBUG 08-06 10:25:45 [cuda_piecewise_backend.py:151] Warming up 1/1 for shape 88
DEBUG 08-06 10:25:45 [cuda_piecewise_backend.py:162] Capturing a cudagraph for shape 88
DEBUG 08-06 10:25:45 [cuda_piecewise_backend.py:151] Warming up 1/1 for shape 80
DEBUG 08-06 10:25:45 [cuda_piecewise_backend.py:162] Capturing a cudagraph for shape 80
DEBUG 08-06 10:25:45 [cuda_piecewise_backend.py:151] Warming up 1/1 for shape 72
DEBUG 08-06 10:25:45 [cuda_piecewise_backend.py:162] Capturing a cudagraph for shape 72
DEBUG 08-06 10:25:45 [cuda_piecewise_backend.py:151] Warming up 1/1 for shape 64
DEBUG 08-06 10:25:45 [cuda_piecewise_backend.py:162] Capturing a cudagraph for shape 64
DEBUG 08-06 10:25:45 [cuda_piecewise_backend.py:151] Warming up 1/1 for shape 56
DEBUG 08-06 10:25:45 [cuda_piecewise_backend.py:162] Capturing a cudagraph for shape 56
DEBUG 08-06 10:25:45 [cuda_piecewise_backend.py:151] Warming up 1/1 for shape 48
DEBUG 08-06 10:25:45 [cuda_piecewise_backend.py:162] Capturing a cudagraph for shape 48
DEBUG 08-06 10:25:45 [cuda_piecewise_backend.py:151] Warming up 1/1 for shape 40
DEBUG 08-06 10:25:45 [cuda_piecewise_backend.py:162] Capturing a cudagraph for shape 40
Capturing CUDA graph shapes:  90%|████████████████████████████████████████████████████▊      | 60/67 [00:00<00:00, 61.99it/s]DEBUG 08-06 10:25:45 [cuda_piecewise_backend.py:151] Warming up 1/1 for shape 32
DEBUG 08-06 10:25:45 [cuda_piecewise_backend.py:162] Capturing a cudagraph for shape 32
DEBUG 08-06 10:25:45 [cuda_piecewise_backend.py:151] Warming up 1/1 for shape 24
DEBUG 08-06 10:25:45 [cuda_piecewise_backend.py:162] Capturing a cudagraph for shape 24
DEBUG 08-06 10:25:45 [cuda_piecewise_backend.py:151] Warming up 1/1 for shape 16
DEBUG 08-06 10:25:45 [cuda_piecewise_backend.py:162] Capturing a cudagraph for shape 16
DEBUG 08-06 10:25:45 [cuda_piecewise_backend.py:151] Warming up 1/1 for shape 8
DEBUG 08-06 10:25:45 [cuda_piecewise_backend.py:162] Capturing a cudagraph for shape 8
DEBUG 08-06 10:25:45 [cuda_piecewise_backend.py:151] Warming up 1/1 for shape 4
DEBUG 08-06 10:25:45 [cuda_piecewise_backend.py:162] Capturing a cudagraph for shape 4
DEBUG 08-06 10:25:45 [cuda_piecewise_backend.py:151] Warming up 1/1 for shape 2
DEBUG 08-06 10:25:45 [cuda_piecewise_backend.py:162] Capturing a cudagraph for shape 2
DEBUG 08-06 10:25:46 [cuda_piecewise_backend.py:151] Warming up 1/1 for shape 1
DEBUG 08-06 10:25:46 [cuda_piecewise_backend.py:162] Capturing a cudagraph for shape 1
Capturing CUDA graph shapes: 100%|███████████████████████████████████████████████████████████| 67/67 [00:01<00:00, 60.89it/s]
INFO 08-06 10:25:46 [gpu_model_runner.py:2575] Graph capturing finished in 1 secs, took 0.31 GiB
INFO 08-06 10:25:46 [core.py:201] init engine (profile, create kv cache, warmup model) took 9.39 seconds
(APIServer pid=192421) DEBUG 08-06 10:25:46 [utils.py:822] READY from local core engine process 0.
DEBUG 08-06 10:25:46 [core.py:681] EngineCore waiting for work.
(APIServer pid=192421) INFO 08-06 10:25:46 [loggers.py:142] Engine 000: vllm cache_config_info with initialization after num_gpu_blocks is: 75972
DEBUG 08-06 10:25:46 [core.py:681] EngineCore waiting for work.
DEBUG 08-06 10:25:46 [core.py:681] EngineCore waiting for work.
(APIServer pid=192421) INFO 08-06 10:25:46 [api_server.py:1595] Supported_tasks: ['generate']
(APIServer pid=192421) WARNING 08-06 10:25:46 [config.py:1616] Default sampling parameters have been overridden by the model's Hugging Face generation config recommended from the model creator. If this is not intended, please relaunch vLLM instance with `--generation-config vllm`.
(APIServer pid=192421) INFO 08-06 10:25:46 [serving_responses.py:89] Using default chat sampling params from model: {'temperature': 0.6, 'top_p': 0.9}
(APIServer pid=192421) INFO 08-06 10:25:46 [serving_chat.py:125] Using default chat sampling params from model: {'temperature': 0.6, 'top_p': 0.9}
(APIServer pid=192421) INFO 08-06 10:25:46 [serving_completion.py:77] Using default completion sampling params from model: {'temperature': 0.6, 'top_p': 0.9}
(APIServer pid=192421) INFO 08-06 10:25:46 [api_server.py:1847] Starting vLLM API server 0 on http://0.0.0.0:8000
(APIServer pid=192421) INFO 08-06 10:25:46 [launcher.py:29] Available routes are:
(APIServer pid=192421) INFO 08-06 10:25:46 [launcher.py:37] Route: /openapi.json, Methods: HEAD, GET
(APIServer pid=192421) INFO 08-06 10:25:46 [launcher.py:37] Route: /docs, Methods: HEAD, GET
(APIServer pid=192421) INFO 08-06 10:25:46 [launcher.py:37] Route: /docs/oauth2-redirect, Methods: HEAD, GET
(APIServer pid=192421) INFO 08-06 10:25:46 [launcher.py:37] Route: /redoc, Methods: HEAD, GET
(APIServer pid=192421) INFO 08-06 10:25:46 [launcher.py:37] Route: /health, Methods: GET
(APIServer pid=192421) INFO 08-06 10:25:46 [launcher.py:37] Route: /load, Methods: GET
(APIServer pid=192421) INFO 08-06 10:25:46 [launcher.py:37] Route: /ping, Methods: POST
(APIServer pid=192421) INFO 08-06 10:25:46 [launcher.py:37] Route: /ping, Methods: GET
(APIServer pid=192421) INFO 08-06 10:25:46 [launcher.py:37] Route: /tokenize, Methods: POST
(APIServer pid=192421) INFO 08-06 10:25:46 [launcher.py:37] Route: /detokenize, Methods: POST
(APIServer pid=192421) INFO 08-06 10:25:46 [launcher.py:37] Route: /v1/models, Methods: GET
(APIServer pid=192421) INFO 08-06 10:25:46 [launcher.py:37] Route: /version, Methods: GET
(APIServer pid=192421) INFO 08-06 10:25:46 [launcher.py:37] Route: /v1/responses, Methods: POST
(APIServer pid=192421) INFO 08-06 10:25:46 [launcher.py:37] Route: /v1/responses/{response_id}, Methods: GET
(APIServer pid=192421) INFO 08-06 10:25:46 [launcher.py:37] Route: /v1/responses/{response_id}/cancel, Methods: POST
(APIServer pid=192421) INFO 08-06 10:25:46 [launcher.py:37] Route: /v1/chat/completions, Methods: POST
(APIServer pid=192421) INFO 08-06 10:25:46 [launcher.py:37] Route: /v1/completions, Methods: POST
(APIServer pid=192421) INFO 08-06 10:25:46 [launcher.py:37] Route: /v1/embeddings, Methods: POST
(APIServer pid=192421) INFO 08-06 10:25:46 [launcher.py:37] Route: /pooling, Methods: POST
(APIServer pid=192421) INFO 08-06 10:25:46 [launcher.py:37] Route: /classify, Methods: POST
(APIServer pid=192421) INFO 08-06 10:25:46 [launcher.py:37] Route: /score, Methods: POST
(APIServer pid=192421) INFO 08-06 10:25:46 [launcher.py:37] Route: /v1/score, Methods: POST
(APIServer pid=192421) INFO 08-06 10:25:46 [launcher.py:37] Route: /v1/audio/transcriptions, Methods: POST
(APIServer pid=192421) INFO 08-06 10:25:46 [launcher.py:37] Route: /v1/audio/translations, Methods: POST
(APIServer pid=192421) INFO 08-06 10:25:46 [launcher.py:37] Route: /rerank, Methods: POST
(APIServer pid=192421) INFO 08-06 10:25:46 [launcher.py:37] Route: /v1/rerank, Methods: POST
(APIServer pid=192421) INFO 08-06 10:25:46 [launcher.py:37] Route: /v2/rerank, Methods: POST
(APIServer pid=192421) INFO 08-06 10:25:46 [launcher.py:37] Route: /scale_elastic_ep, Methods: POST
(APIServer pid=192421) INFO 08-06 10:25:46 [launcher.py:37] Route: /is_scaling_elastic_ep, Methods: POST
(APIServer pid=192421) INFO 08-06 10:25:46 [launcher.py:37] Route: /invocations, Methods: POST
(APIServer pid=192421) INFO 08-06 10:25:46 [launcher.py:37] Route: /metrics, Methods: GET
(APIServer pid=192421) INFO:     Started server process [192421]
(APIServer pid=192421) INFO:     Waiting for application startup.
(APIServer pid=192421) INFO:     Application startup complete.

The Results are:
root@vllm-vm:/app# python3 vllm/benchmarks/benchmark_serving.py --backend vllm --model meta-llama/Llama-3.2-1B --endpoint /v1/completions --dataset-name sharegpt --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json --num-prompts 10
INFO 08-06 10:27:17 [__init__.py:241] Automatically detected platform cuda.
/app/vllm/benchmarks/benchmark_serving.py:1299: DeprecationWarning: benchmark_serving.py is deprecated and will be removed in a future version. Please use 'vllm bench serve' instead.
  main(args)
Namespace(backend='vllm', base_url=None, host='127.0.0.1', port=8000, endpoint='/v1/completions', dataset_name='sharegpt', dataset_path='ShareGPT_V3_unfiltered_cleaned_split.json', no_stream=False, max_concurrency=None, model='meta-llama/Llama-3.2-1B', tokenizer=None, use_beam_search=False, num_prompts=10, logprobs=None, request_rate=inf, burstiness=1.0, seed=0, trust_remote_code=False, disable_tqdm=False, profile=False, save_result=False, save_detailed=False, append_result=False, metadata=None, result_dir=None, result_filename=None, ignore_eos=False, percentile_metrics='ttft,tpot,itl', metric_percentiles='99', goodput=None, custom_output_len=256, custom_skip_chat_template=False, sonnet_input_len=550, sonnet_output_len=150, sonnet_prefix_len=200, sharegpt_output_len=None, random_input_len=1024, random_output_len=128, random_range_ratio=0.0, random_prefix_len=0, hf_subset=None, hf_split=None, hf_output_len=None, top_p=None, top_k=None, min_p=None, temperature=None, tokenizer_mode='auto', served_model_name=None, lora_modules=None, ramp_up_strategy=None, ramp_up_start_rps=None, ramp_up_end_rps=None)
Starting initial single prompt test run...
Initial test run completed. Starting main benchmark run...
Traffic request rate: inf RPS.
Burstiness factor: 1.0 (Poisson process)
Maximum request concurrency: None
100%|████████████████████████████████████████████████████████████████████████████████████████| 10/10 [00:04<00:00,  2.48it/s]
============ Serving Benchmark Result ============
Successful requests:                     10        
Benchmark duration (s):                  4.04      
Total input tokens:                      1369      
Total generated tokens:                  1746      
Request throughput (req/s):              2.48      
Output token throughput (tok/s):         432.26    
Total Token throughput (tok/s):          771.18    
---------------Time to First Token----------------
Mean TTFT (ms):                          37.55     
Median TTFT (ms):                        38.87     
P99 TTFT (ms):                           40.79     
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          5.41      
Median TPOT (ms):                        5.47      
P99 TPOT (ms):                           5.52      
---------------Inter-token Latency----------------
Mean ITL (ms):                           5.24      
Median ITL (ms):                         5.18      
P99 ITL (ms):                            5.78      
==================================================

And the server shows:
request: None.
(APIServer pid=192421) INFO:     127.0.0.1:57636 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=192421) INFO 08-06 10:27:21 [async_llm.py:273] Added request cmpl-6e4c6fc27ab145bba578e9457a380301-0.
DEBUG 08-06 10:27:21 [core.py:687] EngineCore loop active.
DEBUG 08-06 10:27:22 [core.py:681] EngineCore waiting for work.
(APIServer pid=192421) INFO 08-06 10:27:22 [logger.py:41] Received request cmpl-a841fd86e1a24566b13abcb62969febe-0: prompt: 'Do you know the book Traction by Gino Wickman', params: SamplingParams(n=1, presence_penalty=0.0, frequency_penalty=0.0, repetition_penalty=1.0, temperature=0.0, top_p=1.0, top_k=0, min_p=0.0, seed=None, stop=[], stop_token_ids=[], bad_words=[], include_stop_str_in_output=False, ignore_eos=False, max_tokens=120, min_tokens=0, logprobs=None, prompt_logprobs=None, skip_special_tokens=True, spaces_between_special_tokens=True, truncate_prompt_tokens=None, guided_decoding=None, extra_args=None), prompt_token_ids: [128000, 5519, 499, 1440, 279, 2363, 350, 16597, 555, 480, 3394, 75206, 1543], prompt_embeds shape: None, lora_request: None.
(APIServer pid=192421) INFO 08-06 10:27:22 [logger.py:41] Received request cmpl-e5cf3fd11bd5468e837b9cf4cbb4a9b3-0: prompt: 'help me create a rust app that supports the elevenlabs.io api and that can read the contents of clipboard aloud using tts', params: SamplingParams(n=1, presence_penalty=0.0, frequency_penalty=0.0, repetition_penalty=1.0, temperature=0.0, top_p=1.0, top_k=0, min_p=0.0, seed=None, stop=[], stop_token_ids=[], bad_words=[], include_stop_str_in_output=False, ignore_eos=False, max_tokens=770, min_tokens=0, logprobs=None, prompt_logprobs=None, skip_special_tokens=True, spaces_between_special_tokens=True, truncate_prompt_tokens=None, guided_decoding=None, extra_args=None), prompt_token_ids: [128000, 8823, 757, 1893, 264, 23941, 917, 430, 11815, 279, 45314, 71371, 4340, 6464, 323, 430, 649, 1373, 279, 8970, 315, 47134, 71511, 1701, 99640], prompt_embeds shape: None, lora_request: None.
(APIServer pid=192421) INFO 08-06 10:27:22 [logger.py:41] Received request cmpl-3e174a5ae7e74e57889af6fb00fed2d4-0: prompt: 'create new version. we will call it: "second draft". You need to reformat Filters part to be more ease to read', params: SamplingParams(n=1, presence_penalty=0.0, frequency_penalty=0.0, repetition_penalty=1.0, temperature=0.0, top_p=1.0, top_k=0, min_p=0.0, seed=None, stop=[], stop_token_ids=[], bad_words=[], include_stop_str_in_output=False, ignore_eos=False, max_tokens=233, min_tokens=0, logprobs=None, prompt_logprobs=None, skip_special_tokens=True, spaces_between_special_tokens=True, truncate_prompt_tokens=None, guided_decoding=None, extra_args=None), prompt_token_ids: [128000, 3261, 502, 2373, 13, 584, 690, 1650, 433, 25, 330, 5686, 10165, 3343, 1472, 1205, 311, 312, 2293, 46112, 961, 311, 387, 810, 14553, 311, 1373], prompt_embeds shape: None, lora_request: None.
(APIServer pid=192421) INFO 08-06 10:27:22 [logger.py:41] Received request cmpl-7829dcccbd014d1f8d7bdec82440a0cc-0: prompt: 'in the jtbd context whats a push?', params: SamplingParams(n=1, presence_penalty=0.0, frequency_penalty=0.0, repetition_penalty=1.0, temperature=0.0, top_p=1.0, top_k=0, min_p=0.0, seed=None, stop=[], stop_token_ids=[], bad_words=[], include_stop_str_in_output=False, ignore_eos=False, max_tokens=194, min_tokens=0, logprobs=None, prompt_logprobs=None, skip_special_tokens=True, spaces_between_special_tokens=True, truncate_prompt_tokens=None, guided_decoding=None, extra_args=None), prompt_token_ids: [128000, 258, 279, 92176, 9117, 2317, 41209, 264, 4585, 30], prompt_embeds shape: None, lora_request: None.
(APIServer pid=192421) INFO 08-06 10:27:22 [logger.py:41] Received request cmpl-77ff87619d0f40f2bfa07ee32b9f1079-0: prompt: "| Project Charter |  |\n| --- | --- |\n|  | 2. Users may not be satisfied with the functionality or usability of the application, which could affect user adoption. <br> 3. Security breaches or data loss could occur, which could compromise user data and trust. <br> 4. The project budget may exceed expectations due to unforeseen issues or scope changes. |\n| **Approvals:** | The following approvals are required for this project: <br> - Project Charter: [Project Sponsor's Name] <br> - Finalized Design: [Project Sponsor's Name] <br> - User Acceptance Testing: [Project Sponsor's Name] |\n| **Project Success Criteria:** | The success of the project will be measured by the following criteria: <br> 1. Completion of the project on time and within budget. <br> 2. User satisfaction with the application and its features. <br> 3. Reduction in the time and effort required to generate appraisal reports. <br> 4. Improvement in the accuracy and quality of appraisal reports. <br> 5. Increased efficiency in the appraisal process. |\n| **Conclusion:** | This project charter outlines the scope, objectives, deliverables, timeline, budget, project team, assumptions and risks, and approvals required for the development of a web-based commercial appraisal report writing application. The success of the project will be measured by completion on time and within budget, user satisfaction, reduction in time and effort required for appraisal reports, improved accuracy and quality of appraisal reports, and increased efficiency in the appraisal process. |", params: SamplingParams(n=1, presence_penalty=0.0, frequency_penalty=0.0, repetition_penalty=1.0, temperature=0.0, top_p=1.0, top_k=0, min_p=0.0, seed=None, stop=[], stop_token_ids=[], bad_words=[], include_stop_str_in_output=False, ignore_eos=False, max_tokens=101, min_tokens=0, logprobs=None, prompt_logprobs=None, skip_special_tokens=True, spaces_between_special_tokens=True, truncate_prompt_tokens=None, guided_decoding=None, extra_args=None), prompt_token_ids: [128000, 91, 5907, 49705, 765, 220, 9432, 91, 12730, 765, 12730, 9432, 91, 220, 765, 220, 17, 13, 14969, 1253, 539, 387, 20097, 449, 279, 15293, 477, 76160, 315, 279, 3851, 11, 902, 1436, 7958, 1217, 25375, 13, 366, 1347, 29, 220, 18, 13, 8398, 69140, 477, 828, 4814, 1436, 12446, 11, 902, 1436, 30485, 1217, 828, 323, 7095, 13, 366, 1347, 29, 220, 19, 13, 578, 2447, 8199, 1253, 12771, 17078, 4245, 311, 96691, 29412, 4819, 477, 7036, 4442, 13, 9432, 91, 3146, 29688, 26678, 68063, 765, 578, 2768, 83923, 527, 2631, 369, 420, 2447, 25, 366, 1347, 29, 482, 5907, 49705, 25, 510, 8006, 48661, 596, 4076, 60, 366, 1347, 29, 482, 13321, 1534, 7127, 25, 510, 8006, 48661, 596, 4076, 60, 366, 1347, 29, 482, 2724, 21496, 685, 27866, 25, 510, 8006, 48661, 596, 4076, 60, 9432, 91, 3146, 8006, 13346, 14577, 68063, 765, 578, 2450, 315, 279, 2447, 690, 387, 17303, 555, 279, 2768, 13186, 25, 366, 1347, 29, 220, 16, 13, 57350, 315, 279, 2447, 389, 892, 323, 2949, 8199, 13, 366, 1347, 29, 220, 17, 13, 2724, 24617, 449, 279, 3851, 323, 1202, 4519, 13, 366, 1347, 29, 220, 18, 13, 59200, 304, 279, 892, 323, 5149, 2631, 311, 7068, 79392, 6821, 13, 366, 1347, 29, 220, 19, 13, 53751, 304, 279, 13708, 323, 4367, 315, 79392, 6821, 13, 366, 1347, 29, 220, 20, 13, 62697, 15374, 304, 279, 79392, 1920, 13, 9432, 91, 3146, 44534, 68063, 765, 1115, 2447, 38124, 50729, 279, 7036, 11, 26470, 11, 6493, 4893, 11, 25845, 11, 8199, 11, 2447, 2128, 11, 32946, 323, 15635, 11, 323, 83923, 2631, 369, 279, 4500, 315, 264, 3566, 6108, 8518, 79392, 1934, 4477, 3851, 13, 578, 2450, 315, 279, 2447, 690, 387, 17303, 555, 9954, 389, 892, 323, 2949, 8199, 11, 1217, 24617, 11, 14278, 304, 892, 323, 5149, 2631, 369, 79392, 6821, 11, 13241, 13708, 323, 4367, 315, 79392, 6821, 11, 323, 7319, 15374, 304, 279, 79392, 1920, 13, 765], prompt_embeds shape: None, lora_request: None.
(APIServer pid=192421) INFO 08-06 10:27:22 [logger.py:41] Received request cmpl-3685c67b75d94560925ad8e211957495-0: prompt: 'create react and node and express js web app for creating or add dummy data and show and How I can deploy the code after create build.', params: SamplingParams(n=1, presence_penalty=0.0, frequency_penalty=0.0, repetition_penalty=1.0, temperature=0.0, top_p=1.0, top_k=0, min_p=0.0, seed=None, stop=[], stop_token_ids=[], bad_words=[], include_stop_str_in_output=False, ignore_eos=False, max_tokens=741, min_tokens=0, logprobs=None, prompt_logprobs=None, skip_special_tokens=True, spaces_between_special_tokens=True, truncate_prompt_tokens=None, guided_decoding=None, extra_args=None), prompt_token_ids: [128000, 3261, 14085, 323, 2494, 323, 3237, 7139, 3566, 917, 369, 6968, 477, 923, 17741, 828, 323, 1501, 323, 2650, 358, 649, 10739, 279, 2082, 1306, 1893, 1977, 13], prompt_embeds shape: None, lora_request: None.
(APIServer pid=192421) INFO 08-06 10:27:22 [logger.py:41] Received request cmpl-c5c5e5f82d9547e1b10d35312e547d4c-0: prompt: "You can use Django's built-in task scheduling framework, `django-background-tasks`, to schedule the training of your model every `n` number of days.\n\nHere's a high-level overview of how you can implement this:\n\n1. Install the `django-background-tasks` library:\n```css\npip install django-background-tasks\n```\n2. Add `background_tasks` to your `INSTALLED_APPS` in the `settings.py` file:\n```python\nINSTALLED_APPS = [\n    # ...\n    'background_tasks',\n    # ...\n]\n```\n3. Define a task function to train your model:\n```python\nimport pickle\nimport numpy as np\nfrom .models import ModelPath\n\ndef train_model():\n    # Code to train your model\n    model = ...\n    path = ...\n\n    # Save the model to disk\n    pickle.dump(model, open(path, 'wb'))\n\n    # Update the database with the new model path\n    model_path = ModelPath.objects.last()\n    model_path.path = path\n    model_path.save()\n```\n4. Register the task in the `tasks.py` file of your app:\n```python\nfrom background_tasks import background\n\n@background(schedule=60 * 60 * 24 * n)  # Schedule the task to run every n days\ndef run_train_model_task():\n    train_model()\n```\n5. Run the background task worker:\n```\npython manage.py process_tasks\n```\nIn this example, the `train_model` function trains your model, saves it to disk, and updates the database with the new model path. The `run_train_model_task` function is a background task that is scheduled to run every `n` days and calls the `train_model` function. The `process_tasks` command must be run to start the background task worker.\n\nNote: This is just one way to schedule the training of your model. The exact implementation will depend on your specific requirements and constraints.", params: SamplingParams(n=1, presence_penalty=0.0, frequency_penalty=0.0, repetition_penalty=1.0, temperature=0.0, top_p=1.0, top_k=0, min_p=0.0, seed=None, stop=[], stop_token_ids=[], bad_words=[], include_stop_str_in_output=False, ignore_eos=False, max_tokens=9, min_tokens=0, logprobs=None, prompt_logprobs=None, skip_special_tokens=True, spaces_between_special_tokens=True, truncate_prompt_tokens=None, guided_decoding=None, extra_args=None), prompt_token_ids: [128000, 2675, 649, 1005, 53704, 596, 5918, 3502, 3465, 38952, 12914, 11, 1595, 13887, 43034, 2442, 4707, 7964, 311, 9899, 279, 4967, 315, 701, 1646, 1475, 1595, 77, 63, 1396, 315, 2919, 382, 8586, 596, 264, 1579, 11852, 24131, 315, 1268, 499, 649, 4305, 420, 1473, 16, 13, 19796, 279, 1595, 13887, 43034, 2442, 4707, 63, 6875, 512, 74694, 5254, 198, 52601, 4685, 8426, 43034, 2442, 4707, 198, 14196, 4077, 17, 13, 2758, 1595, 6884, 33923, 63, 311, 701, 1595, 65562, 78340, 90854, 63, 304, 279, 1595, 6648, 7345, 63, 1052, 512, 74694, 12958, 198, 65562, 78340, 90854, 284, 2330, 262, 674, 12515, 262, 364, 6884, 33923, 756, 262, 674, 12515, 933, 14196, 4077, 18, 13, 19127, 264, 3465, 734, 311, 5542, 701, 1646, 512, 74694, 12958, 198, 475, 22975, 198, 475, 8760, 439, 2660, 198, 1527, 662, 6644, 1179, 5008, 1858, 271, 755, 5542, 5156, 4019, 262, 674, 6247, 311, 5542, 701, 1646, 198, 262, 1646, 284, 12515, 262, 1853, 284, 5585, 262, 674, 10467, 279, 1646, 311, 13668, 198, 262, 22975, 28026, 7790, 11, 1825, 5698, 11, 364, 20824, 25863, 262, 674, 5666, 279, 4729, 449, 279, 502, 1646, 1853, 198, 262, 1646, 2703, 284, 5008, 1858, 8549, 9288, 746, 262, 1646, 2703, 3960, 284, 1853, 198, 262, 1646, 2703, 5799, 746, 14196, 4077, 19, 13, 8618, 279, 3465, 304, 279, 1595, 25792, 7345, 63, 1052, 315, 701, 917, 512, 74694, 12958, 198, 1527, 4092, 33923, 1179, 4092, 271, 31, 6884, 88812, 28, 1399, 353, 220, 1399, 353, 220, 1187, 353, 308, 8, 220, 674, 24416, 279, 3465, 311, 1629, 1475, 308, 2919, 198, 755, 1629, 7745, 5156, 12461, 4019, 262, 5542, 5156, 746, 14196, 4077, 20, 13, 6588, 279, 4092, 3465, 12128, 512, 14196, 4077, 12958, 10299, 7345, 1920, 33923, 198, 14196, 4077, 644, 420, 3187, 11, 279, 1595, 10613, 5156, 63, 734, 28788, 701, 1646, 11, 27024, 433, 311, 13668, 11, 323, 9013, 279, 4729, 449, 279, 502, 1646, 1853, 13, 578, 1595, 6236, 7745, 5156, 12461, 63, 734, 374, 264, 4092, 3465, 430, 374, 13847, 311, 1629, 1475, 1595, 77, 63, 2919, 323, 6880, 279, 1595, 10613, 5156, 63, 734, 13, 578, 1595, 4734, 33923, 63, 3290, 2011, 387, 1629, 311, 1212, 279, 4092, 3465, 12128, 382, 9290, 25, 1115, 374, 1120, 832, 1648, 311, 9899, 279, 4967, 315, 701, 1646, 13, 578, 4839, 8292, 690, 6904, 389, 701, 3230, 8670, 323, 17413, 13], prompt_embeds shape: None, lora_request: None.
(APIServer pid=192421) INFO 08-06 10:27:22 [logger.py:41] Received request cmpl-3eb8475f0a434bb8bbe1aca321d89783-0: prompt: 'Lila, who sat on the deck, her arms wrapped protectively around the children she had saved. Her eyes were filled with tears, but her expression was resolute.\n\nRoran approached her, offering a handkerchief. "You did what you could," he told her gently. "You saved these children. They\'re alive because of you."\n\nLila took the handkerchief, dabbing at her eyes. "Thank you, Captain. I just wish I could\'ve done more."\n\nAs the ship sailed away from the ruins of the Salakor Shard, Roran gathered his crew, as well as the survivors. Their faces were a mix of shock, sorrow, and determination. Together, they would face the uncertain future and forge a new path for themselves and those they had saved.\n\nThe Falcon\'s Fury became a symbol of hope amidst the chaos, and the story of the Salakor Shard\'s collapse was etched into the hearts and minds of those who survived. The journey ahead would be filled with challenges, but the bonds forged in the face of tragedy would be unbreakable.\n\nAs they sailed toward the Dawn Coast, the survivors of Salakor Shard stared out at the vast expanse of the Aire Sea, their hearts heavy with loss, but also filled with a newfound sense of purpose. In the days and weeks to come, they would work together to rebuild their lives and create a new home on the resilient Dawn Coast. And while the memories of that fateful day would never fade, the resilience of the human spirit would ensure that they continued to endure, adapt, and ultimately, thrive.', params: SamplingParams(n=1, presence_penalty=0.0, frequency_penalty=0.0, repetition_penalty=1.0, temperature=0.0, top_p=1.0, top_k=0, min_p=0.0, seed=None, stop=[], stop_token_ids=[], bad_words=[], include_stop_str_in_output=False, ignore_eos=False, max_tokens=24, min_tokens=0, logprobs=None, prompt_logprobs=None, skip_special_tokens=True, spaces_between_special_tokens=True, truncate_prompt_tokens=None, guided_decoding=None, extra_args=None), prompt_token_ids: [128000, 43, 10746, 11, 889, 7731, 389, 279, 9722, 11, 1077, 11977, 20037, 6144, 3210, 2212, 279, 2911, 1364, 1047, 6924, 13, 6385, 6548, 1051, 10409, 449, 24014, 11, 719, 1077, 7645, 574, 594, 6402, 382, 49, 55504, 25735, 1077, 11, 10209, 264, 1450, 7197, 62626, 13, 330, 2675, 1550, 1148, 499, 1436, 1359, 568, 3309, 1077, 30373, 13, 330, 2675, 6924, 1521, 2911, 13, 2435, 2351, 13989, 1606, 315, 499, 2266, 43, 10746, 3952, 279, 1450, 7197, 62626, 11, 83868, 7278, 520, 1077, 6548, 13, 330, 13359, 499, 11, 22022, 13, 358, 1120, 6562, 358, 1436, 3077, 2884, 810, 2266, 2170, 279, 8448, 76844, 3201, 505, 279, 46762, 315, 279, 8375, 587, 269, 96466, 11, 432, 55504, 20802, 813, 13941, 11, 439, 1664, 439, 279, 32696, 13, 11205, 12580, 1051, 264, 6651, 315, 10988, 11, 58596, 11, 323, 26314, 13, 32255, 11, 814, 1053, 3663, 279, 36218, 3938, 323, 57728, 264, 502, 1853, 369, 5694, 323, 1884, 814, 1047, 6924, 382, 791, 43961, 596, 50479, 6244, 264, 7891, 315, 3987, 65904, 279, 28013, 11, 323, 279, 3446, 315, 279, 8375, 587, 269, 96466, 596, 18678, 574, 1880, 2454, 1139, 279, 23492, 323, 20663, 315, 1884, 889, 26968, 13, 578, 11879, 8469, 1053, 387, 10409, 449, 11774, 11, 719, 279, 27460, 54299, 304, 279, 3663, 315, 31926, 1053, 387, 653, 9137, 481, 382, 2170, 814, 76844, 9017, 279, 35607, 16377, 11, 279, 32696, 315, 8375, 587, 269, 96466, 45135, 704, 520, 279, 13057, 506, 95519, 315, 279, 362, 556, 15379, 11, 872, 23492, 8987, 449, 4814, 11, 719, 1101, 10409, 449, 264, 94621, 5647, 315, 7580, 13, 763, 279, 2919, 323, 5672, 311, 2586, 11, 814, 1053, 990, 3871, 311, 32593, 872, 6439, 323, 1893, 264, 502, 2162, 389, 279, 59780, 35607, 16377, 13, 1628, 1418, 279, 19459, 315, 430, 282, 21508, 1938, 1053, 2646, 15366, 11, 279, 56062, 315, 279, 3823, 9090, 1053, 6106, 430, 814, 8738, 311, 46753, 11, 10737, 11, 323, 13967, 11, 41972, 13], prompt_embeds shape: None, lora_request: None.
(APIServer pid=192421) INFO 08-06 10:27:22 [logger.py:41] Received request cmpl-b04ab4dcf78746278d96369fcd5986c4-0: prompt: '**Assistant**', params: SamplingParams(n=1, presence_penalty=0.0, frequency_penalty=0.0, repetition_penalty=1.0, temperature=0.0, top_p=1.0, top_k=0, min_p=0.0, seed=None, stop=[], stop_token_ids=[], bad_words=[], include_stop_str_in_output=False, ignore_eos=False, max_tokens=6, min_tokens=0, logprobs=None, prompt_logprobs=None, skip_special_tokens=True, spaces_between_special_tokens=True, truncate_prompt_tokens=None, guided_decoding=None, extra_args=None), prompt_token_ids: [128000, 334, 72803, 334], prompt_embeds shape: None, lora_request: None.
(APIServer pid=192421) INFO 08-06 10:27:22 [logger.py:41] Received request cmpl-19484518145041b1a5aa1f0614d24b22-0: prompt: '"test: [noun] a means of testing: such as. something (such as a series of questions or exercises) for measuring the skill, knowledge, intelligence, capacities, or aptitudes of an individual or group. a procedure, reaction, or reagent used to identify or characterize a substance or constituent. a positive result in such a test."\nSource: https://www.merriam-webster.com/dictionary/test\n\n"Define test. test synonyms, test pronunciation, test translation, English dictionary definition of test. n. 1. A procedure for critical evaluation; a means of determining the presence, quality, or truth of something; a trial: a test of ones eyesight;..."\nSource: https://www.thefreedictionary.com/test\n\n"Synonyms for TEST: essay, experiment, experimentation, trial, exam, examination, quiz, sample"\nSource: https://www.merriam-webster.com/thesaurus/test\n\nGiven these web results, answer the following question: test', params: SamplingParams(n=1, presence_penalty=0.0, frequency_penalty=0.0, repetition_penalty=1.0, temperature=0.0, top_p=1.0, top_k=0, min_p=0.0, seed=None, stop=[], stop_token_ids=[], bad_words=[], include_stop_str_in_output=False, ignore_eos=False, max_tokens=80, min_tokens=0, logprobs=None, prompt_logprobs=None, skip_special_tokens=True, spaces_between_special_tokens=True, truncate_prompt_tokens=None, guided_decoding=None, extra_args=None), prompt_token_ids: [128000, 1, 1985, 25, 510, 91209, 60, 264, 3445, 315, 7649, 25, 1778, 439, 13, 2555, 320, 21470, 439, 264, 4101, 315, 4860, 477, 23783, 8, 369, 30090, 279, 10151, 11, 6677, 11, 11478, 11, 59539, 11, 477, 20697, 21237, 315, 459, 3927, 477, 1912, 13, 264, 10537, 11, 13010, 11, 477, 312, 8252, 1511, 311, 10765, 477, 70755, 264, 20278, 477, 75164, 13, 264, 6928, 1121, 304, 1778, 264, 1296, 10246, 3692, 25, 3788, 1129, 2185, 749, 261, 462, 309, 30531, 3751, 916, 3529, 4003, 12986, 271, 1, 36438, 1296, 13, 1296, 86506, 11, 1296, 71722, 11, 1296, 14807, 11, 6498, 11240, 7419, 315, 1296, 13, 308, 13, 220, 16, 13, 362, 10537, 369, 9200, 16865, 26, 264, 3445, 315, 26679, 279, 9546, 11, 4367, 11, 477, 8206, 315, 2555, 26, 264, 9269, 25, 264, 1296, 315, 6305, 6548, 492, 26, 31538, 3692, 25, 3788, 1129, 2185, 13991, 830, 29616, 4003, 916, 12986, 271, 1, 38234, 46703, 369, 13916, 25, 9071, 11, 9526, 11, 66196, 11, 9269, 11, 7151, 11, 24481, 11, 28223, 11, 6205, 702, 3692, 25, 3788, 1129, 2185, 749, 261, 462, 309, 30531, 3751, 916, 14, 6509, 43613, 12986, 271, 22818, 1521, 3566, 3135, 11, 4320, 279, 2768, 3488, 25, 1296], prompt_embeds shape: None, lora_request: None.
(APIServer pid=192421) INFO:     127.0.0.1:57650 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=192421) INFO 08-06 10:27:22 [async_llm.py:273] Added request cmpl-a841fd86e1a24566b13abcb62969febe-0.
(APIServer pid=192421) INFO:     127.0.0.1:57658 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=192421) INFO 08-06 10:27:22 [async_llm.py:273] Added request cmpl-e5cf3fd11bd5468e837b9cf4cbb4a9b3-0.
(APIServer pid=192421) INFO:     127.0.0.1:57660 - "POST /v1/completions HTTP/1.1" 200 OK
DEBUG 08-06 10:27:22 [core.py:687] EngineCore loop active.
(APIServer pid=192421) INFO 08-06 10:27:22 [async_llm.py:273] Added request cmpl-3e174a5ae7e74e57889af6fb00fed2d4-0.
(APIServer pid=192421) INFO:     127.0.0.1:57664 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=192421) INFO 08-06 10:27:22 [async_llm.py:273] Added request cmpl-7829dcccbd014d1f8d7bdec82440a0cc-0.
(APIServer pid=192421) INFO:     127.0.0.1:57676 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=192421) INFO 08-06 10:27:22 [async_llm.py:273] Added request cmpl-77ff87619d0f40f2bfa07ee32b9f1079-0.
(APIServer pid=192421) INFO:     127.0.0.1:57692 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=192421) INFO 08-06 10:27:22 [async_llm.py:273] Added request cmpl-3685c67b75d94560925ad8e211957495-0.
(APIServer pid=192421) INFO:     127.0.0.1:57702 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=192421) INFO 08-06 10:27:22 [async_llm.py:273] Added request cmpl-c5c5e5f82d9547e1b10d35312e547d4c-0.
(APIServer pid=192421) INFO:     127.0.0.1:57712 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=192421) INFO 08-06 10:27:22 [async_llm.py:273] Added request cmpl-3eb8475f0a434bb8bbe1aca321d89783-0.
(APIServer pid=192421) INFO:     127.0.0.1:57714 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=192421) INFO 08-06 10:27:22 [async_llm.py:273] Added request cmpl-b04ab4dcf78746278d96369fcd5986c4-0.
(APIServer pid=192421) INFO:     127.0.0.1:57716 - "POST /v1/completions HTTP/1.1" 200 OK
(APIServer pid=192421) INFO 08-06 10:27:22 [async_llm.py:273] Added request cmpl-19484518145041b1a5aa1f0614d24b22-0.
DEBUG 08-06 10:27:26 [core.py:681] EngineCore waiting for work.