import logging
from multiprocessing import Value
import os
import vllm
import torch 
import types
import gc
from packaging.version import parse

from .flash_quantization import get_quantize_fn
import sys 

# Set up logger
logger = logging.getLogger(__name__)



def fanout_existing_imports():
    from vllm.model_executor.model_loader import utils

    orig = getattr(utils, "beforeflashrl_process_weights_after_loading", None)
    hacked = utils.process_weights_after_loading
    if not callable(orig) or not callable(hacked):
        return
    for name, mod in list(sys.modules.items()):
        if not name.startswith("vllm.model_executor.model_loader"):
            continue
        for attr_name, attr_value in list(vars(mod).items()):
            if attr_value is orig:
                setattr(mod, attr_name, hacked)
                logger.debug(f"[fanout] Replaced {name}.{attr_name} with hacked fn")

def apply_patch():
    from loguru import logger

    # Check if patching is needed based on environment variables
    if "FLASHRL_CONFIG" in os.environ:
        try :
            import vllm 
        except Exception: 
            print("Skipping patching, vllm not installed")
            return

        # Patch the process_weights_after_loading function
        process_weights_status = patch_vllm_process_weights_after_loading()
        fanout_existing_imports()
        logger.debug(f"Patching vllm process_weights_after_loading... status: {process_weights_status}")
        

        # patch_vllm_logprob_compute()
        # logger.debug("patching vllm logprob works")

        # Patch the LLM class
        patch_status = patch_vllm_llm()
        logger.debug(f"Patching the vllm LLM to enable flash_rl quantization... status: {patch_status}")

        if "FLASHRL_TEST_RELOAD" in os.environ:

            reload_status = patch_vllm_llm_test_reload()
            logger.debug(f"Patching vllm LLM init to test reload... status: {reload_status}")

        if os.environ.get("FLASHRL_LMHEAD_FP32", "0") == "1":

            patch_status = patch_vllm_lmhead_to_fp32()
            logger.debug(f"Patching vllm lmhead to fp32... status: {patch_status}")
    else:
        logger.debug("Skipping the patching of vllm")


def vllm_model_finder(vllm_llm):
    if not hasattr(vllm_llm.llm_engine, 'model_executor'):
        vllm_llm.llm_engine.model_executor = vllm_llm.llm_engine.engine_core.model_executor
    
    vllm_model = vllm_llm.llm_engine.model_executor.driver_worker.worker.model_runner.model
    return vllm_model

def bond_method_to_cls(func, obj):
    if hasattr(func, '__self__') or not callable(func):
        # If the function is already bound to an instance, return it as is
        return func
    else:
        return types.MethodType(func, obj)

recorded_loader_keys = [
    'weight_loader',
    'load_qkv_weight',
    'load_row_parallel_weight',
    'load_merged_column_weight',
    'output_dim',
    'input_dim',
    '_assert_and_load',
]
def patch_vllm_process_weights_after_loading():
    try:
        # Store the original process_weights_after_loading function
        from vllm.model_executor.model_loader import utils
        
        if not hasattr(utils, 'beforeflashrl_process_weights_after_loading'):
            
            original_process_weights_after_loading = utils.process_weights_after_loading
            utils.beforeflashrl_process_weights_after_loading = original_process_weights_after_loading
            
            def hacked_process_weights_after_loading(
                model, 
                model_config, 
                target_device, 
            ) -> None:
                if model_config is None and target_device is None:
                    model_config = getattr(model, 'hacked_model_config', None)
                    target_device = getattr(model, 'hacked_target_device', None)
                else:
                    setattr(model, 'hacked_model_config', model_config)
                    setattr(model, 'hacked_target_device', target_device)
            
                if getattr(model, 'hacked_not_need_process_weights_after_loading', False):
                    logger.debug("vllm process_weights_after_loading already processed")
                    return

                # print("Patched process_weights_after_loading function called")
                
                original_weights = dict(model.named_parameters())
                
                # this can be optimized for better memory usage, leave for future work...
                if not hasattr(model, 'hacked_original_weights_rebuild_keys'):
                    model.hacked_original_weights_rebuild_keys = {}
                    for name, p in original_weights.items():
                        model.hacked_original_weights_rebuild_keys[name] = (p.shape, p.stride(), p.dtype, p.untyped_storage().nbytes())
                
                # record weight_loader 
                recorded_loader = {k: dict() for k in recorded_loader_keys}
                for name, p in original_weights.items():
                    for k in recorded_loader.keys():
                        if hasattr(p, k):
                            attr = getattr(p, k)
                            if not callable(attr):
                                recorded_loader[k][name] = attr
                            elif p is attr.__self__:
                                recorded_loader[k][name] = attr.__func__
                            else:
                                recorded_loader[k][name] = attr
                
                original_process_weights_after_loading(model, model_config, target_device)
                        
                model.hacked_recorded_loader = recorded_loader

            # Patch the process_weights_after_loading function
            utils.process_weights_after_loading = hacked_process_weights_after_loading
            
            logger.debug("Successfully patched the process_weights_after_loading function of vllm")
        else:
            logger.debug("vllm process_weights_after_loading already patched")
        
        from vllm.model_executor.layers.quantization.kv_cache import BaseKVCacheMethod
        if not hasattr(BaseKVCacheMethod, 'beforeflashrl_process_weights_after_loading'):
            original_kvcache_process_weights_after_loading = BaseKVCacheMethod.process_weights_after_loading
            BaseKVCacheMethod.beforeflashrl_process_weights_after_loading = original_kvcache_process_weights_after_loading
            def hacked_kvcache_process_weights_after_loading(
                self,
                layer,
            ) -> None:
                # print("Patched kv_cache process_weights_after_loading function called")
                if not hasattr(layer, 'k_scale'):
                    layer.k_scale = -1.0
                if not hasattr(layer, 'v_scale'):
                    layer.v_scale = -1.0
                if not hasattr(layer, 'q_scale'):
                    layer.q_scale = -1.0
                if not hasattr(layer, 'prob_scale'):
                    layer.prob_scale = -1.0
                return original_kvcache_process_weights_after_loading(self, layer)
            BaseKVCacheMethod.process_weights_after_loading = hacked_kvcache_process_weights_after_loading
            
            logger.debug("Successfully patched kv_cache process_weights_after_loading")
        else:
            logger.debug("vllm kv_cache process_weights_after_loading already patched")
            
        return True
        
    except Exception as e:
        logger.debug(f"Error patching vllm process_weights_after_loading: {e}")
        return False

def patch_vllm_lmhead_to_fp32():
    try:
        if not hasattr(vllm.model_executor.layers.vocab_parallel_embedding.VocabParallelEmbedding, 'beforeflashrl_VocabParallelEmbedding'):
            # Store the original LLM init function
            original_init = vllm.model_executor.layers.vocab_parallel_embedding.VocabParallelEmbedding.__init__
            vllm.model_executor.layers.vocab_parallel_embedding.VocabParallelEmbedding.beforeflashrl_VocabParallelEmbedding = original_init

            def hacked_init_(
                self, 
                *args, 
                **kwargs
            ) -> None:
               
                if len(args) >= 3:
                    output_dtype = args[2]
                    args = list(args)
                    args[2] = torch.float32
                    args = tuple(args)
                else:
                    output_dtype = kwargs.get('params_dtype', torch.get_default_dtype())
                    kwargs['params_dtype'] = torch.float32

                if len(args) >= 6:
                    args = list(args)
                    args[5] = None
                    args = tuple(args)
                else:
                    kwargs['quant_config'] = None

                print('Here are the kwargs: ', kwargs)

                init_return = original_init(
                    self, 
                    *args,
                    **kwargs,
                )
                self.output_dtype = output_dtype
                
                if not hasattr(self, 'beforeflashrl_forward'):
                    original_forward = self.forward 
                    self.beforeflashrl_forward = original_forward

                    def hacked_forward(
                        self, 
                        *args, 
                        **kwargs,
                    ):
                        original_forward_return = original_forward(*args, **kwargs)
                        return original_forward_return.to(self.output_dtype)

                    setattr(self, 'forward', bond_method_to_cls(hacked_forward, self))
                    logger.debug("Successfully patched vllm VocabParallelEmbedding forward to fp32 weight")
                else: 
                    logger.debug("vllm VocabParallelEmbedding forward already patched to fp32 weight")
                return init_return
            
            # Patch the LLM init function
            vllm.model_executor.layers.vocab_parallel_embedding.VocabParallelEmbedding.__init__ = hacked_init_
            print("Successfully patched vllm VocabParallelEmbedding at init")
        else:
            print("vllm VocabParallelEmbedding init already patched")
        
        status = True
    except Exception as e:
        
        status = False
        logger.error(f"Error patching VocabParallelEmbedding init: {e}")

    try: 
        if not hasattr(vllm.model_executor.layers.vocab_parallel_embedding.UnquantizedEmbeddingMethod, 'beforeflashrl_apply'):
            # Store the original LLM init function
            original_apply = vllm.model_executor.layers.vocab_parallel_embedding.UnquantizedEmbeddingMethod.apply
            vllm.model_executor.layers.vocab_parallel_embedding.UnquantizedEmbeddingMethod.beforeflashrl_apply = original_apply

            def hacked_apply_(
                self,
                layer: torch.nn.Module,
                x: torch.Tensor,
                bias = None
            ) -> torch.Tensor:
                original_type = x.dtype
                x = x.to(dtype=torch.float32)
                return original_apply(self, layer, x, bias).to(original_type)
             
            vllm.model_executor.layers.vocab_parallel_embedding.UnquantizedEmbeddingMethod.apply = hacked_apply_
            print("Successfully patched vllm UnquantizedEmbeddingMethod at apply")
        else:
            print("vllm UnquantizedEmbeddingMethod apply already patched")
        status = status
    except Exception as e:
        status = False
        logger.error(f"Error patching UnquantizedEmbeddingMethod apply: {e}")
    
    return status 

def apply_top_k_top_p(logits, k, p) -> torch.Tensor:
    """copied from vllm
    """
    if k is None and p is None:
        return logits
    logits_sort, logits_idx = logits.sort(dim=-1, descending=False)

    if k is not None:
        # Apply top-k.
        top_k_mask = logits_sort.size(1) - k.to(torch.long)  # shape: B
        # Get all the top_k values.
        top_k_mask = logits_sort.gather(1, top_k_mask.unsqueeze(dim=1))
        top_k_mask = logits_sort < top_k_mask
        logits_sort.masked_fill_(top_k_mask, -float("inf"))

    if p is not None:
        # Apply top-p.
        probs_sort = logits_sort.softmax(dim=-1)
        probs_sum = probs_sort.cumsum(dim=-1)
        top_p_mask = probs_sum <= 1 - p.unsqueeze(dim=1)
        # at least one
        top_p_mask[:, -1] = False
        logits_sort.masked_fill_(top_p_mask, -float("inf"))

    # Re-sort the probabilities.
    logits = logits_sort.scatter(dim=-1, index=logits_idx, src=logits_sort)
    return logits

def patch_vllm_logprob_compute():
    try:
        from vllm.v1.sample.sampler import Sampler
        if not hasattr(Sampler, 'beforeflashrl_forward'):
            # Store the original LLM init function
            original_forward = Sampler.forward
            Sampler.beforeflashrl_forward = original_forward

            def hacked_logprob_forward(
                self,
                logits: torch.Tensor,
                sampling_metadata,
            ):
                # Use float32 for the logits.
                logits = logits.to(torch.float32)
                
                # Apply temperature.
                if (
                    isinstance(sampling_metadata.temperature, torch.Tensor) 
                    and torch.any(sampling_metadata.temperature != 1.0)
                ) or (
                    isinstance(sampling_metadata.temperature, float)
                    and sampling_metadata.temperature != 1.0
                ):
                    logits = self.apply_temperature(logits, sampling_metadata.temperature)
                    
                # Apply topk and/or topp.
                logits = apply_top_k_top_p(logits, sampling_metadata.top_k, sampling_metadata.top_p)
                
                if sampling_metadata.all_random:
                    greedy_sampled = None
                else:
                    greedy_sampled = logits.argmax(dim=-1).view(-1)
                    
                if sampling_metadata.all_greedy:
                    sampled = greedy_sampled
                else:
                    # Sampling
                    sampled = self.topk_topp_sampler(
                        logits,
                        sampling_metadata.generators,
                        None,
                        None,
                    )
                    
                    if greedy_sampled is not None:
                        sampled = torch.where(
                            sampling_metadata.temperature < 1e-5,
                            greedy_sampled,
                            sampled,
                            out=greedy_sampled,  # Reuse tensor
                        )

                if sampling_metadata.max_num_logprobs is not None:
                    processed_logprobs = self.compute_logprobs(logits)
                    logprobs_tensors = self.gather_logprobs(processed_logprobs, 0, token_ids=sampled.long())
                else:
                    logprobs_tensors = None
                
                sampler_output = vllm.v1.outputs.SamplerOutput(
                    sampled_token_ids=sampled.to(torch.int32).unsqueeze(-1),
                    logprobs_tensors=logprobs_tensors,
                )
                return sampler_output
            
            # Patch the LLM init function
            Sampler.forward = hacked_logprob_forward
            
            logger.debug("Successfully patched Sampler at init")
        else:
            logger.debug("vllm Sampler already patched")
            
        status = True
    
    except Exception as e:
        logger.error(f"Error patching Sampler forward: {e}")
        status = False

    return status

keys_to_overload = [
    'load_format', 
    'quantization', 
    'distributed_executor_backend',
]

def load_flashrl_config(config):
    
    config_path = config.strip()
    
    if config_path in ['bf16', 'fp8', 'fp8_vllm']: 
        logger.info(f"Using profile-free default for: {config_path}")
        
        from .flashrl_configs import get_default_config
        from dataclasses import asdict
        config_data = {'configs': [asdict(get_default_config(config_path))]}
    else:        
        logger.info(f"Loading flash_rl config from: {config_path}")
        
        if not os.path.exists(config_path):
            from huggingface_hub import hf_hub_download
            config_path = config_path.split('/')
            assert len(config_path) >= 3, f'Invalid flash_rl config path: {config_path}'
            config_path = hf_hub_download(repo_id='/'.join(config_path[:2]), filename='/'.join(config_path[2:]))

        import yaml
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)

    return config_data

def patch_load_weights(self: "Worker"):
    config = os.environ.get("FLASHRL_CONFIG", None)
    if not config: 
        print("Skipping patch load weights...")
        return 
    config_data = load_flashrl_config(config)["configs"][0]
    print('config data in patch load weights: ', config_data)
    if 'module_attribute_to_preserve' in config_data:
        print(f"flash_rl module_attribute_to_preserve: {config_data['module_attribute_to_preserve']}")
        self.flash_rl_module_attribute_to_preserve = config_data.get('module_attribute_to_preserve')
    else:
        self.flash_rl_module_attribute_to_preserve = []
    model = self.model_runner.model
    self.flash_rl_profile = None

    if (not hasattr(model, 'beforeflashrl_load_weights')) and \
        (config_data.get('fn', 'int8') != 'bf16'):
        
        quant_fn = config_data.get('fn', 'int8')
        logger.debug(f"flash_rl quantization function: {quant_fn}")
        flash_quantize_fn = get_quantize_fn(quant_fn)
            
        # Store the original load_weights function
        original_load_weights = model.load_weights
        model.beforeflashrl_load_weights = original_load_weights
        def hacked_load_weights(
            weights,
        ):
            # print("flash_rl quant load_weights is called")
            setattr(model, 'hacked_not_need_process_weights_after_loading', False)
            
            if len(self.flash_rl_module_attribute_to_preserve) > 0:
                for _, module in model.named_modules():
                    for attr in self.flash_rl_module_attribute_to_preserve:
                        if torch.is_tensor(getattr(module, attr, None)):
                            # print(f"flash_rl reserving {attr} in module {module}")
                            setattr(module, f'hacked_{attr}', getattr(module, attr))
            
            existing_params = dict(model.named_parameters())
            
            hacked_data_dict = {}
            for name, param in existing_params.items():
                hacked_data_dict[name] = param.data
            
            assert hasattr(model, "hacked_original_weights_rebuild_keys")
            
            for name, (shape, stride, dtype, nbytes) in model.hacked_original_weights_rebuild_keys.items():
                if name in existing_params:
                    existing_params[name].data = torch.empty(shape, dtype=dtype) 
            
            for k, loader_k in model.hacked_recorded_loader.items():
                for n, loader in loader_k.items():
                    if not hasattr(existing_params[n], k):
                        setattr(existing_params[n], k, bond_method_to_cls(loader, existing_params[n]))

            updated_params = original_load_weights(
                flash_quantize_fn(weights, self.flash_rl_profile)
            )
            
            if hasattr(model, 'hacked_model_config') and hasattr(model, 'hacked_target_device'):        
                from vllm.model_executor.model_loader import utils
                utils.process_weights_after_loading(model, None, None)
                setattr(model, 'hacked_not_need_process_weights_after_loading', True)
            else:
                setattr(model, 'hacked_not_need_process_weights_after_loading', False)
            
            skipped_params = list()
            print("I'm inside the hacked load weights, yay")
            for name, p in existing_params.items():
                assert name in hacked_data_dict, f'param {name} is not in hacked_data dict'
                assert hacked_data_dict[name].dtype == p.data.dtype, f'param {name} dtype mismatch: {hacked_data_dict[name].dtype} vs {p.data.dtype}'
                assert hacked_data_dict[name].numel() == p.data.numel(), f'param {name} numel() mismatch: {hacked_data_dict[name].numel()} vs {p.data.numel()}'
                
                if name in updated_params:
                    trided_data = torch.as_strided(p.data, hacked_data_dict[name].shape, hacked_data_dict[name].stride())
                    hacked_data_dict[name].copy_(trided_data)
                else:
                    skipped_params.append(name)
                    
                tmp_data = p.data
                p.data = hacked_data_dict[name]
                del tmp_data
            
            logger.debug(f"flash_rl load_weights skipped params (not accurate for `fp8-vllm`): {skipped_params}")
            del skipped_params
            del hacked_data_dict
            del existing_params
            gc.collect()
            torch.cuda.empty_cache()
            
            if len(self.flash_rl_module_attribute_to_preserve) > 0:
                for _, module in model.named_modules():
                    for attr in self.flash_rl_module_attribute_to_preserve:
                        if torch.is_tensor(getattr(module, attr, None)):
                            assert hasattr(module, f'hacked_{attr}'), f"module {module} does not have attribute hacked_{attr}"
                            setattr(module, attr, getattr(module, f'hacked_{attr}'))
                            delattr(module, f'hacked_{attr}')
                            
            return updated_params
        
        model.load_weights = hacked_load_weights
        logger.debug("Successfully patched the load_weights function of vllm")

def patch_vllm_llm():
    try:                                                                                                                    
        if not hasattr(vllm.LLM, 'beforeflashrl__init__'):
            # Store the original LLM init function
            original_init = vllm.LLM.__init__
            vllm.LLM.beforeflashrl__init__ = original_init
        
            def hacked_init_(
                self, 
                model: str,
                **kwargs
            ) -> None:
                
                # Patch the sampler class
                print("Inside the hacked init function")
                sampler_patch_status = patch_vllm_logprob_compute()
                logger.debug(f"Patching vllm Sampler... status: {sampler_patch_status}")
                
                import vllm.envs as envs
                assert envs.VLLM_USE_V1, 'flash_rl only supports vllm v1 for now'

                config = os.environ.get("FLASHRL_CONFIG", None)
                
                # if 'distributed_executor_backend' not in kwargs or kwargs['distributed_executor_backend'] != 'external_launcher':
                #     logger.error("flash_rl only supports external_launcher for now")
                
                # assert 'RANK' in os.environ and 'WORLD_SIZE' in os.environ, \
                #     'flash_rl only supports external_launcher for now'
                    
                # rank = int(os.environ.get("RANK", None))
                # mp_size = kwargs.get('tensor_parallel_size', 1) * kwargs.get('pipeline_parallel_size', 1)
                # dp_rank = rank // mp_size
                dp_rank = 0
                
                if config is not None:
                    # Load the config file and set the model
                    # Assuming config is a JSON file, you can use json.load() to read it
                    logger.info(f"flash_rl config detected.")
                    config_data = load_flashrl_config(config)
                        
                    config_count = len(config_data['configs'])
                    config_index = dp_rank % config_count
                    logger.info(f"Using config {config_index} of {config_count}")
                    config_data = config_data['configs'][config_index]
                    
                    for k, v in config_data.items():
                        logger.info(f"flash_rl config: {k}: {v}")
                    
                    for key in keys_to_overload:
                        if key in config_data:
                            logger.debug(f"Overloading {key} with {config_data[key]}")
                            kwargs[key] = config_data.get(key)
                    model = config_data.get('model', model)
                    if config_data.get('fn', 'int8') != 'bf16':
                        
                        # assert parse(vllm.__version__) <= parse('0.8.4'), (
                        #     f'detected vLLM version {vllm.__version__}'
                        #     'for vLLM > 0.8.4, `FlashRL` only supports `bf16` patches'
                        #     'for exact logprob compute'
                        # )
                        
                        if config_data.get('fn', 'int8') in ['fp8_vllm', 'fp8']:
                            if 'profile' in config_data:
                                logger.warning(f"flash_rl fp8_vllm profile is not needed, but set as {config_data['profile']}")
                            self.flash_rl_profile = None
                        else:
                            quant_profile = config_data.get('profile', os.path.join(model, 'profile.pt'))
                            logger.debug(f"Loading flash_rl profile from: {quant_profile}")
                            
                            quant_profile_path = quant_profile.strip()
                            if not os.path.exists(quant_profile_path):
                                from huggingface_hub import hf_hub_download
                                quant_profile_path = quant_profile_path.split('/')
                                assert len(quant_profile_path) >= 3, f'Invalid flash_rl profile path: {quant_profile_path}'
                                quant_profile_path = hf_hub_download(repo_id='/'.join(quant_profile_path[:2]), filename='/'.join(quant_profile_path[2:]))
                            
                            self.flash_rl_profile = torch.load(quant_profile_path)
                        
                    if 'module_attribute_to_preserve' in config_data:
                        logger.debug(f"flash_rl module_attribute_to_preserve: {config_data['module_attribute_to_preserve']}")
                        self.flash_rl_module_attribute_to_preserve = config_data.get('module_attribute_to_preserve')
                    else:
                        self.flash_rl_module_attribute_to_preserve = []
                        
                else:
                    logger.info(f"flash_rl config not detected.")
                    logger.info(f"Using the original model: {model}")
                
                print("using the kwargs: ", kwargs)
                # Call the parent's __init__ with the custom model
                init_return = original_init(
                    self, 
                    model, 
                    **kwargs,
                )

                logger.debug("vllm init patching successful")
                
                return init_return
            
            # Patch the LLM init function
            vllm.LLM.__init__ = hacked_init_
            
            logger.debug("Successfully patched vllm")
        else:
            logger.debug("vllm LLM already patched")
        return True
        
    except Exception as e:
        logger.error(f"Error patching vllm LLM: {e}")
        return False

def patch_vllm_llm_test_reload():
    try:
        if not hasattr(vllm.LLM, 'beforeflashrl_test__init__'):
            # Store the original LLM init function
            test_vllm_init = vllm.LLM.__init__
            vllm.LLM.beforeflashrl_test__init__ = test_vllm_init
        
            def test_reload_at_init_(
                self, 
                *args, 
                **kwargs
            ) -> None:
                # Call the parent's __init__ with the custom model
                init_return = test_vllm_init(self, *args, **kwargs)
                
                config = os.environ.get("FLASHRL_TEST_RELOAD", None)
                
                if config is not None:
                    config = config.split(',')
                    from transformers import AutoModelForCausalLM
                    for config_i in config:
                        model_to_be_reloaded = AutoModelForCausalLM.from_pretrained(config_i, device_map="cpu", torch_dtype="auto")
                        device = torch.cuda.current_device()
                        print(f"FLASH_RL re-loading model {config_i} to device {device}")
                        model = vllm_model_finder(self)
                        model.load_weights(
                            ((name, param.to(device)) for name, param in model_to_be_reloaded.named_parameters())
                        )
                else:
                    print(f"FLASH_RL re-loading model not detected.")
                
                return init_return
            
            # Patch the LLM init function
            vllm.LLM.__init__ = test_reload_at_init_
            
            print("Successfully patched vllm reload at init")
        else:
            print("vllm LLM reload already patched")
        return True
        
    except Exception as e:
        logger.error(f"Error patching vllm reload LLM: {e}")
        return False
