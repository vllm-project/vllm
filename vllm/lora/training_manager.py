# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import re
import json
import os
from typing import Dict, List, Optional, Set
from functools import partial

import math
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import LinearLR
from safetensors.torch import save_file

from vllm.config.lora import LoRAConfig
from vllm.logger import init_logger
from vllm.lora.layers.column_parallel_linear import MergedQKVParallelLinearWithLoRA
from vllm.lora.models import LoRAModel
from vllm.lora.worker_manager import WorkerLoRAManager
from vllm.lora.request import LoRARequest
from vllm.lora.training_state import TrainingState


logger = init_logger(__name__)


def _get_cosine_schedule_with_warmup_lr_lambda(
    current_step: int, *, num_warmup_steps: int, num_training_steps: int, num_cycles: float, min_lr_rate: float = 0.0
):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    factor = 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
    factor = factor * (1 - min_lr_rate) + min_lr_rate
    return max(0, factor)


def get_decay_parameter_names(model) -> list[str]:
    """
    Get all parameter names that weight decay will be applied to.

    This function filters out parameters in two ways:
    1. By layer type (instances of layers specified in ALL_LAYERNORM_LAYERS)
    2. By parameter name patterns (containing 'bias', or variation of 'norm')
    """
    forbidden_name_patterns = [r"bias", r"layernorm", r"rmsnorm", r"(?:^|\.)norm(?:$|\.)", r"_norm(?:$|\.)"]
    decay_parameters = get_parameter_names(model, [nn.LayerNorm], forbidden_name_patterns)
    return decay_parameters


def get_parameter_names(model, forbidden_layer_types, forbidden_layer_names=None):
    """
    Returns the names of the model parameters that are not inside a forbidden layer.
    """
    forbidden_layer_patterns = (
        [re.compile(pattern) for pattern in forbidden_layer_names] if forbidden_layer_names is not None else []
    )
    print(f"forbidden: {forbidden_layer_patterns}")
    result = []
    for name, child in model.named_children():
        child_params = get_parameter_names(child, forbidden_layer_types, forbidden_layer_names)
        result += [
            f"{name}.{n}"
            for n in child_params
            if not isinstance(child, tuple(forbidden_layer_types))
            and not any(pattern.search(f"{name}.{n}".lower()) for pattern in forbidden_layer_patterns)
        ]
    # Add model specific parameters that are not in any child
    result += [
        k for k in model._parameters if not any(pattern.search(k.lower()) for pattern in forbidden_layer_patterns)
    ]
    print(f"result: {result}")
    return result


# TODO(girfan): Either use this or remove this. Need to fix the way decay parameters are obtained. Is that correct?
def create_optimizer(model, weight_decay: float = 0.0, lr: float = 0.0001):
    decay_parameters = get_decay_parameter_names(model)
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters() if (n in decay_parameters and p.requires_grad)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters() if (n not in decay_parameters and p.requires_grad)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer_kwargs = {'lr': lr, 'betas': (0.9, 0.999), 'eps': 1e-08, 'fused': True}
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, **optimizer_kwargs)
    return optimizer


class TrainingManager:
    """
    Training manager for managing LoRA adapters during training.
    
    This class implements the singleton pattern to ensure only one instance
    exists throughout the application lifecycle.

    TODO(girfan): Add support for multiple LoRA adapters.
    """
    
    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(TrainingManager, cls).__new__(cls)
        return cls._instance

    def __init__(
        self,
        model_runner: "GPUModelRunner",
        lora_manager: WorkerLoRAManager,
        lora_config: LoRAConfig,
        device: torch.device,
        dtype: torch.dtype,
    ):
        # Only initialize once
        if self._initialized:
            return

        self.model_runner = model_runner
        self.lora_manager = lora_manager
        self.lora_config = lora_config
        self.device = device
        self.dtype = dtype

        self.rank = lora_config.max_lora_rank
        self.alpha = lora_config.lora_alpha
        self.target_modules = lora_config.training_target_modules

        # Track trainable parameters and training state
        self.trainable_lora_ids: Set[int] = set()
        self.in_prog_lora_ids: Set[int] = set()
        self.trainable_lora_params: Dict[str, nn.Parameter] = {}
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None
        self.training_step: int = 0
        self.max_grad_norm: float = 1.0

        # TODO(girfan): Take the params from elsewhere.
        self.training_state = TrainingState(grad_accumulation_steps=1)

        # # Register with LoRAModelManager
        # if hasattr(self.lora_manager, '_adapter_manager'):
        #     self.lora_manager._adapter_manager._training_manager = self
        #     logger.info("[TrainingManager] Registered with LoRAModelManager")
        
        # Mark as initialized
        self._initialized = True

    @classmethod
    def get_instance(cls):
        """
        Get the singleton instance of TrainingManager.
        
        Returns:
            TrainingManager: The singleton instance
            
        Raises:
            RuntimeError: If the instance has not been initialized yet
        """
        if cls._instance is None:
            raise RuntimeError("TrainingManager has not been initialized yet. "
                             "Create an instance first with TrainingManager(...)")
        return cls._instance

    @classmethod
    def reset_instance(cls):
        """
        Reset the singleton instance. 
        
        This is primarily useful for testing or when you need to create
        a fresh instance. Use with caution as this will lose all state.
        """
        if cls._instance is not None:
            cls._instance = None
            cls._initialized = False

    @classmethod
    def is_initialized(cls) -> bool:
        """
        Check if the singleton instance has been initialized.
        
        Returns:
            bool: True if initialized, False otherwise
        """
        return cls._instance is not None and cls._initialized

    @property
    def model(self):
        """Get the actual model, unwrapping if needed."""
        if hasattr(self.model_runner, 'model'):
            model = self.model_runner.model
            # Unwrap if it's a UBatchWrapper or similar
            if hasattr(model, 'unwrap'):
                return model.unwrap()
            return model
        return self.model_runner

    @property
    def loss(self) -> Optional[torch.Tensor]:
        return self.training_state.loss

    def add_loss(self, loss: torch.Tensor):
        self.training_state.add_loss(loss)

    def step(self):
        self.training_state.step()

    def reset(self):
        self.training_state.reset()

    def should_run_optimizer_step(self) -> bool:
        return self.training_state.steps % self.training_state.grad_accumulation_steps == 0 and self.training_state.steps > 0

    # TODO(girfan): Cache this result?
    def get_qkv_indices_for_training(self) -> List[int]:
        """
        Determine which Q/K/V indices to train based on target_patterns.
        Returns (a_indices, b_indices) where indices map to: 0=Q, 1=K, 2=V.

        This makes training configurable - only specified projections are trained.
        """
        # Map projection names to indices
        projection_map = {"q_proj": 0, "k_proj": 1, "v_proj": 2}

        # Determine which indices to enable based on target_patterns
        enabled_indices = []
        for pattern in self.target_modules:
            for proj_name, idx in projection_map.items():
                if proj_name in pattern:
                    enabled_indices.append(idx)
                    break

        # Remove duplicates and sort
        enabled_indices = sorted(list(set(enabled_indices)))

        if not enabled_indices:
            raise ValueError(f"No specific projections found for target_patterns={self.target_patterns}")

        return enabled_indices

    def _freeze_base_model(self) -> Dict[str, int]:
        """Freeze all base model parameters (non-LoRA)."""
        frozen_count = 0
        total_count = 0

        for name, param in self.model.named_parameters():
            total_count += 1
            # Check if this is a LoRA parameter (including stacked tensors)
            is_lora_param = any([
                'lora_a' in name.lower(), 'lora_b' in name.lower(),
                'lora' in name.lower() and 'stacked' in name.lower(),
                'lora' in name.lower() and ('weight' in name.lower() or 'bias' in name.lower())
            ])

            if is_lora_param:
                raise ValueError(f"Base model parameter {name} is a LoRA parameter")

            param.requires_grad = False
            frozen_count += 1

        return {
            'total': total_count,
            'frozen': frozen_count,
        }

    def is_registered_by_id(self, lora_id: int) -> bool:
        return lora_id in self.trainable_lora_ids or lora_id in self.in_prog_lora_ids

    def is_registered_by_index(self, index: int) -> bool:
        mapping = self.lora_manager._adapter_manager.lora_index_to_id
        if index < 0 or index >= len(mapping):
            raise ValueError(f"LoRA index {index} out of range")
        lora_id = mapping[index]
        if lora_id is None:
            raise ValueError(f"LoRA index {index} not found in LoRA manager")
        return self.is_registered_by_id(lora_id)

    def make_lora_trainable(
        self,
        lora_request: LoRARequest,
        learning_rate: float = 1e-4,
        num_training_steps: Optional[int] = None,
        num_warmup_steps: int = 0,
        gradient_accumulation_steps: int = 1,
        weight_decay: float = 0.0,
        scheduler_type: str = "cosine",
    ):
        """Convert a loaded LoRA adapter to trainable parameters and setup optimizer."""
        lora_id = lora_request.lora_int_id

        # Check if the LoRA adapter is already trainable
        if lora_id in self.trainable_lora_ids:
            logger.warning(f"LoRA adapter {lora_id} is already trainable")
            return

        # Mark the LoRA adapter setup as in progress
        self.in_prog_lora_ids.add(lora_id)

        # Get QKV indices for training
        indices = self.get_qkv_indices_for_training()

        # Add the LoRA adapter to the LoRAManager
        self.lora_manager.add_adapter(lora_request, is_trainable=True, trainable_slices=indices)

        # Freeze the base model
        base_model_stats = self._freeze_base_model()

        # Get LoRA model and stacked tensor index
        lora_adapters = self.lora_manager.list_adapters()
        if lora_id not in lora_adapters:
            raise ValueError(f"LoRA adapter {lora_id} not found in LoRA manager")

        lora_model = self.lora_manager._adapter_manager.get_adapter(lora_id)
        if lora_model is None:
            raise ValueError(f"LoRA adapter {lora_id} could not be retrieved")

        # Clear existing trainable parameters and make stacked tensors trainable
        trainable_params = []
        trainable_count = 0
        self.trainable_lora_params.clear()

        # Make stacked tensors trainable
        for module_name, module in self.model.named_modules():
            # Check if this module has loaded LoRA weights
            if module_name not in lora_model.loras:
                continue  # Skip modules that don't have LoRA loaded

            # Determine slice indices to train per module
            if isinstance(module, MergedQKVParallelLinearWithLoRA):
                # Only use indices that exist in the stacked tensors
                a_indices = [i for i in indices if i < len(module.lora_a_stacked)]
                b_indices = [i for i in indices if i < len(module.lora_b_stacked)]
            else:
                # Single-slice (includes QKVParallelLinearWithLoRA which packs qkv into one slice)
                a_indices = [0] if len(module.lora_a_stacked) > 0 else []
                b_indices = [0] if len(module.lora_b_stacked) > 0 else []

            if not a_indices or not b_indices:
                raise ValueError(f"No valid indices found for module {module_name}")

            # Process lora_a_stacked
            for idx in a_indices:
                stacked_tensor = module.lora_a_stacked[idx]

                param_name = f"{module_name}.lora_a_stacked[{idx}]"
                self.trainable_lora_params[param_name] = stacked_tensor
                trainable_params.append(stacked_tensor)

            # Process lora_b_stacked
            for idx in b_indices:
                stacked_tensor = module.lora_b_stacked[idx]

                param_name = f"{module_name}.lora_b_stacked[{idx}]"
                self.trainable_lora_params[param_name] = stacked_tensor
                trainable_params.append(stacked_tensor)

        # Setup optimizer
        # self.optimizer = create_optimizer(self.model, lr=learning_rate, weight_decay=weight_decay)
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        # Setup scheduler
        if num_training_steps is not None and num_training_steps > 0:
            self.scheduler = self.setup_scheduler(
                optimizer=self.optimizer,
                num_training_steps=num_training_steps,
                num_warmup_steps=num_warmup_steps,
                scheduler_type=scheduler_type,
            )
        else:
            self.scheduler = None

        # Mark the LoRA adapter as trainable
        self.trainable_lora_ids.add(lora_id)
        self.in_prog_lora_ids.remove(lora_id)

        lora_stats = {
            "trainable_params": len(trainable_params),
        }

        all_stats = lora_stats | base_model_stats

        return all_stats

    def setup_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
        num_training_steps: int,
        num_warmup_steps: int = 0,
        scheduler_type: str = "cosine",
    ):
        """Setup learning rate scheduler."""
        if scheduler_type == "cosine":
            lr_lambda = partial(
                _get_cosine_schedule_with_warmup_lr_lambda,
                num_warmup_steps=100, # TODO(girfan): variable
                num_training_steps=300, # TODO(girfan): variable
                num_cycles=0.5,
            )
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=-1)
        elif scheduler_type == "linear":
            scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=num_training_steps)
        else:
            raise ValueError(f"Unsupported scheduler type: {scheduler_type}")

        return scheduler

    def optimizer_step(self):
        """Perform optimizer step with optional gradient clipping."""
        if self.max_grad_norm is not None and self.max_grad_norm > 0:
            # Clip gradients to prevent exploding gradients during training
            # This matches PEFT's TrainingArguments(max_grad_norm=1.0) default
            trainable_params = [p for p in self.optimizer.param_groups[0]['params'] if p.grad is not None]
            grad_norm = torch.nn.utils.clip_grad_norm_(
                trainable_params,
                max_norm=self.max_grad_norm,
            )
        else:
            # Calculate gradient norm without clipping
            total_norm = 0.0
            for param in self.optimizer.param_groups[0]['params']:
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2)
                    total_norm += param_norm.item()**2
            total_norm = total_norm**0.5

        self.optimizer.step()
        self.scheduler.step()

    def zero_grad(self) -> None:
        """Zero out all gradients."""
        # TODO(girfan): Zero grad of the model?
        # self.model.zero_grad()
        self.optimizer.zero_grad()

    def save_lora_checkpoint(
        self,
        lora_model: LoRAModel,
        output_dir: str,
        adapter_name: str = "adapter",
    ) -> str:
        """Save LoRA adapter weights to disk."""
        os.makedirs(output_dir, exist_ok=True)

        # Prepare tensors for saving
        tensors = {}
        for module_name, lora_weights in lora_model.loras.items():
            base_name = f"base_model.model.{module_name}"

            # Handle packed layers vs regular layers
            if isinstance(lora_weights.lora_a, list):
                # Packed layer - save each component separately
                for i, (lora_a_tensor, lora_b_tensor) in enumerate(zip(lora_weights.lora_a, lora_weights.lora_b)):
                    if lora_a_tensor is not None:
                        tensor_cpu = lora_a_tensor.detach().cpu() if isinstance(lora_a_tensor, torch.nn.Parameter) else lora_a_tensor.cpu()
                        tensors[f"{base_name}.lora_A.weight.{i}"] = tensor_cpu.contiguous()
                    if lora_b_tensor is not None:
                        tensor_cpu = lora_b_tensor.detach().cpu() if isinstance(lora_b_tensor, torch.nn.Parameter) else lora_b_tensor.cpu()
                        tensors[f"{base_name}.lora_B.weight.{i}"] = tensor_cpu.contiguous()
            else:
                # Regular layer - save as single tensor
                if lora_weights.lora_a is not None:
                    tensor_cpu = lora_weights.lora_a.detach().cpu() if isinstance(lora_weights.lora_a, torch.nn.Parameter) else lora_weights.lora_a.cpu()
                    tensors[f"{base_name}.lora_A.weight"] = tensor_cpu.contiguous()
                if lora_weights.lora_b is not None:
                    tensor_cpu = lora_weights.lora_b.detach().cpu() if isinstance(lora_weights.lora_b, torch.nn.Parameter) else lora_weights.lora_b.cpu()
                    tensors[f"{base_name}.lora_B.weight"] = tensor_cpu.contiguous()

        # Save adapter_config.json
        config = {
            "peft_type": "LORA",
            "r": lora_model.rank,
            "lora_alpha": self.alpha,
            "lora_dropout": 0.0,
            "target_modules": list(self.sub_modules),
            "bias": "none",
            "task_type": "CAUSAL_LM",
        }

        with open(os.path.join(output_dir, "adapter_config.json"), "w") as f:
            json.dump(config, f, indent=2)

        # Save weights
        save_file(tensors, os.path.join(output_dir, "adapter_model.safetensors"))

        logger.info(f"[TrainingManager] Saved LoRA adapter to {output_dir}")
        logger.info(f"[TrainingManager] Saved {len(tensors)} tensors")

        return output_dir