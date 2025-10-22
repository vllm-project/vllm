# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os
from typing import Dict, List, Optional, Set

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


logger = init_logger(__name__)


class TrainingManager:
    """
    Training manager for managing LoRA adapters during training.

    TODO(girfan): Add support for multiple LoRA adapters.
    """

    def __init__(
        self,
        model_runner: "GPUModelRunner",
        lora_manager: WorkerLoRAManager,
        lora_config: LoRAConfig,
        device: torch.device,
        dtype: torch.dtype,
    ):
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
        self.trainable_lora_params: Dict[str, nn.Parameter] = {}
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None
        self.training_step: int = 0
        self.gradient_accumulation_steps: int = 1
        self.gradient_accumulation_counter: int = 0

        # # Register with LoRAModelManager
        # if hasattr(self.lora_manager, '_adapter_manager'):
        #     self.lora_manager._adapter_manager._training_manager = self
        #     logger.info("[TrainingManager] Registered with LoRAModelManager")

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

    def _get_qkv_indices_for_training(self) -> List[int]:
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
        """Convert a loaded LoRA adapter to trainable Parameters and setup optimizer."""
        lora_id = lora_request.lora_int_id

        if lora_id in self.trainable_lora_ids:
            logger.warning(f"LoRA adapter {lora_id} is already trainable")
            return

        # Add the LoRA adapter to the LoRAManager
        self.lora_manager.add_adapter(lora_request)

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

        # Get QKV indices for training
        indices = self._get_qkv_indices_for_training()

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
                stacked_tensor.requires_grad_(True)

                param_name = f"{module_name}.lora_a_stacked[{idx}]"
                self.trainable_lora_params[param_name] = stacked_tensor
                trainable_params.append(stacked_tensor)

            # Process lora_b_stacked
            for idx in b_indices:
                stacked_tensor = module.lora_b_stacked[idx]
                stacked_tensor.requires_grad_(True)

                param_name = f"{module_name}.lora_b_stacked[{idx}]"
                self.trainable_lora_params[param_name] = stacked_tensor
                trainable_params.append(stacked_tensor)

        # Setup optimizer
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
            if num_warmup_steps > 0:

                def lr_lambda(current_step: int):
                    # PyTorch LambdaLR correctly starts from step 0
                    if current_step < num_warmup_steps:
                        lr_factor = float(current_step) / float(max(1, num_warmup_steps))
                        return lr_factor
                    progress = float(current_step - num_warmup_steps) / float(
                        max(1, num_training_steps - num_warmup_steps))
                    lr_factor = max(0.0, 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.141592653589793))))
                    return lr_factor

                scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
            else:
                scheduler = CosineAnnealingLR(optimizer, T_max=num_training_steps)
        elif scheduler_type == "linear":
            scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=num_training_steps)
        else:
            raise ValueError(f"Unsupported scheduler type: {scheduler_type}")

        return scheduler

    def optimizer_step(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        max_grad_norm: Optional[float] = 1.0,
    ) -> Dict[str, float]:
        """Perform optimizer step with optional gradient clipping."""
        stats = {}

        if max_grad_norm is not None and max_grad_norm > 0:
            # Clip gradients to prevent exploding gradients during training
            # This matches PEFT's TrainingArguments(max_grad_norm=1.0) default
            trainable_params = [p for p in optimizer.param_groups[0]['params'] if p.grad is not None]
            total_norm_before_clip = torch.nn.utils.clip_grad_norm_(
                trainable_params,
                max_norm=max_grad_norm,
                norm_type=2.0  # L2 norm (default)
            )
            stats['grad_norm'] = float(total_norm_before_clip)  # Store pre-clip norm for logging
            stats['grad_norm_before_clip'] = float(total_norm_before_clip)
            stats['grad_norm_after_clip'] = min(float(total_norm_before_clip), max_grad_norm)
        else:
            # Calculate gradient norm without clipping
            total_norm = 0.0
            for param in optimizer.param_groups[0]['params']:
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2)
                    total_norm += param_norm.item()**2
            total_norm = total_norm**0.5
            stats['grad_norm'] = total_norm

        # Optimizer step
        optimizer.step()
        stats['learning_rate'] = optimizer.param_groups[0]['lr']

        # Scheduler step if provided
        if scheduler is not None:
            scheduler.step()
            # Log LR after scheduler step for first 10 steps
            if self.training_step <= 10:
                new_lr = optimizer.param_groups[0]['lr']
                stats['learning_rate'] = new_lr

        return stats

    def zero_grad(self, optimizer: torch.optim.Optimizer) -> None:
        """Zero out all gradients."""
        optimizer.zero_grad()

    def step_with_accumulation(
        self,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: Optional[float] = 1.0,
    ) -> Optional[Dict[str, float]]:
        """Handle gradient accumulation and optimizer step."""
        if self.optimizer is None:
            logger.warning("[TrainingManager] No optimizer configured, skipping step")
            return None

        # Check if gradients are computed and log norms (every step)
        grad_count = 0
        total_grad_l2 = 0.0
        sample_b_grads: list[tuple[str, float]] = []
        for name, param in self.trainable_lora_params.items():
            if param.grad is not None:
                grad_count += 1
                try:
                    gnorm = float(param.grad.data.norm().detach().cpu())
                    total_grad_l2 += gnorm * gnorm
                    if 'lora_b_stacked' in name and len(sample_b_grads) < 3:
                        sample_b_grads.append((name, gnorm))
                except Exception:
                    print("Error calculating gradient norm")
        if grad_count == 0:
            logger.warning(f"[TrainingManager] No gradients found in {len(self.trainable_lora_params)} LoRA parameters")
            return None
        try:
            msg = f"[LORA/GRAD] step={self.training_step} grads_present={grad_count}/{len(self.trainable_lora_params)} total_grad_norm={(total_grad_l2 ** 0.5):.6e}"
            if sample_b_grads:
                msg += " sample_B_grad_norms=" + ",".join(f"{n}:{g:.3e}" for n,g in sample_b_grads)
            logger.info(msg)
        except Exception:
            print("Error logging gradient presence after step")

        # ✅ FIX: Check if high-level training loop is controlling gradient accumulation
        external_control = hasattr(self, '_should_step_optimizer')

        if external_control:
            # External control from high-level loop - respect the flag
            should_step = self._should_step_optimizer
            if should_step:
                self.training_step += 1
        else:
            self.gradient_accumulation_counter += 1
            self.training_step += 1
            should_step = self.gradient_accumulation_counter >= gradient_accumulation_steps

        # Check if we should perform an optimizer step
        if should_step:
            # Snapshot params before step for delta logging
            params_before = {}
            try:
                for param_name, param in self.trainable_lora_params.items():
                    params_before[param_name] = param.detach().clone()
            except Exception:
                params_before = {}

            # Perform optimizer step (directly updates the stacked tensor parameters)
            stats = self.optimizer_step(
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                max_grad_norm=max_grad_norm,
            )

            # Zero gradients and reset accumulation counter
            self.zero_grad(self.optimizer)
            if not external_control:
                # Only reset internal counter if not using external control
                self.gradient_accumulation_counter = 0
            stats['training_step'] = self.training_step

            # Clear external control flag if set
            if external_control:
                delattr(self, '_should_step_optimizer')

            return stats
        else:
            # Clear external control flag if set (even when not stepping)
            if external_control:
                delattr(self, '_should_step_optimizer')

            return None

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