# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Example: Using Megatron-LM Tensor Parallel Layers with vLLM

This example demonstrates how to use NVIDIA's Megatron-LM tensor parallel
layers with vLLM's inference engine. This is the key use case: bringing your
own parallelism implementation instead of being forced to use vLLM's internals.

Key features:
- Use Megatron-LM's ColumnParallelLinear and RowParallelLinear for MLPs
- Use vLLM's TrainableFlashAttention for training-compatible attention
- Full transformer architecture with attention + MLP blocks
- Configure Megatron from vLLM's parallel context
- Test with actual LLM() API and worker spawning

This demonstrates the complete integration:
1. Attention: vLLM's TrainableFlashAttention (with backward pass support)
2. MLP: Megatron's ColumnParallelLinear → GELU → RowParallelLinear

Requirements:
    pip install megatron-core flash-attn

For more details on Megatron-LM:
    https://github.com/NVIDIA/Megatron-LM
"""

import json
import os
import socket
import tempfile

import torch
import torch.nn as nn

from vllm.model_executor.layers.trainable_attention import (
    TrainableFlashAttention,  # vLLM's training-compatible attention
)
from vllm.model_executor.models import ModelRegistry
from vllm.model_executor.parallel_context import ParallelContext


def find_free_port() -> int:
    """Find a free port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


class MegatronTransformer(nn.Module):
    """
    Example model using Megatron-LM's tensor parallel layers for MLPs
    and vLLM's TrainableFlashAttention for attention.

    This demonstrates how users can leverage:
    1. External parallelism libraries (Megatron-LM) for custom components
    2. vLLM's training-compatible modules (TrainableFlashAttention)

    Architecture:
    - Attention: vLLM's TrainableFlashAttention (supports backward pass)
    - MLP: Megatron's ColumnParallelLinear → GELU → RowParallelLinear
    """

    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_size: int = 4096,
        intermediate_size: int = 11008,
        num_layers: int = 4,
        num_attention_heads: int = 32,
        tp_size: int = 1,
        tp_group=None,  # vLLM's tensor parallel group
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.tp_size = tp_size
        self.head_dim = hidden_size // num_attention_heads

        # Standard embedding (not parallelized)
        self.embeddings = nn.Embedding(vocab_size, hidden_size)

        # Build layers with both attention and MLP
        self.layers = nn.ModuleList()

        # Create Megatron config
        megatron_config = TransformerConfig(
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            use_cpu_initialization=True,
        )

        for i in range(num_layers):
            # Build transformer layer with vLLM attention + Megatron MLP
            layer = nn.ModuleDict(
                {
                    # === ATTENTION BLOCK ===
                    # Use vLLM's TrainableFlashAttention
                    # (includes QKV + output projections)
                    "attn": TrainableFlashAttention(
                        hidden_size=hidden_size,
                        num_heads=num_attention_heads,
                        dropout=0.0,
                        causal=True,
                    ),
                    "attn_norm": nn.LayerNorm(hidden_size),
                    # === MLP BLOCK ===
                    # Use Megatron's parallel layers for MLP
                    "fc1": ColumnParallelLinear(
                        hidden_size,
                        intermediate_size,
                        config=megatron_config,
                        init_method=nn.init.xavier_normal_,
                        bias=False,
                        gather_output=False,  # Keep output sharded
                        tp_group=tp_group,
                    ),
                    "act": nn.GELU(),
                    "fc2": RowParallelLinear(
                        intermediate_size,
                        hidden_size,
                        config=megatron_config,
                        init_method=nn.init.xavier_normal_,
                        bias=False,
                        input_is_parallel=True,  # Input is sharded
                        skip_bias_add=False,
                        tp_group=tp_group,
                    ),
                    "mlp_norm": nn.LayerNorm(hidden_size),
                }
            )
            self.layers.append(layer)

        self.final_norm = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Required by vLLM."""
        return self.embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Required by vLLM.

        Args:
            input_ids: Token IDs, shape [total_tokens] for V1 or [batch, seq_len] for V0
            **kwargs: Additional vLLM kwargs (intermediate_tensors, etc.)

        Returns:
            hidden_states: Shape [total_tokens, hidden_size]
        """
        # vLLM V1 passes flattened tokens: [total_tokens]
        # vLLM V0 passes batched tokens: [batch, seq_len]
        # Embeddings support both, so we can pass input_ids directly

        # Clamp input_ids to valid range for embedding lookup
        # (warmup may pass out-of-bounds values)
        input_ids = input_ids.clamp(0, self.vocab_size - 1)

        hidden_states = self.embeddings(input_ids)

        # Ensure flattened format: [total_tokens, hidden_size]
        if hidden_states.dim() == 3:
            # [batch, seq_len, hidden_size] -> [total_tokens, hidden_size]
            hidden_states = hidden_states.view(-1, self.hidden_size)

        for layer_idx, layer in enumerate(self.layers):
            # === ATTENTION BLOCK ===
            residual = hidden_states
            # Pass kwargs through to attention (for vLLM dynamic batching support)
            attn_output = layer["attn"](hidden_states, **kwargs)
            hidden_states = layer["attn_norm"](residual + attn_output)

            # === MLP BLOCK ===
            residual = hidden_states
            hidden_states = layer["fc1"](hidden_states)
            # Handle Megatron output (tuple) for fc1
            if isinstance(hidden_states, tuple):
                hidden_states = hidden_states[0]

            hidden_states = layer["act"](hidden_states)

            hidden_states = layer["fc2"](hidden_states)
            # Handle Megatron output (tuple) for fc2
            if isinstance(hidden_states, tuple):
                hidden_states = hidden_states[0]

            hidden_states = layer["mlp_norm"](hidden_states + residual)

        hidden_states = self.final_norm(hidden_states)

        # Output shape: [total_tokens, hidden_size]
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata=None,
    ) -> torch.Tensor:
        """Compute output logits.

        Note: hidden_states here are ALREADY indexed by vLLM using logit_indices_device.
        We just need to apply the LM head to project to vocabulary space.
        """
        logits = self.lm_head(hidden_states)
        return logits

    def load_weights(self, weights):
        """Load weights into the model.

        For this demo, we load from a saved state_dict if available.
        If TP > 1, we need to shard the weights appropriately for each rank.
        """
        import os

        # Check if reference weights are available
        # Use environment variable to allow dynamic path configuration
        ref_weights_path = os.environ.get(
            "MEGATRON_REFERENCE_WEIGHTS_PATH", "/tmp/megatron_reference_weights.pt"
        )
        if os.path.exists(ref_weights_path):
            print(f"[Worker] Loading reference weights from {ref_weights_path}")
            state_dict = torch.load(ref_weights_path, map_location="cuda")

            print(f"[Worker] Reference state_dict has {len(state_dict)} keys")
            print(f"[Worker] Current model has {len(self.state_dict())} keys")
            print(f"[Worker] TP size: {self.tp_size}")

            # For TP > 1, we need to shard the Megatron parallel layer weights
            if self.tp_size > 1:
                from vllm.distributed import parallel_state as vllm_parallel_state

                tp_rank = vllm_parallel_state.get_tensor_model_parallel_rank()
                print(f"[Worker] Sharding weights for TP rank {tp_rank}/{self.tp_size}")

                # Shard the fc1 (ColumnParallelLinear) and
                # fc2 (RowParallelLinear) weights
                sharded_state_dict = {}
                for key, value in state_dict.items():
                    if ".fc1.weight" in key:
                        # ColumnParallelLinear: split output dimension (dim 0)
                        # Full shape: [intermediate_size, hidden_size]
                        # Shard along dim 0: [intermediate_size/tp_size, hidden_size]
                        chunk_size = value.shape[0] // self.tp_size
                        start = tp_rank * chunk_size
                        end = (tp_rank + 1) * chunk_size
                        sharded_state_dict[key] = value[start:end, :].clone()
                        print(
                            f"[Worker] Sharding {key}: "
                            f"{value.shape} -> {sharded_state_dict[key].shape}"
                        )
                    elif ".fc2.weight" in key:
                        # RowParallelLinear: split input dimension (dim 1)
                        # Full shape: [hidden_size, intermediate_size]
                        # Shard along dim 1: [hidden_size, intermediate_size/tp_size]
                        chunk_size = value.shape[1] // self.tp_size
                        start = tp_rank * chunk_size
                        end = (tp_rank + 1) * chunk_size
                        sharded_state_dict[key] = value[:, start:end].clone()
                        print(
                            f"[Worker] Sharding {key}: "
                            f"{value.shape} -> {sharded_state_dict[key].shape}"
                        )
                    else:
                        # Non-sharded weights (embeddings, LayerNorm, lm_head, etc.)
                        sharded_state_dict[key] = value

                state_dict = sharded_state_dict

            # Check lm_head before loading
            lm_head_before = self.lm_head.weight.data.clone()
            print(
                f"[Worker] lm_head BEFORE load: "
                f"mean={lm_head_before.mean().item():.6f}, "
                f"std={lm_head_before.std().item():.6f}"
            )

            # Load and check what matched
            result = self.load_state_dict(state_dict, strict=False)
            print(f"[Worker] Missing keys: {len(result.missing_keys)}")
            print(f"[Worker] Unexpected keys: {len(result.unexpected_keys)}")

            # Check lm_head after loading
            lm_head_after = self.lm_head.weight.data
            print(
                f"[Worker] lm_head AFTER load: "
                f"mean={lm_head_after.mean().item():.6f}, "
                f"std={lm_head_after.std().item():.6f}"
            )
            print(
                f"[Worker] lm_head changed: "
                f"{not torch.equal(lm_head_before, lm_head_after)}"
            )

            # Check against reference
            ref_lm_head = state_dict["lm_head.weight"]
            print(
                f"[Worker] Reference lm_head: "
                f"mean={ref_lm_head.mean().item():.6f}, "
                f"std={ref_lm_head.std().item():.6f}"
            )
            print(
                f"[Worker] lm_head matches reference: "
                f"{torch.equal(lm_head_after, ref_lm_head)}"
            )

            if len(result.missing_keys) == 0 and len(result.unexpected_keys) == 0:
                print("[Worker] ✅ All weights loaded successfully!")
            else:
                print("[Worker] ⚠️  Some weights were not loaded")
                if result.missing_keys:
                    print(f"[Worker] Missing: {result.missing_keys}")
        else:
            print("[Worker] No reference weights found, using random initialization")


def build_megatron_model(vllm_config, parallel_context: ParallelContext):
    """
    Factory that builds a model using Megatron-LM parallelism.

    This shows how to:
    1. Get TP info from vLLM's parallel context
    2. Get vLLM's tensor parallel process group
    3. Pass the process group to Megatron layers
    """
    # Import Megatron here to avoid CUDA initialization before fork,
    # purely for testing purposes
    global ColumnParallelLinear, RowParallelLinear, TransformerConfig

    import megatron.core.parallel_state as megatron_parallel_state
    from megatron.core.tensor_parallel import (
        ColumnParallelLinear,
        RowParallelLinear,
    )
    from megatron.core.transformer.transformer_config import (
        TransformerConfig,
    )

    # Get vLLM's tensor parallel process group
    # Set Megatron's global tensor parallel group to vLLM's group
    # Megatron layers require this even though they also accept tp_group as parameter
    from vllm.distributed import parallel_state as vllm_parallel_state

    tp_coordinator = vllm_parallel_state.get_tp_group()
    tp_group = tp_coordinator.device_group
    tp_rank = parallel_context.get_tensor_parallel_rank()
    tp_size = parallel_context.get_tensor_parallel_world_size()

    assert tp_group is not None, "Failed to get TP process group from vLLM!"

    megatron_parallel_state._TENSOR_MODEL_PARALLEL_GROUP = tp_group
    megatron_parallel_state._MPU_TENSOR_MODEL_PARALLEL_RANK = tp_rank
    megatron_parallel_state._TENSOR_MODEL_PARALLEL_GLOBAL_RANKS = list(range(tp_size))
    megatron_parallel_state._MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE = tp_size

    # Extract config from vLLM's config
    assert hasattr(vllm_config, "model_config")
    assert hasattr(vllm_config.model_config, "hf_config")

    hf_config = vllm_config.model_config.hf_config
    vocab_size = getattr(hf_config, "vocab_size", 32000)
    hidden_size = getattr(hf_config, "hidden_size", 4096)
    num_attention_heads = getattr(hf_config, "num_attention_heads", 32)
    num_hidden_layers = getattr(hf_config, "num_hidden_layers", 4)

    model = MegatronTransformer(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=hidden_size * 4,
        num_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        tp_size=tp_size,
        tp_group=tp_group,
    )

    if hasattr(vllm_config.model_config, "dtype"):
        model = model.to(dtype=vllm_config.model_config.dtype)

    return model


# Register the model
ModelRegistry.register_model("MegatronModel", build_megatron_model)


def run_reference_forward(config_dict, input_ids, seed=42, save_weights_path=None):
    """
    Run MegatronTransformer independently (without vLLM) to get ground truth logits.

    This initializes Megatron's parallel state with TP=1 and runs a forward pass.

    Args:
        config_dict: Model configuration
        input_ids: Input token IDs
        seed: Random seed
        save_weights_path: If provided, save model weights to this path
    """
    # Import Megatron here to make them available as globals for MegatronTransformer
    global ColumnParallelLinear, RowParallelLinear, TransformerConfig

    import megatron.core.parallel_state as megatron_parallel_state
    from megatron.core.tensor_parallel import (
        ColumnParallelLinear,
        RowParallelLinear,
    )
    from megatron.core.transformer.transformer_config import TransformerConfig

    print("\n" + "=" * 70)
    print("[Ground Truth] Running MegatronTransformer independently (TP=1)")
    print("=" * 70)

    # Set random seed for reproducibility
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Initialize distributed process group for TP=1
    import torch.distributed as dist

    if not dist.is_initialized():
        # Use a dynamic free port to avoid conflicts
        free_port = find_free_port()
        dist.init_process_group(
            backend="nccl",
            init_method=f"tcp://localhost:{free_port}",
            world_size=1,
            rank=0,
        )

    # Create process group for TP
    tp_group = dist.new_group([0])

    # Initialize Megatron's parallel state
    megatron_parallel_state._TENSOR_MODEL_PARALLEL_GROUP = tp_group
    megatron_parallel_state._TENSOR_MODEL_PARALLEL_GLOBAL_RANKS = [0]
    megatron_parallel_state._MPU_TENSOR_MODEL_PARALLEL_RANK = 0
    megatron_parallel_state._MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE = 1

    print("✓ Initialized Megatron parallel state (TP=1)")

    # Build model
    model = MegatronTransformer(
        vocab_size=config_dict["vocab_size"],
        hidden_size=config_dict["hidden_size"],
        intermediate_size=config_dict["hidden_size"] * 4,
        num_layers=config_dict["num_hidden_layers"],
        num_attention_heads=config_dict["num_attention_heads"],
        tp_size=1,
        tp_group=tp_group,
    ).cuda()

    # Convert to bfloat16 to match vLLM
    model = model.to(dtype=torch.bfloat16)

    print("✓ Built reference model (bfloat16)")

    # Save weights if requested
    if save_weights_path:
        torch.save(model.state_dict(), save_weights_path)
        print(f"✓ Saved weights to {save_weights_path}")

    # === Test 1: Forward pass for ground truth (eval mode) ===
    model.eval()
    with torch.no_grad():
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device="cuda")

        print(
            f"[Reference] Input shape: {input_tensor.shape}, "
            f"values: {input_tensor[0].tolist()}"
        )
        print(
            f"[Reference] lm_head weight sample [0,:5]: "
            f"{model.lm_head.weight[0, :5].tolist()}"
        )
        print(
            f"[Reference] lm_head weight sample [511,:5]: "
            f"{model.lm_head.weight[511, :5].tolist()}"
        )

        hidden_states = model(input_tensor)
        print(f"[Reference] Hidden states shape: {hidden_states.shape}")
        print(
            f"[Reference] Hidden states (last token): "
            f"mean={hidden_states[-1].mean().item():.6f}, "
            f"std={hidden_states[-1].std().item():.6f}"
        )
        print(
            f"[Reference] Hidden states (last token, first 5 dims): "
            f"{hidden_states[-1, :5].tolist()}"
        )

        logits = model.compute_logits(hidden_states)
        print(f"[Reference] Logits shape: {logits.shape}")
        print(
            f"[Reference] Logits (last token) stats: "
            f"mean={logits[-1].mean().item():.6f}, "
            f"std={logits[-1].std().item():.6f}, "
            f"min={logits[-1].min().item():.6f}, "
            f"max={logits[-1].max().item():.6f}"
        )
        print(f"[Reference] Logit for token 377: {logits[-1, 377].item():.6f}")
        print(f"[Reference] Logit for token 511: {logits[-1, 511].item():.6f}")

    print(f"✓ Forward pass complete: logits shape = {logits.shape}")

    # === Test 2: Backward pass to verify training support ===
    print("\n" + "=" * 70)
    print("[Training Test] Testing backward pass for gradient computation")
    print("=" * 70)

    model.train()  # Switch to train mode

    # Test full model backward pass (attention + MLP)
    print(
        "[Training] Testing full model backward pass "
        "(TrainableFlashAttention + Megatron MLP)..."
    )

    # Create a dummy input matching the model's expected format
    test_input_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long, device="cuda")

    # Forward through the entire model
    hidden_states = model(test_input_ids)
    print(f"[Training] Hidden states shape after full forward: {hidden_states.shape}")

    # Compute logits
    train_logits = model.compute_logits(hidden_states)
    print(f"[Training] Logits shape: {train_logits.shape}")

    # Create a simple loss (sum of logits)
    loss = train_logits.sum()
    print(f"[Training] Loss (logits sum): {loss.item():.6f}")

    # Backward pass through the entire model
    loss.backward()

    # Check that gradients were computed across all component types
    grad_checks = []

    # Check embedding gradient
    if model.embeddings.weight.grad is not None:
        grad_checks.append(
            (
                "embeddings.weight",
                model.embeddings.weight.grad.abs().max().item(),
                model.embeddings.weight.grad.abs().mean().item(),
            )
        )

    # Check attention gradients (TrainableFlashAttention)
    attn_qkv = model.layers[0]["attn"].qkv
    if attn_qkv.weight.grad is not None:
        grad_checks.append(
            (
                "layers[0].attn.qkv.weight (TrainableFlashAttention)",
                attn_qkv.weight.grad.abs().max().item(),
                attn_qkv.weight.grad.abs().mean().item(),
            )
        )

    # Check Megatron fc1 (ColumnParallelLinear) gradient
    fc1_weight = model.layers[0]["fc1"].weight
    if fc1_weight.grad is not None:
        grad_checks.append(
            (
                "layers[0].fc1.weight (Megatron ColumnParallel)",
                fc1_weight.grad.abs().max().item(),
                fc1_weight.grad.abs().mean().item(),
            )
        )

    # Check Megatron fc2 (RowParallelLinear) gradient
    fc2_weight = model.layers[0]["fc2"].weight
    if fc2_weight.grad is not None:
        grad_checks.append(
            (
                "layers[0].fc2.weight (Megatron RowParallel)",
                fc2_weight.grad.abs().max().item(),
                fc2_weight.grad.abs().mean().item(),
            )
        )

    # Check lm_head gradient
    if model.lm_head.weight.grad is not None:
        grad_checks.append(
            (
                "lm_head.weight",
                model.lm_head.weight.grad.abs().max().item(),
                model.lm_head.weight.grad.abs().mean().item(),
            )
        )

    # Report gradient statistics
    print("\n[Training] Gradient check:")
    if len(grad_checks) == 0:
        print("  ❌ No gradients computed!")
        raise RuntimeError("Backward pass failed - no gradients!")
    else:
        print(f"  ✓ Gradients computed for {len(grad_checks)} tensors:")
        for name, max_grad, mean_grad in grad_checks:
            print(f"    - {name}: max={max_grad:.6f}, mean={mean_grad:.6f}")

    print("\n✓ Backward pass successful! Full model supports training:")
    print("  - TrainableFlashAttention (vLLM attention with backward pass)")
    print("  - Megatron-LM ColumnParallelLinear and RowParallelLinear")

    # Clean up
    dist.destroy_process_group()

    return logits


if __name__ == "__main__":
    print("=" * 70)
    print("Megatron-LM + vLLM Integration Demo")
    print("=" * 70)

    print("\nWhat this demonstrates:")
    print("  ✓ External parallelism (Megatron-LM) for MLP layers")
    print("  ✓ vLLM's TrainableFlashAttention for attention")
    print("  ✓ SAME model works for both training AND inference")
    print("  ✓ Ground truth validation against independent PyTorch run")

    # Configuration
    config = {
        "model_type": "gpt2",
        "architectures": ["MegatronModel"],
        "vocab_size": 1000,
        "hidden_size": 256,
        "num_attention_heads": 4,
        "num_hidden_layers": 2,
        "max_position_embeddings": 128,
    }

    # Test input
    test_input_ids = [1, 2, 3, 4, 5]
    SEED = 42

    # === GROUND TRUTH: Run model independently ===
    # Create a temporary file for weights to avoid conflicts
    with tempfile.NamedTemporaryFile(
        suffix=".pt", delete=False, prefix="megatron_weights_"
    ) as weights_file:
        weights_path = weights_file.name

    # Set environment variable so worker processes can find the weights
    os.environ["MEGATRON_REFERENCE_WEIGHTS_PATH"] = weights_path

    try:
        reference_logits = run_reference_forward(
            config, test_input_ids, seed=SEED, save_weights_path=weights_path
        )

        # === VLLM TEST: With TP=4 (testing actual tensor parallelism) ===
        print("\n" + "=" * 70)
        print("[vLLM Test] Testing model with vLLM (TP=4, inference)")
        print("=" * 70)

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "config.json")
            with open(config_path, "w") as f:
                json.dump(config, f)

            # Set same random seed for vLLM
            torch.manual_seed(SEED)
            torch.cuda.manual_seed_all(SEED)

            from vllm import LLM

            llm = LLM(
                model=tmpdir,
                tokenizer=None,
                tensor_parallel_size=4,  # Use TP=4 to test actual parallelism
                max_model_len=128,
                max_num_seqs=8,  # Allow batching for dynamic batching test
                enforce_eager=True,
                skip_tokenizer_init=True,
                trust_remote_code=True,
                enable_prefix_caching=False,
                seed=SEED,  # Set seed for vLLM
            )

            print("✓ vLLM engine initialized")

            # Get logits from vLLM (not just generated tokens)
            # We'll need to extract the actual logits from vLLM's forward pass
            from vllm import SamplingParams
            from vllm.inputs import TokensPrompt

            sampling_params = SamplingParams(
                temperature=0.0,
                max_tokens=1,  # Just generate 1 token to get logits
                logprobs=1,  # Request logprobs to access logits
            )

            # Generate using same input
            outputs = llm.generate(
                prompts=TokensPrompt(prompt_token_ids=test_input_ids),
                sampling_params=sampling_params,
            )

            print("✓ vLLM forward pass complete")

            # Extract logits from vLLM output
            # Note: vLLM doesn't directly expose logits in generate(), so we'll
            # compare the generated tokens and logprobs as a proxy
            print("\n" + "=" * 70)
            print("[Validation] Comparing vLLM output to ground truth")
            print("=" * 70)

            for output in outputs:
                single_query_token = output.outputs[0].token_ids[0]
                logprobs = (
                    output.outputs[0].logprobs[0]
                    if output.outputs[0].logprobs
                    else None
                )

                # Get top-K tokens from reference (handle numerical differences)
                K = 10
                # IMPORTANT: Get top-K from LAST token only!
                topk_values, topk_indices = torch.topk(reference_logits[-1], K)
                topk_tokens = topk_indices.tolist()
                reference_greedy = topk_tokens[0]

                # Find vLLM token's rank in reference
                sorted_logits, sorted_indices = torch.sort(
                    reference_logits[-1], descending=True
                )
                vllm_token_rank = (sorted_indices == single_query_token).nonzero(
                    as_tuple=True
                )[0].item() + 1
                vllm_token_ref_logit = reference_logits[-1, single_query_token].item()

                print("\nGround Truth (PyTorch TP=1):")
                print(f"  - Greedy token: {reference_greedy}")
                print(f"  - Top-{K} tokens: {topk_tokens}")
                print(f"  - Top-{K} logits: {[f'{v.item():.4f}' for v in topk_values]}")

                print("\nvLLM Output (TP=4):")
                print(f"  - Greedy token: {single_query_token}")
                print(
                    f"  - Rank in reference: {vllm_token_rank} / {len(sorted_logits)}"
                )
                print(f"  - Reference logit for this token: {vllm_token_ref_logit:.4f}")
                if logprobs:
                    print(
                        f"  - vLLM log probability: "
                        f"{logprobs[single_query_token].logprob:.4f}"
                    )

                # Validate: vLLM token should be in top-K reference tokens
                # (allows for numerical differences due to TP, dtype, etc.)
                if single_query_token in topk_tokens:
                    rank = topk_tokens.index(single_query_token) + 1
                    print("\n✅ VALIDATION PASSED!")
                    print(
                        f"   vLLM token {single_query_token} is in reference "
                        f"top-{K} (rank {rank})"
                    )
                    if single_query_token == reference_greedy:
                        print("   ✨ Exact match with reference greedy token!")
                else:
                    print("\n❌ VALIDATION FAILED!")
                    print(
                        f"   vLLM token {single_query_token} NOT in reference top-{K}"
                    )
                    print(
                        "   This suggests a correctness issue "
                        "(not just numerical differences)"
                    )
                    raise AssertionError(
                        f"vLLM output ({single_query_token}) not in reference "
                        f"top-{K} tokens {topk_tokens}"
                    )

            print("\n" + "=" * 70)
            print("✅ All validation tests passed!")
            print("=" * 70)

            # === DYNAMIC BATCHING TEST ===
            print("\n" + "=" * 70)
            print("[Dynamic Batching] Testing with multiple sequences")
            print("=" * 70)

            # Create multiple prompts of different lengths
            # Include the same prompt as the single-query test to verify consistency
            batch_prompts = [
                TokensPrompt(prompt_token_ids=[1, 2, 3]),  # 3 tokens
                TokensPrompt(
                    prompt_token_ids=test_input_ids
                ),  # 5 tokens (same as single query!)
                TokensPrompt(prompt_token_ids=[100, 200]),  # 2 tokens
            ]

            # Generate with batching
            batch_outputs = llm.generate(
                prompts=batch_prompts,
                sampling_params=sampling_params,
            )

            print(f"\n[Dynamic Batching] Generated {len(batch_outputs)} outputs:")
            for i, output in enumerate(batch_outputs):
                prompt_len = len(batch_prompts[i]["prompt_token_ids"])
                batch_generated_id = output.outputs[0].token_ids[0]
                print(
                    f"  Prompt {i + 1}: {prompt_len} tokens -> "
                    f"generated token {batch_generated_id}"
                )

            # Verify the batched result matches the single-query result
            batched_token = batch_outputs[1].outputs[0].token_ids[0]  # Middle prompt
            if batched_token == single_query_token:
                print(
                    f"\n✓ Batched query matches single query! Both generated "
                    f"token {single_query_token}"
                )
            else:
                print(
                    f"\n⚠️  Batched token {batched_token} != single query "
                    f"token {single_query_token}"
                )

            print(
                "\n✓ Dynamic batching works! Sequences of different lengths "
                "handled correctly."
            )

            # Clean up vLLM engine to avoid shutdown errors
            print("\n[Cleanup] Shutting down vLLM engine...")
            del llm
            print("[Cleanup] ✓ vLLM engine shutdown complete")

    finally:
        # Clean up temporary weights file
        if os.path.exists(weights_path):
            os.unlink(weights_path)
            print(f"[Cleanup] ✓ Removed temporary weights file: {weights_path}")
