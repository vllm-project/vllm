# Introduction
Recent papers QuaRot and SpinQuant papers propose a quantization method for large language models (LLMs) that use rotations to simplify quantization. It quantizes all components, including weights, activations, and KV cache, into 4 bits by removing outliers without changing the output. It accomplishes this by inserting four fundamental rotations (called R1, R2, R3 and R4 in SpinQuant), into Llama models.

To support these features, the ability to add online **hadamard** rotations - explicitly rotating the input before it is passed into the quantized layer - is necessary (see figure below).

![Alt text](../assets/features/quantization/{506D28EB-F734-48C3-A3C8-0A7CF5F31141}.PNG)
# Features
- Specify specific layers, by regular expression, and corresponding online rotation to be applied to their inputs
- Specify a custom Python module and method for the rotation function

# Requirements
The custom Python module method for the rotation function must satisfy the following: 
- Must accept `torch.Tensor` in the form $b \times n$ where $b$ denotes batch size and $n$ denotes embedding dimension  
- Must be a CUDA-graph-compatible Python function (cannot involve dynamic memory allocation in the nn.Module forward pass etc.)
# Current Limitations
- Requires rotations to be defined internally in vLLM `quark/schemes` directory - user must define every rotation as a method inside a Python file in this directory

# Proof of Concept (PoC) & Performance Impact
To demonstrate the effectiveness of implementing online rotations into vLLM, I have implemented QuaRot into vLLM and demonstrated significant end-end performance benefits.
### Throughput / Latency:

Below is a specific example using `benchmarks/benchmark_throughput.py`:
```bash
python3 benchmark_throughput.py --model {name} --num-prompts 50 --input-len 64 --output-len 128
```

![Alt text](../assets/features/quantization/image.png)
QuaRot demonstrates very strong performance improvements when integrated with vLLM, close to int8.

### Accuracy:
To ensure the implementation accuracy matched the claims from the research paper ([[2404.00456] QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs](https://arxiv.org/abs/2404.00456)) comprehensive perplexity benchmark as well as empirical validation were used.

To measure perplexity, I used `benchmarks/P3L.py`. Note: this benchmark has not yet been merged into the main branch of vLLM. Please refer to it here: [vllm/benchmarks/P3L.py at main · ROCm/vllm](https://github.com/ROCm/vllm/blob/main/benchmarks/P3L.py)
The perplexity score matches the claims of the paper: achieve near int8 efficiency with near accuracy of the unquantized model.

| unquantized | quarot FHT 512 BMM Groups | int8  |
| ----------- | ------------------------- | ----- |
| 4.2         | 7.1                       | 210.5 |
## Reproducibility
### Model
Utilize Quark to quantize the model:
[Viva Engage - Conversation](https://engage.cloud.microsoft/main/org/amd.com/threads/eyJfdHlwZSI6IlRocmVhZCIsImlkIjoiMzEwMzE5NjkwNzQxMzUwNCJ9?search=rotation&groupScope=eyJfdHlwZSI6Ikdyb3VwIiwiaWQiOiIyMDYzMjQ0NjU2NjQifQ)
Use this command to export to huggingface
```bash
python quantize_quark.py --output_dir {name} --model_export hf_format --model_dir meta-llama/Meta-Llama-3-8B --quant_scheme w_int8_a_int8_per_tensor_sym --pre_quantization_optimization quarot
```
### vLLM

# Usage Example
- Inside `config.json` include a field `online_rotations`
    - Include the layer name regex
        - Under `func_name`, define the name of the rotation function class - this must be registered inside `model_executor/layers/quantization/quark/schemes/hadamard_transform.py`
        - Under `func_args`, define any arguments that rotation function class constructor takes
```json
"online_rotations": {
  "mlp.down_proj": {
    "func_name": "quarot_r4",
    "func_args": []
  }
}
```

# System Architecture
## Configuration
For reference vLLM contains the following classes:

`class QuarkConfig(QuantizationConfig)`
- `def get_quant_method`
	- Maps a layer to a particular `LinearMethodBase` quant method that will be applied to that layer. Associates the layer with a `QuarkScheme`. See `QuantizationConfig` for param details.


`class QuarkLinearMethod(LinearMethodBase)`
- `def apply_weights`
	- Use the output of `create_weights` and the `QuarkScheme` associated with the layer to apply the forward pass with the layer input. See `LinearMethodBase` for param details.

`class QuarkW8A8Int8(QuarkScheme)`
- `def apply_weights`
	- Apply the forward pass with the layer input. See `QuarkScheme` for param details.

*To achieve online rotations, I use the `QuarkScheme` associated with the layer to apply the correct rotation to its input.*

### *Below is a detailed explanation of how I implemented online rotations:*

*End goal:*
*A way to apply online rotation to the input of a particular layer.*

Modifications:
- `apply_weights` method, inside the `QuarkW8A8Int8` class, takes in a layer and the input, and applies the forward pass. *I modify the `apply_weights` method to apply online rotation (if it exists).*
```python
if self.online_rotation_method:
	x=self.online_rotation_method_callable(x)
```
- `create_weights` method, inside the `QuarkW8A8Int8` class, takes in a layer and initializes the quantized weights. *I modify the `create_weights` method to initialize the rotation function class (if it exists)*
```python
if self.online_rotation_method:
	func_name,func_args=self.online_rotation_method
	self.online_rotation_method_callable=func_name(layer,*func_args)
```
- `get_quant_method` method, inside `QuarkConfig` class, takes in the layer, associates it with a `QuarkW8A8Int8` scheme, and returns a `QuarkLinearMethod` object, that will later use the `QuarkW8A8Int8` associated with the layer to apply the forward pass with the layer input.
	- The helper method `_get_scheme_from_config` of the class `QuarkConfig` takes in a layer config and retrieves the appropriate `QuarkW8A8Int8` scheme. *I modify the `_get_scheme_from_config` method to check if the online rotation function is specified. If it is, ensure it is found in the registry, then pass the class name and constructor arguments of the online rotation function in while initializing the `QuarkW8A8Int8` quant schema.*
	```python
	online_rotation_config = cast(Dict[str, Any], config.get("online_rotations"))
	if online_rotation_config:
	func_name,func_args=online_rotation_config['func_name'],online_rotation_config['func_args']
	if func_name in hadamard_transform_registry:
		func_name=hadamard_transform_registry[func_name]
		online_rotation_method=func_name,func_args
	else:
		raise ValueError("hadamard rotation func_name is not found in registry")
else:
	online_rotation_method = None
	```
	- The helper method `_find_matched_config` of the class `QuarkConfig`, takes in the layer name, and retrieves the matching layer config. *I modify `_find_matched_config` to, based on whether the layer name is listed in the configuration, add the online rotation method specification to the layer config.*
	```python
	if "online_rotations" in self.quant_config:
		rot_info = next((value for key, value in self.quant_config['online_rotations'].items() if layer_name.endswith(key)), None)
		rv['online_rotations']=rot_info
	```

## Rotation Function design
### High Level Explanation
- `HadamardTransform` superclass
	- `__init__`:
		- Takes the layer (`nn.Module` object) - retrieves the device its on, and sets the device field to this
- `QuaRotR4` subclass:
	- `__init__`
		- Select the appropriate hadamard matrix size.
		- Initialize the hadamard matrix as a `RowParallelLinear` layer:
			- There are as many devices as tensor parallel size
		- Initialize the weight with `RowParallelLinear.weight_loader`:
			- *Based off of the input_size of the following down projection layer, it will initialize the weight, from the appropriate hadamard_k matrix*
			- In tensor parallelism, the input is sharded across multiple devices. The matching shard of the hadamard weight matrix will be initialized on each device.
				- The dimension of the hadamard weight matrix is `(hadamard_k.input_size,hadamard_k.input_size)`
				- The dimension of the shard of the hadamard weight matrix is `(hadamard_k.input_size, hadamard_k.input_size_per_partition)`
		- Select the appropriate Fast Hadamard Transform (FHT) function to preprocess each chunk of the input.
	- `forward`: 
		- Apply hadamard rotation to input shard, and return it.
			1.   The FHT preprocessing step: group input into chunks, and preprocess each chunk. (Each chunk can be processed independently so this this can be done in parallel on each processor).
				1.  Input shard size: `layer.input_size_per_partition` (where `layer` is the Llama MLP down projection layer). Input is reshaped into chunks: `(hadamard_k.input_size_per_partition,chunk_size)`. 
				2. Each chunk is processed using Fast Hadamard Transform.
			2. The chunked input is multiplied by a randomized hadamard matrix:
				- (A) Mathematically, this is: $AX=\begin{bmatrix} A_0 & A_1 & \cdots & A_p \end{bmatrix} \begin{bmatrix} X_0 \\ X_1 \\ \cdots \\ X_p \end{bmatrix}$ 
					- Where $A, X$ are the randomized hadamard matrix, input.
					- And where $A_i, X_i$ are the sharded $A, X$ on processor $i$.
				1. \*\*On processor $i$, the matrix $A_i$ `(hadamard_k.input_size,hadamard_k.input_size_per_partition)` is multiplied by the input $X_i$ `(hadamard_k.input_size_per_partition,chunk_size)` to yield a `(hadamard_k.input_size,chunk_size)` size tensor. The result is reduced, so the final $AX$ is on each processor (which is equal to $\sum A_i X_i$)
				2.  Finally, $AX$ is sharded, where processor $i$ takes `hadamard_k.input_size_per_partition*i:hadamard_k.input_size_per_partition*(i+1)` slice along dimension 0.

### Low Level Explanation - Matrix Multiply Implementation
- My implementation of the Matrix Multiplication in step 2.1\*\*:
	- Consider the expression for $AX$ in (A). I rewrite this as the transpose of the following expression: $\begin{bmatrix} X_0^\top & X_1^\top & \cdots & X_p^\top \end{bmatrix} \begin{bmatrix} A_0 \\ A_1 \\ \cdots \\ A_p \end{bmatrix}$ 
		- Note that:
			- The `RowParallelLinear` uses `nn.Linear`, where $y=XA$, and $A$ is the fixed weight, and $X$ is the activation. Therefore, we need to rewrite it to match this form:
				- $AX$ = $(X^\top A^\top)^\top$
				- Note that $A$ is a symmetric matrix, so it is equivalent to its transpose
	- To compute the expression, I use the `RowParallelLinear.forward`,  which will compute the result of each $X_i^\top A_i$ shard, then perform a reduction operation leaving the entire result of $X^\top A$ on all processors.
	- Finally, each processor performs the transpose to get the final result, $(X^\top A)^\top=AX$
### A note about configurability:
I improved the configurability of the online rotation, by allowing the user to specify the size of the FHT chunk to preprocess and/or the size of the randomized Hadamard matrix. It will ensure that the configuration matches the input size (`s_FHT*s_k==self.actual_input_size`) and there are existing FHT configurations and Hadamard matrices for the selected sizes. 
## Custom Operators (Compiled PyTorch ops for the rotations)
Have the online rotation module (Fast Hadamard Transform) registered as a custom operator, bundled within the vLLM codebase.

# Appendix
## Fast Hadamard Transform - Algorithm Design Reference
### Algorithm Overview
R4 corresponds to performing the Fast Hadamard Transform (FHT, aka the Butterfly Algorithm) on the input before multiplying by a hadamard matrix.

More concretely, it involves the following steps:
- Apply FHT to each 512-element group of the length 14336 input
    - Performed in $O(n \cdot \log n)$
        - Recursively combine smaller Hadamard transforms into larger ones using butterfly operations.
        - Each butterfly operation merges pairs of elements by summing and subtracting them.
- Reshape input into a 28 x 512 matrix, and post-multiply the 28 x 28 hadamard matrix by input
### Tri Dao's implementation
[Dao-AILab/fast-hadamard-transform: Fast Hadamard transform in CUDA, with a PyTorch interface](https://github.com/Dao-AILab/fast-hadamard-transform)
For each 512 chunk of the 14336 input, apply FHT butterfly algorithm iteratively for $\log n$ iterations. This I/O aware implementation effectively minimizes memory access.
- Load `kNelts` from segment $i$ of input to registers ofthread $i$ 
    - Apply for $\log$ `kNelts` iterations, thread wise
    - Apply for $\log$ `kWarpSize` iterations, warp wise
        - Use `__shfl_xor_sync` for each pair to receive their element from complement thread
    - Apply for $\log$ `kNWarps` iterations, block wise
        - Use shared memory to copy each pair complement to be on same warp
        - Use `__shfl_xor_sync` for each pair to receive their element from complement thread
### Modifications for AMD Compatibility
- Ensure warp size is defined to be 64
- ensure `__shfl_xor_sync` is replaced with `__shfl` (which seems more broadly compatible with ROCm)
- ensure the correct `lane_id` of the thread in its warp is calculated

### Implementation Nuance
If given the full 14336 size input, the Dao Lab kernel performs both the FHT and the 28 x 28 hadamard post matrix multiplication (using shared memory), in the same kernel. In my implementation, I perform only the FHT in the Dao Lab kernel, and perform the matrix multiply separately. 
- I chose this implementation because I found it performed slightly better than performing the matrix multiply in the Dao Lab kernel - I believe this is because the separate matrix multiply uses the matrix cores for the 28 x 28 post-multiply, which are more efficient than performing the matrix multiply in shared memory. This also uses bfloat16 (while the shared memory uses float32).
