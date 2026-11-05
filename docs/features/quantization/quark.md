# AMD Quark

Quantization can effectively reduce memory and bandwidth usage, accelerate computation and improve
throughput while with minimal accuracy loss. vLLM can leverage [Quark](https://quark.docs.amd.com/latest/),
the flexible and powerful quantization toolkit, to produce performant quantized models to run on AMD GPUs. Quark has specialized support for quantizing large language models with weight,
activation and kv-cache quantization and cutting-edge quantization algorithms like
AWQ, GPTQ, Rotation and SmoothQuant.

## Quark Installation

Before quantizing models, you need to install Quark. The latest release of Quark can be installed with pip:

```bash
pip install amd-quark
```

You can refer to [Quark installation guide](https://quark.docs.amd.com/latest/install.html)
for more installation details.

Additionally, install `vllm` and `lm-evaluation-harness` for evaluation:

```bash
pip install vllm "lm-eval[api]>=0.4.12"
```

## Quantization Process

After installing Quark, we will use an example to illustrate how to use Quark.
The Quark quantization process can be listed for 5 steps as below:

1. Load the model
2. Prepare the calibration dataloader
3. Set the quantization configuration
4. Quantize the model and export
5. Evaluation in vLLM

### 1. Load the Model

Quark uses [Transformers](https://huggingface.co/docs/transformers/en/index)
to fetch model and tokenizer.

??? code

    ```python
    from transformers import AutoTokenizer, AutoModelForCausalLM

    MODEL_ID = "meta-llama/Llama-2-70b-chat-hf"
    MAX_SEQ_LEN = 512

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        dtype="auto",
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, model_max_length=MAX_SEQ_LEN)
    tokenizer.pad_token = tokenizer.eos_token
    ```

### 2. Prepare the Calibration Dataloader

Quark uses the [PyTorch Dataloader](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html)
to load calibration data. For more details about how to use calibration datasets efficiently, please refer
to [Adding Calibration Datasets](https://quark.docs.amd.com/latest/pytorch/calibration_datasets.html).

??? code

    ```python
    from datasets import load_dataset
    from torch.utils.data import DataLoader

    BATCH_SIZE = 1
    NUM_CALIBRATION_DATA = 512

    # Load the dataset and get calibration data.
    dataset = load_dataset("mit-han-lab/pile-val-backup", split="validation")
    text_data = dataset["text"][:NUM_CALIBRATION_DATA]

    tokenized_outputs = tokenizer(
        text_data,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_SEQ_LEN,
    )
    calib_dataloader = DataLoader(
        tokenized_outputs['input_ids'],
        batch_size=BATCH_SIZE,
        drop_last=True,
    )
    ```

### 3. Set the Quantization Configuration

We need to set the quantization configuration, you can check
[quark config guide](https://quark.docs.amd.com/latest/pytorch/user_guide_config_description.html)
for further details. Here we use FP8 per-tensor quantization on weight, activation,
kv-cache and the quantization algorithm is AutoSmoothQuant.

!!! note
    Note the quantization algorithm needs a JSON config file and the config file is located in
    [Quark PyTorch examples](https://quark.docs.amd.com/latest/pytorch/pytorch_examples.html),
    under the directory `examples/torch/language_modeling/llm_ptq/models`. For example,
    AutoSmoothQuant config file for Llama is
    `examples/torch/language_modeling/llm_ptq/models/llama/autosmoothquant_config.json`.

??? code

    ```python
    from quark.torch.quantization import (Config, QuantizationConfig,
                                        FP8E4M3PerTensorSpec,
                                        load_quant_algo_config_from_file)

    # Define fp8/per-tensor/static spec.
    FP8_PER_TENSOR_SPEC = FP8E4M3PerTensorSpec(
        observer_method="min_max",
        is_dynamic=False,
    ).to_quantization_spec()

    # Define global quantization config, input tensors and weight apply FP8_PER_TENSOR_SPEC.
    global_quant_config = QuantizationConfig(
        input_tensors=FP8_PER_TENSOR_SPEC,
        weight=FP8_PER_TENSOR_SPEC,
    )

    # Define quantization config for kv-cache layers, output tensors apply FP8_PER_TENSOR_SPEC.
    KV_CACHE_SPEC = FP8_PER_TENSOR_SPEC
    kv_cache_layer_names_for_llama = ["*k_proj", "*v_proj"]
    kv_cache_quant_config = {
        name: QuantizationConfig(
            input_tensors=global_quant_config.input_tensors,
            weight=global_quant_config.weight,
            output_tensors=KV_CACHE_SPEC,
        )
        for name in kv_cache_layer_names_for_llama
    }
    layer_quant_config = kv_cache_quant_config.copy()

    # Define algorithm config by config file.
    LLAMA_AUTOSMOOTHQUANT_CONFIG_FILE = "examples/torch/language_modeling/llm_ptq/models/llama/autosmoothquant_config.json"
    algo_config = load_quant_algo_config_from_file(LLAMA_AUTOSMOOTHQUANT_CONFIG_FILE)

    EXCLUDE_LAYERS = ["lm_head"]
    quant_config = Config(
        global_quant_config=global_quant_config,
        layer_quant_config=layer_quant_config,
        kv_cache_quant_config=kv_cache_quant_config,
        exclude=EXCLUDE_LAYERS,
        algo_config=algo_config,
    )
    ```

### 4. Quantize the Model and Export

Then we can apply the quantization. After quantizing, we need to freeze the
quantized model first before exporting. Note that we need to export model with format of
HuggingFace `safetensors`, you can refer to
[HuggingFace format exporting](https://quark.docs.amd.com/latest/pytorch/export/quark_export_hf.html)
for more exporting format details.

??? code

    ```python
    import torch
    from quark.torch import ModelQuantizer, ModelExporter
    from quark.torch.export import ExporterConfig, JsonExporterConfig

    # Apply quantization.
    quantizer = ModelQuantizer(quant_config)
    quant_model = quantizer.quantize_model(model, calib_dataloader)

    # Freeze quantized model to export.
    freezed_model = quantizer.freeze(model)

    # Define export config.
    LLAMA_KV_CACHE_GROUP = ["*k_proj", "*v_proj"]
    export_config = ExporterConfig(json_export_config=JsonExporterConfig())
    export_config.json_export_config.kv_cache_group = LLAMA_KV_CACHE_GROUP

    # Model: Llama-2-70b-chat-hf-w-fp8-a-fp8-kvcache-fp8-pertensor-autosmoothquant
    EXPORT_DIR = MODEL_ID.split("/")[1] + "-w-fp8-a-fp8-kvcache-fp8-pertensor-autosmoothquant"
    exporter = ModelExporter(config=export_config, export_dir=EXPORT_DIR)
    with torch.no_grad():
        exporter.export_safetensors_model(
            freezed_model,
            quant_config=quant_config,
            tokenizer=tokenizer,
        )
    ```

### 5. Evaluation in vLLM

Now, you can load and run the Quark quantized model directly through the LLM entrypoint:

??? code

    ```python
    from vllm import LLM, SamplingParams

    # Sample prompts.
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    # Create a sampling params object.
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

    # Create an LLM.
    llm = LLM(
        model="Llama-2-70b-chat-hf-w-fp8-a-fp8-kvcache-fp8-pertensor-autosmoothquant",
        kv_cache_dtype="fp8",
        quantization="quark",
    )
    # Generate texts from the prompts. The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.
    outputs = llm.generate(prompts, sampling_params)
    # Print the outputs.
    print("\nGenerated Outputs:\n" + "-" * 60)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt:    {prompt!r}")
        print(f"Output:    {generated_text!r}")
        print("-" * 60)
    ```

Or, you can use `lm_eval` to evaluate accuracy:

```bash
lm_eval --model vllm \
  --model_args pretrained=Llama-2-70b-chat-hf-w-fp8-a-fp8-kvcache-fp8-pertensor-autosmoothquant,kv_cache_dtype='fp8',quantization='quark' \
  --tasks gsm8k
```

## Quark Quantization Script

In addition to the example of Python API above, Quark also offers a
[quantization script](https://quark.docs.amd.com/latest/pytorch/example_quark_torch_llm_ptq.html)
to quantize large language models more conveniently. It supports quantizing models with variety
of different quantization schemes and optimization algorithms. It can export the quantized model
and run evaluation tasks on the fly. With the script, the example above can be:

```bash
python3 quantize_quark.py --model_dir meta-llama/Llama-2-70b-chat-hf \
                          --output_dir /path/to/output \
                          --quant_scheme w_fp8_a_fp8 \
                          --kv_cache_dtype fp8 \
                          --quant_algo autosmoothquant \
                          --num_calib_data 512 \
                          --model_export hf_format \
                          --tasks gsm8k
```

## Using OCP MX (MXFP4, MXFP6) models

### RDNA4 MXFP4 weights with FP8 activations

On ROCm gfx1200/gfx1201, the `flydsl` linear backend supports Quark checkpoints
with OCP MXFP4 weights, E8M0 scales per 32 weights, and static per-tensor or
dynamic per-token FP8 E4M3 activations. Install a FlyDSL build that provides
`flydsl.kernels.gemm.rdna4_mxfp4`; vLLM imports the implementation from that
package. Weight preparation runs once during model loading and retains
compressed weights. GEMM uses native FP8 WMMA with FP32 accumulation.
The [FlyDSL kernel source and deployment guide](https://github.com/big-yellow-duck/FlyDSL/tree/ce625eb6be6d521eab3b0226703c6a03346d470d/kernels/gemm/rdna4_mxfp4)
provide the implementation and its analytical host selector. The refined selector
uses padding, wave coverage and legal K lookahead to reduce configuration cliffs
across unfamiliar shapes while keeping the register and hybrid strategies.

```bash
vllm serve amd/Qwen3.8-27B-Quark-AWQ-MXFP4 \
  --dtype bfloat16 \
  --quantization-config '{"linear":{"activation":"fp8_per_token"}}' \
  --kernel-config '{"linear_backend_per_quant":{"mxfp4":"flydsl"}}'
```

This checkpoint declares dynamic MXFP4 activations. The explicit override above
changes it to W4A8; retain the same override when comparing backends or evaluating
accuracy. Checkpoints that already declare static FP8 activations do not need an
override. Auto selection enables this backend only on RDNA4 when the FlyDSL
kernel package and a supported FP8 scheme are available. To compare against
dequantization and half-precision GEMM with the same FP8 activation quantizer,
select `"mxfp4":"emulation"` instead. Explicitly
selecting FlyDSL on unsupported hardware or without its kernels raises a
kernel-selection error. Dynamic MXFP4 activations and weight-only MXFP4
schemes continue to use their existing backends unless explicitly overridden.

The FlyDSL host selector chooses register or hybrid GEMM from input shape and
cached processor, cache and LDS capacities. It uses a small configuration palette
with explicit rules for tile coverage, tails, K divisibility and compressed
weight footprint. Runtime selection has no JSON, learned model or tuning cache,
and performs no GPU configuration search.

Prefill executables share runtime row counts, with separate aligned and
partial-tile epilogues. The selected linear layers pass device capacities into
vLLM's normal JIT warmup registry so startup and runtime use the same selector.
Startup compiles the selected configurations through `max_num_batched_tokens`,
using meta tensors without dummy GPU launches.
This covers new prefill row counts within the configured budget while keeping
the specialized decode paths. It requires a FlyDSL build providing
`get_device_capabilities`, `get_warmup_keys` and `compile_gemm` in the kernel
package. Eager mode normally disables vLLM's JIT warmup, so it can still compile
configurations on first use.
For an emulation reference that retains BF16 weights between calls, set
`VLLM_MXFP4_EMULATION_DEQUANT_AT_LOAD=1`; this uses more model memory than the
compressed FlyDSL backend.

For deployment experiments, FlyDSL's offline benchmark can compare explicit
configurations without changing production dispatch. The analytical selector
does not guarantee the fastest possible configuration for every unseen shape.

Validation on gfx1201 covers Qwen3.8-27B and Gemma4-12B/31B linear shapes at
TP1/2/4, including checkpoint rank shards and BF16/FP16 output. TP4 coverage is
per-rank kernels on two physical GPUs. Gemma12 MLX checkpoint operands are
adapted for kernel tests; this does not establish vLLM loader compatibility.
The [analytical selector evidence](https://github.com/big-yellow-duck/FlyDSL/blob/121e29927474770f38a129a3a85eb0e0be710db5/tests/perf/data/rdna4_mxfp4_analytical.json)
records paired kernel timings, individual regressions, independent cohort
provenance, startup coverage, ISA resources and model evaluation scope.
The separate [selector refinement evidence](https://github.com/big-yellow-duck/FlyDSL/blob/ce625eb6be6d521eab3b0226703c6a03346d470d/tests/perf/data/rdna4_mxfp4_refinement.json)
reports the subsequent host-rule changes, consumed development cohorts, final
unseen-shape validation and remaining regressions. Qwen TP1/TP2 retain identical
rotating-policy configurations for every row count from 1 through 8,192; a fresh
TP1 server smoke test also passes with the refined package. These checks preserve
the scope of the earlier serving and accuracy measurements.
Qwen TP1 agent throughput is nearly flat against the preceding profile selector.
One TP2 warmed run regressed substantially; matched follow-ups with the same
source were nearly flat, so the initial regression remains unresolved.
Full TP1 GSM8K scores are 1,269/1,319 versus 1,272/1,319, with zero invalid
answers. Those full runs used the first analytical freeze; final-source TP1
dispatch equivalence and a separate final-source smoke are recorded. Both
cases use the same FP8 activation override, rather than the checkpoint's default
MXFP4 activation path. Earlier
[profile-selector measurements](https://github.com/big-yellow-duck/FlyDSL/blob/6f7b095accb1ed4c8a43ebf60237e6a975baac56/tests/perf/data/rdna4_mxfp4_comprehensive.json)
remain historical evidence.

vLLM supports loading MXFP4 and MXFP6 models quantized offline through AMD Quark, compliant with [Open Compute Project (OCP) specification](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf).

The scheme currently only supports dynamic quantization for activations.

Example usage, after installing the latest AMD Quark release:

```bash
vllm serve fxmarty/qwen_1.5-moe-a2.7b-mxfp4 --tensor-parallel-size 1
# or, for a model using fp6 activations and fp4 weights:
vllm serve fxmarty/qwen1.5_moe_a2.7b_chat_w_fp4_a_fp6_e2m3 --tensor-parallel-size 1
```

A simulation of the matrix multiplication execution in MXFP4/MXFP6 can be run on devices that do not support OCP MX operations natively (e.g. AMD Instinct MI325, MI300 and MI250), dequantizing weights from FP4/FP6 to half precision on the fly, using a fused kernel. This is useful e.g. to evaluate FP4/FP6 models using vLLM, or alternatively to benefit from the ~2.5-4x memory savings (compared to float16 and bfloat16).

To generate offline models quantized using MXFP4 data type, the easiest approach is to use AMD Quark's [quantization script](https://quark.docs.amd.com/latest/pytorch/example_quark_torch_llm_ptq.html), as an example:

```bash
python quantize_quark.py --model_dir Qwen/Qwen1.5-MoE-A2.7B-Chat \
    --quant_scheme w_mxfp4_a_mxfp4 \
    --output_dir qwen_1.5-moe-a2.7b-mxfp4 \
    --skip_evaluation \
    --model_export hf_format \
    --group_size 32
```

The current integration supports [all combination of FP4, FP6_E3M2, FP6_E2M3](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/utils/ocp_mx_utils.py) used for either weights or activations.

## Using Quark Quantized layerwise Auto Mixed Precision (AMP) Models

vLLM also supports loading layerwise mixed precision model quantized using AMD Quark. Currently, mixed scheme of {MXFP4, FP8} is supported, where FP8 here denotes for FP8 per-tensor scheme. More mixed precision schemes are planned to be supported in a near future, including

- Unquantized Linear and/or MoE layer(s) as an option for each layer, i.e., mixed of {MXFP4, FP8, BF16/FP16}
- MXFP6 quantization extension, i.e., {MXFP4, MXFP6, FP8, BF16/FP16}

Although one can maximize serving throughput using the lowest precision supported on a given device (e.g. MXFP4 for AMD Instinct MI355, FP8 for AMD Instinct MI300), these aggressive schemes can be detrimental to accuracy recovering from quantization on target tasks. Mixed precision allows to strike a balance between maximizing accuracy and throughput.

There are two steps to generate and deploy a mixed precision model quantized with AMD Quark, as shown below.

### 1. Quantize a model using mixed precision in AMD Quark

Firstly, the layerwise mixed-precision configuration for a given LLM model is searched and then quantized using AMD Quark. We will provide a detailed tutorial with Quark APIs later.

As examples, we provide some ready-to-use quantized mixed precision model to show the usage in vLLM and the accuracy benefits. They are:

- amd/Llama-2-70b-chat-hf-WMXFP4FP8-AMXFP4FP8-AMP-KVFP8
- amd/Mixtral-8x7B-Instruct-v0.1-WMXFP4FP8-AMXFP4FP8-AMP-KVFP8
- amd/Qwen3-8B-WMXFP4FP8-AMXFP4FP8-AMP-KVFP8

### 2. inference the quantized mixed precision model in vLLM

Models quantized with AMD Quark using mixed precision can natively be reload in vLLM, and e.g. evaluated using lm-evaluation-harness as follows:

```bash
lm_eval --model vllm \
    --model_args pretrained=amd/Llama-2-70b-chat-hf-WMXFP4FP8-AMXFP4FP8-AMP-KVFP8,tensor_parallel_size=4,dtype=auto,gpu_memory_utilization=0.8,trust_remote_code=False \
    --tasks mmlu \
    --batch_size auto
```

## Online Quantization

All the workflows above are *offline* quantization: you run a script, write a new quantized checkpoint to disk, and later load it for serving. This produces the most accurate and smallest checkpoints because it can run accuracy-recovery algorithms (rotation, SmoothQuant, GPTQ, AWQ) during export, but it requires a separate quantization step and a second copy of the model on disk.

*Online* quantization instead quantizes the weights at load time, directly from a high-precision checkpoint, and offers several advantages over the offline flow:

- **No export step** — serve directly from the original `bf16`/`fp16` checkpoint; no separate quantization run before deployment.
- **No extra disk footprint** — nothing new is written to disk, so there is no second copy of the model to store or manage.
- **No calibration data** — activations are scaled dynamically at runtime, so no calibration dataset is needed.
- **Fast iteration** — switch schemes or per-layer selections instantly by changing a config, without re-exporting.

### vLLM online quantization

vLLM has [built-in online quantization](online.md) that skips the export step: it loads an ordinary high-precision (`bf16`) checkpoint and quantizes each layer's weights to schemes such as FP8 or MXFP4 at load time, without a pre-quantized checkpoint or calibration data. No new checkpoint is written to disk. See [Online Quantization](online.md) for the supported schemes and configuration.

### Quark online quantization

AMD Quark provides its own online path, exposed as the `quark_online` quantization backend, for users who want parity with Quark's offline export and Quark's per-layer mixed-precision configs. Like vLLM's built-in online quantization, it quantizes weights inside the weight-loading hook just before serving and writes no new checkpoint.

Its quant math is aligned byte-for-byte with Quark's offline export, so what you validate online is what you get offline. Use it to *find* the scheme and per-layer selection you want.

Compared to vLLM's built-in online quantization, the `quark_online` plugin adds:

- **Flexible config parsing** — a terse config expands into Quark's verbose per-layer config, delegating all per-layer matching to a real `QuarkConfig`.
- **Per-layer / mixed schemes** — dispatch a different method per layer (e.g. MXFP4 experts with FP8 attention on an MoE model). Each online method subclasses the matching offline scheme, so a load-time quantized layer runs the identical inference kernel as an offline one.
- **Re-quantizing an already-quantized checkpoint** — an FP8 block-scale checkpoint (e.g. DeepSeek-R1) is dequantized and re-quantized to a target scheme layer-locally at load time, with no new checkpoint.

The plugin ships in AMD Quark — no fork of vLLM, no patched checkpoint format; see the [Quark documentation](https://quark.docs.amd.com/latest/) for details. Three presets ship ready to use:

| Preset key | Scheme |
| --- | --- |
| `ptpc_fp8` | FP8 E4M3, per-channel weight + dynamic per-token activation |
| `mxfp4` | MXFP4, per-group (group size 32) with E8M0 block scale |
| `linear_ptpc_fp8_moe_mxfp4` | Mixed: attention in FP8, MoE experts in MXFP4 |

### Python API

```python
from vllm import LLM, SamplingParams
from quark.online_quantization.vllm import HF_QUANTIZATION_CONFIGS

llm = LLM(
    model="Qwen/Qwen3-30B-A3B-Thinking-2507",
    quantization="quark_online",
    hf_overrides=HF_QUANTIZATION_CONFIGS["ptpc_fp8"],
    enforce_eager=True,
    tensor_parallel_size=1,
)
out = llm.generate(["The capital of France is"], SamplingParams(temperature=0.0, max_tokens=100))
print(out)
```

### Native vLLM CLI

```bash
export VLLM_PLUGINS="${VLLM_PLUGINS:-quark_online_quant}"
ONLINE_QUANT_CONFIG='{"online_quant_config": {"global_quant_config": "ptpc_fp8", "exclude_layer": ["lm_head"]}}'

vllm serve Qwen/Qwen3-8B \
  --trust-remote-code \
  --tensor-parallel-size 1 \
  --additional-config "$ONLINE_QUANT_CONFIG"
```

### Re-quantizing an offline checkpoint

No extra arguments — the same `hf_overrides` detects the checkpoint's existing config and merges it automatically:

```python
llm = LLM(
    model="deepseek-ai/DeepSeek-R1",           # ships quant_method: "fp8"
    quantization="quark_online",
    hf_overrides=HF_QUANTIZATION_CONFIGS["ptpc_fp8"],
    tensor_parallel_size=8,
)
```

The plugin can be disabled with `QUARK_DISABLE_VLLM_PLUGIN=1`, and it steps aside automatically if another platform plugin (e.g. ATOM) has already taken over the backend.
