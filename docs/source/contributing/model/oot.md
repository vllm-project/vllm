(new-model-oot)=

# Out-of-Tree Model Integration

You can integrate a model using a plugin without modifying the vLLM codebase.

```{seealso}
[vLLM's Plugin System](#plugin-system)
```

To register the model, use the following code:

```python
from vllm import ModelRegistry
from your_code import YourModelForCausalLM
ModelRegistry.register_model("YourModelForCausalLM", YourModelForCausalLM)
```

If your model imports modules that initialize CUDA, consider lazy-importing it to avoid errors like `RuntimeError: Cannot re-initialize CUDA in forked subprocess`:

```python
from vllm import ModelRegistry

ModelRegistry.register_model("YourModelForCausalLM", "your_code:YourModelForCausalLM")
```

```{important}
If your model is a multimodal model, ensure the model class implements the {class}`~vllm.model_executor.models.interfaces.SupportsMultiModal` interface.
Read more about that [here](#enabling-multimodal-inputs).
```

```{note}
Although you can directly put these code snippets in your script using `vllm.LLM`, the recommended way is to place these snippets in a vLLM plugin. This ensures compatibility with various vLLM features like distributed inference and the API server.
```
