# Configuration

API documentation for vLLM's configuration classes.

<!-- autodoc2 options are TOML. Therefore, use single quotes for regex -->
<!-- The current regexes remove snake case (global functions/variables) and "Supports*" protocols -->

```{autodoc2-object} vllm.config
hidden_regexes = [
    '.*\.(\w+_.*|[a-z]+)',
    '.*Supports.*',
]
```
