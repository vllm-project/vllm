target "xpu-base" {
  context    = "."
  dockerfile = "docker/Dockerfile.xpu_base"
}

target "_xpu" {
  context    = "."
  dockerfile = "docker/Dockerfile.xpu"
  # Build the base in this graph unless the caller supplies an image.
  contexts = {
    vllm-base = "target:xpu-base"
  }
}

target "vllm-build" {
  inherits = ["_xpu"]
  target   = "vllm-build"
  tags     = ["vllm-xpu-env"]
}

target "vllm-openai" {
  inherits = ["_xpu"]
  target   = "vllm-openai"
  tags     = ["vllm-xpu-env"]
}

group "default" { targets = ["vllm-openai"] }
