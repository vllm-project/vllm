target "xpu-base" {
  context    = "."
  dockerfile = "docker/Dockerfile.xpu"
  target     = "vllm-base"
}

target "_xpu" {
  context    = "."
  dockerfile = "docker/Dockerfile.xpu"
  # Build the base in this graph unless the caller supplies an image.
  contexts = {
    vllm-base = "target:xpu-base"
  }
}

target "vllm-openai" {
  inherits = ["_xpu"]
  target   = "vllm-openai"
  tags     = ["vllm-xpu-env"]
}

group "default" { targets = ["vllm-openai"] }
