# ROCm images (docker/Dockerfile.rocm). CI/release drive this via .buildkite/scripts/rocm/build.sh. Locally:
#   docker buildx bake -f docker/docker-bake-rocm.hcl                    # test image, everything from source
#   STACK_IMAGE=vllm/vllm-openai-rocm:base-nightly RUNTIME=vllm/vllm-openai-rocm:base-nightly \
#     docker buildx bake -f docker/docker-bake-rocm.hcl vllm-openai        # compile only vLLM
variable "STACK"        { default = "rocm" }   # rocm: TheRock ROCm 10
variable "STACK_IMAGE"  { default = "" }       # prebuilt stack ref@digest; empty: build it in this graph
variable "STACK_WHEELS" { default = "" }       # that stack's -wheels image
variable "RUNTIME"      { default = "" }       # prebuilt runtime ref@digest; empty: build it in this graph
variable "TEST_DEPS"    { default = "" }       # prebuilt test-deps ref@digest; empty: build it in this graph
variable "CI_ARCH"      { default = "gfx90a;gfx942;gfx950" }  # MI250, MI300/325, MI355: what CI can test
variable "TAG"          { default = "vllm-rocm:dev" }
variable "PUSH"         { default = false }
variable "CACHE"        { default = "" }       # trusted cache repo (rocm/vllm-ci-cache): every build reads it
variable "CACHE_TO"     { default = "" }       # ref prefix this build writes ("" = none); build.sh sets it per repo
variable "AITER_SPEC"   { default = "" }       # pins the AITER nightly wheel (pip spec)
variable "AITER_NIGHTLY" { default = "" }      # "1": test the newest AITER nightly wheel
variable "VLLM_VERSION" { default = "" }       # release: image version label (build.sh reuse check)
variable "CACHE_CI"     { default = true }     # false (PR builds): skip the full ci cache; compile-cache still exports

function "cfrom" {   # trusted cache, plus this PR's own cache when CACHE_TO points elsewhere
  params = [names]
  result = flatten([for p in distinct(compact([CACHE == "" ? "" : "${CACHE}:", CACHE_TO])) :
                    [for n in names : "type=registry,ref=${p}${STACK}-${n}"]])
}
function "cto" {
  params = [name]
  result = CACHE_TO == "" ? [] : ["type=registry,ref=${CACHE_TO}${STACK}-${name},mode=max,image-manifest=true,ignore-error=true,compression=zstd,compression-level=3,force-compression=true"]
}
# Images and cache layers are zstd: on the CI fleet, unpacking dominates pull time (vllm#60258,
# vllm#28656). Pulling needs Docker >= 23 or containerd >= 1.5. force-compression keeps every layer
# zstd: without it, a builder exports whichever copy of a layer it holds, e.g. a gzip one imported
# from an older cache. Images built once per key (stack, runtime) also get reproducible timestamps.
function "out" {
  params = [rewrite]
  result = PUSH ? join(",", concat(["type=registry", "compression=zstd", "compression-level=3", "force-compression=true", "oci-mediatypes=true"],
                                   rewrite ? ["rewrite-timestamp=true"] : [])) : "type=docker"
}
function "ctx" {   # named contexts: prebuilt refs when given, else in-graph targets
  params = []
  result = merge(
    { stack = STACK_IMAGE == "" ? "target:stack-${STACK}" : "docker-image://${STACK_IMAGE}" },
    RUNTIME == "" ? {} : { runtime = "docker-image://${RUNTIME}" },
    TEST_DEPS == "" ? {} : { test-deps = "docker-image://${TEST_DEPS}" })
}

target "_rocm" {
  context = "."
  args    = { SOURCE_DATE_EPOCH = "0", STACK = STACK }
  attest  = ["type=provenance,disabled=true"]   # provenance would change the pushed digest on every build
  output  = [out(true)]
}
target "_stack" {
  inherits   = ["_rocm"]
  target     = "final"
  tags       = [TAG]
  cache-from = cfrom(["stack"])
  cache-to   = cto("stack")
}
# literal dockerfile paths: ci-infra ci_selector (images.py) only follows `dockerfile = "..."` literals
target "stack-rocm" {
  inherits   = ["_stack"]
  dockerfile = "docker/Dockerfile.rocm_base"
}
target "stack-wheels" {
  inherits = ["stack-${STACK}"]
  target   = "wheels"
  tags     = ["${TAG}-wheels"]
  cache-to = []
}
target "runtime" {
  inherits   = ["_rocm"]
  dockerfile = "docker/Dockerfile.rocm"
  target     = "runtime"
  contexts   = ctx()
  tags       = [TAG]
  cache-from = cfrom(["runtime"])
  cache-to   = cto("runtime")
}
target "test-deps" {   # content-keyed like the runtime (build.sh base mode)
  inherits   = ["_rocm"]
  dockerfile = "docker/Dockerfile.rocm"
  target     = "test-deps"
  contexts   = ctx()
  tags       = [TAG]
  output     = [out(false)]   # rewriting would hand it a second copy of the runtime's base layer
  cache-from = cfrom(["test-deps", "runtime", "ci"])
  cache-to   = cto("test-deps")
}
target "_app" {
  inherits   = ["_rocm"]
  dockerfile = "docker/Dockerfile.rocm"
  contexts   = ctx()
  args       = { ARCH = "", AITER_SPEC = AITER_SPEC, AITER_NIGHTLY = AITER_NIGHTLY, VLLM_VERSION = VLLM_VERSION }
  output     = [out(false)]   # per-commit images: rewriting would hand them second copies of inherited layers
  cache-from = cfrom(["runtime", "ci", "compile", "csrc", "release"])   # dev/csrc records are shared
}
target "test" {
  inherits = ["_app"]
  target   = "test"
  args     = { ARCH = CI_ARCH }
  tags     = [TAG]
  cache-to = CACHE_CI ? cto("ci") : []
  no-cache-filter = AITER_SPEC == "" && AITER_NIGHTLY != "1" ? [] : ["test-base"]   # AITER nightly: always fetch the newest wheel
}
target "test-smoke" {
  inherits = ["test"]
  target   = "test-smoke"
  output   = ["type=cacheonly"]
  cache-to = []
}
target "compile-cache" {   # csrc + rust-deps on their own: a fresh builder (or a new PR) reuses them
  inherits = ["_app"]
  target   = "compile"
  args     = { ARCH = CI_ARCH }
  output   = ["type=cacheonly"]
  cache-to = cto("compile")
}
target "vllm-wheel" {      # the python-only install test's wheel, uploaded as a build artifact
  inherits = ["test"]
  target   = "vllm-wheel"
  output   = ["type=local,dest=.rocm-wheel"]
  cache-to = []
}
target "vllm-openai" {
  inherits = ["_app"]
  target   = "vllm-openai"
  tags     = [TAG]
  cache-to = cto("release")
}
# cacheonly checks
target "openai-smoke" {
  inherits = ["_app"]
  target   = "openai-smoke"
  output   = ["type=cacheonly"]
}
target "csrc-all" {       # release: the serialized all-arch compile, cache only
  inherits = ["openai-smoke"]
  target   = "csrc"
  cache-to = cto("csrc")
}
# STACK_WHEELS must be set (a full ref@digest); build.sh release mode sets it via load() before invoking this target.
target "wheels" {
  inherits = ["_app"]
  target   = "wheels"
  tags     = ["${TAG}-wheels"]
  contexts = merge(ctx(), { stack-wheels = "docker-image://${STACK_WHEELS}" })
}
target "version" {
  inherits = ["_app"]
  target   = "version"
  output   = ["type=local,dest=.rocm-version"]
}

group "default" { targets = ["test"] }
group "base"    { targets = ["stack-${STACK}", "stack-wheels"] }   # build.sh pushes runtime separately (own key)
group "ci"      { targets = ["test", "test-smoke", "compile-cache", "vllm-wheel"] }
