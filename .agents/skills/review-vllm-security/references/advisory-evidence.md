# Published examples

- **Resource budgets.** Limit transport reads, media materialization, and
  total prompt or media work across the request.
  [HTTP headers](https://github.com/vllm-project/vllm/security/advisories/GHSA-rxc4-3w6r-4v47)
  · [Media downloads](https://github.com/vllm-project/vllm/security/advisories/GHSA-p6g9-7v3x-m8mv)
  · [Request-wide media budget](https://github.com/vllm-project/vllm/security/advisories/GHSA-5r3c-5x4v-8gjh)
  · [Prompt list fan-out](https://github.com/vllm-project/vllm/security/advisories/GHSA-87x5-vmc3-756j)
- **Input invariants and fault containment.** Use the effective model bounds in
  each frontend and keep invalid request or compilation errors request-local.
  [Rust and Python token ID bounds](https://github.com/vllm-project/vllm/security/advisories/GHSA-qff2-492f-9fm4)
  · [Structured-output error containment](https://github.com/vllm-project/vllm/security/advisories/GHSA-85xf-c7hm-whqw)
- **Cache, request, and peer ownership.** Preserve cache ownership and namespace
  through transformations, errors, and eviction.
  [Shared cache identity](https://github.com/vllm-project/vllm/security/advisories/GHSA-rh6f-3x46-j33q)
  · [Cache salt through tool continuations](https://github.com/vllm-project/vllm/security/advisories/GHSA-935w-9g4m-p28p)
  · [Mirrored cache consistency](https://github.com/vllm-project/vllm/security/advisories/GHSA-ph3r-5jfg-f84f)
- **Parser and rendering complexity.** Bound worker occupancy and apply
  compilation controls across supported backends.
  [Template rendering resources](https://github.com/vllm-project/vllm/security/advisories/GHSA-542m-9234-8gvf)
  · [Structured-output regex compilation](https://github.com/vllm-project/vllm/security/advisories/GHSA-48jh-3gj7-fg8v)
- **Artifact trust and unsafe interpretation.** Verify both artifact
  resolution and actual code or object loading.
  [Model code trust control](https://github.com/vllm-project/vllm/security/advisories/GHSA-7972-pg2x-xr59)
  · [Artifact revision propagation](https://github.com/vllm-project/vllm/security/advisories/GHSA-3ww4-5jv9-j5gm)
  · [Shared-memory deserialization boundary](https://github.com/vllm-project/vllm/security/advisories/GHSA-6757-hggm-hp2w)
- **Fetch, filesystem, and decoder policy.** Check media acquisition and
  format restrictions throughout model-specific paths.
  [Media connector coverage](https://github.com/vllm-project/vllm/security/advisories/GHSA-4hhp-h66f-j5j7)
  · [Process-wide decoder safeguards](https://github.com/vllm-project/vllm/security/advisories/GHSA-mm38-5g96-7j6x)
- **Authorization and capability gates.** Verify authorization and trust
  settings at the route and operation that consume input.
  [Route authentication](https://github.com/vllm-project/vllm/security/advisories/GHSA-94f4-hr76-p5j6)
  · [Template trust gate](https://github.com/vllm-project/vllm/security/advisories/GHSA-x567-49g5-p4cf)
- **Sensitive output and error responses.** Check diagnostics for both
  disclosure and unbounded output.
  [Validation error disclosure](https://github.com/vllm-project/vllm/security/advisories/GHSA-hwrm-c4cx-rf4j)
  · [Error representation across frontends](https://github.com/vllm-project/vllm/security/advisories/GHSA-hgg8-fqqc-vfmw)
  · [Metric label cardinality](https://github.com/vllm-project/vllm/security/advisories/GHSA-5fj9-pfhr-6j48)
