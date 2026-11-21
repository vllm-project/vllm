# Advisory evidence

The 2026-11-21 review enumerated the vLLM repository's draft and published
advisories using paginated `gh api` calls. It inspected 312 entries and processed
1,439 GitHub comments across 301 threads. Comment counts matched the advisory
list for every entry.

- 126 published advisories, including project-authored advisories with no
  submission flag.
- 186 drafts. GitHub marked 171 submissions accepted. The other 15 were
  maintainer-authored drafts with a proposed fix or maintainer discussion of
  an active patch.
- One draft marked accepted in GitHub was excluded from the ranking after a
  maintainer scoped its internal-only network scenario outside the project's
  threat model.

Each of the 311 ranked entries was assigned to its primary prevention control.
Counts are per advisory. Related reports and follow-ups can involve the same
control, and a single advisory can cover several sites. The counts describe this
review snapshot and use reviewer classification where controls overlap.

| Primary prevention control | Total | Published | Draft |
| --- | ---: | ---: | ---: |
| Resource budgets | 103 | 37 | 66 |
| Input invariants and fault containment | 66 | 27 | 39 |
| Cache, request, and peer ownership | 44 | 15 | 29 |
| Parser and rendering complexity | 29 | 6 | 23 |
| Artifact trust and unsafe interpretation | 26 | 19 | 7 |
| Fetch, filesystem, and decoder policy | 23 | 11 | 12 |
| Authorization and capability gates | 15 | 7 | 8 |
| Sensitive output and error responses | 5 | 4 | 1 |

Denial of service recurred across the first, second, and fourth categories.
Common causes were late limits, amplification during expansion or fan-out,
fatal errors escaping the request boundary, and superlinear parsing. Maintainer
discussions confirmed that related routes can need separate checks, including
frontend-only fields and backend-specific input validation.

See [Detailed security failure modes](failure-modes.md) for the prevention
checks across all eight categories.

## Published examples

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
