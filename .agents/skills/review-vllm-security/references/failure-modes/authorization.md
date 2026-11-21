# Authorization and capability gates

- **Effective route.** Inspect registered routes, mount prefixes, methods,
  WebSocket upgrades, middleware order, and the deployed frontend. Verify that
  authorization and routing see consistent paths under supported prefix and
  proxy configurations.
- **Access boundary.** Check API-key coverage for the specific route. Apply
  the documented trusted-network model to internal transports. Restrict
  administrative, debug, and dynamic model-loading actions to their intended
  principals.
- **Browser and credential handling.** Check Host and origin behavior when
  changing browser-facing listeners or local endpoints. Verify that gated
  routes use the existing credential comparison path and test expected
  credential handling.
- **Operator-owned switches.** Check enablement of request templates,
  multimodal overrides, insecure serialization, plugins, and privileged
  tools at the operation that uses them. Include nested overrides, backend
  selection, and continuation paths.
- **Internal authority.** Use typed, server-owned context for peer selection,
  tensor handles, cache transfer state, and internal routing. Review merging
  of generic extra arguments with privileged fields.
- **Tool dispatch.** Resolve tool identity and namespace from trusted
  definitions. Enforce the allowlist at execution and share applicable call
  budget counters across continuations. Check same-name tools and who may
  turn a parsed tool call into an action. Treat remote tool results as data
  with the trust of their source. Keep authorization decisions in trusted
  server state.
- **Tool retries.** Track dispatch and outcome by call identity when execution
  can have side effects. Account for duplicate events and disconnects.
  A timeout can leave the result unknown. Safe retries need an idempotency
  guarantee or a way to verify the original outcome.
- **Sessions and peers.** Bind tool sessions and connector actions to the
  intended principal, instance, or authorized peer. Verify that caller
  headers and IDs cannot take over a session or redirect an action.

## Published advisory examples

Verify authorization and trust
settings at the route and operation that consume input.

- [Route authentication](https://github.com/vllm-project/vllm/security/advisories/GHSA-94f4-hr76-p5j6)
- [Template trust gate](https://github.com/vllm-project/vllm/security/advisories/GHSA-x567-49g5-p4cf)
