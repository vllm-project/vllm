# Fetch, filesystem, and decoder policy

- **Every acquisition path.** Reuse the shared media connector or equivalent
  controls. Trace sync and async fetches, model or tokenizer hooks, file and
  inline handlers, and secondary reads during decoding or rendering.
- **Effective destination.** Enforce the configured scheme, host, address,
  and redirect policy on the destination actually used by the client. Keep
  validation and fetch URL interpretation consistent. Reapply applicable
  policy to each redirect.
- **Request context.** Scope credentials and headers to their intended
  destination. Keep ambient authorization, shared cookies, and caller session
  state out of unrelated media fetches. Check retries and redirects for
  unintended forwarding.
- **Path confinement.** Apply configured local-media, artifact, and
  shared-storage roots to the resolved path. Check absolute paths, symlinks,
  archive members, and caller or artifact names used to construct paths.
  Validate the target used by the actual open or write.
- **Detected format.** Use the detected media format and nested stream type to
  choose an allowed decoder. Review backend selection, fallbacks, and formats
  that can launch helpers or open another file or network resource.
- **Decoder safeguards.** Preserve pixel, sample, duration, and frame
  protections across decoder choices. Keep time and byte limits active during
  download and decode. Review model initialization for changes to shared
  decoder safety settings.
- **Decode failure.** Return a request error for unusable results, partial
  reads, or unexpected dimensions. Verify that a failed decode cannot return
  a buffer from an earlier request, including on native and hardware backends.

## Published advisory examples

Check media acquisition and
format restrictions throughout model-specific paths.

- [Media connector coverage](https://github.com/vllm-project/vllm/security/advisories/GHSA-4hhp-h66f-j5j7)
- [Process-wide decoder safeguards](https://github.com/vllm-project/vllm/security/advisories/GHSA-mm38-5g96-7j6x)
