# Gemma 4 redeployment reference

This fork preserves two implementations:

- **`plugin.py` and `grammar.py` here:** the server-side lab workaround tested
  with vLLM **0.32.0** and XGrammar **0.2.9**. The human operator confirmed code
  review and staging-call testing. It includes the JSON-reply whitespace cap.
- **The changes under `vllm/tool_parsers/`:** a focused port to upstream main,
  based on `c282bd3b72475904b6bbf0e34850d0487171299c`. Local parser and grammar
  tests passed, but this port has **not** been deployed or validated with live
  inference. It deliberately leaves whitespace handling to existing upstream
  work.

Use the lab overlay with its pinned runtime when reproducing the staging setup.
Do not treat this fork's main-source port as the exact staging image.
No upstream PR was submitted.

## Lab overlay

The tested model was `nvidia/Gemma-4-31B-IT-NVFP4`, revision
`4135a98a9b728a548947683219633b25682223ac`, on one NVIDIA RTX PRO 6000 GPU.
The tested overlay image digest was
`sha256:2965f1b4aa608d353bb9e0376dc34f82e51db100f33ccc623079935fec30afa4`.
The image and deployment configuration are environment-specific; this directory
contains the portable patch rather than cloud credentials or endpoint settings.

1. Start with a working vLLM 0.32.0 environment for the model and hardware.
2. Install XGrammar 0.2.9 for that environment's Python version and architecture.
   The tested container used CPython 3.12 on Linux x86-64.
3. Copy `plugin.py` and `grammar.py` together to `/opt/gemma-lab`, and add that
   directory to `PYTHONPATH`.
4. Add these flags to the existing server command:

   ```text
   --tool-parser-plugin /opt/gemma-lab/plugin.py
   --tool-call-parser gemma4_lab
   --reasoning-parser gemma4
   --enable-auto-tool-choice
   ```

5. Disable native thinking through the model's chat-template settings
   (`enable_thinking: false`). The overlay supports the application's JSON
   `reasoning` field; it does not support a native thinking channel.
6. Run `check_grammar.py` before starting the server. Optionally pass the pinned
   checkpoint's `tokenizer.json` to check its actual delimiter IDs:

   ```bash
   .venv/bin/python deployment/gemma4-lab/check_grammar.py /path/to/tokenizer.json
   ```

Clients use ordinary `response_format: json_schema`, `tool_choice: auto`, and
strict function definitions. The server combines the request's schema with the
native tool grammar; it does not hard-code any response fields. Plain-text
requests omit the response format. Preserve application speech and reasoning
history normally.

## Behavior and limits

- The required-tool branch starts with the actual tool-call token rather than
  allowing a free-text prefix or printable delimiter lookalikes.
- Delimiter names are resolved through the tokenizer before tokenizer-free
  grammar validation.
- Formatting whitespace is capped only in the composed JSON reply branch.
  Spaces inside JSON strings are preserved. JSON-only/no-tools paths are not
  covered by this cap.
- Required tools and JSON replies are different grammar alternatives. Valid
  syntax does not establish that the chosen action is appropriate.
- Synthetic structured prompts sometimes induced unwanted actions. Recorded
  call replays had identical action choices across constrained/unconstrained
  modes and no unwanted hang-ups, but both missed some explicit hang-up requests.
- Named forced-tool responses sometimes used `finish_reason: "stop"` despite
  returning the correct arguments. This reference does not claim to fix that.
- The original live validation covered streaming, non-streaming, plain replies,
  structured replies, and native tools. It was performed on the lab overlay,
  not on the upstream-source port.

## Upstream context

[vLLM #69709](https://github.com/vllm-project/vllm/pull/69709) already proposes
Gemma's native-grammar registration. The source port in this branch also needs
those two registration lines, then tightens the required-call branch.
Registration alone still admitted ordinary-text delimiters in all eight new
regression cases. Whitespace changes overlap
[#61205](https://github.com/vllm-project/vllm/pull/61205) and
[#67031](https://github.com/vllm-project/vllm/pull/67031).

Run the focused upstream checks from the repository root:

```bash
.venv/bin/python -m pytest \
  tests/tool_use/test_gemma4_responses_adjust_request.py \
  tests/tool_parsers/test_gemma4_tool_parser.py -q
```

For the code in commit `851ee07470b3dd822cc04ca4acbe3dc5bc97f914`, the command
above returned:

```text
80 passed, 1 warning in 10.11s
```

The warning was missing generated `vllm._version` metadata in the source checkout.
The commit message was amended after testing; its code tree is unchanged
(`dec5e04c8e5a1363be3b6be742a7d7f1e0652f8a`).
The new eight-case regression initially failed on base main (`8 failed, 8
deselected, 1 warning in 2.05s`) and with the registration-only changes (`8 failed,
8 deselected, 1 warning in 1.22s`). The latter failure demonstrated acceptance of
printable delimiter lookalikes, rather than merely missing grammar registration.

All applicable pre-commit checks passed, including Ruff, mypy and documentation
lint. The reference CPU check passed both synthetic-token and actual-checkpoint
tokenizer validation. An AST comparison confirmed that the executable logic in
the copied `grammar.py` and `plugin.py` matches the staging overlay; formatting
and import ordering were adjusted for repository conventions.

AI assistance was used in the investigation, implementation, tests, and notes.
