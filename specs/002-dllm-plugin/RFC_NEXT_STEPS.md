# Next steps: Opening the dLLM Plugin RFC

This document describes how to open the RFC in vLLM’s repository and follow the project’s process. The **RFC body** is in [rfc-draft.md](rfc-draft.md). **Alternative RFC (minimal-core variant):** For the design that reuses the spec-decode path with a single engine change and maximal plugin encapsulation, see [rfc-draft-spec-decode-path.md](rfc-draft-spec-decode-path.md).

---

## 1. Detailed design (link from RFC)

The RFC draft already links to this folder as the **detailed design**:

- **[spec.md](spec.md)** – Feature specification (user stories, requirements, success criteria)
- **[plan.md](plan.md)** – Implementation plan (technical context, project structure)
- **[data-model.md](data-model.md)** – Entities, state, validation
- **[contracts/](contracts/)** – Plugin registration contract, dLLM step contract reference
- **[research.md](research.md)** – Decisions and rationale (bart-plugin patterns, core vs plugin)
- **[quickstart.md](quickstart.md)** – How to run and test once implemented

When you paste the RFC into the GitHub issue, the relative links `(.)`, `(data-model.md)`, `(contracts/)` will need to be turned into absolute links (e.g. to this repo or a rendered doc) if the issue is in vllm-project/vllm. Alternatively, keep the RFC body self-contained and add a single “Detailed design” URL (e.g. to the branch or a shared doc).

---

## 2. Open the RFC (vLLM GitHub)

1. **Create a new issue** in [vllm-project/vllm](https://github.com/vllm-project/vllm).
2. **Title:** `[RFC]: dLLM (Blocked Masked Diffusion LLM) support via plugin` (or `[RFC][Plugin]: ...` if that matches other plugin RFCs).
3. **Body:** Paste the content of [rfc-draft.md](rfc-draft.md). Fix links to the detailed design (see above).
4. **Labels:** Add `RFC` and any applicable labels (e.g. `plugin`, `inference`).
5. **Feedback period:** Ensure the “Feedback period” date in the body is set (e.g. “Two weeks from &lt;date&gt;”).
6. **CC list:** Fill in the CC list with relevant committers/area owners (scheduler, worker, plugins, model loader) from [vLLM committers](https://docs.vllm.ai/en/latest/governance/committers/).

---

## 3. Follow vLLM process

- **Post** the issue link in the **#contributors** channel on vLLM Slack.
- **Tag** the CC list for feedback; respond to comments and update the issue (or linked design) as needed.
- If an **assignee** is nominated, coordinate with them for next steps (tracking issue, implementation PRs).

---

## 4. Optional: Check for RFC issue template

When creating the issue, check whether vLLM offers an “RFC” or “Design” issue template. If yes, align the draft with its headings and checklist (e.g. add any required checkboxes or sections).

---

## 5. After acceptance

- Open a **tracking issue** (or reference an existing branch/spec) for implementation; link it in the RFC.
- Implementation can follow [plan.md](plan.md) and the rest of this folder (core dLLM path in vLLM + plugin package with LLaDA2.x as MVP).
