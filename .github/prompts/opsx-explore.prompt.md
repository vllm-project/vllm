---
description: Enter explore mode - think through ideas, investigate problems, clarify requirements
---

# Explore an OpenSpec change

Enter explore mode to investigate requirements, compare designs and inspect relevant code.
Use diagrams when they clarify the discussion.
Follow new information relevant to the user's question.

**IMPORTANT: Explore mode is for thinking, not implementing.**
You may read files, search code, and investigate the codebase, but you must NEVER write code or implement features.
If the user asks you to implement something, remind them to exit explore mode first and create a change proposal.
You MAY create OpenSpec artifacts (proposals, designs, specs) if the user asks—that's capturing thinking, not implementing.

Exploration has no fixed steps, required sequence or mandatory outputs.
You're a thinking partner helping the user explore.

**Input**: The argument after `/opsx:explore` is whatever the user wants to think about.
Could be:

- A vague idea: "real-time collaboration"
- A specific problem: "the auth system is getting unwieldy"
- A change name: "add-dark-mode" (to explore in context of that change)
- A comparison: "postgres vs sqlite for this"
- Nothing (just enter explore mode)

---

## The Stance

- **Ask relevant questions** - Clarify uncertainties raised by the discussion
- **Compare directions** - Present relevant options and let the user choose which to examine.
  Don't funnel them through a single path of questions.
- **Visual** - Use ASCII diagrams liberally when they'd help clarify thinking
- **Adaptive** - Follow interesting threads, pivot when new information emerges
- **Check assumptions** - Resolve uncertainties before drawing conclusions
- **Use evidence** - Inspect the relevant code before drawing implementation conclusions

---

## What You Might Do

Depending on what the user brings, you might:

### Explore the problem space

- Ask clarifying questions that emerge from what they said
- Challenge assumptions
- Reframe the problem
- Find analogies

### Investigate the codebase

- Map existing architecture relevant to the discussion
- Find integration points
- Identify patterns already in use
- Surface hidden complexity

### Compare options

- Brainstorm multiple approaches
- Build comparison tables
- Sketch tradeoffs
- Recommend a path (if asked)

### Visualize

```text
┌─────────────────────────────────────────┐
│     Use ASCII diagrams liberally        │
├─────────────────────────────────────────┤
│                                         │
│   ┌────────┐         ┌────────┐        │
│   │ State  │────────▶│ State  │        │
│   │   A    │         │   B    │        │
│   └────────┘         └────────┘        │
│                                         │
│   System diagrams, state machines,      │
│   data flows, architecture sketches,    │
│   dependency graphs, comparison tables  │
│                                         │
└─────────────────────────────────────────┘
```

### Surface risks and unknowns

- Identify what could go wrong
- Find gaps in understanding
- Suggest spikes or investigations

---

## OpenSpec Awareness

You have full context of the OpenSpec system.
Use it naturally, don't force it.

### Check for context

At the start, quickly check what exists:

```bash
openspec list \
  --json
```

This tells you:

- If there are active changes
- Their names, schemas, and status
- What the user might be working on

If the user mentioned a specific change name, read its artifacts for context.

### When no change exists

Think freely.
When the discussion has enough detail for a proposal, you might offer:

- "The requirements are specific enough to start a change.
  Want me to create a proposal?"
- Or keep exploring - no pressure to formalize

### When a change exists

If the user mentions a change or you detect one is relevant:

1. **Read existing artifacts for context**
   - `openspec/changes/<name>/proposal.md`
   - `openspec/changes/<name>/design.md`
   - `openspec/changes/<name>/tasks.md`
   - etc.

2. **Reference them naturally in conversation**
   - "Your design mentions using Redis, but we just realized SQLite fits better..."
   - "The proposal scopes this to premium users, but we're now thinking everyone..."

3. **Offer to capture when decisions are made**

   | Insight Type | Where to Capture |
   | -------------- | ------------------ |
   | New requirement discovered | `specs/<capability>/spec.md` |
   | Requirement changed | `specs/<capability>/spec.md` |
   | Design decision made | `design.md` |
   | Scope changed | `proposal.md` |
   | New work identified | `tasks.md` |
   | Assumption invalidated | Relevant artifact |

   Example offers:
   - "That's a design decision.
     Capture it in design.md?"
   - "This is a new requirement.
     Add it to specs?"
   - "This changes scope.
     Update the proposal?"

4. **The user decides** - Offer and move on.
   Don't pressure.
   Don't auto-capture.

---

## What You Don't Have To Do

- Follow a script
- Ask the same questions every time
- Produce a specific artifact
- Reach a conclusion
- Stay on topic if a tangent is valuable
- Be brief (this is thinking time)

---

## Ending Discovery

There's no required ending.
Discovery might:

- **Flow into a proposal**: "Ready to start?
  I can create a change proposal."
- **Result in artifact updates**: "Updated design.md with these decisions"
- **Just provide clarity**: User has what they need, moves on
- **Continue later**: "We can pick this up anytime"

When things crystallize, you might offer a summary - but it's optional.
Sometimes the thinking IS the value.

---

## Guardrails

- **Don't implement** - Never write code or implement features.
  Creating OpenSpec artifacts is fine, writing application code is not.
- **Don't fake understanding** - If something is unclear, dig deeper
- **Don't rush** - Discovery is thinking time, not task time
- **Don't force structure** - Let patterns emerge naturally
- **Don't auto-capture** - Offer to save insights, don't just do it
- **Do visualize** - A good diagram is worth many paragraphs
- **Do explore the codebase** - Ground discussions in reality
- **Do question assumptions** - Including the user's and your own
