# 🔗 Inter-Agent Communication Protocol

## Overview

This document defines how the 6 specialized agents communicate, coordinate, and collaborate to create the VLLM learning materials. Effective communication is critical for parallel development and quality outcomes.

**Core Principle**: Asynchronous-first communication with structured synchronization points.

---

## 📋 Central Communication Hub

### MULTI_AGENT_WORK_STATUS.md

**Purpose**: Single source of truth for all agent coordination

**Location**: `/MULTI_AGENT_WORK_STATUS.md` (repository root)

**Update Frequency**: Real-time (agents update as work progresses)

**Structure**:

```markdown
# Multi-Agent Work Status

**Last Updated**: [Timestamp] by [Agent Name]

---

## 🎯 Current Phase: [Phase Name]

**Phase Goal**: [Description]
**Target Completion**: [Date]
**Overall Progress**: [X/Y tasks complete]

---

## 📊 Today's Focus (YYYY-MM-DD)

### Agent 1: Curriculum Architect
**Status**: [Working/Blocked/Complete]
**Current Task**: [Task description]
**Progress**: [X%]
**Blockers**: [None/Description]
**ETA**: [Timestamp]

### Agent 2: Content Builder
**Status**: [Working/Blocked/Complete]
**Current Task**: [Task description]
**Progress**: [X%]
**Blockers**: [None/Description]
**Needs from**: [Agent name - specific request]
**ETA**: [Timestamp]

### Agent 3: Code Engineer
**Status**: [Working/Blocked/Complete]
**Current Task**: [Task description]
**Progress**: [X%]
**Files Created**: [List]
**Blockers**: [None/Description]
**ETA**: [Timestamp]

### Agent 4: Technical Validator
**Status**: [Working/Blocked/Complete]
**Current Task**: [Task description]
**Review Queue**: [X items]
**Blockers**: [None/Description]
**ETA**: [Timestamp]

### Agent 5: Assessment Designer
**Status**: [Working/Blocked/Complete]
**Current Task**: [Task description]
**Progress**: [X%]
**Blockers**: [None/Description]
**ETA**: [Timestamp]

### Agent 6: Scribe/Integrator
**Status**: [Working/Blocked/Complete]
**Current Task**: [Task description]
**Polish Queue**: [X items]
**Blockers**: [None/Description]
**ETA**: [Timestamp]

---

## 📬 Inter-Agent Messages

### Message #1: [Agent X] → [Agent Y]
**Timestamp**: [YYYY-MM-DD HH:MM]
**Subject**: [Brief subject]
**Priority**: [High/Medium/Low]

**Message**:
[Content]

**Action Required**: [Yes/No]
**Deadline**: [If applicable]

**Status**: [Unread/Read/Resolved]

---

### Message #2: [Agent X] → [Agent Y]
...

---

## 📝 Completed Today

- [x] [Task description] - Agent X
- [x] [Task description] - Agent Y
- [x] [Task description] - Agent Z

---

## 🚧 Blockers & Dependencies

### Blocker #1
**Affected Agent**: [Agent name]
**Issue**: [Description]
**Impact**: [High/Medium/Low]
**Needs**: [What's needed to unblock]
**Owner**: [Who can resolve]
**Status**: [Open/In Progress/Resolved]

---

## 📅 Tomorrow's Plan

### Agent 1: Curriculum Architect
- [ ] [Task 1]
- [ ] [Task 2]

### Agent 2: Content Builder
- [ ] [Task 1]
- [ ] [Task 2]

[... for all agents]

---

## 🎯 Weekly Goals Progress

| Goal | Owner | Status | Progress |
|------|-------|--------|----------|
| [Goal 1] | Agent X | In Progress | 60% |
| [Goal 2] | Agent Y | Blocked | 30% |
| [Goal 3] | Agent Z | Complete | 100% |

---

## 📈 Metrics

- **Files Created This Week**: X
- **Files Reviewed**: Y
- **Tests Passing**: Z/Z (100%)
- **Blocker Resolution Time**: Avg X hours
```

---

## 🔄 Communication Workflows

### Workflow 1: New Module Creation

**Participants**: Curriculum Architect → Content Builder → Code Engineer → Technical Validator → Scribe

**Steps**:

1. **Curriculum Architect** (Day 1):
   - Creates module specification
   - Posts message in MULTI_AGENT_WORK_STATUS.md:
   ```markdown
   ### Message: Architect → Content Builder
   **Subject**: New Module Spec Ready - Module 3 CUDA Kernels
   **Priority**: High

   Module 3 specification is complete and ready for content creation.

   **Location**: `/specs/module_03_spec.md`
   **Files Needed**: 15 tutorial files (see spec)
   **Deadline**: Week 5

   **Action Required**: Yes - please review spec and confirm feasibility
   ```

2. **Content Builder** (Day 2):
   - Reviews spec, asks clarifying questions if needed
   - Updates status: "Working on Module 3 tutorials"
   - Creates first drafts
   - Posts message when ready for code examples:
   ```markdown
   ### Message: Content Builder → Code Engineer
   **Subject**: Code Examples Needed for Module 3
   **Priority**: High

   I've completed 5 tutorial drafts that need code examples:
   - `cuda_memory_hierarchy.md` - needs 3 examples (lines 45, 78, 120)
   - `kernel_fusion.md` - needs 2 examples (lines 34, 89)

   **Specifications**: See inline comments in files
   **Deadline**: End of week

   **Action Required**: Yes - create code examples
   ```

3. **Code Engineer** (Day 3-4):
   - Creates code examples
   - Writes tests
   - Updates status: "Working on Module 3 code examples"
   - Posts when ready for validation:
   ```markdown
   ### Message: Code Engineer → Technical Validator
   **Subject**: Module 3 Code Ready for Validation
   **Priority**: High

   All code examples for Module 3 tutorials complete:
   - 15 code files created
   - 15 test files created
   - All tests passing locally

   **Location**: `/examples/module_03/`
   **Action Required**: Yes - please validate and benchmark
   ```

4. **Technical Validator** (Day 5):
   - Reviews code, runs tests, benchmarks
   - Posts validation report:
   ```markdown
   ### Message: Technical Validator → Code Engineer, Content Builder
   **Subject**: Module 3 Validation Report
   **Priority**: Medium

   **Status**: APPROVED with minor fixes needed

   **Issues**:
   - Example 3: Memory leak in cleanup (line 78) - FIXED
   - Example 7: Performance 20% below expected - INVESTIGATING

   **Recommendations**:
   - Add error handling to example 5
   - Update tutorial text to match actual performance

   **Action Required**: Code Engineer - fix example 3, Content Builder - update perf claims
   ```

5. **Scribe/Integrator** (Day 6):
   - Polishes all content
   - Checks cross-references
   - Posts final approval:
   ```markdown
   ### Message: Scribe → All Agents
   **Subject**: Module 3 Complete and Published
   **Priority**: Low

   Module 3 has been polished and merged to main branch.

   **Summary**:
   - 15 tutorial files ✅
   - 15 code files + tests ✅
   - All cross-references working ✅
   - Style guide compliant ✅

   **Metrics**:
   - Total pages: 87
   - Code examples: 25
   - Time: 6 days (on schedule)

   Great work team! 🎉
   ```

---

### Workflow 2: Code Lab Creation

**Participants**: Curriculum Architect → Code Engineer → Content Builder → Technical Validator → Scribe

**Steps**:

1. **Curriculum Architect**:
   - Defines lab learning objectives
   - Specifies difficulty level
   - Posts lab specification

2. **Code Engineer**:
   - Creates starter code
   - Implements complete solution
   - Writes test suite
   - Posts code completion message

3. **Content Builder**:
   - Writes lab instructions
   - Creates hints and guidance
   - Posts instructions ready message

4. **Technical Validator**:
   - Tests both starter code and solution
   - Verifies difficulty level appropriate
   - Posts validation report

5. **Scribe/Integrator**:
   - Polishes instructions
   - Ensures consistent formatting
   - Posts completion

---

### Workflow 3: Interview Question Creation

**Participants**: Assessment Designer → Technical Validator → Curriculum Architect

**Steps**:

1. **Assessment Designer**:
   - Creates interview questions
   - Writes model solutions
   - Develops evaluation rubrics
   - Posts question set for review

2. **Technical Validator**:
   - Verifies solution correctness
   - Tests code problems
   - Validates difficulty rating
   - Posts validation report

3. **Curriculum Architect**:
   - Reviews alignment with learning objectives
   - Approves or requests revisions
   - Posts approval message

---

## 📨 Message Types and Templates

### 1. Task Handoff

**Use**: When passing work to another agent

```markdown
### Message: [Your Agent] → [Target Agent]
**Timestamp**: [YYYY-MM-DD HH:MM]
**Subject**: [Clear subject line]
**Priority**: [High/Medium/Low]

**Context**: [Brief background]

**Deliverables Ready**:
- [Item 1] - location: [path]
- [Item 2] - location: [path]

**Your Action Needed**:
- [Specific action 1]
- [Specific action 2]

**Deadline**: [Date/time]
**Blockers**: [Any issues to be aware of]

**Status**: Ready for your review
```

### 2. Question/Clarification

**Use**: When needing information from another agent

```markdown
### Message: [Your Agent] → [Target Agent]
**Timestamp**: [YYYY-MM-DD HH:MM]
**Subject**: Question about [Topic]
**Priority**: [High/Medium/Low]

**Question**: [Clear question]

**Context**: [Why you need to know]

**Impact if not resolved**: [Consequences]

**Preferred response time**: [Timeframe]

**Status**: Awaiting response
```

### 3. Blocker Notification

**Use**: When blocked and cannot proceed

```markdown
### Message: [Your Agent] → ALL
**Timestamp**: [YYYY-MM-DD HH:MM]
**Subject**: 🚨 BLOCKER: [Brief description]
**Priority**: HIGH

**Issue**: [Clear description of blocker]

**Impact**:
- Cannot proceed with: [Tasks]
- Affects: [Other agents/deliverables]
- Estimated delay: [Time]

**What I Need**:
- [Specific help needed]
- [From whom]

**Workarounds Attempted**: [What you tried]

**Status**: BLOCKED - urgent assistance needed
```

### 4. Validation Report

**Use**: Technical Validator reporting results

```markdown
### Message: Technical Validator → [Original Author]
**Timestamp**: [YYYY-MM-DD HH:MM]
**Subject**: Validation Report: [Item name]
**Priority**: [High/Medium/Low]

**Item Reviewed**: [Name and location]

**Validation Status**: [APPROVED/NEEDS REVISION/REJECTED]

**Test Results**:
- [Test category]: [X/Y passed]
- [Test category]: [X/Y passed]

**Issues Found**: [Number]

**Critical Issues**:
- [Issue 1 with location]

**Recommendations**:
- [Recommendation 1]

**Approval Conditions**: [If conditional approval]

**Next Steps**: [What should happen next]
```

### 5. Completion Notification

**Use**: When finishing a deliverable

```markdown
### Message: [Your Agent] → ALL
**Timestamp**: [YYYY-MM-DD HH:MM]
**Subject**: ✅ Completed: [Deliverable name]
**Priority**: Low

**Deliverable**: [Name]
**Location**: [Path]

**Summary**:
- [Key metric 1]
- [Key metric 2]

**Quality Checks**:
- [x] Tests passing
- [x] Reviewed by [Agent]
- [x] Style guide compliant

**Status**: Complete and merged to [branch]

**Next**: [What's next in pipeline]
```

### 6. Weekly Summary

**Use**: End of week recap (Curriculum Architect)

```markdown
### Message: Curriculum Architect → ALL
**Timestamp**: [YYYY-MM-DD] Week [X] Summary
**Subject**: 📊 Week [X] Summary - [Phase Name]
**Priority**: Medium

**Week Goal**: [Original goal]
**Achieved**: [What was completed]

**Metrics**:
| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| Files created | X | Y | ✅/⚠️/❌ |
| Tests passing | 100% | Z% | ✅/⚠️/❌ |

**Highlights**:
- [Major achievement 1]
- [Major achievement 2]

**Challenges**:
- [Challenge 1 and how resolved]

**Next Week Goals**:
- [Goal 1]
- [Goal 2]

**Action Items**:
- Agent X: [Action]
- Agent Y: [Action]

**Team Notes**: [Any general notes]
```

---

## 🔔 Communication Rules

### Response Time SLAs

| Priority | Response Time | Action Time |
|----------|--------------|-------------|
| **High** | 4 hours | 24 hours |
| **Medium** | 12 hours | 48 hours |
| **Low** | 24 hours | 1 week |
| **Blocker** | 2 hours | Immediate |

### Communication Etiquette

1. **Be Specific**: Include file paths, line numbers, specific requirements
2. **Be Concise**: Respect other agents' time
3. **Be Proactive**: Don't wait for problems to escalate
4. **Be Positive**: Celebrate wins, support during challenges
5. **Be Professional**: Constructive feedback, no blame

### Update Frequency

**Required Updates**:
- **Daily**: Status update in MULTI_AGENT_WORK_STATUS.md (EOD)
- **When blocking**: Immediate notification
- **When handing off**: Within 1 hour of completion
- **When blocked**: Within 2 hours of discovering blocker

**Optional Updates**:
- Mid-day progress check-ins
- Questions and clarifications as needed
- Proactive heads-up about upcoming needs

---

## 🎯 Dependency Management

### Common Dependencies

**Content → Code**:
- Content Builder needs code examples from Code Engineer
- Use priority: Medium (plan ahead)

**Code → Validation**:
- Code Engineer always needs Technical Validator review
- Use priority: High (on critical path)

**All → Scribe**:
- All content needs final polish from Scribe
- Use priority: Low (happens at end)

**Assessment → Validation**:
- Assessment Designer needs Technical Validator for solution verification
- Use priority: Medium

### Dependency Tracking Template

In MULTI_AGENT_WORK_STATUS.md:

```markdown
## 🔗 Current Dependencies

| Waiting For | Needs From | Item | Requested | Needed By | Status |
|-------------|-----------|------|-----------|-----------|--------|
| Content Builder | Code Engineer | CUDA examples | 2025-11-15 | 2025-11-18 | In Progress |
| Code Engineer | Technical Validator | Lab 5 review | 2025-11-16 | 2025-11-19 | Pending |
```

---

## 📊 Synchronization Points

### Daily Stand-up (Async)

**Time**: End of each work day
**Format**: Status update in MULTI_AGENT_WORK_STATUS.md

**Each agent posts**:
- What I completed today
- What I'm working on tomorrow
- Any blockers or help needed

### Weekly Review (Sync or Detailed Async)

**Time**: End of week
**Format**: Comprehensive review in MULTI_AGENT_WORK_STATUS.md

**Agenda**:
1. Review week's goals vs. achievements
2. Review metrics and KPIs
3. Discuss blockers and resolutions
4. Plan next week's priorities
5. Process improvements

### Phase Retrospective (Sync or Detailed Async)

**Time**: End of each phase (every 2-4 weeks)
**Format**: Dedicated retrospective document

**Agenda**:
1. What went well
2. What could be improved
3. Action items for next phase
4. Celebrate successes

---

## 🚀 Getting Started: Agent Initialization

### Step 1: Repository Setup

Each agent should:
1. Clone the repository
2. Create their working branch: `git checkout -b agent[X]/[name]`
3. Read MULTI_AGENT_PROJECT_PLAN.md
4. Read AGENT_PERSONAS.md (focus on their role)
5. Read this communication protocol

### Step 2: Initial Check-in

Post to MULTI_AGENT_WORK_STATUS.md:

```markdown
### Agent [X]: [Name] - INITIALIZED ✅
**Timestamp**: [YYYY-MM-DD HH:MM]

I am Agent [X] - [Role Name]. I have:
- [x] Read all planning documents
- [x] Understood my role and responsibilities
- [x] Created my working branch
- [x] Ready to begin assigned tasks

**My first task**: [Description]
**Expected completion**: [Date]

**Questions**: [Any questions or None]

**Status**: Ready to work 🚀
```

### Step 3: Begin Work

- Check TASK_DISTRIBUTION_MATRIX.md for assignments
- Update status in MULTI_AGENT_WORK_STATUS.md
- Begin creating deliverables
- Communicate early and often

---

## 🛠️ Tools and Automation

### Automated Checks

Set up automated checks for:
- **Link checker**: Validate all cross-references daily
- **Test runner**: Run all tests on each commit
- **Style checker**: Validate markdown formatting
- **Metric tracker**: Update KPI dashboard

### Communication Helpers

**Quick Status Update Script**:
```bash
# agents/update_status.sh
# Quick script to update agent status

AGENT_NAME="Agent 1: Curriculum Architect"
CURRENT_TASK="$1"
PROGRESS="$2"
BLOCKERS="${3:-None}"

cat >> MULTI_AGENT_WORK_STATUS.md <<EOF

### $AGENT_NAME - Updated $(date)
**Current Task**: $CURRENT_TASK
**Progress**: $PROGRESS%
**Blockers**: $BLOCKERS

EOF
```

---

## 📖 Best Practices

### Do's ✅

- **Do** update status daily
- **Do** communicate blockers immediately
- **Do** ask clarifying questions early
- **Do** celebrate team wins
- **Do** provide specific feedback with line numbers
- **Do** offer help when you have capacity
- **Do** read messages meant for all agents

### Don'ts ❌

- **Don't** work in silence - communicate progress
- **Don't** assume others know what you need
- **Don't** wait for blockers to resolve themselves
- **Don't** skip quality checks to meet deadlines
- **Don't** duplicate work - check what others are doing
- **Don't** merge to main without proper reviews

---

## 🔍 Troubleshooting

### Issue: Agent Not Responding

**Solutions**:
1. Check MULTI_AGENT_WORK_STATUS.md for their last update
2. Post high-priority message
3. If blocker, escalate to Curriculum Architect
4. Find workaround if possible

### Issue: Conflicting Approaches

**Solutions**:
1. Discuss in MULTI_AGENT_WORK_STATUS.md
2. Reference style guide or specifications
3. Curriculum Architect makes final decision
4. Document decision for future reference

### Issue: Quality Concerns

**Solutions**:
1. Technical Validator raises concern with specific examples
2. Original agent revises
3. If disagreement, escalate to Curriculum Architect
4. Update quality standards if needed

---

## 📚 Appendix: Example Work Status Document

See `/examples/MULTI_AGENT_WORK_STATUS_example.md` for a complete example of a filled-out work status document.

---

**Last Updated**: 2025-11-18
**Version**: 1.0
**Next Review**: Week 2
**Maintained By**: All agents (collaborative document)
