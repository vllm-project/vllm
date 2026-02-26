# Governance Process

vLLM's success comes from our strong open source community. We favor informal, meritocratic norms over formal policies. This document clarifies our governance philosophy and practices.

## Values

vLLM aims to be the fastest and easiest-to-use LLM inference and serving engine. We stay current with advances, enable innovation, and support diverse models, modalities, and hardware.

### Design Values

1. **Top performance**: System performance is our top priority. We monitor overheads, optimize kernels, and publish benchmarks. We never leave performance on the table.
2. **Ease of use**: vLLM must be simple to install, configure, and operate. We provide clear documentation, fast startup, clean logs, helpful error messages, and monitoring guides. Many users fork our code or study it deeply, so we keep it readable and modular.
3. **Wide coverage**: vLLM supports frontier models and high-performance accelerators. We make it easy to add new models and hardware. vLLM + PyTorch form a simple interface that avoids complexity.
4. **Production ready**: vLLM runs 24/7 in production. It must be easy to operate and monitor for health issues.
5. **Extensibility**: vLLM serves as fundamental LLM infrastructure. Our codebase cannot cover every use case, so we design for easy forking and customization.

### Collaboration Values

1. **Tightly Knit and Fast-Moving**: Our maintainer team is aligned on vision, philosophy, and roadmap. We work closely to unblock each other and move quickly.
2. **Individual Merit**: No one buys their way into governance. Committer status belongs to individuals, not companies. We reward contribution, maintenance, and project stewardship.

## Project Maintainers

Maintainers form a hierarchy based on sustained, high-quality contributions and alignment with our design philosophy.

### Core Maintainers

Core Maintainers function like a project planning and decision making committee. In other convention, they might be called a Technical Steering Committee (TSC). In vLLM vocabulary, they are often known as "Project Leads". They meet weekly to coordinate roadmap priorities and allocate engineering resources.

**Project Leads:**

- Woosuk Kwon ([@WoosukKwon](https://github.com/WoosukKwon))
- Zhuohan Li ([@zhuohan123](https://github.com/zhuohan123))
- Simon Mo ([@simon-mo](https://github.com/simon-mo))
- Kaichao You ([@youkaichao](https://github.com/youkaichao))
- Robert Shaw ([@robertgshaw2-redhat](https://github.com/robertgshaw2-redhat))
- Tyler Michael Smith ([@tlrmchlsmth](https://github.com/tlrmchlsmth))
- Michael Goin ([@mgoin](https://github.com/mgoin))
- Nick Hill ([@njhill](https://github.com/njhill))
- Roger Wang ([@ywang96](https://github.com/ywang96))
- Lu Fang ([@houseroad](https://github.com/houseroad))
- Ye (Charlotte) Qi ([@yeqcharlotte](https://github.com/yeqcharlotte))
- Yihua Cheng ([@ApostaC](https://github.com/ApostaC))

**Responsibilities:**

- Author quarterly roadmap and responsible for each development effort.
- Making major changes to the technical direction or scope of vLLM and vLLM projects.
- Defining the project's release strategy.
- Work with model providers, hardware vendors, and key users of vLLM to ensure the project is on the right track.

### Lead Maintainers

While Core maintainers assume the day-to-day responsibilities of the project, Lead maintainers are responsible for the overall direction and strategy of the project. The following committee currently shares this role with divided responsibilities:

- Woosuk Kwon ([@WoosukKwon](https://github.com/WoosukKwon))
- Zhuohan Li ([@zhuohan123](https://github.com/zhuohan123))
- Simon Mo ([@simon-mo](https://github.com/simon-mo))
- Kaichao You ([@youkaichao](https://github.com/youkaichao))
- Robert Shaw ([@robertgshaw2-redhat](https://github.com/robertgshaw2-redhat))

**Responsibilities:**

- Making decisions where consensus among core maintainers cannot be reached.
- Adopting changes to the project's technical governance.
- Organizing the voting process for new committers.

### Committers and Area Owners

Committers have write access and merge rights. They typically have deep expertise in specific areas and help the community.

**Responsibilities:**

- Reviewing PRs and providing feedback.
- Addressing issues and questions from the community.
- Own specific areas of the codebase and development efforts: reviewing PRs, addressing issues, answering questions, improving documentation.

Specially, committers are almost all area owners. They author subsystems, review PRs, refactor code, monitor tests, and ensure compatibility with other areas. All area owners are committers with deep expertise in that area, but not all committers own areas.

For a full list of committers and their respective areas, see the [committers](./committers.md) page.

#### Committer Proposal Process

Any committer can nominate candidates via our private committer mailing list. The process runs as follows:

1. **Nominate**: A committer sends email to the committer group to nominate a candidate, highlighting the candidate’s contributions (e.g., links to PRs, reviews, RFCs, issues, benchmarks, and adoption evidence) and how they map to the standards below.
2. **Discuss and vote**: The committer group discusses the nomination, votes, and voices concerns if needed. Shared concerns can stop the process. For concerns, the group discusses clear criteria for the person to be nominated again. Most cases are decided by consensus; in contentious cases, the lead maintainers resolve conflicts and make the decision.
3. **Feedback period**: After a two-week feedback period (allowing time for any last input or concerns), if no blocking concerns arise and the nominator confirms with lead maintainer group to move forward (via the mailing list or committers slack channel), the nominator sends an invitation to the candidate asking them to open a PR to update their code ownership (e.g., CODEOWNERS and committers list).
4. **Permissions and onboarding**: In parallel, the lead maintainers assign the necessary permissions in GitHub and add the new member to the committer mailing list, the committer-only Slack channel, and other communications channels as appropriate.
5. **Finalize**: Once the CODEOWNERS/committer PR is ready and permissions are in place, the PR is merged and the new committer is welcomed.

Committership is highly selective and merit based. The selection criteria requires:

- **Area expertise**: leading design/implementation of core subsystems, material performance or reliability improvements adopted project‑wide, or accepted RFCs that shape technical direction.
- **Sustained contributions**: high‑quality merged contributions and reviews across releases, responsiveness to feedback, and stewardship of code health.
- **Community leadership**: mentoring contributors, triaging issues, improving docs, and elevating project standards.

To further illustrate, a committer typically satisfies at least two of the following accomplishment patterns:

- Author of an accepted RFC or design that materially shaped project direction
- Measurable, widely adopted performance or reliability improvement in core paths
- Long‑term ownership of a subsystem with demonstrable quality and stability gains
- Significant cross‑project compatibility or ecosystem enablement work (models, hardware, tooling)

While there isn't a quantitative bar, past committers have:

- Submitted approximately 30+ PRs of substantial quality and scope
- Provided high-quality reviews of approximately 10+ substantial external contributor PRs
- Addressed multiple issues and questions from the community in issues/forums/Slack
- Led concentrated efforts on RFCs and their implementation, or significant performance or reliability improvements adopted project‑wide

### Working Groups

vLLM runs informal working groups such as CI, CI infrastructure, torch compile, and startup UX. These can be loosely tracked via `#sig-` (or `#feat-`) channels in vLLM Slack. Some groups have regular sync meetings.

### Advisory Board

vLLM project leads consult with an informal advisory board that is composed of model providers, hardware vendors, and ecosystem partners. This manifests as a collaboration channel in Slack and frequent communications.

## Process

### Project Roadmap

Project Leads publish quarterly roadmaps as GitHub issues. These clarify current priorities. Unlisted topics aren't excluded but may get less review attention. See [https://roadmap.vllm.ai/](https://roadmap.vllm.ai/).

### Decision Making

We make technical decisions in Slack and GitHub using RFCs and design docs. Discussion may happen elsewhere, but we maintain public records of significant changes: problem statements, rationale, and alternatives considered.

### Merging Code

Contributors and maintainers often collaborate closely on code changes, especially within organizations or specific areas. Maintainers should give others appropriate review opportunities based on change significance.

PRs requires at least one committer review and approval. If the code is covered by CODEOWNERS, the PR should be reviewed by the CODEOWNERS. There are cases where the code is trivial or hotfix, the PR can be merged by the lead maintainers directly.

In case where CI didn't pass due to the failure is not related to the PR, the PR can be merged by the lead maintainers using "force merge" option that overrides the CI checks.

### Slack

Contributors are encouraged to join `#pr-reviews` and `#contributors` channels.

There are `#sig-` and `#feat-` channels for discussion and coordination around specific topics.

The project maintainer group also uses a private channel for high-bandwidth collaboration.

### Meetings

We hold weekly contributor syncs with standup-style updates on progress, blockers, and plans. You can refer to the notes [standup.vllm.ai](https://standup.vllm.ai) for joining instructions.
