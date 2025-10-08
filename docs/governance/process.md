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

The core maintainers make final decisions. A committee of @WoosukKwon, @zhuohan123, @simon-mo, and @youkaichao currently shares this role with divided responsibilities.

The responsibilities of the core maintainers are:

* Making decisions where consensus among maintainers cannot be reached.
* Approving or removing committers.
* Adopting changes to the project's technical governance.
* Making major changes to the technical direction or scope of vLLM itself.
* Managing the scope of the vLLM project – e.g., the addition or removal of new sub-projects in the GitHub org.
* Defining the project's release strategy.

Core maintainers serve until they step down. Core maintainers will choose successors. We prefer natural succession with mentoring over sudden votes.

### Project Leads

Projects leads function like a project planning committee. They meet weekly to coordinate roadmap priorities and allocate engineering resources. Current leads: @WoosukKwon, @zhuohan123, @simon-mo, @youkaichao, @robertshaw2-redhat, @tlrmchlsmth, @mgoin, @njhill, @ywang96, @houseroad, @yeqcharlotte, @comaniac, @ApostaC

The leads actively work with model providers, hardware vendors, and key users of vLLM to ensure the project is on the right track.

### Committers and Area Owners

Committers have write access and merge rights. They typically have deep expertise in specific areas and help the community.

Area owners run subsystems. They review PRs, refactor code, monitor tests, and ensure compatibility.

All area owners are committers with deep expertise in that area, but not all committers own areas. Committers maintain their areas and help across the project: reviewing PRs, addressing issues, answering questions, improving documentation.

For a full list of committers and their respective areas, see the [committers](./committers.md) page.

#### Nomination Process

Any committer can nominate candidates via our private mailing list:

1. **Nominate**: Cite accomplishments and area expertise.
2. **Vote**: Group voices support or concerns. Concerns can stop the process. The vote typically last 3 working days. For concerns, committers group discuss the clear criteria for such person to be nominated again. The lead maintainers will make the final decision.
3. **Confirm**: The lead maintainers send invitation, update CODEOWNERS, assign permissions, add to communications channels (mailing list and Slack).

Committership requires:

* **Area expertise**: Deep involvement in design, implementation, and maintenance.
* **Community**: Support users, shepherd PRs, enhance project direction.
* **Vision and mentorship**: Good design taste, clean code, teaching ability.

While there isn't a quantitative bar, past committers have:

* Submitted 30+ PRs
* Provided high-quality reviews of 10+ contributor PRs
* Addressed multiple issues and questions from the community in issues/forums/Slack
* Led concentrated efforts on RFCs and their implementation

### Working Groups

vLLM runs informal working groups such as CI, CI infrastructure, torch compile, and startup UX. These can be loosely tracked via `#sig-` channels in vLLM Slack. Some groups have regular sync meetings.

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

Contributors are encouraged to join #pr-reviews and #contributors channels.

There are `#sig-` and `#feat-` channels for discussion and coordination around specific topics.

The project maintainer group also uses a private channel for high-bandwidth collaboration.

### Meetings

We hold weekly contributor syncs with standup-style updates on progress, blockers, and plans. You can refer to the notes [standup.vllm.ai](https://standup.vllm.ai) for joining instructions.