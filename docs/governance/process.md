# Governance Process

We believe vLLM’s success comes from the power of the strong open source community. We operate under technical governance that favors informal, meritocratic norms over more formalized policies and procedures. This document seeks to provide clarity for the broader community on the project’s governance philosophy and practices.

The goal of the vLLM project is to become a thriving open source community building the de facto, open-source standard as the fastest and easiest-to-use LLM inference and serving engine for LLMs. The project intends to keep pace with the state of the art, be a platform for innovation, and support a breadth of models, modalities, and hardware.

## Values

### Design Values

1. **Top performance**: we treat performance of the system as top priority. We pursue efficiency via continuously monitoring and lowering critical overheads, highly optimized kernels, and publication of benchmarks. Do not leave performance on the table.   
2. **Ease of use**: vLLM should be easy to install, configure, and operate. That means great documentation, fast startup time, clean logs, actionable error message, and extensive monitoring guide. Additionally, many users of vLLM operate on a fork or study the code extensively, we strive to make the code readable and modular.   
3. **Wide coverage** for important models and hardware: vLLM is a platform for frontier model architecture and the most performant accelerators to shine. We make it easy for the community to add and adopt new models and hardware. vLLM \+ PyTorch functions as the narrow waists to avoid complexity.   
4. **Production ready**: vLLM should be continuously tested as it is defined to run 24/7. It should be easy to operate and identify when it becomes unhealthy.   
5. **Extensibility**: vLLM is designed to be a fundamental infrastructure for LLMs. That means the open source codebase will not be able to cover all use cases; we should make sure the code organization and feature design is welcoming 3rd parties to fork and customize. 

### Collaboration Values

1. **Tightly Knit and Fast-Moving**: The vLLM project maintainer team is a relatively small group of developers aligned on the project’s vision, design philosophy, and roadmap. We work closely together to unblock each other, endeavoring to allow the maintainer's work to progress efficiently.  
2. **Technical Governance is for Individuals**: No one "buys their way into” technical governance. When someone becomes a committer, the status stays with the individual for life, regardless of their companies. We award contributions, maintenance, and overall stewardship of the project. 

## Project Maintainers

The project maintainer group is hierarchical in nature. All maintainers have demonstrated sustained and high-quality contributions to the project and are expected to have a strong bias toward the project’s design philosophy.

### Lead Core Maintainer (BDFL)

The lead core maintainer \- also known as the BDFL \- is the catch-all decision-maker for the project. The role is currently held by a small committee composed of @WoosukKown, @zhuohan123, @simon-mo, and @youkaichao, according to a division of responsibilities that has served the project very well to date.

The responsibilities of the lead core maintainer are:

* Making decisions where consensus among maintainers cannot be reached.  
* Approving or removing core maintainers.  
* Adopting changes to the project’s technical governance.  
* Making major changes to the technical direction or scope change of vLLM itself.  
* Changing the scope of the vLLM project \- e.g., the addition or removal of new sub-projects in the GitHub org.  
* Defining the project’s release strategy

The lead core maintainer will continue to serve in the role until they choose to step down. In the event of all of them stepping down, a new lead core maintainer will be chosen by the core maintainers. Ideally, a natural successor will emerge well in advance and the existing lead core maintainer will support the successor in the transition. The core maintainer group will resort to a vote only in the event of a sudden departure or obvious disagreement on the appointment of a successor.

### Project Leads

The lead core maintainer convenes a Planning Committee that meets weekly in order to ensure close coordination on the project’s current priorities as described in the roadmap. The goal is coordinating resources to ensure there’s no competing priorities and engineering resources are properly allocated to priorities. Current leads are: @WoosukKown, @zhuohan123, @simon-mo, @youkaichao, @robertshaw2-redhat, @tlrmchlsmth, @mgoin, @njhill, @ywang96, @houseroad, @yeqcharlotte, @comaniac, @ApostaC

The leads actively work with model providers, hardware vendors, and key users of vLLM to ensure the project is on the right track.

### Committers and Area Owners

Committers have write access to the repository and have the ability to merge pull requests. When someone becomes committers, it is typically they have deep expertise and stewardship over a certain area of the code base, and overall being helpful around the community. 

Area owners are responsible for the subsystem within the vLLM code base. They are the decision makers and individuals responsible for the evolution of those components. In particular, they will review PRs, refactor codebase, monitor tests, and ensure its compatibility with other components. 

Not all committers are area owners. But all area owners are senior committers. Committers are expected to not only cover their area, but also maintain the project as they see fit (for example, help others review PRs, address issues, answer common questions, improve documentations). 

For a full list of committers and their respective areas, see \[committers\] page. 

#### Nomination Process

Committers can be nominated by any committers via a mailing list private to all committers. It is done via a 3 step process:

1. Nominate: a committer nominates someone from the community, citing their accomplishments and deep area expertise.   
2. Vote: the committer groups voice their support or concerns. If there are concerns, the process may not move forward. Currently, we trust the judgement among the groups to decide the bar.  
3. Confirmation: once the committers agree, the nominator (if available) will send invitation to serve as committer to the nominee, the nominees is expected to submit a PR to update their areas of ownership in CODEOWNERS, then administrators (see Lead Core Maintainers) will assign appropriate permission via GitHub. Committers will also be added to a mailing list and Slack channel. 

We define the bar of committerships as following

* Area expertise: deep involvement in the design, implementation, and maintenance of a given part of the vLLM code base.  
* Community: support users who open issues, shepherd relevant PRs, work with other contributors to enhance the overall direction of vLLM.  
* Vision and mentorship: with a good design taste and coding style, the committer will be able to help maintain the readability of the codebase and teach others to write clean and maintainable for in vLLM. 

While there isn’t a quantitative bar, past committers have

* Submitted 30+ PRs  
* High quality reviews of 10+ contributor PRs  
* Addressed multiple issues and questions from community in issues/forums/Slack  
* Lead concentrated efforts on RFCs and implementation of it

### Working Groups

vLLM runs informal working groups such as CI, CI infrastructure, torch compile, and startup ux. These can be loosely tracked via `#sig-` channels in vLLM Slack. Some groups have regular sync meetings. 

### Advisory Board

vLLM project leads consultation with an informal advisory board that is composed of model providers, hardware vendors, and ecosystem partners. The board includes but not limits to NVIDIA, AMD, AWS, Mistral, ,,,. 

## Process

### Project Roadmap

The Project Leads produce a quarterly roadmap. This roadmap clarifies the current priorities being worked on by those most heavily involved in the project. Topics not included in the roadmap are not excluded but may receive less attention from scarce review bandwidth.Roadmaps are published as a GitHub issue. The latest roadmap can be accessed via [https://roadmap.vllm.ai/](https://roadmap.vllm.ai/). 

### Decision Making

Technical decision-making happens in Slack and GitHub. The project uses RFC issues and design docs in PRs for significant changes. While significant collaboration and discussion on particular topics may well happen outside of GitHub, the project endeavours to ensure there is a public record of each significant change, including the original problem statement, the rationale for the chosen approach, and the alternative approaches considered.

### Merging Code

It is common for contributors and project maintainers to work closely together on code changes. This is most often the case for contributor groups from the same company or organization, but it can also be true for diverse contributors collaborating on a specific area. The project has an informal understanding that project maintainers in such a situation should allow an appropriate opportunity for other project maintainers to review a change, depending on its significance.

### Slack

All community members are encouraged to join \#announcements, \#questions, and \#introduction.

There are \#sig- and \#feat- channels for discussion and coordination around specific topics.

The project maintainer group also uses a private channel for high-bandwidth collaboration.

### Meetings

In addition to the Planning Committee meeting described above, there is a Weekly Contributor Sync meeting that is structured as a standup-style update on progress, blockers, and plans. Any established contributor can request an invite to the meeting [here](https://zoom-lfx.platform.linuxfoundation.org/meeting/92292660775?password=60c32ccb-3dde-457b-9cf3-dff16bc1c116).

### In-Person Meetups

The vLLM project maintainer team understands the importance of interpersonal trust and that spending time physically together greatly accelerates the development of that trust. We endeavour to come together at [our regular meetups](https://docs.vllm.ai/en/v0.5.5/community/meetups.html), which are usually held in San Francisco.

