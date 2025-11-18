# Scenario 01: Evaluation Rubric

## Overall Scoring Framework

**Total Points: 100**

| Category | Points | Description |
|----------|--------|-------------|
| Requirements Clarification | 10 | Asks relevant questions to understand scope |
| High-Level Design | 20 | Proposes clear, scalable architecture |
| Deep Technical Dive | 35 | Demonstrates deep understanding of key components |
| Trade-offs & Alternatives | 15 | Analyzes trade-offs and proposes alternatives |
| Operational Considerations | 10 | Considers monitoring, deployment, debugging |
| Communication | 10 | Clear explanation, structured thinking |

## Detailed Scoring Criteria

### 1. Requirements Clarification (10 points)

**Outstanding (9-10 points):**
- Asks 5+ insightful clarifying questions
- Covers functional, non-functional, and operational requirements
- Identifies ambiguities in requirements
- Proposes reasonable assumptions when information is missing
- Documents assumptions clearly

**Example questions:**
- "What's the expected input/output token distribution?"
- "Do we need to support multi-turn conversations with context?"
- "What's the acceptable failure rate for requests?"
- "Are there peak hours or traffic patterns we should optimize for?"

**Strong (7-8 points):**
- Asks 3-4 relevant questions
- Covers main functional and non-functional requirements
- Makes reasonable assumptions

**Acceptable (5-6 points):**
- Asks 1-2 basic questions
- Understands basic requirements
- May miss some important aspects

**Needs Improvement (0-4 points):**
- Doesn't ask clarifying questions
- Misunderstands requirements
- Makes unrealistic assumptions

### 2. High-Level Design (20 points)

**Components to Include (5 points each):**

a) **API Layer** (5 points)
- ✓ Load balancer / API Gateway
- ✓ Rate limiting
- ✓ Authentication/authorization
- ✓ Request validation

b) **Inference Service** (5 points)
- ✓ Request routing
- ✓ Batching mechanism
- ✓ Multi-model support
- ✓ Proper service abstraction

c) **Model Workers** (5 points)
- ✓ GPU-based inference
- ✓ Model loading strategy
- ✓ Worker pool management
- ✓ Health monitoring

d) **Supporting Services** (5 points)
- ✓ Model registry
- ✓ Monitoring & metrics
- ✓ Logging
- ✓ Configuration management

**Scoring Guidelines:**
- Full points: All components present with clear responsibilities
- Partial credit: Missing 1-2 minor components
- Low score: Missing major components or unclear separation

### 3. Deep Technical Dive (35 points)

This is the most important section. Candidate should demonstrate depth in multiple areas.

#### 3a. Batching Strategy (12 points)

**Outstanding (10-12 points):**
- Explains continuous/iteration-level batching clearly
- Discusses dynamic batch formation
- Compares with static batching
- Mentions specific techniques (e.g., vLLM's approach)
- Can calculate batching impact on latency/throughput
- Discusses fairness and priority handling

**Example explanation:**
"Unlike static batching where we wait for all sequences to complete, continuous batching allows new requests to join as slots free up. For example, if we have a batch of 32 sequences and one completes after 50 iterations, we can immediately add a new request from the queue instead of wasting that GPU slot. This improves both latency and throughput..."

**Strong (7-9 points):**
- Understands difference between static and continuous batching
- Explains basic batching strategy
- Discusses batch size trade-offs

**Acceptable (4-6 points):**
- Mentions batching is important
- Basic understanding of why batching helps
- May not understand continuous batching

**Needs Improvement (0-3 points):**
- Doesn't understand batching
- Proposes request-level batching only
- Can't explain trade-offs

#### 3b. Memory Management & KV Cache (12 points)

**Outstanding (10-12 points):**
- Explains KV cache purpose and memory requirements
- Can calculate memory consumption
- Discusses PagedAttention or similar optimization
- Mentions prefix caching for shared context
- Discusses eviction policies
- Understands memory-throughput trade-offs

**Example calculation:**
"For Llama-70B with 80 layers and 8192 hidden size, each token's KV cache is approximately 2.6MB in FP16. For a batch of 32 sequences with 2048 tokens each, we'd need about 171GB just for KV cache, which exceeds a single A100's capacity. PagedAttention helps by using fixed-size blocks and reducing fragmentation..."

**Strong (7-9 points):**
- Understands KV cache concept
- Can estimate memory requirements
- Mentions at least one optimization technique

**Acceptable (4-6 points):**
- Knows KV cache exists
- Basic understanding of memory constraints
- May not know optimization techniques

**Needs Improvement (0-3 points):**
- Doesn't know what KV cache is
- Can't explain memory requirements
- No understanding of optimizations

#### 3c. Model Parallelism & Scaling (11 points)

**Outstanding (9-11 points):**
- Clearly distinguishes tensor, pipeline, data parallelism
- Explains when to use each
- Discusses communication overhead
- Mentions hardware requirements (NVLink, InfiniBand)
- Can calculate parallelism strategy for given model size
- Proposes hybrid approaches

**Example:**
"For a 70B model, we need tensor parallelism to fit the model in memory. I'd use 4-way TP on GPUs within the same node connected via NVLink for low-latency all-reduce operations. Each forward pass requires all-reduce after every attention and FFN layer, which adds ~15% overhead with NVLink but would be prohibitive with PCIe..."

**Strong (6-8 points):**
- Understands main parallelism types
- Can explain basic trade-offs
- Knows when to use TP vs PP

**Acceptable (3-5 points):**
- Mentions model parallelism
- Basic understanding of scaling
- May confuse different types

**Needs Improvement (0-2 points):**
- Doesn't understand model parallelism
- Can't explain scaling strategy
- Proposes unrealistic approaches

### 4. Trade-offs & Alternatives (15 points)

**Outstanding (13-15 points):**
- Discusses 3+ major trade-offs with clear analysis
- Compares multiple implementation options
- Quantifies trade-offs when possible
- Proposes alternative architectures
- Explains why chosen approach is optimal for requirements

**Key Trade-offs to Discuss:**
1. **Latency vs Throughput:** Batch size impact
2. **Cost vs Performance:** GPU type, quantization, replication
3. **Simplicity vs Flexibility:** Off-the-shelf vs custom
4. **Isolation vs Efficiency:** Dedicated vs shared resources

**Example:**
"For batching, there's a direct trade-off between latency and throughput. Larger batches (64-128) give us 2-3x throughput but increase P99 latency to 200-300ms. Smaller batches (8-16) give 50-80ms latency but half the throughput. Given our 100ms P99 requirement and 1000 QPS target, a batch size of 32 with continuous batching optimally balances both..."

**Strong (9-12 points):**
- Discusses 2-3 trade-offs clearly
- Mentions alternatives
- Basic cost-benefit analysis

**Acceptable (5-8 points):**
- Mentions 1-2 trade-offs
- Aware of alternatives
- Limited analysis

**Needs Improvement (0-4 points):**
- Doesn't discuss trade-offs
- Single solution mindset
- No consideration of alternatives

### 5. Operational Considerations (10 points)

**Outstanding (9-10 points):**
- Comprehensive monitoring strategy with specific metrics
- Clear deployment and rollback plan
- Discusses failure modes and recovery
- Includes cost tracking and optimization
- Mentions debugging and troubleshooting

**Must Cover:**
- ✓ Key metrics to monitor (latency, throughput, GPU utilization, errors)
- ✓ Alerting strategy
- ✓ Deployment approach (blue-green, canary, gradual rollout)
- ✓ Failure handling (GPU OOM, worker crash, model loading failure)

**Strong (7-8 points):**
- Good monitoring strategy
- Basic deployment plan
- Mentions some failure scenarios

**Acceptable (4-6 points):**
- Mentions monitoring
- Basic awareness of operational needs
- Limited detail

**Needs Improvement (0-3 points):**
- Doesn't consider operations
- No monitoring strategy
- Unrealistic deployment plan

### 6. Communication & Presentation (10 points)

**Outstanding (9-10 points):**
- Clear, structured presentation
- Uses diagrams effectively
- Explains concepts at appropriate level
- Actively engages with interviewer
- Manages time well
- Responds well to feedback

**Strong (7-8 points):**
- Generally clear communication
- Logical flow
- Good engagement
- Minor organizational issues

**Acceptable (5-6 points):**
- Understandable but disorganized
- Some unclear explanations
- Adequate engagement

**Needs Improvement (0-4 points):**
- Unclear communication
- Disorganized thoughts
- Poor time management
- Doesn't listen to feedback

## Level-Specific Expectations

### L6/Senior Staff Engineer (85-100 points)

**Must Demonstrate:**
- Deep technical expertise in LLM inference
- Strong understanding of distributed systems
- Proactive identification of edge cases
- Multiple solution alternatives with trade-offs
- Production operations mindset
- Ability to estimate costs and performance
- Leadership in technical discussions

**Distinguishing Factors:**
- References research papers or industry best practices
- Proposes novel optimizations
- Considers second-order effects
- Thinks about team scalability and maintenance

### L5/Staff Engineer (70-84 points)

**Must Demonstrate:**
- Solid technical foundation in LLM serving
- Good understanding of system design principles
- Can discuss main trade-offs
- Reasonable operational awareness
- Can design for scale

**May Need Prompting For:**
- Advanced optimization techniques
- Complex failure scenarios
- Cost optimization strategies
- Multi-region considerations

### L4/Senior Engineer (55-69 points)

**Must Demonstrate:**
- Basic understanding of LLM inference
- Can design simple serving architecture
- Aware of batching and scaling
- Basic monitoring awareness

**Will Need Guidance On:**
- Advanced batching strategies
- Memory management details
- Model parallelism
- Production operations

### L3/Mid-Level Engineer (<55 points)

**Typical Performance:**
- Generic ML serving design
- Limited LLM-specific knowledge
- Needs significant guidance
- May struggle with calculations
- Limited systems thinking

## Common Scoring Scenarios

### Scenario A: Strong Systems, Weak ML
**Profile:** Candidate from distributed systems background, new to LLMs

**Typical Score:** 60-75
- High on: Architecture, scaling, operations
- Low on: LLM specifics, memory management, model parallelism
- Recommendation: L4-L5 depending on learning ability

### Scenario B: Strong ML, Weak Systems
**Profile:** Candidate from ML research, limited production experience

**Typical Score:** 55-70
- High on: Model understanding, optimization techniques
- Low on: Distributed systems, operations, cost awareness
- Recommendation: L4, potentially L5 with operational support

### Scenario C: Well-Rounded
**Profile:** Experience in ML systems/MLOps

**Typical Score:** 75-90
- Solid across all areas
- Good balance of theory and practice
- Recommendation: L5-L6 depending on depth

### Scenario D: Deep Specialist
**Profile:** Heavy experience in LLM inference (e.g., from inference company)

**Typical Score:** 85-100
- Exceptional depth in LLM-specific topics
- References specific implementations
- Proposes optimized solutions
- Recommendation: L6+

## Calibration Examples

### Example 1: Outstanding Response (12/12 on Batching)

**Candidate:**
"I'd implement continuous batching, similar to vLLM's approach. Here's how it differs from traditional batching:

Traditional static batching waits for all sequences in a batch to complete. If we have 32 sequences with lengths [100, 150, 200, ..., 500] tokens, the GPU idles while waiting for the longest sequence.

Continuous batching works at the iteration level. After each decoding step, we check for completed sequences and immediately replace them with new requests from the queue. This has several benefits:

1. **Higher Throughput:** No wasted GPU cycles waiting for stragglers
2. **Better Latency:** New requests don't wait for entire batch to complete
3. **Flexible Batching:** Dynamic batch size based on current load

Implementation-wise, we need:
- Request queue with priority support
- Batch scheduler that runs every iteration
- Memory manager to allocate/deallocate KV cache blocks
- Careful handling of variable-length sequences

The main challenge is memory management. With static batching, we pre-allocate KV cache for max_seq_len. With continuous batching, we need dynamic allocation, which is where PagedAttention helps...

For our 1000 QPS requirement with average 200 output tokens, continuous batching gives us about 2x throughput compared to static batching, allowing us to meet latency targets with fewer GPUs."

**Score: 12/12**
- Excellent explanation of concept
- Compares approaches clearly
- Discusses implementation details
- Quantifies benefits
- Identifies challenges

### Example 2: Acceptable Response (6/12 on Batching)

**Candidate:**
"Batching is important for GPU efficiency. I'd collect requests for maybe 100ms and then process them together in a batch. This way we can use the GPU more efficiently and get better throughput. We might use batch sizes of 32 or 64 depending on memory."

**Score: 6/12**
- Understands batching helps efficiency
- Mentions concrete numbers (100ms, batch size 32-64)
- Missing: Continuous vs static batching, memory implications, trade-offs
- Doesn't understand LLM-specific challenges

### Example 3: Weak Response (2/12 on Batching)

**Candidate:**
"We should batch requests together for better performance. Just collect multiple requests and send to GPU at once."

**Score: 2/12**
- Basic awareness that batching exists
- No understanding of how it works for LLMs
- No specifics on implementation
- No discussion of trade-offs

## Scoring Decision Tree

```
Does candidate ask clarifying questions?
├─ No (0-4 points) → Likely <55 total
└─ Yes
    └─ Does candidate propose reasonable architecture?
        ├─ No (0-10 points) → Likely <60 total
        └─ Yes
            └─ Does candidate understand continuous batching?
                ├─ No (0-6 points) → Likely <70 total (L4)
                └─ Yes
                    └─ Can candidate calculate memory requirements?
                        ├─ No → Likely 70-80 (L5)
                        └─ Yes
                            └─ Does candidate discuss advanced topics unprompted?
                                ├─ No → Likely 75-85 (L5)
                                └─ Yes → Likely 85+ (L6)
```

## Feedback Templates

### For Outstanding Performance (85+)

"Excellent work! You demonstrated deep understanding of LLM inference systems with specific knowledge of continuous batching, memory management, and model parallelism. Your cost calculations were accurate and your operational considerations were comprehensive. You'd be a strong fit for [Senior Staff/Staff] role."

**Strengths:**
- [Specific examples]

**Areas for growth:**
- [Minor points if any]

### For Strong Performance (70-84)

"Strong performance overall. You showed solid system design skills and good understanding of LLM serving. [Specific strength]. To reach the next level, focus on [specific area like cost optimization, advanced batching strategies, etc.]."

### For Acceptable Performance (55-69)

"You demonstrated a reasonable understanding of system design fundamentals. Your architecture covered the main components. To improve, I'd recommend diving deeper into LLM-specific optimizations like continuous batching and memory management. [Specific resources]."

### For Weak Performance (<55)

"You showed some understanding of [basic concepts], but we'd need to see stronger fundamentals in [specific areas]. I'd recommend focusing on understanding how autoregressive generation works, why batching is challenging for LLMs, and basic distributed systems concepts."

## Interviewer Notes Section

Use this section during interview to track:

```
Candidate: _________________
Date: _____________________
Level: ____________________

Quick Scores:
[ ] Requirements (___/10)
[ ] Architecture (___/20)
[ ] Batching (___/12)
[ ] Memory (___/12)
[ ] Parallelism (___/11)
[ ] Trade-offs (___/15)
[ ] Operations (___/10)
[ ] Communication (___/10)

Total: ___/100

Key Strengths:
1. ________________________
2. ________________________
3. ________________________

Development Areas:
1. ________________________
2. ________________________
3. ________________________

Notable Moments:
_________________________
_________________________

Recommendation:
[ ] Strong Hire (85+)
[ ] Hire (70-84)
[ ] Borderline (55-69)
[ ] No Hire (<55)

Level: [ ] L6  [ ] L5  [ ] L4  [ ] L3
```

## Red Flags Checklist

Check any that apply:

- [ ] Didn't ask any clarifying questions
- [ ] Doesn't understand what KV cache is
- [ ] Proposes physically impossible performance numbers
- [ ] No awareness of operational concerns
- [ ] Can't explain trade-offs
- [ ] Doesn't understand batching fundamentals
- [ ] Confuses different types of parallelism
- [ ] Can't do basic calculations
- [ ] Overly complicated design without justification
- [ ] Dismissive of interviewer feedback
- [ ] Poor time management (spent 40min on one component)

**2+ red flags:** Likely No Hire
**1 red flag:** Investigate further, may be recoverable

## Green Flags Checklist

Check any that apply:

- [ ] Asked insightful clarifying questions
- [ ] Used back-of-envelope calculations
- [ ] Referenced specific tools/papers (vLLM, PagedAttention, etc.)
- [ ] Proposed multiple alternatives
- [ ] Quantified trade-offs
- [ ] Thought about cost optimization
- [ ] Comprehensive monitoring strategy
- [ ] Considered failure scenarios
- [ ] Good visual communication (diagrams)
- [ ] Actively engaged with feedback
- [ ] Structured approach to problem-solving
- [ ] Production operations mindset

**5+ green flags:** Likely Strong Hire
**3-4 green flags:** Likely Hire
**1-2 green flags:** Borderline
