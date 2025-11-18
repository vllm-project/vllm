# System Design: [System Name]

<!--
INSTRUCTIONS FOR USING THIS TEMPLATE:
1. Replace all [bracketed] sections with actual content
2. Be specific about requirements and constraints
3. Include multiple design alternatives where applicable
4. Provide detailed diagrams (ASCII art or references)
5. Address scalability, reliability, and performance
6. Consider trade-offs explicitly
7. Include real-world examples and references
-->

<!--
METADATA:
Difficulty: [Entry-Level/Mid-Level/Senior/Staff]
Domain: [Web Services/Data Systems/Infrastructure/etc.]
Estimated Interview Time: [45-60 minutes]
Key Concepts: [Concept1, Concept2, Concept3]
Related Designs: [Links to similar systems]
-->

## Problem Statement

**Design:** [One-sentence description of the system to design]

**Real-World Context:**
[2-3 paragraphs explaining the real-world scenario, business context, and why this system is needed]

**Example Use Cases:**
1. [Concrete use case 1]
2. [Concrete use case 2]
3. [Concrete use case 3]

---

## Step 1: Requirements Gathering

<!--
Interview Tip: Always start by clarifying requirements.
Never jump straight to design.
-->

### Functional Requirements

**Core Features (Must Have):**

1. **[Feature 1]:** [Detailed description]
   - [Sub-requirement 1.1]
   - [Sub-requirement 1.2]

2. **[Feature 2]:** [Detailed description]
   - [Sub-requirement 2.1]
   - [Sub-requirement 2.2]

3. **[Feature 3]:** [Detailed description]
   - [Sub-requirement 3.1]
   - [Sub-requirement 3.2]

**Secondary Features (Nice to Have):**

1. [Feature 1]: [Description]
2. [Feature 2]: [Description]
3. [Feature 3]: [Description]

**Out of Scope:**

- [Explicitly state what is NOT included]
- [This helps bound the problem]
- [Clarifies focus areas]

### Non-Functional Requirements

**Performance:**
- **Latency:** [e.g., p99 latency < 100ms for reads]
- **Throughput:** [e.g., 10,000 requests per second]
- **Response Time:** [e.g., 95% of requests < 200ms]

**Scalability:**
- **Users:** [e.g., 100M active users]
- **Data Volume:** [e.g., 1TB of data per day]
- **Growth Rate:** [e.g., 20% YoY growth]

**Availability:**
- **Uptime SLA:** [e.g., 99.99% uptime (52 minutes downtime/year)]
- **Disaster Recovery:** [e.g., RPO < 1 hour, RTO < 4 hours]

**Consistency:**
- **Consistency Model:** [Strong/Eventual/Causal]
- **Justification:** [Why this model is appropriate]

**Reliability:**
- **Fault Tolerance:** [Requirements for handling failures]
- **Data Durability:** [e.g., 99.999999999% durability]

**Security:**
- **Authentication:** [Requirements]
- **Authorization:** [Requirements]
- **Data Protection:** [Encryption, PII handling]

**Other:**
- **Cost Constraints:** [Budget considerations]
- **Compliance:** [GDPR, HIPAA, etc.]
- **Observability:** [Monitoring, logging requirements]

---

## Step 2: Capacity Planning and Estimations

<!--
Interview Tip: Show you can think quantitatively about systems.
Use back-of-envelope calculations.
-->

### Traffic Estimates

**Assumptions:**
- Total users: [X million]
- Daily active users (DAU): [Y million]
- Monthly active users (MAU): [Z million]

**Read/Write Ratio:**
- Read operations: [X%]
- Write operations: [Y%]
- Ratio: [X:Y]

**Request Rate:**

```
DAU: [Y million]
Average requests per user per day: [N]
Total requests per day: [Y million × N] = [total]
Requests per second (QPS): [total / 86,400] ≈ [X] QPS
Peak QPS (assume 2x average): [2X] QPS
```

### Storage Estimates

**Data Model:**

Single [entity] record:
```
- Field 1: [size] bytes
- Field 2: [size] bytes
- Field 3: [size] bytes
- Metadata: [size] bytes
Total per record: [total] bytes ≈ [X] KB
```

**Storage Calculation:**

```
Records per day: [N]
Storage per day: [N × X KB] = [Y] GB/day
Storage per month: [Y × 30] = [Z] GB/month
Storage per year: [Z × 12] = [A] TB/year

With replication factor of [R]: [A × R] TB/year
With backup and overhead (30%): [A × R × 1.3] TB/year
```

### Bandwidth Estimates

**Incoming (Write):**

```
Data written per day: [X] GB
Bandwidth: [X GB / 86,400 seconds] ≈ [Y] MB/s
Peak bandwidth (2x): [2Y] MB/s
```

**Outgoing (Read):**

```
Data read per day: [X] GB
Bandwidth: [X GB / 86,400 seconds] ≈ [Y] MB/s
Peak bandwidth (2x): [2Y] MB/s
```

### Memory Estimates (for caching)

**Caching Strategy:** Cache [X]% of hot data

```
Total data: [X] TB
Hot data (20% of total): [0.2X] TB = [Y] GB
Cache with overhead: [Y × 1.2] GB
```

**Summary Table:**

| Metric | Estimate |
|--------|----------|
| QPS (average) | [X] |
| QPS (peak) | [Y] |
| Storage (per year) | [Z] TB |
| Incoming bandwidth | [A] MB/s |
| Outgoing bandwidth | [B] MB/s |
| Cache size | [C] GB |

---

## Step 3: High-Level Design

<!--
Interview Tip: Start with a simple design, then iterate.
Draw diagrams. Think about data flow.
-->

### System Architecture Diagram

```
┌──────────────┐
│   Clients    │
│ (Web/Mobile) │
└──────┬───────┘
       │
       ▼
┌─────────────────────────────────────────────┐
│          Load Balancer (Layer 7)            │
└──────────────────┬──────────────────────────┘
                   │
       ┌───────────┴───────────┐
       ▼                       ▼
┌─────────────┐         ┌─────────────┐
│   API       │         │   API       │
│  Server 1   │   ...   │  Server N   │
└──────┬──────┘         └──────┬──────┘
       │                       │
       └───────────┬───────────┘
                   │
       ┌───────────┴───────────────────┐
       │                               │
       ▼                               ▼
┌─────────────┐                 ┌─────────────┐
│   Cache     │                 │  Database   │
│   (Redis)   │                 │  (Primary)  │
└─────────────┘                 └──────┬──────┘
                                       │
                                ┌──────┴──────┐
                                ▼             ▼
                         ┌───────────┐ ┌───────────┐
                         │ Database  │ │ Database  │
                         │ (Replica) │ │ (Replica) │
                         └───────────┘ └───────────┘
```

### Core Components

#### 1. [Component Name - e.g., API Gateway]

**Responsibility:**
- [Responsibility 1]
- [Responsibility 2]

**Technology Choices:**
- [Technology option 1]: [Pros/cons]
- [Technology option 2]: [Pros/cons]

**Chosen:** [Technology X] because [justification]

---

#### 2. [Component Name - e.g., Application Servers]

**Responsibility:**
- [Responsibility 1]
- [Responsibility 2]

**Technology Choices:**
- [Technology option 1]: [Pros/cons]
- [Technology option 2]: [Pros/cons]

**Chosen:** [Technology X] because [justification]

---

#### 3. [Component Name - e.g., Database]

**Responsibility:**
- [Responsibility 1]
- [Responsibility 2]

**Technology Choices:**

| Database | Pros | Cons | Use Case |
|----------|------|------|----------|
| [Option 1] | [Pros] | [Cons] | [When to use] |
| [Option 2] | [Pros] | [Cons] | [When to use] |
| [Option 3] | [Pros] | [Cons] | [When to use] |

**Chosen:** [Technology X] because [justification]

---

#### 4. [Component Name - e.g., Caching Layer]

**Responsibility:**
- [Responsibility 1]
- [Responsibility 2]

**Caching Strategy:**
- **Cache-aside:** [When used]
- **Write-through:** [When used]
- **Write-behind:** [When used]

**Chosen:** [Strategy X] because [justification]

---

### Data Flow

#### Read Path

```
1. Client sends request
   └─> Load Balancer
       └─> API Server
           └─> Check Cache
               ├─> Cache Hit: Return data
               └─> Cache Miss:
                   └─> Query Database
                       └─> Update Cache
                           └─> Return data
```

**Pseudocode:**

```python
def handle_read_request(request):
    """
    Handle read request with caching.
    """
    # 1. Validate request
    if not validate(request):
        return error_response("Invalid request")

    # 2. Check cache
    cache_key = generate_cache_key(request)
    cached_data = cache.get(cache_key)

    if cached_data:
        # Cache hit
        log_metric("cache_hit")
        return cached_data

    # 3. Cache miss - query database
    log_metric("cache_miss")
    data = database.query(request.query)

    # 4. Update cache
    cache.set(cache_key, data, ttl=3600)

    # 5. Return response
    return data
```

#### Write Path

```
1. Client sends write request
   └─> Load Balancer
       └─> API Server
           └─> Validate & Process
               └─> Write to Database
                   ├─> Replicate to replicas
                   ├─> Invalidate cache
                   └─> Return confirmation
```

**Pseudocode:**

```python
def handle_write_request(request):
    """
    Handle write request with cache invalidation.
    """
    # 1. Validate request
    if not validate(request):
        return error_response("Invalid request")

    # 2. Write to database
    try:
        result = database.write(request.data)
    except DatabaseError as e:
        log_error(e)
        return error_response("Write failed")

    # 3. Invalidate cache
    cache_keys = get_affected_cache_keys(request.data)
    for key in cache_keys:
        cache.delete(key)

    # 4. Publish event (if using event-driven architecture)
    event_bus.publish("data_updated", result)

    # 5. Return confirmation
    return success_response(result)
```

---

## Step 4: Detailed Component Design

### API Design

#### Endpoint Specifications

**1. [Endpoint Name]**

```
POST /api/v1/[resource]
```

**Request:**

```json
{
  "field1": "value",
  "field2": 123,
  "field3": {
    "nested_field": "value"
  }
}
```

**Response (Success - 200 OK):**

```json
{
  "id": "unique_id",
  "field1": "value",
  "created_at": "2024-01-01T00:00:00Z",
  "status": "success"
}
```

**Response (Error - 400 Bad Request):**

```json
{
  "error": {
    "code": "INVALID_INPUT",
    "message": "Field validation failed",
    "details": {
      "field1": "Required field missing"
    }
  }
}
```

**Rate Limiting:**
- [X] requests per [time period] per user
- [Y] requests per [time period] per IP

---

**2. [Endpoint Name]**

```
GET /api/v1/[resource]/{id}
```

[Similar structure as above]

---

### Database Schema

#### Tables/Collections

**Table: [table_name]**

```sql
CREATE TABLE [table_name] (
    id                  BIGINT PRIMARY KEY AUTO_INCREMENT,
    user_id             BIGINT NOT NULL,
    field1              VARCHAR(255) NOT NULL,
    field2              TEXT,
    field3              TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,

    INDEX idx_user_id (user_id),
    INDEX idx_created_at (created_at),
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
```

**Rationale:**
- [Index 1]: For [query pattern]
- [Index 2]: For [query pattern]
- [Partitioning]: By [field] for [reason]

---

**Table: [another_table]**

[Similar structure]

---

#### Data Model Diagram

```
┌─────────────────┐         ┌─────────────────┐
│     Users       │         │   [Resource]    │
├─────────────────┤         ├─────────────────┤
│ id (PK)         │────┐    │ id (PK)         │
│ username        │    │    │ user_id (FK)    │
│ email           │    └───▶│ field1          │
│ created_at      │         │ field2          │
└─────────────────┘         │ created_at      │
                            └─────────────────┘
```

---

### Caching Strategy

**What to Cache:**

1. **[Data Type 1]:**
   - TTL: [X] seconds
   - Invalidation: [Strategy]
   - Key Format: `[prefix]:[identifier]`

2. **[Data Type 2]:**
   - TTL: [X] seconds
   - Invalidation: [Strategy]
   - Key Format: `[prefix]:[identifier]`

**Cache Eviction Policy:** [LRU/LFU/TTL]

**Cache Warming:** [Strategy for pre-populating cache]

---

## Step 5: Scalability and Performance

### Horizontal Scaling

**Stateless Services:**
- API servers can be scaled horizontally
- Use load balancer to distribute traffic
- Auto-scaling based on [metric - CPU/memory/request rate]

**Stateful Services:**
- Database: Use sharding/replication
- Cache: Use Redis Cluster or Memcached
- Sessions: Store in distributed cache/database

### Database Scaling

#### Read Scaling

**Strategy 1: Read Replicas**

```
┌─────────────┐
│   Primary   │ (writes)
│   Database  │
└──────┬──────┘
       │ (replication)
       ├───────────┬───────────┐
       ▼           ▼           ▼
┌──────────┐ ┌──────────┐ ┌──────────┐
│ Replica  │ │ Replica  │ │ Replica  │
│    1     │ │    2     │ │    3     │
└──────────┘ └──────────┘ └──────────┘
  (reads)      (reads)      (reads)
```

**Configuration:**
- Number of replicas: [N]
- Replication lag acceptable: [X] seconds
- Failover strategy: [Automatic/manual]

---

#### Write Scaling

**Strategy: Database Sharding**

**Sharding Key:** [field_name]

**Reasoning:** [Why this key distributes data evenly]

```
┌──────────────────────────────────────────┐
│         Application Layer                │
│      (Determines shard routing)          │
└──────────────┬───────────────────────────┘
               │
      ┌────────┴────────┬──────────┐
      ▼                 ▼          ▼
┌──────────┐      ┌──────────┐  ┌──────────┐
│ Shard 1  │      │ Shard 2  │  │ Shard N  │
│ (Key: 0- │      │ (Key: X- │  │ (Key: Y- │
│  X)      │      │  Y)      │  │  Z)      │
└──────────┘      └──────────┘  └──────────┘
```

**Shard Distribution Logic:**

```python
def get_shard(user_id, num_shards):
    """
    Determine which shard to route the request to.
    """
    return user_id % num_shards
```

**Challenges:**
- [ ] Rebalancing when adding shards
- [ ] Cross-shard queries
- [ ] Distributed transactions

**Solutions:**
- [Solution to challenge 1]
- [Solution to challenge 2]
- [Solution to challenge 3]

---

### Caching Strategy for Scale

**Multi-Level Caching:**

```
Client
  └─> CDN (Static content)
      └─> Application Cache (L1 - Local)
          └─> Distributed Cache (L2 - Redis)
              └─> Database
```

**Cache Hierarchy:**

| Level | Technology | TTL | What to Cache |
|-------|------------|-----|---------------|
| L1 | Local Memory | 60s | Hot data |
| L2 | Redis | 1hr | Frequently accessed |
| L3 | CDN | 24hr | Static assets |

---

### Performance Optimization

**1. Query Optimization:**
- Use appropriate indexes
- Avoid N+1 queries
- Use batch queries where possible
- Implement pagination

**2. Connection Pooling:**
```python
# Database connection pool configuration
DB_POOL_SIZE = 20
DB_MAX_OVERFLOW = 10
DB_POOL_TIMEOUT = 30
```

**3. Async Processing:**
- Use message queues for non-critical operations
- Implement background jobs for heavy processing

**4. Compression:**
- Compress API responses (gzip)
- Use efficient data formats (protobuf vs JSON)

---

## Step 6: Reliability and Fault Tolerance

### Failure Modes and Mitigation

**1. API Server Failure**

**Impact:** Service degradation
**Mitigation:**
- Run multiple instances (N+2)
- Health checks and auto-recovery
- Circuit breakers to prevent cascade failures

```python
from circuitbreaker import circuit

@circuit(failure_threshold=5, recovery_timeout=60)
def call_external_service():
    """Call with circuit breaker protection."""
    pass
```

---

**2. Database Failure**

**Impact:** Data unavailability, potential data loss
**Mitigation:**
- Primary-replica setup with automatic failover
- Regular backups (point-in-time recovery)
- Multi-AZ deployment

**Failover Process:**

```
1. Health check detects primary failure
2. Promote replica to primary (30-60 seconds)
3. Update DNS/connection strings
4. Redirect traffic to new primary
5. Investigate and repair failed instance
```

---

**3. Cache Failure**

**Impact:** Increased database load, higher latency
**Mitigation:**
- Cache cluster with multiple nodes
- Gradual cache warming on restart
- Rate limiting to protect database

**Mitigation Code:**

```python
def get_with_fallback(key):
    """Get from cache with database fallback."""
    try:
        return cache.get(key)
    except CacheError:
        log_error("Cache unavailable, falling back to DB")
        return database.get(key)
```

---

**4. Network Partition**

**Impact:** Service inconsistency
**Mitigation:**
- Implement retry logic with exponential backoff
- Use idempotency keys
- Design for eventual consistency

---

### Disaster Recovery

**Backup Strategy:**
- Full backup: [frequency - daily/weekly]
- Incremental backup: [frequency - hourly]
- Transaction log backup: [frequency - every 15 min]
- Retention period: [duration - 30 days]

**Recovery Procedures:**

```markdown
1. Identify failure scope and impact
2. Assess recovery options (failover vs restore)
3. Execute recovery plan
4. Verify data integrity
5. Restore normal operations
6. Post-mortem analysis
```

---

## Step 7: Security

### Authentication and Authorization

**Authentication:**
- Method: [OAuth 2.0/JWT/API Keys]
- Token expiration: [time period]
- Refresh token rotation: [Yes/No]

**Authorization:**
- Model: [RBAC/ABAC]
- Permissions: [List of permission types]

**Implementation:**

```python
from functools import wraps

def require_auth(permission):
    """Decorator for endpoint authorization."""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            token = request.headers.get('Authorization')
            if not validate_token(token):
                return error_response("Unauthorized", 401)

            user = get_user_from_token(token)
            if not user.has_permission(permission):
                return error_response("Forbidden", 403)

            return f(*args, **kwargs)
        return decorated_function
    return decorator


@app.route('/api/v1/resource', methods=['POST'])
@require_auth('resource:create')
def create_resource():
    """Create resource endpoint."""
    pass
```

### Data Protection

**Encryption:**
- **In Transit:** TLS 1.3
- **At Rest:** AES-256
- **Sensitive Fields:** Application-level encryption

**PII Handling:**
- Tokenization of sensitive data
- Data masking in logs
- Compliance with regulations (GDPR, CCPA)

### Rate Limiting

**Strategy:** Token bucket algorithm

```python
class RateLimiter:
    """Token bucket rate limiter."""

    def __init__(self, rate, capacity):
        self.rate = rate  # tokens per second
        self.capacity = capacity  # max tokens
        self.tokens = capacity
        self.last_update = time.time()

    def allow_request(self):
        """Check if request is allowed."""
        now = time.time()
        elapsed = now - self.last_update

        # Add tokens based on elapsed time
        self.tokens = min(
            self.capacity,
            self.tokens + elapsed * self.rate
        )
        self.last_update = now

        if self.tokens >= 1:
            self.tokens -= 1
            return True
        return False
```

**Limits:**
- Per user: [X] requests/minute
- Per IP: [Y] requests/minute
- Global: [Z] requests/second

---

## Step 8: Monitoring and Observability

### Key Metrics

**Application Metrics:**

| Metric | Threshold | Action |
|--------|-----------|--------|
| Request rate | > [X] QPS | Scale up |
| Error rate | > [Y]% | Alert |
| p99 latency | > [Z]ms | Investigate |
| CPU usage | > 80% | Scale up |
| Memory usage | > 85% | Scale up/investigate |

**Infrastructure Metrics:**

- Database connections
- Cache hit/miss ratio
- Queue depth
- Network throughput

### Logging Strategy

**Log Levels:**
- ERROR: Critical issues requiring immediate attention
- WARN: Potential issues
- INFO: Important business events
- DEBUG: Detailed debugging information

**Structured Logging:**

```python
import logging
import json

def log_request(request_id, user_id, endpoint, duration):
    """Log request with structured format."""
    log_data = {
        "timestamp": time.time(),
        "request_id": request_id,
        "user_id": user_id,
        "endpoint": endpoint,
        "duration_ms": duration,
        "level": "INFO"
    }
    logging.info(json.dumps(log_data))
```

**Log Aggregation:**
- Tool: [ELK Stack/Splunk/CloudWatch]
- Retention: [X] days

### Alerting

**Alert Conditions:**

1. **Critical:**
   - Service down
   - Database connection failures
   - Error rate > [X]%

2. **Warning:**
   - High latency
   - Resource usage > 80%
   - Cache hit rate < [Y]%

**Notification Channels:**
- PagerDuty for critical alerts
- Slack for warnings
- Email for daily summaries

---

## Step 9: Cost Optimization

### Cost Breakdown

| Component | Monthly Cost | Notes |
|-----------|--------------|-------|
| Compute (API servers) | $[X] | [N] instances |
| Database | $[Y] | [Size] with [replicas] |
| Cache | $[Z] | [Size] cluster |
| Storage | $[A] | [X] TB |
| Network | $[B] | [Bandwidth] |
| CDN | $[C] | [Traffic volume] |
| **Total** | **$[Total]** | |

### Optimization Strategies

1. **Compute:**
   - Use auto-scaling to match demand
   - Reserved instances for baseline capacity
   - Spot instances for batch processing

2. **Storage:**
   - Implement data lifecycle policies
   - Use cheaper storage tiers for old data
   - Compress data

3. **Network:**
   - CDN for static content
   - Data compression
   - Optimize API payload sizes

---

## Step 10: Trade-offs and Alternatives

### Design Decisions

**Decision 1: [SQL vs NoSQL]**

| Option | Pros | Cons | Chosen? |
|--------|------|------|---------|
| SQL (PostgreSQL) | Strong consistency, ACID, relationships | Harder to scale horizontally | ✓ |
| NoSQL (MongoDB) | Easy horizontal scaling, flexible schema | Weaker consistency guarantees | ✗ |

**Reasoning:** [Detailed justification for choice]

---

**Decision 2: [Caching Strategy]**

| Option | Pros | Cons | Chosen? |
|--------|------|------|---------|
| Cache-aside | Simple, flexible | Potential stale data | ✓ |
| Write-through | Always fresh data | Higher write latency | ✗ |
| Write-behind | Lower write latency | Risk of data loss | ✗ |

**Reasoning:** [Detailed justification]

---

### Alternative Architectures

#### Alternative 1: Microservices Architecture

**Approach:**
[Description of microservices approach]

**Pros:**
- Independent scaling
- Technology flexibility
- Fault isolation

**Cons:**
- Increased complexity
- Network overhead
- Distributed transactions

**When to Use:**
[Scenarios where this is better]

---

#### Alternative 2: Event-Driven Architecture

**Approach:**
[Description with diagram]

```
[Service A] ─┐
             │
[Service B] ─┼─> [Message Queue] ─┬─> [Service X]
             │                     │
[Service C] ─┘                     ├─> [Service Y]
                                   │
                                   └─> [Service Z]
```

**Pros:**
- Loose coupling
- Scalability
- Resilience

**Cons:**
- Eventual consistency
- Debugging complexity
- Message ordering challenges

**When to Use:**
[Scenarios where this is better]

---

## Step 11: Future Enhancements

### Phase 2 Features

1. **[Feature 1]:**
   - Description: [What it adds]
   - Impact: [How it changes the system]
   - Complexity: [High/Medium/Low]

2. **[Feature 2]:**
   - Description: [What it adds]
   - Impact: [How it changes the system]
   - Complexity: [High/Medium/Low]

### Scaling for 10x Growth

**Challenges at 10x scale:**
- [Challenge 1]
- [Challenge 2]
- [Challenge 3]

**Solutions:**
- [Solution 1]
- [Solution 2]
- [Solution 3]

---

## Interview Discussion Points

### Questions to Anticipate

1. **"How would you handle [scenario]?"**
   - Answer: [Approach]

2. **"What if we need to support [requirement]?"**
   - Answer: [Modifications needed]

3. **"How would you optimize for [metric]?"**
   - Answer: [Optimization strategy]

### What Interviewers Look For

- [ ] Clear requirement gathering
- [ ] Structured thinking
- [ ] Consideration of trade-offs
- [ ] Scalability awareness
- [ ] Practical experience
- [ ] Communication skills
- [ ] Diagrams and visualization
- [ ] Depth in areas of expertise

---

## References and Resources

### Similar Systems

- [System 1](link): [How it's similar/different]
- [System 2](link): [How it's similar/different]

### Technical Resources

- [Resource 1](link): [What it covers]
- [Resource 2](link): [What it covers]

### Books and Papers

- [Book/Paper 1](link): [Key concepts]
- [Book/Paper 2](link): [Key concepts]

---

## Appendix

### Appendix A: Complete API Specification

[Full API documentation]

### Appendix B: Database Queries

[Common queries with optimizations]

### Appendix C: Configuration Examples

[Sample configuration files]

---

**Design Version:** 1.0
**Created:** [YYYY-MM-DD]
**Last Updated:** [YYYY-MM-DD]
**Interview Level:** [Level]
**Estimated Discussion Time:** [X] minutes
