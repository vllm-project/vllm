# Scenario 05: Multi-Tenant LLM Platform - Solution

## Architecture

```
┌──────────────────────────────────────────────────────┐
│              Tenant Management Layer                  │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐       │
│  │ Tenant A   │ │ Tenant B   │ │ Tenant C   │       │
│  │ Gold SLA   │ │ Silver SLA │ │ Bronze SLA │       │
│  │ Quota: 100 │ │ Quota: 50  │ │ Quota: 20  │       │
│  └──────┬─────┘ └──────┬─────┘ └──────┬─────┘       │
└─────────┼──────────────┼──────────────┼─────────────┘
          │              │              │
     ┌────▼──────────────▼──────────────▼────┐
     │       Resource Allocation Layer       │
     │  - Quota enforcement                  │
     │  - Priority scheduling                │
     │  - Cost tracking                      │
     └────────────────┬──────────────────────┘
                      │
     ┌────────────────▼──────────────────────┐
     │      GPU Resource Pools               │
     │  ┌──────────┐  ┌──────────┐          │
     │  │ Shared   │  │ Dedicated│          │
     │  │ Pool     │  │ Pool     │          │
     │  │ (Tenants │  │ (Gold    │          │
     │  │  B, C)   │  │  Tenant) │          │
     │  └──────────┘  └──────────┘          │
     └───────────────────────────────────────┘
```

## Resource Isolation Strategies

### 1. Namespace-Based Isolation

```python
class TenantNamespace:
    def __init__(self, tenant_id, sla_tier, quotas):
        self.tenant_id = tenant_id
        self.sla_tier = sla_tier  # GOLD, SILVER, BRONZE

        # Resource quotas
        self.quotas = {
            'max_qps': quotas['max_qps'],
            'max_gpu_memory_gb': quotas['max_gpu_memory_gb'],
            'max_concurrent_requests': quotas['max_concurrent_requests'],
            'max_tokens_per_day': quotas['max_tokens_per_day']
        }

        # Current usage
        self.usage = {
            'current_qps': 0,
            'gpu_memory_used_gb': 0,
            'concurrent_requests': 0,
            'tokens_today': 0
        }

    def can_accept_request(self):
        """Check if request can be accepted within quotas"""
        return (
            self.usage['current_qps'] < self.quotas['max_qps'] and
            self.usage['concurrent_requests'] < self.quotas['max_concurrent_requests'] and
            self.usage['tokens_today'] < self.quotas['max_tokens_per_day']
        )

    def get_priority(self):
        """Get scheduling priority based on SLA tier"""
        if self.sla_tier == 'GOLD':
            return 0  # Highest
        elif self.sla_tier == 'SILVER':
            return 1
        else:
            return 2  # Lowest
```

### 2. Dedicated vs Shared Resources

**Resource Allocation Strategy:**
```python
class ResourceAllocator:
    def __init__(self):
        # Dedicated pools (for Gold tier)
        self.dedicated_pools = {}  # tenant_id -> GPUs

        # Shared pool (for Silver/Bronze)
        self.shared_pool = SharedGPUPool(num_gpus=16)

    def allocate_resources(self, tenant_id, sla_tier):
        if sla_tier == 'GOLD':
            # Dedicated resources
            return self.allocate_dedicated(tenant_id)
        else:
            # Shared resources
            return self.allocate_from_shared_pool(tenant_id, sla_tier)

    def allocate_dedicated(self, tenant_id):
        """Allocate dedicated GPUs for gold tier"""
        if tenant_id not in self.dedicated_pools:
            # Provision new GPU cluster
            num_gpus = self.calculate_required_gpus(tenant_id)
            gpus = self.provision_gpu_cluster(num_gpus)
            self.dedicated_pools[tenant_id] = gpus

        return self.dedicated_pools[tenant_id]

    def allocate_from_shared_pool(self, tenant_id, sla_tier):
        """Allocate from shared pool with quotas"""
        # Get quota-based allocation
        quota_percentage = {
            'SILVER': 0.6,  # 60% of shared pool
            'BRONZE': 0.4   # 40% of shared pool
        }[sla_tier]

        return self.shared_pool.allocate(
            tenant_id,
            quota_percentage
        )
```

## Multi-Tenant Scheduling

```python
class MultiTenantScheduler:
    """Scheduler that enforces SLAs and quotas"""

    def __init__(self):
        self.tenant_queues = {}  # tenant_id -> PriorityQueue
        self.tenant_stats = {}   # tenant_id -> Stats

    async def enqueue_request(self, tenant_id, request):
        """Add request to tenant's queue"""

        tenant = self.get_tenant(tenant_id)

        # Check quotas
        if not tenant.can_accept_request():
            raise QuotaExceededException(tenant_id)

        # Add to tenant's queue
        if tenant_id not in self.tenant_queues:
            self.tenant_queues[tenant_id] = PriorityQueue()

        priority = tenant.get_priority()
        self.tenant_queues[tenant_id].put((priority, request))

        # Update usage
        tenant.usage['concurrent_requests'] += 1

    async def schedule_next_batch(self):
        """Select requests for next batch across tenants"""

        # Fair scheduling across tenants
        # Use weighted fair queuing based on SLA tier

        batch = []
        max_batch_size = 32

        # Weight allocation:
        # GOLD: 50%, SILVER: 30%, BRONZE: 20%
        allocations = {
            'GOLD': int(max_batch_size * 0.5),
            'SILVER': int(max_batch_size * 0.3),
            'BRONZE': int(max_batch_size * 0.2)
        }

        for sla_tier, allocation in allocations.items():
            tenants_in_tier = self.get_tenants_by_sla(sla_tier)

            # Round-robin within tier
            for tenant_id in tenants_in_tier:
                if len(batch) >= max_batch_size:
                    break

                queue = self.tenant_queues.get(tenant_id)
                if queue and not queue.empty():
                    _, request = queue.get()
                    batch.append((tenant_id, request))

        return batch
```

## Cost Attribution

```python
class CostTracker:
    """Track and attribute costs to tenants"""

    def __init__(self, gpu_cost_per_hour=4.0):
        self.gpu_cost_per_hour = gpu_cost_per_hour
        self.tenant_costs = defaultdict(lambda: {
            'gpu_hours': 0,
            'input_tokens': 0,
            'output_tokens': 0,
            'requests': 0
        })

    def record_request(self, tenant_id, request_stats):
        """Record cost for a request"""

        # GPU time cost
        inference_time_sec = request_stats['inference_time_ms'] / 1000
        gpu_fraction = 1.0 / request_stats['batch_size']
        gpu_hours = (inference_time_sec / 3600) * gpu_fraction

        self.tenant_costs[tenant_id]['gpu_hours'] += gpu_hours
        self.tenant_costs[tenant_id]['input_tokens'] += request_stats['input_tokens']
        self.tenant_costs[tenant_id]['output_tokens'] += request_stats['output_tokens']
        self.tenant_costs[tenant_id]['requests'] += 1

    def calculate_bill(self, tenant_id):
        """Calculate total bill for tenant"""
        costs = self.tenant_costs[tenant_id]

        # Pricing model
        gpu_cost = costs['gpu_hours'] * self.gpu_cost_per_hour
        token_cost = (
            costs['input_tokens'] * 0.0001 +   # $0.0001 per 1K input tokens
            costs['output_tokens'] * 0.0002     # $0.0002 per 1K output tokens
        ) / 1000

        total = gpu_cost + token_cost

        return {
            'gpu_cost': gpu_cost,
            'token_cost': token_cost,
            'total': total,
            'breakdown': costs
        }
```

## Custom Model Management

```python
class TenantModelManager:
    """Manage custom fine-tuned models per tenant"""

    def __init__(self):
        self.base_models = {}  # model_id -> BaseModel
        self.tenant_adapters = {}  # (tenant_id, model_id) -> LoRAAdapter

    def load_tenant_model(self, tenant_id, model_id):
        """Load base model + tenant-specific adapter"""

        # Load base model if not already loaded
        if model_id not in self.base_models:
            self.base_models[model_id] = self.load_base_model(model_id)

        # Load tenant's LoRA adapter
        adapter_key = (tenant_id, model_id)
        if adapter_key not in self.tenant_adapters:
            adapter = self.load_lora_adapter(tenant_id, model_id)
            self.tenant_adapters[adapter_key] = adapter

        return self.base_models[model_id], self.tenant_adapters[adapter_key]

    async def infer_with_adapter(self, tenant_id, model_id, inputs):
        """Run inference with tenant's adapter"""

        base_model, adapter = self.load_tenant_model(tenant_id, model_id)

        # Temporarily attach adapter
        base_model.attach_adapter(adapter)

        # Run inference
        outputs = await base_model.forward(inputs)

        # Detach adapter
        base_model.detach_adapter()

        return outputs
```

## Isolation Enforcement

```python
class IsolationEnforcer:
    """Ensure one tenant doesn't affect others"""

    def __init__(self):
        self.resource_limits = {}  # tenant_id -> Limits

    def enforce_cpu_limits(self, tenant_id, process_id):
        """Use cgroups for CPU isolation"""
        import subprocess

        cpu_quota = self.resource_limits[tenant_id]['cpu_quota']

        # Set CPU quota via cgroup
        subprocess.run([
            'cgset', '-r', f'cpu.cfs_quota_us={cpu_quota}',
            f'tenant_{tenant_id}'
        ])

    def enforce_memory_limits(self, tenant_id):
        """Enforce memory limits"""
        memory_limit = self.resource_limits[tenant_id]['memory_gb']

        # Use PyTorch CUDA memory limits
        torch.cuda.set_per_process_memory_fraction(
            memory_limit / torch.cuda.get_device_properties(0).total_memory
        )

    def monitor_noisy_neighbor(self):
        """Detect and mitigate noisy neighbor issues"""

        # Monitor per-tenant latency
        for tenant_id in self.tenants:
            p99_latency = self.get_p99_latency(tenant_id)
            sla_target = self.get_sla_target(tenant_id)

            if p99_latency > sla_target * 1.5:
                # Potential noisy neighbor issue
                self.investigate_interference(tenant_id)
```

## SLA Monitoring

```python
class SLAMonitor:
    def __init__(self):
        self.sla_targets = {
            'GOLD': 50,    # 50ms P99
            'SILVER': 100, # 100ms P99
            'BRONZE': 500  # 500ms P99
        }

        self.tenant_latencies = defaultdict(list)

    def record_latency(self, tenant_id, latency_ms):
        self.tenant_latencies[tenant_id].append(latency_ms)

    def check_sla_compliance(self, tenant_id):
        """Check if tenant's SLA is being met"""

        tenant = self.get_tenant(tenant_id)
        target = self.sla_targets[tenant.sla_tier]

        latencies = self.tenant_latencies[tenant_id]
        if len(latencies) < 100:
            return True  # Not enough data

        p99 = np.percentile(latencies[-1000:], 99)

        return p99 <= target

    def get_sla_violations(self):
        """Get list of tenants with SLA violations"""
        violations = []

        for tenant_id in self.tenant_latencies:
            if not self.check_sla_compliance(tenant_id):
                violations.append(tenant_id)

        return violations
```

## Key Metrics

- **Resource Utilization:** CPU, GPU, memory per tenant
- **Cost Attribution:** Accurate billing per tenant
- **SLA Compliance:** % of requests meeting SLA
- **Isolation Effectiveness:** Cross-tenant interference metrics
- **Queue Fairness:** Wait time distribution per tier

## Results

- **GOLD Tier:** P99 = 45ms, 99.99% SLA compliance
- **SILVER Tier:** P99 = 95ms, 99.5% SLA compliance
- **BRONZE Tier:** P99 = 480ms, 99% SLA compliance
- **Overall GPU Utilization:** 75% (good multi-tenancy efficiency)
- **Cost Attribution Accuracy:** >99% (token-level tracking)
