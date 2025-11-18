# Scenario 08: Cost Optimization for LLM Serving - Solution

## Cost Optimization Strategy

```
┌───────────────────────────────────────────────────────┐
│         Multi-Tier Cost Optimization                  │
│                                                        │
│  Tier 1: Infrastructure Optimization                  │
│  - Spot instances (60% savings)                       │
│  - Reserved instances for baseline                    │
│  - Auto-scaling (right-sizing)                        │
│                                                        │
│  Tier 2: Model Optimization                           │
│  - Quantization (INT4) - 4x memory reduction          │
│  - Smaller models for simple queries                  │
│  - Model distillation                                 │
│                                                        │
│  Tier 3: Request Optimization                         │
│  - Aggressive batching                                │
│  - Request deduplication                              │
│  - Response caching                                   │
└───────────────────────────────────────────────────────┘
```

## Spot Instance Strategy

```python
class SpotInstanceManager:
    """Manage spot instances with fallback"""

    def __init__(self):
        # Mix of instance types
        self.instance_mix = {
            'on_demand_reserved': 2,  # Baseline capacity
            'spot': 4                 # Burst capacity
        }

        self.spot_interruption_handler = SpotInterruptionHandler()

    async def provision_with_spot(self, target_capacity):
        """Provision capacity with spot instances"""

        # Calculate mix
        baseline = min(target_capacity, self.instance_mix['on_demand_reserved'])
        burst = target_capacity - baseline

        instances = []

        # 1. Provision baseline (reserved on-demand)
        for i in range(baseline):
            instance = await self.provision_on_demand('p4d.24xlarge', reserved=True)
            instances.append(instance)

        # 2. Provision burst (spot)
        spot_pools = ['p4d.24xlarge', 'p4de.24xlarge', 'p5.48xlarge']

        for i in range(burst):
            # Diversify across instance types for better availability
            instance_type = spot_pools[i % len(spot_pools)]
            instance = await self.provision_spot(instance_type)
            instances.append(instance)

        return instances

    async def handle_spot_interruption(self, instance_id):
        """Handle spot instance interruption (2-minute warning)"""

        logger.warning(f"Spot interruption warning for {instance_id}")

        # 1. Stop accepting new requests
        await self.drain_instance(instance_id, graceful=True)

        # 2. Checkpoint in-flight requests
        await self.checkpoint_requests(instance_id)

        # 3. Provision replacement (try spot first, fallback to on-demand)
        try:
            replacement = await self.provision_spot_replacement()
        except SpotUnavailableException:
            replacement = await self.provision_on_demand_temporary()

        # 4. Resume checkpointed requests on replacement
        await self.resume_requests(replacement)

# Cost savings:
# Spot: $9.83/hr vs On-Demand: $32.77/hr = 70% savings
# Risk: ~5% interruption rate
# Mitigation: Diversification, graceful handling
```

## Dynamic Scaling for Cost

```python
class CostAwareAutoscaler:
    """Autoscale based on cost and performance"""

    def __init__(self, cost_budget_per_hour=50):
        self.cost_budget = cost_budget_per_hour
        self.current_cost = 0

    async def scale_decision(self, metrics):
        """Decide scaling based on cost constraints"""

        current_qps = metrics['current_qps']
        target_latency = 200  # ms
        current_p99 = metrics['p99_latency_ms']

        # Calculate required capacity
        if current_p99 > target_latency * 1.2:
            # Need to scale up
            additional_capacity_needed = self.calculate_additional_capacity(
                current_qps, current_p99, target_latency
            )

            # Can we afford it?
            additional_cost = additional_capacity_needed * 9.83  # spot price/hr

            if self.current_cost + additional_cost <= self.cost_budget:
                # Scale up with spot
                await self.scale_up_spot(additional_capacity_needed)
            else:
                # At budget limit, optimize instead of scaling
                await self.trigger_aggressive_optimizations()

        elif current_p99 < target_latency * 0.7:
            # Scale down to save cost
            await self.scale_down_opportunistic()

    async def trigger_aggressive_optimizations(self):
        """Optimize when can't scale due to budget"""

        # 1. Increase batch size (latency for throughput)
        self.set_batch_size(64)  # from 32

        # 2. Enable response caching
        self.enable_caching(ttl=3600)

        # 3. Route simple queries to smaller model
        self.enable_model_routing()
```

## Model Optimization for Cost

```python
class ModelCostOptimizer:
    """Optimize model selection and configuration"""

    def __init__(self):
        self.models = {
            '70b_fp16': {
                'cost_per_token': 0.0005,
                'quality_score': 0.95,
                'latency_ms': 100
            },
            '70b_int4': {
                'cost_per_token': 0.0001,
                'quality_score': 0.94,
                'latency_ms': 50
            },
            '13b_int4': {
                'cost_per_token': 0.00002,
                'quality_score': 0.85,
                'latency_ms': 20
            },
            '7b_int4': {
                'cost_per_token': 0.00001,
                'quality_score': 0.75,
                'latency_ms': 10
            }
        }

    def route_to_optimal_model(self, request):
        """Route request to cost-optimal model"""

        # Classify request complexity
        complexity = self.classify_complexity(request)

        if complexity == 'simple':
            # Use 7B model (10x cheaper)
            return '7b_int4'
        elif complexity == 'medium':
            # Use 13B model (5x cheaper than 70B FP16)
            return '13b_int4'
        else:
            # Use 70B INT4 (5x cheaper than FP16, minimal quality loss)
            return '70b_int4'

    def classify_complexity(self, request):
        """Classify request complexity"""

        # Heuristics:
        # - Short prompts (<50 tokens) → simple
        # - Keywords like "summarize", "translate" → medium
        # - Complex reasoning, code → complex

        prompt_len = len(request.prompt_tokens)
        prompt_text = request.prompt_text.lower()

        if prompt_len < 50 and any(kw in prompt_text for kw in ['hello', 'hi', 'what is']):
            return 'simple'
        elif any(kw in prompt_text for kw in ['summarize', 'translate', 'explain']):
            return 'medium'
        else:
            return 'complex'
```

## Response Caching

```python
class ResponseCache:
    """Cache responses to reduce compute cost"""

    def __init__(self, redis_client):
        self.redis = redis_client
        self.ttl = 3600  # 1 hour

    async def get_or_compute(self, request):
        """Check cache before computing"""

        # Generate cache key
        cache_key = self.generate_cache_key(request)

        # Check cache
        cached = await self.redis.get(cache_key)
        if cached:
            logger.info(f"Cache hit for {cache_key}")
            return json.loads(cached)

        # Compute
        response = await self.model.generate(request)

        # Cache if deterministic (temperature=0)
        if request.temperature == 0:
            await self.redis.setex(
                cache_key,
                self.ttl,
                json.dumps(response)
            )

        return response

    def generate_cache_key(self, request):
        """Generate cache key from request"""

        # Include: model, prompt, params
        key_data = {
            'model': request.model_id,
            'prompt': request.prompt_text,
            'params': {
                'temperature': request.temperature,
                'top_p': request.top_p,
                'max_tokens': request.max_tokens
            }
        }

        return hashlib.sha256(
            json.dumps(key_data, sort_keys=True).encode()
        ).hexdigest()

# Cache hit rate: ~30% for typical workloads
# Cost savings: 30% reduction in inference cost
```

## Request Deduplication

```python
class RequestDeduplicator:
    """Deduplicate identical concurrent requests"""

    def __init__(self):
        self.in_flight = {}  # request_hash → Future

    async def deduplicate_and_execute(self, request):
        """Execute request or wait for identical in-flight request"""

        req_hash = self.hash_request(request)

        if req_hash in self.in_flight:
            # Wait for in-flight request
            logger.info(f"Deduplicating request {req_hash}")
            return await self.in_flight[req_hash]

        # Create new future
        future = asyncio.Future()
        self.in_flight[req_hash] = future

        try:
            # Execute
            result = await self.model.generate(request)
            future.set_result(result)
            return result
        finally:
            del self.in_flight[req_hash]
```

## Cost Tracking & Analysis

```python
class CostAnalytics:
    """Track and analyze costs"""

    def calculate_cost_per_request(self, request, response):
        """Calculate actual cost for request"""

        # GPU time cost
        inference_time_sec = response.inference_time_ms / 1000
        gpu_cost_per_sec = 9.83 / 3600  # spot price per second
        gpu_cost = inference_time_sec * gpu_cost_per_sec

        # Token cost (alternative pricing model)
        total_tokens = request.input_tokens + response.output_tokens
        token_cost = total_tokens * 0.00001  # $0.01 per 1K tokens

        return {
            'gpu_cost': gpu_cost,
            'token_cost': token_cost,
            'total_cost': gpu_cost,
            'tokens': total_tokens
        }

    def analyze_cost_breakdown(self):
        """Analyze where costs are going"""

        breakdown = {
            'gpu_compute': 0.65,      # 65% of cost
            'networking': 0.10,       # 10%
            'storage': 0.05,          # 5%
            'monitoring': 0.02,       # 2%
            'other': 0.18             # 18%
        }

        # Optimization opportunities:
        # 1. GPU compute: Use spot, quantization, smaller models
        # 2. Networking: Compress responses, CDN for static assets
        # 3. Storage: Lifecycle policies, compress models
```

## Cost Optimization Results

```python
# Before optimization:
baseline_cost = {
    'gpu_instances': 6 * 32.77 * 730,  # 6 on-demand p4d.24xlarge
    'networking': 2000,
    'storage': 1000,
    'total_monthly': 144,000
}

# After optimization:
optimized_cost = {
    'gpu_instances': (
        2 * 21.18 * 730 +  # 2 reserved instances
        4 * 9.83 * 730 * 0.6  # 4 spot (60% uptime after interruptions)
    ),
    'networking': 1000,  # Compression reduced by 50%
    'storage': 500,      # Model compression
    'total_monthly': 50,000
}

cost_reduction = (144000 - 50000) / 144000  # 65% reduction! (exceeds 50% target)

# Performance maintained:
# - P99 latency: 180ms (< 200ms target ✓)
# - Throughput: No degradation
# - Availability: 99.9% (acceptable)
# - Quality: <1% degradation (INT4 quantization)
```

## Trade-offs

| Optimization | Cost Savings | Performance Impact | Risk |
|--------------|--------------|-------------------|------|
| Spot Instances | 70% | None (with handling) | Medium |
| INT4 Quantization | 75% (memory) | <1% quality loss | Low |
| Model Routing | 50% (avg) | Quality varies by query | Medium |
| Response Caching | 30% (hit rate dependent) | None | Low |
| Aggressive Batching | 40% | +50ms latency | Low |
| **Combined** | **65%** | **P99: 180ms** | **Medium** |

## Key Takeaways

1. **Spot instances** provide massive savings (70%) with proper handling
2. **Quantization (INT4)** reduces cost with minimal quality impact
3. **Model routing** uses right-sized models for each query
4. **Caching** eliminates duplicate compute
5. **Combined optimizations** achieve 65% cost reduction while maintaining performance
