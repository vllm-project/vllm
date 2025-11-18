# Scenario 09: A/B Testing Framework for LLMs - Solution

## A/B Testing Architecture

```
┌─────────────────────────────────────────────────────────┐
│                 Traffic Router                           │
│  ┌─────────────────────────────────────────────────┐   │
│  │  Experiment Config: model_v2 vs model_v1        │   │
│  │  Traffic Split: 10% variant, 90% control       │   │
│  │  Randomization: User ID hash-based              │   │
│  └─────────────────────────────────────────────────┘   │
└─────────┬─────────────────────────────┬─────────────────┘
          │                             │
     ┌────▼──────┐               ┌──────▼──────┐
     │ Control   │               │  Variant    │
     │ (v1)      │               │  (v2)       │
     │ 90% traffic│               │ 10% traffic │
     └────┬──────┘               └──────┬──────┘
          │                             │
     ┌────▼─────────────────────────────▼──────┐
     │       Metrics Collection                 │
     │  - Latency per variant                   │
     │  - Quality scores (human eval, LLM judge)│
     │  - Error rates                           │
     │  - Cost per variant                      │
     └──────────────┬───────────────────────────┘
                    │
     ┌──────────────▼───────────────────────────┐
     │     Statistical Analysis Engine          │
     │  - T-tests for latency                   │
     │  - Chi-square for quality                │
     │  - Confidence intervals                  │
     │  - Sequential testing                    │
     └──────────────────────────────────────────┘
```

## Traffic Splitting

```python
class ABTestRouter:
    """Route traffic to experiment variants"""

    def __init__(self, experiment_config):
        self.experiments = {}  # experiment_id -> ExperimentConfig
        self.load_experiment(experiment_config)

    def load_experiment(self, config):
        """Load experiment configuration"""
        self.experiments[config.id] = {
            'id': config.id,
            'variants': {
                'control': {
                    'model': config.control_model,
                    'allocation': 0.9  # 90%
                },
                'treatment': {
                    'model': config.treatment_model,
                    'allocation': 0.1  # 10%
                }
            },
            'status': 'RUNNING',
            'start_time': time.time()
        }

    def assign_variant(self, user_id, experiment_id):
        """Consistently assign user to variant"""

        # Hash user ID for consistent assignment
        hash_value = int(hashlib.md5(
            f"{user_id}:{experiment_id}".encode()
        ).hexdigest(), 16)

        # Normalize to [0, 1]
        assignment_value = (hash_value % 10000) / 10000

        experiment = self.experiments[experiment_id]

        # Assign based on allocation
        if assignment_value < experiment['variants']['control']['allocation']:
            return 'control'
        else:
            return 'treatment'

    async def route_request(self, request):
        """Route request to appropriate variant"""

        # Get active experiments for this request
        experiment_id = self.get_active_experiment(request)

        if not experiment_id:
            # No active experiment, use default model
            return await self.default_model.generate(request)

        # Assign variant
        variant = self.assign_variant(request.user_id, experiment_id)

        # Track assignment
        self.track_assignment(request.id, experiment_id, variant)

        # Route to variant
        model = self.experiments[experiment_id]['variants'][variant]['model']
        response = await model.generate(request)

        # Tag response with experiment metadata
        response.experiment_id = experiment_id
        response.variant = variant

        return response
```

## Metrics Collection

```python
class MetricsCollector:
    """Collect metrics for each variant"""

    def __init__(self):
        self.metrics = defaultdict(lambda: {
            'latencies': [],
            'quality_scores': [],
            'errors': 0,
            'requests': 0,
            'total_cost': 0
        })

    async def record_request(self, request_id, experiment_id, variant, response):
        """Record metrics for request"""

        key = (experiment_id, variant)

        # Latency
        latency_ms = response.latency_ms
        self.metrics[key]['latencies'].append(latency_ms)

        # Quality (if available)
        if response.quality_score:
            self.metrics[key]['quality_scores'].append(response.quality_score)

        # Errors
        if response.error:
            self.metrics[key]['errors'] += 1

        # Cost
        cost = self.calculate_cost(response)
        self.metrics[key]['total_cost'] += cost

        # Count
        self.metrics[key]['requests'] += 1

    def get_summary(self, experiment_id):
        """Get summary statistics"""

        control_key = (experiment_id, 'control')
        treatment_key = (experiment_id, 'treatment')

        control = self.metrics[control_key]
        treatment = self.metrics[treatment_key]

        return {
            'control': {
                'n': control['requests'],
                'p50_latency': np.percentile(control['latencies'], 50),
                'p99_latency': np.percentile(control['latencies'], 99),
                'avg_quality': np.mean(control['quality_scores']),
                'error_rate': control['errors'] / control['requests'],
                'cost_per_request': control['total_cost'] / control['requests']
            },
            'treatment': {
                'n': treatment['requests'],
                'p50_latency': np.percentile(treatment['latencies'], 50),
                'p99_latency': np.percentile(treatment['latencies'], 99),
                'avg_quality': np.mean(treatment['quality_scores']),
                'error_rate': treatment['errors'] / treatment['requests'],
                'cost_per_request': treatment['total_cost'] / treatment['requests']
            }
        }
```

## Statistical Analysis

```python
class StatisticalAnalyzer:
    """Analyze experiment results"""

    def analyze_experiment(self, experiment_id):
        """Perform statistical analysis"""

        summary = self.metrics_collector.get_summary(experiment_id)
        control = summary['control']
        treatment = summary['treatment']

        # 1. Latency comparison (T-test)
        latency_result = self.compare_latency(
            control['latencies'],
            treatment['latencies']
        )

        # 2. Quality comparison
        quality_result = self.compare_quality(
            control['quality_scores'],
            treatment['quality_scores']
        )

        # 3. Error rate comparison (Chi-square)
        error_result = self.compare_error_rates(
            control['errors'], control['requests'],
            treatment['errors'], treatment['requests']
        )

        return {
            'latency': latency_result,
            'quality': quality_result,
            'errors': error_result,
            'recommendation': self.make_recommendation(
                latency_result, quality_result, error_result
            )
        }

    def compare_latency(self, control_latencies, treatment_latencies):
        """T-test for latency difference"""
        from scipy import stats

        # Two-sample t-test
        t_stat, p_value = stats.ttest_ind(
            control_latencies,
            treatment_latencies
        )

        # Effect size (Cohen's d)
        control_mean = np.mean(control_latencies)
        treatment_mean = np.mean(treatment_latencies)
        pooled_std = np.sqrt(
            (np.var(control_latencies) + np.var(treatment_latencies)) / 2
        )
        cohens_d = (treatment_mean - control_mean) / pooled_std

        # Confidence interval
        ci = stats.t.interval(
            0.95,
            len(control_latencies) + len(treatment_latencies) - 2,
            loc=treatment_mean - control_mean,
            scale=pooled_std
        )

        return {
            'control_mean_ms': control_mean,
            'treatment_mean_ms': treatment_mean,
            'difference_ms': treatment_mean - control_mean,
            'p_value': p_value,
            'statistically_significant': p_value < 0.05,
            'effect_size': cohens_d,
            'confidence_interval_95': ci
        }

    def compare_quality(self, control_scores, treatment_scores):
        """Compare quality scores"""

        # For LLM outputs, quality is often measured by:
        # - Human evaluation scores
        # - LLM-as-judge scores
        # - Task-specific metrics (BLEU, ROUGE, etc.)

        from scipy import stats

        # Mann-Whitney U test (non-parametric)
        u_stat, p_value = stats.mannwhitneyu(
            control_scores,
            treatment_scores,
            alternative='two-sided'
        )

        return {
            'control_median': np.median(control_scores),
            'treatment_median': np.median(treatment_scores),
            'p_value': p_value,
            'statistically_significant': p_value < 0.05,
            'winner': 'treatment' if np.median(treatment_scores) > np.median(control_scores) else 'control'
        }

    def make_recommendation(self, latency_result, quality_result, error_result):
        """Make rollout recommendation"""

        # Decision criteria:
        # 1. Quality must not degrade significantly
        # 2. Latency improvement is bonus
        # 3. Error rate must not increase

        if quality_result['p_value'] < 0.05 and quality_result['winner'] == 'control':
            return 'ROLLBACK'  # Quality regression

        if error_result['treatment_error_rate'] > error_result['control_error_rate'] * 1.5:
            return 'ROLLBACK'  # Error rate too high

        if quality_result['p_value'] < 0.05 and quality_result['winner'] == 'treatment':
            return 'ROLLOUT'  # Quality improvement

        if latency_result['statistically_significant'] and latency_result['difference_ms'] < 0:
            return 'ROLLOUT'  # Latency improvement

        return 'CONTINUE'  # Need more data
```

## Quality Measurement

```python
class QualityEvaluator:
    """Evaluate LLM output quality"""

    async def evaluate_response(self, request, response):
        """Evaluate response quality"""

        # Method 1: Human evaluation (gold standard, slow)
        if self.should_sample_for_human_eval():
            score = await self.request_human_eval(request, response)
            return score

        # Method 2: LLM-as-judge (fast, scalable)
        score = await self.llm_judge_eval(request, response)
        return score

    async def llm_judge_eval(self, request, response):
        """Use LLM to judge quality"""

        judge_prompt = f"""
        Evaluate the following AI response on a scale of 1-5:

        User Request: {request.prompt}
        AI Response: {response.text}

        Criteria:
        - Accuracy: Is the response factually correct?
        - Relevance: Does it answer the question?
        - Helpfulness: Is it useful to the user?
        - Coherence: Is it well-structured?

        Provide a score from 1 (poor) to 5 (excellent).
        """

        judge_response = await self.judge_model.generate(judge_prompt)

        # Extract score from response
        score = self.extract_score(judge_response)

        return score

    def should_sample_for_human_eval(self):
        """Sample 1% for human evaluation"""
        return random.random() < 0.01
```

## Gradual Rollout

```python
class GradualRolloutManager:
    """Manage gradual rollout of winning variant"""

    async def rollout(self, experiment_id):
        """Gradually increase traffic to treatment"""

        stages = [
            {'percentage': 0.1, 'duration': 86400},   # 10% for 1 day
            {'percentage': 0.25, 'duration': 86400},  # 25% for 1 day
            {'percentage': 0.5, 'duration': 86400},   # 50% for 1 day
            {'percentage': 1.0, 'duration': 0}        # 100% (complete)
        ]

        for stage in stages:
            # Update traffic allocation
            await self.update_allocation(experiment_id, stage['percentage'])

            # Monitor for stage duration
            await self.monitor_stage(experiment_id, stage['duration'])

            # Check for regressions
            if self.detect_regression(experiment_id):
                await self.rollback(experiment_id)
                return 'ROLLED_BACK'

        # Rollout complete
        await self.finalize_rollout(experiment_id)
        return 'COMPLETE'

    def detect_regression(self, experiment_id):
        """Detect quality or error rate regression"""

        current_metrics = self.get_recent_metrics(experiment_id)

        # Check error rate
        if current_metrics['error_rate'] > 0.05:  # 5%
            logger.error("Error rate regression detected")
            return True

        # Check quality
        if current_metrics['avg_quality'] < 3.0:  # Below acceptable
            logger.error("Quality regression detected")
            return True

        return False
```

## Automatic Rollback

```python
class SafetyMonitor:
    """Monitor experiment safety and trigger rollback"""

    async def monitor(self, experiment_id):
        """Continuous safety monitoring"""

        while self.is_experiment_running(experiment_id):
            metrics = self.get_recent_metrics(experiment_id, window='5min')

            # Check safety thresholds
            if metrics['treatment']['error_rate'] > 0.1:  # 10% errors
                await self.trigger_rollback(experiment_id, reason='high_error_rate')

            if metrics['treatment']['p99_latency'] > 1000:  # 1 second
                await self.trigger_rollback(experiment_id, reason='high_latency')

            await asyncio.sleep(60)  # Check every minute

    async def trigger_rollback(self, experiment_id, reason):
        """Immediate rollback"""

        logger.critical(f"Triggering rollback for {experiment_id}: {reason}")

        # Set allocation to 0% for treatment
        await self.update_allocation(experiment_id, 0.0)

        # Alert team
        await self.alert_oncall(f"Experiment {experiment_id} rolled back: {reason}")
```

## Results

**Experiment Example:** Model v2 vs Model v1

**Metrics After 10,000 Requests:**
```
Control (v1):
- P99 Latency: 120ms
- Avg Quality: 4.2 / 5.0
- Error Rate: 0.5%
- Cost per 1K tokens: $0.15

Treatment (v2):
- P99 Latency: 95ms (21% improvement)
- Avg Quality: 4.4 / 5.0 (4.8% improvement)
- Error Rate: 0.4% (20% improvement)
- Cost per 1K tokens: $0.10 (33% savings)

Statistical Significance:
- Latency: p < 0.001 (highly significant)
- Quality: p < 0.05 (significant)
- Errors: p < 0.1 (marginally significant)

Recommendation: ROLLOUT ✓
```

## Key Takeaways

1. **Hash-based randomization** ensures consistent user experience
2. **LLM-as-judge** enables scalable quality evaluation
3. **Statistical rigor** prevents false positives
4. **Gradual rollout** minimizes risk
5. **Automatic rollback** ensures safety
