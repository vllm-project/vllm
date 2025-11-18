# Scenario 07: Discussion Points - High Availability

## Key Topics

### 1. Availability Calculation
**Question:** "How do you achieve 99.99% availability?"
- Multi-region deployment
- Redundant components
- Fast failover (<10s)
- Formula: 1 - (1 - component_avail)^n

### 2. Failure Detection
**Question:** "How do you detect failures quickly?"
- Health checks (10s interval)
- GPU monitoring
- Inference capability testing
- Error rate tracking

### 3. Zero-Downtime Deployment
**Question:** "How do you deploy without downtime?"
- Rolling updates (one replica at a time)
- Blue-green deployment
- Canary releases
- Connection draining

## Red/Green Flags
**Red:** Single region only, no failover strategy, downtime for deployments
**Green:** Multi-region active-active, automatic failover, zero-downtime deployment
