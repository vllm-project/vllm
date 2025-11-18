# Scenario 07: High-Availability LLM Service

## Problem Statement

Design a high-availability LLM serving system that achieves 99.99% uptime (< 1 hour downtime/year) with zero-downtime deployments, multi-region failover, and disaster recovery capabilities.

## Requirements

### Functional Requirements
1. **Multi-Region Deployment:** Active-active or active-passive
2. **Health Monitoring:** Comprehensive health checks
3. **Automatic Failover:** Detect and route around failures
4. **Zero-Downtime Deployment:** Rolling updates, blue-green
5. **Disaster Recovery:** Backup and restore procedures

### Non-Functional Requirements
1. **Availability:** 99.99% (four nines)
2. **RTO:** Recovery Time Objective < 5 minutes
3. **RPO:** Recovery Point Objective < 1 minute
4. **Failover Latency:** < 10 seconds

## Key Challenges
- Detecting failures quickly
- Maintaining state during failover
- Cost of multi-region deployment
- Consistency during updates

## Difficulty: ★★★★★ (Very Hard)
