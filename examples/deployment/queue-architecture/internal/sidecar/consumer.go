package sidecar

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"time"

	"github.com/redis/go-redis/v9"
	"github.com/rvo-redplatform/vllm/examples/deployment/queue-architecture/internal/queue"
)

// processJob handles the per-job processing logic: idempotency check, forwarding,
// result writing, and acking. This shared helper is used by both ConsumerLoop and
// ReclaimLoop to ensure identical semantics.
//
// Order matters for crash protection: write the result BEFORE acking -- so a crash
// between those two steps causes at most wasted redundant work, never a lost answer.
//
// Parameters:
//   - ctx: context for cancellation
//   - rdb: Redis client
//   - job: the job to process
//   - entryID: the Redis stream entry ID (for acking)
//   - streamName: name of the Redis stream
//   - groupName: consumer group name
//   - target: base URL of the vLLM instance
//   - resultExpiry: TTL for result keys in Redis
//
// Returns error only for infrastructure failures; per-job errors are logged and
// the function returns nil so the caller can continue processing other jobs.
func processJob(ctx context.Context, rdb *redis.Client, job queue.Job, entryID, streamName, groupName, target string, resultExpiry time.Duration) error {
	// Check if result already exists (idempotency)
	exists, err := ResultExists(ctx, rdb, job.JobID)
	if err != nil {
		log.Printf("ERROR: check result exists for job %s: %v", job.JobID, err)
		return nil // Per-job error, don't fail the loop
	}

	if exists {
		// Result already exists, just ack and continue
		if err := queue.Ack(ctx, rdb, streamName, groupName, entryID); err != nil {
			log.Printf("ERROR: ack job %s: %v", job.JobID, err)
			return nil // Per-job error, don't fail the loop
		}
		return nil
	}

	// Dispatch to the appropriate forwarder based on job.Stream
	if job.Stream {
		// Streaming job: forward to ForwardStreaming
		if err := ForwardStreaming(ctx, rdb, job, target); err != nil {
			log.Printf("ERROR: forward streaming job %s: %v", job.JobID, err)
			return nil // Per-job error, don't fail the loop
		}
	} else {
		// Non-streaming job: forward and write result
		status, headers, body, err := ForwardNonStreaming(ctx, job, target)
		if err != nil {
			log.Printf("ERROR: forward non-streaming job %s: %v", job.JobID, err)
			return nil // Per-job error, don't fail the loop
		}

		// Build the result object
		result := map[string]interface{}{
			"status":  status,
			"headers": headers,
			"body":    string(body),
		}

		// Marshal result to JSON
		resultJSON, err := json.Marshal(result)
		if err != nil {
			log.Printf("ERROR: marshal result for job %s: %v", job.JobID, err)
			return nil // Per-job error, don't fail the loop
		}

		// Write result to Redis with expiry (BEFORE acking for crash protection)
		resultKey := "result:" + job.JobID
		if err := rdb.RPush(ctx, resultKey, resultJSON).Err(); err != nil {
			log.Printf("ERROR: rpush result for job %s: %v", job.JobID, err)
			return nil // Per-job error, don't fail the loop
		}
		if err := rdb.Expire(ctx, resultKey, resultExpiry).Err(); err != nil {
			log.Printf("ERROR: expire result for job %s: %v", job.JobID, err)
			return nil // Per-job error, don't fail the loop
		}
	}

	// Ack the job (after writing result for non-streaming)
	if err := queue.Ack(ctx, rdb, streamName, groupName, entryID); err != nil {
		log.Printf("ERROR: ack job %s: %v", job.JobID, err)
		return nil // Per-job error, don't fail the loop
	}

	return nil
}

// ConsumerLoop reads jobs from the queue, checks for idempotency, dispatches to the
// appropriate forwarder, writes results for non-streaming jobs, and acks.
//
// Order matters for crash protection: write the result BEFORE acking -- so a crash
// between those two steps causes at most wasted redundant work, never a lost answer.
//
// Before claiming each message, this gates on vLLM's real-time load (see
// waitForCapacity in health.go): if vLLM already reports being at or above
// maxConcurrent (running+waiting requests), the loop deliberately holds off
// on reading the next message rather than claiming work it can't make
// timely progress on. This makes the sidecar's concurrency ceiling
// self-adjusting to vLLM's actual measured capacity instead of a blind
// one-at-a-time assumption, and leaves un-claimed backlog visible to KEDA's
// stream-lag scaling metric instead of hidden in a claimed-but-blocked
// consumer.
//
// Error handling:
//   - Infrastructure errors (queue.Read failures) are fatal and return from the loop.
//   - Per-job errors (idempotency check, forwarding, marshaling, result write, ack) are
//     logged with the job ID and the loop continues to process subsequent jobs. Failed
//     jobs remain un-acked and stay in the Pending Entries List for later reclaim/retry.
//
// Parameters:
//   - ctx: context for cancellation
//   - rdb: Redis client
//   - streamName: name of the Redis stream to read from
//   - groupName: consumer group name
//   - consumerName: consumer name within the group
//   - target: base URL of the vLLM instance (e.g., "http://localhost:8000")
//   - resultExpiry: TTL for result keys in Redis
//   - httpClient: HTTP client used for the lightweight /health and /metrics probes
//   - maxConcurrent: max vLLM running+waiting requests before pausing claims (<=0 disables the gate)
//   - capacityPollInterval: how often to re-check vLLM's load while waiting for capacity
func ConsumerLoop(ctx context.Context, rdb *redis.Client, streamName, groupName, consumerName, target string, resultExpiry time.Duration, httpClient *http.Client, maxConcurrent int, capacityPollInterval time.Duration) error {
	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		// Gate on vLLM's real-time load before claiming the next message.
		if err := waitForCapacity(ctx, httpClient, target, maxConcurrent, capacityPollInterval); err != nil {
			return err
		}

		// Read a job from the queue
		job, entryID, err := queue.Read(ctx, rdb, streamName, groupName, consumerName)
		if err != nil {
			return fmt.Errorf("read job: %w", err)
		}

		// Process the job using the shared helper
		if err := processJob(ctx, rdb, job, entryID, streamName, groupName, target, resultExpiry); err != nil {
			return err
		}
	}
}

// ReclaimLoop periodically claims idle jobs from the consumer group and processes them.
// It runs on a ticker with interval reclaimInterval, calling queue.Claim to reclaim
// jobs that have been idle longer than idleThreshold. This implements the crash-protection
// redelivery mechanism: if a sidecar dies mid-job, another sidecar will eventually reclaim
// and complete that job.
//
// The idempotency check in processJob ensures that if the crashed sidecar had already
// written the result before dying, the reclaiming sidecar will see the result exists
// and just ack the job without re-executing it.
//
// Each claimed job is gated through the same waitForCapacity check as ConsumerLoop
// (see health.go) before being forwarded -- reclaim traffic must respect vLLM's
// real-time load just as much as fresh reads do.
//
// Parameters:
//   - ctx: context for cancellation
//   - rdb: Redis client
//   - streamName: name of the Redis stream
//   - groupName: consumer group name
//   - consumerName: consumer name within the group
//   - target: base URL of the vLLM instance
//   - resultExpiry: TTL for result keys in Redis
//   - idleThreshold: minimum idle time before a job is eligible for reclaim
//   - reclaimInterval: how often to run the reclaim check
//   - httpClient: HTTP client used for the lightweight /health and /metrics probes
//   - maxConcurrent: max vLLM running+waiting requests before pausing claims (<=0 disables the gate)
//   - capacityPollInterval: how often to re-check vLLM's load while waiting for capacity
func ReclaimLoop(ctx context.Context, rdb *redis.Client, streamName, groupName, consumerName, target string, resultExpiry, idleThreshold, reclaimInterval time.Duration, httpClient *http.Client, maxConcurrent int, capacityPollInterval time.Duration) error {
	ticker := time.NewTicker(reclaimInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-ticker.C:
			// Claim idle jobs
			claimedJobs, err := queue.Claim(ctx, rdb, streamName, groupName, consumerName, idleThreshold)
			if err != nil {
				log.Printf("ERROR: claim idle jobs: %v", err)
				// Don't return; continue trying on the next tick
				continue
			}

			// Process each claimed job, gating on vLLM's real-time capacity first
			for _, claimedJob := range claimedJobs {
				if err := waitForCapacity(ctx, httpClient, target, maxConcurrent, capacityPollInterval); err != nil {
					return err
				}
				if err := processJob(ctx, rdb, claimedJob.Job, claimedJob.EntryID, streamName, groupName, target, resultExpiry); err != nil {
					return err
				}
			}
		}
	}
}
