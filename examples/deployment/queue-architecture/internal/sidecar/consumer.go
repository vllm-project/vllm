package sidecar

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"sync"
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

// ConsumerLoop runs up to maxConcurrent independent worker goroutines, each
// looping: gate on vLLM's real-time load, read one job from the queue,
// process it, repeat. Running multiple workers concurrently (rather than a
// single sequential loop) is what actually lets the sidecar have more than
// one in-flight request to vLLM at a time -- the capacity gate alone only
// decides *whether* to claim the next message; it's the worker pool that
// determines *how many* can be claimed and forwarded concurrently.
//
// Multiple goroutines issuing concurrent XReadGroup calls under the same
// consumer name is safe: Redis Streams hands each blocking reader its own
// distinct message as new ones arrive, there's no risk of two workers
// getting the same message.
//
// Order matters for crash protection: write the result BEFORE acking -- so a crash
// between those two steps causes at most wasted redundant work, never a lost answer.
//
// Each worker gates on vLLM's real-time load (see waitForCapacity in
// health.go) before claiming its next message: if vLLM already reports
// being at or above maxConcurrent (running+waiting requests), that worker
// holds off rather than claiming work it can't make timely progress on.
// This makes the sidecar's aggregate concurrency ceiling self-adjusting to
// vLLM's actual measured capacity instead of a blind assumption, and
// leaves un-claimed backlog visible to KEDA's stream-lag scaling metric
// instead of hidden behind claimed-but-blocked workers.
//
// Error handling:
//   - Infrastructure errors (queue.Read failures) cancel all sibling workers
//     and are returned from the loop.
//   - Per-job errors (idempotency check, forwarding, marshaling, result write, ack) are
//     logged with the job ID by processJob, which returns nil so that worker continues to
//     the next job. Failed jobs remain un-acked and stay in the Pending Entries List for
//     later reclaim/retry.
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
//   - maxConcurrent: number of concurrent worker goroutines, and the vLLM running+waiting
//     threshold each worker gates on before claiming (<=0 is treated as 1 -- always at
//     least one worker; the capacity gate itself is separately disabled by health.go's
//     own <=0 check on this same value)
//   - capacityPollInterval: how often to re-check vLLM's load while waiting for capacity
func ConsumerLoop(ctx context.Context, rdb *redis.Client, streamName, groupName, consumerName, target string, resultExpiry time.Duration, httpClient *http.Client, maxConcurrent int, capacityPollInterval time.Duration) error {
	workers := maxConcurrent
	if workers <= 0 {
		workers = 1
	}

	workerCtx, cancel := context.WithCancel(ctx)
	defer cancel()

	var wg sync.WaitGroup
	var firstErrOnce sync.Once
	var firstErr error

	recordFatal := func(err error) {
		if err == nil {
			return
		}
		firstErrOnce.Do(func() {
			firstErr = err
			cancel() // stop sibling workers on a real infrastructure error
		})
	}

	runWorker := func(workerID int) {
		defer wg.Done()
		for {
			select {
			case <-workerCtx.Done():
				return
			default:
			}

			// Gate on vLLM's real-time load before claiming the next message.
			if err := waitForCapacity(workerCtx, httpClient, target, maxConcurrent, capacityPollInterval); err != nil {
				if ctx.Err() == nil {
					recordFatal(err)
				}
				return
			}

			// Read a job from the queue.
			job, entryID, err := queue.Read(workerCtx, rdb, streamName, groupName, consumerName)
			if err != nil {
				if ctx.Err() == nil {
					recordFatal(fmt.Errorf("read job (worker %d): %w", workerID, err))
				}
				return
			}

			// Process the job using the shared helper. processJob itself
			// never returns a non-nil error for per-job failures (only
			// infrastructure errors), so this worker keeps going.
			if err := processJob(workerCtx, rdb, job, entryID, streamName, groupName, target, resultExpiry); err != nil {
				recordFatal(err)
				return
			}
		}
	}

	wg.Add(workers)
	for i := 0; i < workers; i++ {
		go runWorker(i)
	}
	wg.Wait()

	if firstErr != nil {
		return firstErr
	}
	return ctx.Err()
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
// Each tick's claimed batch is processed by up to maxConcurrent worker
// goroutines pulling from a shared channel (the same bounded-concurrency
// model as ConsumerLoop -- see health.go's waitForCapacity), so a burst of
// reclaimed jobs after a crash is drained with the same real-time
// vLLM-capacity awareness as fresh reads, not one at a time.
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
//   - maxConcurrent: number of concurrent worker goroutines per batch, and the vLLM
//     running+waiting threshold each worker gates on (<=0 is treated as 1 worker)
//   - capacityPollInterval: how often to re-check vLLM's load while waiting for capacity
func ReclaimLoop(ctx context.Context, rdb *redis.Client, streamName, groupName, consumerName, target string, resultExpiry, idleThreshold, reclaimInterval time.Duration, httpClient *http.Client, maxConcurrent int, capacityPollInterval time.Duration) error {
	ticker := time.NewTicker(reclaimInterval)
	defer ticker.Stop()

	workers := maxConcurrent
	if workers <= 0 {
		workers = 1
	}

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
			if len(claimedJobs) == 0 {
				continue
			}

			jobCh := make(chan queue.ClaimedJob, len(claimedJobs))
			for _, cj := range claimedJobs {
				jobCh <- cj
			}
			close(jobCh)

			batchCtx, cancelBatch := context.WithCancel(ctx)
			var wg sync.WaitGroup
			var firstErrOnce sync.Once
			var firstErr error

			recordFatal := func(e error) {
				if e == nil {
					return
				}
				firstErrOnce.Do(func() {
					firstErr = e
					cancelBatch()
				})
			}

			wg.Add(workers)
			for i := 0; i < workers; i++ {
				go func() {
					defer wg.Done()
					for cj := range jobCh {
						select {
						case <-batchCtx.Done():
							return
						default:
						}
						if err := waitForCapacity(batchCtx, httpClient, target, maxConcurrent, capacityPollInterval); err != nil {
							if ctx.Err() == nil {
								recordFatal(err)
							}
							return
						}
						if err := processJob(batchCtx, rdb, cj.Job, cj.EntryID, streamName, groupName, target, resultExpiry); err != nil {
							recordFatal(err)
							return
						}
					}
				}()
			}
			wg.Wait()
			cancelBatch()

			if firstErr != nil {
				return firstErr
			}
		}
	}
}
