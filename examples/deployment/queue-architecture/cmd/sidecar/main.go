package main

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"strconv"
	"syscall"
	"time"

	"github.com/redis/go-redis/v9"
	"github.com/rvo-redplatform/vllm/examples/deployment/queue-architecture/internal/queue"
	"github.com/rvo-redplatform/vllm/examples/deployment/queue-architecture/internal/sidecar"
)

func main() {
	// Read configuration from environment variables
	redisAddr := os.Getenv("REDIS_ADDR")
	if redisAddr == "" {
		log.Fatal("REDIS_ADDR environment variable is required")
	}

	streamName := os.Getenv("STREAM_NAME")
	if streamName == "" {
		log.Fatal("STREAM_NAME environment variable is required")
	}

	consumerGroup := os.Getenv("CONSUMER_GROUP")
	if consumerGroup == "" {
		log.Fatal("CONSUMER_GROUP environment variable is required")
	}

	consumerName := os.Getenv("CONSUMER_NAME")
	if consumerName == "" {
		log.Fatal("CONSUMER_NAME environment variable is required")
	}

	vllmTarget := os.Getenv("VLLM_TARGET")
	if vllmTarget == "" {
		log.Fatal("VLLM_TARGET environment variable is required")
	}

	// Parse IDLE_THRESHOLD (default 10m)
	idleThresholdStr := os.Getenv("IDLE_THRESHOLD")
	if idleThresholdStr == "" {
		idleThresholdStr = "10m"
	}
	idleThreshold, err := time.ParseDuration(idleThresholdStr)
	if err != nil {
		log.Fatalf("Invalid IDLE_THRESHOLD: %v", err)
	}

	// Parse RECLAIM_INTERVAL (default 30s)
	reclaimIntervalStr := os.Getenv("RECLAIM_INTERVAL")
	if reclaimIntervalStr == "" {
		reclaimIntervalStr = "30s"
	}
	reclaimInterval, err := time.ParseDuration(reclaimIntervalStr)
	if err != nil {
		log.Fatalf("Invalid RECLAIM_INTERVAL: %v", err)
	}

	// Parse MAX_CONCURRENT_REQUESTS (default 2 -- matches the measured real
	// concurrency ceiling for large-context requests on a single GPU; see
	// internal/sidecar/health.go for why this gates claims against vLLM's
	// own real-time /metrics rather than being a purely static limit).
	// Set to 0 or a negative value to disable this gate entirely.
	maxConcurrentStr := os.Getenv("MAX_CONCURRENT_REQUESTS")
	maxConcurrent := 2
	if maxConcurrentStr != "" {
		parsed, err := strconv.Atoi(maxConcurrentStr)
		if err != nil {
			log.Fatalf("Invalid MAX_CONCURRENT_REQUESTS: %v", err)
		}
		maxConcurrent = parsed
	}

	// Parse CAPACITY_POLL_INTERVAL (default 2s) -- how often to re-check
	// vLLM's real-time load while waiting for capacity to free up.
	capacityPollIntervalStr := os.Getenv("CAPACITY_POLL_INTERVAL")
	if capacityPollIntervalStr == "" {
		capacityPollIntervalStr = "2s"
	}
	capacityPollInterval, err := time.ParseDuration(capacityPollIntervalStr)
	if err != nil {
		log.Fatalf("Invalid CAPACITY_POLL_INTERVAL: %v", err)
	}

	// Parse HEALTH_CHECK_INTERVAL (default 5s) -- how often to poll
	// VLLM_TARGET's /health endpoint at startup before consuming any jobs.
	healthCheckIntervalStr := os.Getenv("HEALTH_CHECK_INTERVAL")
	if healthCheckIntervalStr == "" {
		healthCheckIntervalStr = "5s"
	}
	healthCheckInterval, err := time.ParseDuration(healthCheckIntervalStr)
	if err != nil {
		log.Fatalf("Invalid HEALTH_CHECK_INTERVAL: %v", err)
	}

	// Construct Redis client
	rdb := redis.NewClient(&redis.Options{
		Addr: redisAddr,
	})

	// Lightweight HTTP client used only for the /health and /metrics probes
	// against VLLM_TARGET -- separate from the (unbounded-timeout) client
	// used inside the actual forwarders, since these probes should always
	// be fast and shouldn't ever block on a slow/hung backend.
	probeClient := &http.Client{Timeout: 5 * time.Second}

	// Set result expiry to 24 hours
	resultExpiry := 24 * time.Hour

	// Log startup configuration
	log.Printf("Starting sidecar consumer: REDIS_ADDR=%s, STREAM_NAME=%s, CONSUMER_GROUP=%s, CONSUMER_NAME=%s, VLLM_TARGET=%s, RESULT_EXPIRY=%v, IDLE_THRESHOLD=%v, RECLAIM_INTERVAL=%v, MAX_CONCURRENT_REQUESTS=%d, CAPACITY_POLL_INTERVAL=%v, HEALTH_CHECK_INTERVAL=%v",
		redisAddr, streamName, consumerGroup, consumerName, vllmTarget, resultExpiry, idleThreshold, reclaimInterval, maxConcurrent, capacityPollInterval, healthCheckInterval)

	// Ensure consumer group exists before starting consumer loop
	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()
	if err := queue.EnsureConsumerGroup(ctx, rdb, streamName, consumerGroup); err != nil {
		log.Fatalf("Failed to ensure consumer group: %v", err)
	}

	// Block here until vLLM's /health endpoint actually responds 200, before
	// touching the queue at all. vLLM's own multi-minute cold start (model
	// load, cudagraph capture) runs concurrently with this container
	// starting -- with no gate, the sidecar would otherwise start reading
	// jobs and failing to forward them for that entire window.
	log.Printf("Waiting for VLLM_TARGET=%s to become healthy before consuming from the queue...", vllmTarget)
	if err := sidecar.WaitForHealthy(ctx, probeClient, vllmTarget, healthCheckInterval); err != nil {
		log.Fatalf("Gave up waiting for vLLM to become healthy: %v", err)
	}

	// Create error channel to collect errors from both loops
	errChan := make(chan error, 2)

	// Launch ConsumerLoop in a goroutine
	go func() {
		if err := sidecar.ConsumerLoop(ctx, rdb, streamName, consumerGroup, consumerName, vllmTarget, resultExpiry, probeClient, maxConcurrent, capacityPollInterval); err != nil {
			errChan <- fmt.Errorf("consumer loop: %w", err)
		}
	}()

	// Launch ReclaimLoop in a goroutine
	go func() {
		if err := sidecar.ReclaimLoop(ctx, rdb, streamName, consumerGroup, consumerName, vllmTarget, resultExpiry, idleThreshold, reclaimInterval, probeClient, maxConcurrent, capacityPollInterval); err != nil {
			errChan <- fmt.Errorf("reclaim loop: %w", err)
		}
	}()

	// Wait for either loop to return an error or context to be cancelled
	select {
	case err := <-errChan:
		log.Fatalf("Loop error: %v", err)
	case <-ctx.Done():
		log.Printf("Shutdown signal received, stopping loops")
	}
}
