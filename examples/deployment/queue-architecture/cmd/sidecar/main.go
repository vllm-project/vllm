package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"os/signal"
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

	// Construct Redis client
	rdb := redis.NewClient(&redis.Options{
		Addr: redisAddr,
	})

	// Set result expiry to 24 hours
	resultExpiry := 24 * time.Hour

	// Log startup configuration
	log.Printf("Starting sidecar consumer: REDIS_ADDR=%s, STREAM_NAME=%s, CONSUMER_GROUP=%s, CONSUMER_NAME=%s, VLLM_TARGET=%s, RESULT_EXPIRY=%v, IDLE_THRESHOLD=%v, RECLAIM_INTERVAL=%v",
		redisAddr, streamName, consumerGroup, consumerName, vllmTarget, resultExpiry, idleThreshold, reclaimInterval)

	// Ensure consumer group exists before starting consumer loop
	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()
	if err := queue.EnsureConsumerGroup(ctx, rdb, streamName, consumerGroup); err != nil {
		log.Fatalf("Failed to ensure consumer group: %v", err)
	}

	// Create error channel to collect errors from both loops
	errChan := make(chan error, 2)

	// Launch ConsumerLoop in a goroutine
	go func() {
		if err := sidecar.ConsumerLoop(ctx, rdb, streamName, consumerGroup, consumerName, vllmTarget, resultExpiry); err != nil {
			errChan <- fmt.Errorf("consumer loop: %w", err)
		}
	}()

	// Launch ReclaimLoop in a goroutine
	go func() {
		if err := sidecar.ReclaimLoop(ctx, rdb, streamName, consumerGroup, consumerName, vllmTarget, resultExpiry, idleThreshold, reclaimInterval); err != nil {
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
