package main

import (
	"context"
	"fmt"
	"log/slog"
	"net/http"
	"os/signal"
	"sync"
	"syscall"
	"time"

	"github.com/rvo-redplatform/vllm/examples/deployment/queue-architecture/internal/queue"
	"github.com/rvo-redplatform/vllm/examples/deployment/queue-architecture/internal/sidecar"
	"github.com/spf13/cobra"
	"github.com/spf13/viper"
)

func newSidecarCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "sidecar",
		Short: "Starts the router sidecar",
		Long:  `Starts the router sidecar.`,
		PreRunE: func(cmd *cobra.Command, args []string) error {
			return requireStrings(vllmTargetKey)
		},
		RunE: runSidecar,
	}

	flags := cmd.Flags()
	flags.String(vllmTargetKey, "", "vLLM upstream URL (env: RTR_VLLM_TARGET)")
	flags.String(consumerNameKey, defaultConsumerName, "JetStream consumer name (env: RTR_CONSUMER_NAME)")
	flags.Int(maxConcurrencyKey, defaultMaxConcurrency, "Max concurrent requests (env: RTR_MAX_CONCURRENCY)")
	flags.Duration(capacityPollIntervalKey, defaultCapacityPollInterval, "Capacity poll interval (env: RTR_CAPACITY_POLL_INTERVAL)")
	flags.Duration(healthCheckIntervalKey, defaultHealthCheckInterval, "vLLM health check interval (env: RTR_HEALTH_CHECK_INTERVAL)")
	flags.Duration(maxDrainTimeoutKey, defaultMaxDrainTimeout, "Shutdown drain timeout (env: RTR_MAX_DRAIN_TIMEOUT)")

	if err := viper.BindPFlags(flags); err != nil {
		panic(fmt.Errorf("bind sidecar flags: %w", err))
	}

	return cmd
}

func runSidecar(cmd *cobra.Command, _ []string) error {
	natsURL := viper.GetString(natsURLKey)
	streamName := viper.GetString(streamNameKey)
	streamSubject := viper.GetString(streamSubjectKey)
	vllmTarget := viper.GetString(vllmTargetKey)
	consumerName := viper.GetString(consumerNameKey)
	maxConcurrency := viper.GetInt(maxConcurrencyKey)
	capacityPollInterval := viper.GetDuration(capacityPollIntervalKey)
	healthCheckInterval := viper.GetDuration(healthCheckIntervalKey)
	maxDrainTimeout := viper.GetDuration(maxDrainTimeoutKey)

	slog.InfoContext(cmd.Context(), "starting sidecar",
		"nats_url", natsURL,
		"stream_name", streamName,
		"stream_subject", streamSubject,
		"vllm_target", vllmTarget,
		"consumer_name", consumerName,
		"max_concurrency", maxConcurrency,
		"capacity_poll_interval", capacityPollInterval,
		"health_check_interval", healthCheckInterval,
		"max_drain_timeout", maxDrainTimeout,
	)

	shutdownCtx, stop := signal.NotifyContext(cmd.Context(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()
	workCtx := context.Background()

	qClient := queue.NewClient(
		cmd.Context(),
		queue.WithNatsURL(natsURL),
		queue.WithStreamName(streamName),
		queue.WithStreamSubject(streamSubject),
	)
	consumer := queue.NewConsumer(qClient, queue.WithConsumerName(consumerName))
	err := consumer.Connect(cmd.Context())
	if err != nil {
		return fmt.Errorf("error connecting to consumer: %w", err)
	}
	defer consumer.Close()

	probeClient := &http.Client{Timeout: 5 * time.Second}

	slog.InfoContext(
		cmd.Context(),
		fmt.Sprintf("Waiting for VLLM_TARGET=%s to become healthy before consuming from the queue...", vllmTarget),
	)
	err = sidecar.WaitForHealthy(shutdownCtx, probeClient, vllmTarget, healthCheckInterval)
	if err != nil {
		return fmt.Errorf("gave up waiting for vllm to become healthy: &w", err)
	}

	errChan := make(chan error, 1)
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		err := sidecar.NewConsumer(consumer).ConsumerLoop(
			shutdownCtx,
			workCtx,
			vllmTarget,
			probeClient,
			maxConcurrency,
			capacityPollInterval,
		)
		if err != nil {
			errChan <- fmt.Errorf("consumer loop error: %w", err)
		}

	}()

	select {
	case err := <-errChan:
		return fmt.Errorf("loop error: %w", err)
	case <-shutdownCtx.Done():
		slog.InfoContext(cmd.Context(), "shutdown sig received, draining in-flight work")
	}

	drained := make(chan struct{})
	go func() {
		wg.Wait()
		close(drained)
	}()

	select {
	case <-drained:
		slog.InfoContext(cmd.Context(), "all in-flight work has finished, exiting cleanly")
	case <-time.After(maxDrainTimeout):
		return fmt.Errorf(
			"WARNING: drain timeout (%v) exceeded with work still in flight -- exiting anyway. Any unfinished job was never acked, so JetStream may redeliver it (MaxDeliver=2).",
			maxDrainTimeout,
		)
	}

	return nil
}
