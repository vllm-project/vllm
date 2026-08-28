package main

import "time"

const (
	// Shared (root persistent flags).
	natsURLKey       = "nats-url"
	streamNameKey    = "stream-name"
	streamSubjectKey = "stream-subject"

	// Proxy-only flags.
	portKey           = "port"
	maxBodyBytesKey   = "max-body-bytes"
	requestTimeoutKey = "request-timeout"
	streamTimeoutKey  = "stream-timeout"

	// Sidecar-only flags.
	consumerNameKey         = "consumer-name"
	vllmTargetKey           = "vllm-target"
	maxConcurrencyKey       = "max-concurrency"
	capacityPollIntervalKey = "capacity-poll-interval"
	healthCheckIntervalKey  = "health-check-interval"
	maxDrainTimeoutKey      = "max-drain-timeout"

	defaultStreamName    = "vllm_requests"
	defaultStreamSubject = "vllm.requests"
	defaultConsumerName  = "vllm-sidecars"

	defaultMaxConcurrency       = 2
	defaultMaxBodyBytes         = 10 << 20 // 10 MiB
	defaultRequestTimeout       = time.Hour
	defaultStreamTimeout        = time.Hour
	defaultCapacityPollInterval = 2 * time.Second
	defaultHealthCheckInterval  = 5 * time.Second
	defaultMaxDrainTimeout      = 660 * time.Second
)
