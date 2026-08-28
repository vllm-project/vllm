package sidecar

import (
	"bufio"
	"context"
	"fmt"
	"log/slog"
	"net/http"
	"strconv"
	"strings"
	"time"
)

// vLLM's own Prometheus metric names that reflect its real-time engine load.
// num_requests_running = actively being computed on the GPU right now.
// num_requests_waiting = accepted by vLLM but queued internally, not yet
// running (vLLM has its own internal admission queue independent of ours).
const (
	metricNumRequestsRunning = "vllm:num_requests_running"
	metricNumRequestsWaiting = "vllm:num_requests_waiting"
)

// WaitForHealthy polls target's /health endpoint until it returns HTTP 200,
// or ctx is cancelled.
//
// Without this, the sidecar (a separate container in the same pod, with no
// startup-ordering guarantee relative to the vLLM container) would begin
// consuming from the queue and attempt to forward the very first job to a
// vLLM instance that is still mid multi-minute cold start (model load,
// cudagraph capture, etc.) -- getting connection-refused, burning the
// client-facing request timeout for no reason, even though the job
// eventually succeeds much later via reclaim.
func WaitForHealthy(ctx context.Context, client *http.Client, target string, pollInterval time.Duration) error {
	healthURL := target + "/health"
	attempt := 0
	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		attempt++
		req, err := http.NewRequestWithContext(ctx, http.MethodGet, healthURL, nil)
		if err == nil {
			resp, doErr := client.Do(req)
			if doErr == nil {
				resp.Body.Close()
				if resp.StatusCode == http.StatusOK {
					slog.InfoContext(ctx, "vLLM target is health",
						"target", target,
						"attempt", attempt,
					)
					return nil
				}
				slog.InfoContext(ctx, "waiting for vLLM target to become healthy",
					"target", target,
					"status", resp.StatusCode,
					"attempt", attempt,
				)
			} else {
				slog.ErrorContext(ctx, "error waiting on vLLM target",
					"target", target,
					"attempt", attempt,
					"err", doErr,
				)
			}
		}

		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(pollInterval):
		}
	}
}

// currentLoad scrapes target's /metrics endpoint and returns the sum of
// vllm:num_requests_running and vllm:num_requests_waiting -- vLLM's own
// real-time count of requests it is actively processing or has internally
// queued. Scraped fresh on every call so capacity decisions reflect the
// engine's true current state rather than an assumed/hardcoded ceiling.
func currentLoad(ctx context.Context, client *http.Client, target string) (float64, error) {
	metricsURL := target + "/metrics"
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, metricsURL, nil)
	if err != nil {
		return 0, fmt.Errorf("build metrics request: %w", err)
	}

	resp, err := client.Do(req)
	if err != nil {
		return 0, fmt.Errorf("fetch metrics: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return 0, fmt.Errorf("metrics endpoint returned status %d", resp.StatusCode)
	}

	var total float64
	scanner := bufio.NewScanner(resp.Body)
	for scanner.Scan() {
		line := scanner.Text()
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		for _, name := range []string{metricNumRequestsRunning, metricNumRequestsWaiting} {
			if !strings.HasPrefix(line, name) {
				continue
			}
			// Prometheus text exposition format: "<metric_name>{labels} <value>"
			// or "<metric_name> <value>" with no labels. The value is always
			// the last whitespace-separated field on the line.
			fields := strings.Fields(line)
			if len(fields) < 2 {
				continue
			}
			val, parseErr := strconv.ParseFloat(fields[len(fields)-1], 64)
			if parseErr != nil {
				continue
			}
			total += val
		}
	}
	if err := scanner.Err(); err != nil {
		return 0, fmt.Errorf("scan metrics response: %w", err)
	}

	return total, nil
}

// waitForCapacity blocks, polling target's real-time load via currentLoad,
// until vLLM's combined running+waiting request count is below
// maxConcurrent, or ctx is cancelled.
//
// This is the gate that makes the sidecar's claim/forward decision aware of
// its host vLLM instance's actual real-time capacity, instead of blindly
// claiming and forwarding every message it reads regardless of whether
// vLLM has room to make timely progress on it. If vLLM already reports
// being at or above the configured ceiling, a message is deliberately left
// unclaimed in the stream (visible to KEDA's lag-based scaling metric)
// rather than pulled and immediately queued up behind other in-flight work.
//
// maxConcurrent <= 0 disables this gate entirely (always returns
// immediately) -- useful for local/dev environments without a real vLLM
// /metrics endpoint, or if an operator wants to opt out.
//
// A single metrics-scrape failure is tolerated (retried after pollInterval)
// -- a flaky one-off scrape should never permanently wedge job processing.
// But if failures are SUSTAINED (maxConsecutiveFailures in a row), vLLM
// itself is treated as unhealthy rather than blindly claiming work it
// can't handle: this re-enters the same WaitForHealthy loop used at
// startup and blocks until vLLM genuinely recovers (or ctx is cancelled).
func waitForCapacity(ctx context.Context, client *http.Client, target string, maxConcurrent int, pollInterval time.Duration) error {
	if maxConcurrent <= 0 {
		return nil
	}

	const maxConsecutiveFailures = 3
	consecutiveFailures := 0

	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		load, err := currentLoad(ctx, client, target)
		if err != nil {
			consecutiveFailures++
			if consecutiveFailures < maxConsecutiveFailures {
				slog.WarnContext(ctx, fmt.Sprintf("failed to check vLLM capacity (%d/%d consecutive failures)", consecutiveFailures, maxConsecutiveFailures),
					"err", err,
				)
				select {
				case <-ctx.Done():
					return ctx.Err()
				case <-time.After(pollInterval):
				}
				continue
			}
			slog.ErrorContext(ctx, fmt.Sprintf("vLLM capacity check failed %d times in a row -- treating vLLM as unhealthy, waiting for it to recover before claiming more work", consecutiveFailures),
				"err", err,
			)
			if err := WaitForHealthy(ctx, client, target, pollInterval); err != nil {
				return err
			}
			consecutiveFailures = 0
			continue
		}
		consecutiveFailures = 0

		if load < float64(maxConcurrent) {
			return nil
		}

		slog.InfoContext(ctx, fmt.Sprintf("vLLM at capacity (running+waiting=%.0f >= max=%d), holding off on claiming next message", load, maxConcurrent))

		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(pollInterval):
		}
	}
}
