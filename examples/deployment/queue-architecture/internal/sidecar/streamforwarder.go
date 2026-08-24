package sidecar

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"

	"github.com/redis/go-redis/v9"
	"github.com/rvo-redplatform/vllm/examples/deployment/queue-architecture/internal/queue"
)

// ForwardStreaming makes a streaming HTTP call to the target endpoint and relays
// each SSE chunk to a Redis Pub/Sub channel as it arrives. After the response body
// is fully read, it publishes a final done sentinel message.
func ForwardStreaming(ctx context.Context, rdb *redis.Client, job queue.Job, target string) error {
	// Construct the full URL
	url := target + job.Path

	// Create the HTTP request with the body attached
	req, err := http.NewRequestWithContext(ctx, job.Method, url, bytes.NewReader(job.Body))
	if err != nil {
		return fmt.Errorf("failed to create request: %w", err)
	}

	// Set headers from the job
	for key, value := range job.Headers {
		req.Header.Set(key, value)
	}

	// Make the HTTP call
	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		return fmt.Errorf("failed to make HTTP request: %w", err)
	}
	defer resp.Body.Close()

	// Channel name for this job's stream
	channel := fmt.Sprintf("stream:%s", job.JobID)

	// If non-2xx response, relay the error status and body to subscribers
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		body, _ := io.ReadAll(resp.Body)
		// Publish error status and body as a JSON object to the channel
		errorMessage := map[string]interface{}{
			"error":  true,
			"status": resp.StatusCode,
			"body":   string(body),
		}
		errorJSON, err := json.Marshal(errorMessage)
		if err != nil {
			// If we can't marshal the error, still publish what we can
			errorJSON = []byte(fmt.Sprintf(`{"error":true,"status":%d,"body":"failed to marshal error"}`, resp.StatusCode))
		}
		if err := rdb.Publish(ctx, channel, string(errorJSON)).Err(); err != nil {
			// Log but don't fail the job for publish errors
			fmt.Printf("failed to publish error message: %v\n", err)
		}
		// Publish the done sentinel to signal end of stream
		doneMessage := map[string]bool{"__done": true}
		doneJSON, err := json.Marshal(doneMessage)
		if err != nil {
			return fmt.Errorf("failed to marshal done message: %w", err)
		}
		if err := rdb.Publish(ctx, channel, string(doneJSON)).Err(); err != nil {
			return fmt.Errorf("failed to publish done message: %w", err)
		}
		// Return nil so the job is acked normally
		return nil
	}

	// Read and publish each chunk from the response body
	scanner := bufio.NewScanner(resp.Body)
	for scanner.Scan() {
		line := scanner.Bytes()
		if len(line) == 0 {
			continue
		}

		// Parse SSE framing: strip "data: " or "data:" prefix
		payload := string(line)
		
		// Skip non-data SSE lines (event, id, retry, comments starting with :)
		if len(payload) > 0 && payload[0] == ':' {
			continue
		}
		if len(payload) >= 6 && payload[:6] == "event:" {
			continue
		}
		if len(payload) >= 3 && payload[:3] == "id:" {
			continue
		}
		if len(payload) >= 6 && payload[:6] == "retry:" {
			continue
		}
		
		// Strip "data: " or "data:" prefix
		if len(payload) >= 6 && payload[:6] == "data: " {
			payload = payload[6:]
		} else if len(payload) >= 5 && payload[:5] == "data:" {
			payload = payload[5:]
		} else {
			// Not a data line, skip it
			continue
		}
		
		// Trim any leading whitespace after the prefix
		payload = string(bytes.TrimLeft([]byte(payload), " \t"))
		
		// Handle the [DONE] sentinel: translate to __done JSON
		if payload == "[DONE]" {
			doneMessage := map[string]bool{"__done": true}
			doneJSON, err := json.Marshal(doneMessage)
			if err != nil {
				return fmt.Errorf("failed to marshal done message: %w", err)
			}
			if err := rdb.Publish(ctx, channel, string(doneJSON)).Err(); err != nil {
				return fmt.Errorf("failed to publish done message: %w", err)
			}
			continue
		}
		
		// Validate that the payload is valid JSON before publishing
		var jsonPayload interface{}
		if err := json.Unmarshal([]byte(payload), &jsonPayload); err != nil {
			// Skip lines that aren't valid JSON
			continue
		}
		
		// Publish the bare JSON chunk to Redis
		if err := rdb.Publish(ctx, channel, payload).Err(); err != nil {
			return fmt.Errorf("failed to publish chunk: %w", err)
		}
	}

	if err := scanner.Err(); err != nil {
		return fmt.Errorf("error reading response body: %w", err)
	}

	// Publish the done sentinel if we didn't receive [DONE] from upstream
	doneMessage := map[string]bool{"__done": true}
	doneJSON, err := json.Marshal(doneMessage)
	if err != nil {
		return fmt.Errorf("failed to marshal done message: %w", err)
	}

	if err := rdb.Publish(ctx, channel, string(doneJSON)).Err(); err != nil {
		return fmt.Errorf("failed to publish done message: %w", err)
	}

	return nil
}
