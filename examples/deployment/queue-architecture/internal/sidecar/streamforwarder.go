package sidecar

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"

	"github.com/rvo-redplatform/vllm/examples/deployment/queue-architecture/internal/model"
)

// ForwardStreaming makes a streaming HTTP call to the target endpoint and relays
// each SSE chunk to the proxy inbox as it arrives. After the response body is
// fully read, it publishes a final done sentinel message.
func ForwardStreaming(ctx context.Context, mailer ConsumerClient, replyTo string, job model.Job, target string) error {
	url := target + job.Path

	req, err := http.NewRequestWithContext(ctx, job.Method, url, bytes.NewReader(job.Body))
	if err != nil {
		return fmt.Errorf("failed to create request: %w", err)
	}

	for key, value := range job.Headers {
		req.Header.Set(key, value)
	}

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		return fmt.Errorf("failed to make HTTP request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		body, _ := io.ReadAll(resp.Body)
		errorMessage := map[string]interface{}{
			"error":  true,
			"status": resp.StatusCode,
			"body":   string(body),
		}
		errorJSON, err := json.Marshal(errorMessage)
		if err != nil {
			errorJSON = []byte(fmt.Sprintf(`{"error":true,"status":%d,"body":"failed to marshal error"}`, resp.StatusCode))
		}
		err = mailer.Mail(replyTo, errorJSON)
		if err != nil {
			slog.WarnContext(ctx, "failed to mail error reply",
				"reply_to", replyTo,
				"err", err,
			)
		}

		doneJSON, err := json.Marshal(map[string]bool{"__done": true})
		if err != nil {
			return fmt.Errorf("failed to marshal done message: %w", err)
		}

		err = mailer.Mail(replyTo, doneJSON)
		if err != nil {
			slog.WarnContext(ctx, "failed to mail done reply",
				"reply_to", replyTo,
				"err", err,
			)
		}
		return nil
	}

	scanner := bufio.NewScanner(resp.Body)
	for scanner.Scan() {
		line := scanner.Bytes()
		if len(line) == 0 {
			continue
		}

		payload := string(line)

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

		if len(payload) >= 6 && payload[:6] == "data: " {
			payload = payload[6:]
		} else if len(payload) >= 5 && payload[:5] == "data:" {
			payload = payload[5:]
		} else {
			continue
		}

		payload = string(bytes.TrimLeft([]byte(payload), " \t"))

		if payload == "[DONE]" {
			doneJSON, err := json.Marshal(map[string]bool{"__done": true})
			if err != nil {
				return fmt.Errorf("failed to marshal done message: %w", err)
			}

			err = mailer.Mail(replyTo, doneJSON)
			if err != nil {
				slog.WarnContext(ctx, "failed to mail done message",
					"reply_to", replyTo,
					"err", err,
				)
			}
			continue
		}

		var jsonPayload interface{}
		if err := json.Unmarshal([]byte(payload), &jsonPayload); err != nil {
			continue
		}

		err = mailer.Mail(replyTo, []byte(payload))
		if err != nil {
			slog.WarnContext(ctx, "failed to mail payload",
				"reply_to", replyTo,
				"err", err,
			)
		}
	}

	if err := scanner.Err(); err != nil {
		return fmt.Errorf("error reading response body: %w", err)
	}

	doneJSON, err := json.Marshal(map[string]bool{"__done": true})
	if err != nil {
		return fmt.Errorf("failed to marshal done message: %w", err)
	}

	err = mailer.Mail(replyTo, doneJSON)
	if err != nil {
		slog.WarnContext(ctx, "failed to mail done message",
			"reply_to", replyTo,
			"err", err,
		)
	}

	return nil
}
