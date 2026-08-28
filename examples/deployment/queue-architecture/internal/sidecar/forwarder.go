package sidecar

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"net/http"

	"github.com/rvo-redplatform/vllm/examples/deployment/queue-architecture/internal/model"
)

// ForwardNonStreaming forwards a non-streaming job to the local vLLM instance
// and returns its raw response.
//
// Parameters:
//   - ctx: context for the HTTP request
//   - job: the job containing method, path, headers, and body
//   - target: the base URL of the vLLM instance (e.g., "http://localhost:8000")
//
// Returns:
//   - status: HTTP status code
//   - headers: response headers as a map
//   - body: raw response body as bytes
//   - err: error if the request failed
func ForwardNonStreaming(ctx context.Context, job model.Job, target string) (status int, headers map[string]string, body []byte, err error) {
	// Construct the full URL
	url := target + job.Path

	// Create the HTTP request
	req, err := http.NewRequestWithContext(ctx, job.Method, url, bytes.NewReader(job.Body))
	if err != nil {
		return 0, nil, nil, fmt.Errorf("failed to create request: %w", err)
	}

	// Set headers from the job
	for key, value := range job.Headers {
		req.Header.Set(key, value)
	}

	// Execute the request
	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		return 0, nil, nil, fmt.Errorf("failed to execute request: %w", err)
	}
	defer resp.Body.Close()

	// Read the response body
	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return 0, nil, nil, fmt.Errorf("failed to read response body: %w", err)
	}

	// Convert response headers to map
	respHeaders := make(map[string]string)
	for key, values := range resp.Header {
		if len(values) > 0 {
			respHeaders[key] = values[0]
		}
	}

	return resp.StatusCode, respHeaders, respBody, nil
}
