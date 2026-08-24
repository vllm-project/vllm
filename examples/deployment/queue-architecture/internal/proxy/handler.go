package proxy

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/redis/go-redis/v9"
	"github.com/oklog/ulid/v2"

	"github.com/rvo-redplatform/vllm/examples/deployment/queue-architecture/internal/queue"
)

// Result represents the stored result of a processed job.
type Result struct {
	Status  int               `json:"status"`
	Headers map[string]string `json:"headers"`
	Body    string            `json:"body"`
}

// HandleNonStreaming returns an HTTP handler that processes non-streaming requests.
// It reads the request into a Job, enqueues it, waits for the result via BLPop,
// and writes the response back to the client.
func HandleNonStreaming(rdb *redis.Client, streamName string, timeout time.Duration) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		ctx := r.Context()

		// Read request body
		body, err := io.ReadAll(r.Body)
		if err != nil {
			http.Error(w, fmt.Sprintf("failed to read body: %v", err), http.StatusBadRequest)
			return
		}
		defer r.Body.Close()

		// Convert headers to map[string]string
		headers := make(map[string]string)
		for key, values := range r.Header {
			if len(values) > 0 {
				headers[key] = values[0]
			}
		}

		// Create Job with new ULID
		jobID := ulid.Make().String()
		job := queue.Job{
			JobID:   jobID,
			Method:  r.Method,
			Path:    r.RequestURI,
			Headers: headers,
			Body:    body,
			Stream:  false,
		}

		// Enqueue the job
		_, err = queue.Enqueue(ctx, rdb, streamName, job)
		if err != nil {
			http.Error(w, fmt.Sprintf("failed to enqueue job: %v", err), http.StatusInternalServerError)
			return
		}

		// Wait for result with configured timeout
		resultKey := fmt.Sprintf("result:%s", jobID)

		// Use BLPop to block and wait for the result
		results, err := rdb.BLPop(ctx, timeout, resultKey).Result()
		if err != nil {
			if err == redis.Nil {
				// Timeout occurred
				http.Error(w, "request timeout", http.StatusGatewayTimeout)
				return
			}
			http.Error(w, fmt.Sprintf("failed to wait for result: %v", err), http.StatusInternalServerError)
			return
		}

		// BLPop returns [key, value], we want the value (results[1])
		if len(results) < 2 {
			http.Error(w, "invalid result format", http.StatusInternalServerError)
			return
		}

		// Unmarshal the result
		var result Result
		err = json.Unmarshal([]byte(results[1]), &result)
		if err != nil {
			http.Error(w, fmt.Sprintf("failed to unmarshal result: %v", err), http.StatusInternalServerError)
			return
		}

		// Write response headers
		for key, value := range result.Headers {
			w.Header().Set(key, value)
		}

		// Write status code
		w.WriteHeader(result.Status)

	// Write response body
	_, err = w.Write([]byte(result.Body))
	if err != nil {
		// Log error but don't write to response (already started)
		fmt.Printf("failed to write response body: %v\n", err)
	}
	}
}
