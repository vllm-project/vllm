package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"net"
	"net/http"
	"os"
	"strconv"
	"time"
)

// ChatCompletionRequest represents the incoming chat completion request
type ChatCompletionRequest struct {
	Model    string `json:"model"`
	Messages []struct {
		Role    string `json:"role"`
		Content string `json:"content"`
	} `json:"messages"`
	Stream bool `json:"stream"`
}

// ChatCompletionResponse represents a non-streaming chat completion response
type ChatCompletionResponse struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	Model   string `json:"model"`
	Choices []struct {
		Index   int `json:"index"`
		Message struct {
			Role    string `json:"role"`
			Content string `json:"content"`
		} `json:"message"`
		FinishReason string `json:"finish_reason"`
	} `json:"choices"`
	Usage struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
		TotalTokens      int `json:"total_tokens"`
	} `json:"usage"`
}

// Server holds the HTTP server and configuration
type Server struct {
	*http.Server
}

// NewServer creates a new mock vLLM server
func NewServer(listenPort string) *Server {
	mux := http.NewServeMux()

	mux.HandleFunc("GET /health", handleHealth)
	mux.HandleFunc("POST /v1/chat/completions", handleChatCompletions)

	return &Server{
		Server: &http.Server{
			Addr:    net.JoinHostPort("", listenPort),
			Handler: mux,
		},
	}
}

// handleHealth responds to health checks
func handleHealth(w http.ResponseWriter, r *http.Request) {
	w.WriteHeader(http.StatusOK)
	fmt.Fprintf(w, "OK")
}

// handleChatCompletions handles POST /v1/chat/completions requests
func handleChatCompletions(w http.ResponseWriter, r *http.Request) {
	var req ChatCompletionRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("invalid request: %v", err), http.StatusBadRequest)
		return
	}

	if req.Stream {
		handleStreamingCompletion(w, r, &req)
	} else {
		handleNonStreamingCompletion(w, r, &req)
	}
}

// handleStreamingCompletion sends SSE chunks for streaming requests
func handleStreamingCompletion(w http.ResponseWriter, r *http.Request, req *ChatCompletionRequest) {
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")

	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "streaming not supported", http.StatusInternalServerError)
		return
	}

	// Send a few canned SSE chunks
	chunks := []string{
		`{"id":"chatcmpl-mock-1","object":"text_completion.chunk","created":1234567890,"model":"mock-model","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}`,
		`{"id":"chatcmpl-mock-1","object":"text_completion.chunk","created":1234567890,"model":"mock-model","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}`,
		`{"id":"chatcmpl-mock-1","object":"text_completion.chunk","created":1234567890,"model":"mock-model","choices":[{"index":0,"delta":{"content":" from"},"finish_reason":null}]}`,
		`{"id":"chatcmpl-mock-1","object":"text_completion.chunk","created":1234567890,"model":"mock-model","choices":[{"index":0,"delta":{"content":" mock"},"finish_reason":null}]}`,
		`{"id":"chatcmpl-mock-1","object":"text_completion.chunk","created":1234567890,"model":"mock-model","choices":[{"index":0,"delta":{"content":" vLLM"},"finish_reason":"stop"}]}`,
	}

	for _, chunk := range chunks {
		fmt.Fprintf(w, "data: %s\n\n", chunk)
		flusher.Flush()
		time.Sleep(100 * time.Millisecond) // Small delay between chunks
	}

	// Send the [DONE] marker
	fmt.Fprintf(w, "data: [DONE]\n\n")
	flusher.Flush()
}

// handleNonStreamingCompletion sends a single canned response after a delay
func handleNonStreamingCompletion(w http.ResponseWriter, r *http.Request, req *ChatCompletionRequest) {
	// Simulate processing delay
	time.Sleep(1 * time.Second)

	response := ChatCompletionResponse{
		ID:      "chatcmpl-mock-1",
		Object:  "text_completion",
		Created: time.Now().Unix(),
		Model:   req.Model,
	}

	response.Choices = make([]struct {
		Index   int `json:"index"`
		Message struct {
			Role    string `json:"role"`
			Content string `json:"content"`
		} `json:"message"`
		FinishReason string `json:"finish_reason"`
	}, 1)

	response.Choices[0].Index = 0
	response.Choices[0].Message.Role = "assistant"
	response.Choices[0].Message.Content = "Hello from mock vLLM"
	response.Choices[0].FinishReason = "stop"

	response.Usage.PromptTokens = 10
	response.Usage.CompletionTokens = 5
	response.Usage.TotalTokens = 15

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

func main() {
	listenPort := os.Getenv("LISTEN_PORT")
	if listenPort == "" {
		listenPort = "8000"
	}

	// Validate port is a valid number
	if _, err := strconv.Atoi(listenPort); err != nil {
		log.Fatalf("Invalid LISTEN_PORT value: %v", err)
	}

	// Check for --healthcheck flag
	healthcheck := flag.Bool("healthcheck", false, "Run healthcheck and exit")
	flag.Parse()

	if *healthcheck {
		// Perform healthcheck: GET http://localhost:LISTEN_PORT/health
		client := &http.Client{
			Timeout: 2 * time.Second,
		}
		resp, err := client.Get(fmt.Sprintf("http://localhost:%s/health", listenPort))
		if err != nil {
			log.Printf("Healthcheck failed: %v", err)
			os.Exit(1)
		}
		defer resp.Body.Close()

		if resp.StatusCode != http.StatusOK {
			log.Printf("Healthcheck failed: status code %d", resp.StatusCode)
			os.Exit(1)
		}

		log.Printf("Healthcheck passed")
		os.Exit(0)
	}

	server := NewServer(listenPort)

	log.Printf("Starting mock vLLM server on port %s", listenPort)
	if err := server.ListenAndServe(); err != nil {
		log.Fatalf("Server error: %v", err)
	}
}
