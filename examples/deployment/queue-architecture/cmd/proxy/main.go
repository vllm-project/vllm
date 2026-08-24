package main

import (
	"log"
	"net"
	"os"
	"strconv"
	"time"

	"github.com/redis/go-redis/v9"
	"github.com/rvo-redplatform/vllm/examples/deployment/queue-architecture/internal/proxy"
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

	listenPort := os.Getenv("LISTEN_PORT")
	if listenPort == "" {
		log.Fatal("LISTEN_PORT environment variable is required")
	}

	// Read MAX_BODY_BYTES with default of 10 MiB
	maxBodyBytesStr := os.Getenv("MAX_BODY_BYTES")
	maxBodyBytes := int64(10 * 1024 * 1024) // 10 MiB default
	if maxBodyBytesStr != "" {
		parsed, err := strconv.ParseInt(maxBodyBytesStr, 10, 64)
		if err != nil {
			log.Fatalf("Invalid MAX_BODY_BYTES value: %v", err)
		}
		maxBodyBytes = parsed
	}

	// Read REQUEST_TIMEOUT with default of 30 seconds
	requestTimeoutStr := os.Getenv("REQUEST_TIMEOUT")
	requestTimeout := 30 * time.Second
	if requestTimeoutStr != "" {
		parsed, err := time.ParseDuration(requestTimeoutStr)
		if err != nil {
			log.Fatalf("Invalid REQUEST_TIMEOUT value: %v", err)
		}
		requestTimeout = parsed
	}

	// Read STREAM_TIMEOUT with default of 5 minutes
	streamTimeoutStr := os.Getenv("STREAM_TIMEOUT")
	streamTimeout := 5 * time.Minute
	if streamTimeoutStr != "" {
		parsed, err := time.ParseDuration(streamTimeoutStr)
		if err != nil {
			log.Fatalf("Invalid STREAM_TIMEOUT value: %v", err)
		}
		streamTimeout = parsed
	}

	// Construct Redis client
	rdb := redis.NewClient(&redis.Options{
		Addr: redisAddr,
	})

	// Create the HTTP server
	server := proxy.NewServer(rdb, streamName, maxBodyBytes, requestTimeout, streamTimeout)

	// Set the server address
	listenAddr := net.JoinHostPort("", listenPort)
	server.Addr = listenAddr

	// Log startup configuration
	log.Printf("Starting proxy server: REDIS_ADDR=%s, STREAM_NAME=%s, LISTEN_PORT=%s, MAX_BODY_BYTES=%d, REQUEST_TIMEOUT=%v, STREAM_TIMEOUT=%v", redisAddr, streamName, listenPort, maxBodyBytes, requestTimeout, streamTimeout)

	// Start listening and serving
	if err := server.ListenAndServe(); err != nil {
		log.Fatalf("Server error: %v", err)
	}
}
