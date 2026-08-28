package main

import (
	"fmt"
	"log/slog"
	"net"

	"github.com/rvo-redplatform/vllm/examples/deployment/queue-architecture/internal/proxy"
	"github.com/rvo-redplatform/vllm/examples/deployment/queue-architecture/internal/queue"
	"github.com/spf13/cobra"
	"github.com/spf13/viper"
)

func newProxyCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "proxy",
		Short: "Starts the router proxy",
		Long:  `Starts the router proxy.`,
		PreRunE: func(cmd *cobra.Command, args []string) error {
			return requireStrings(natsURLKey, portKey)
		},
		RunE: runProxy,
	}

	flags := cmd.Flags()
	flags.String(portKey, "", "HTTP listen port (env: RTR_PORT)")
	flags.Int64(maxBodyBytesKey, defaultMaxBodyBytes, "Max request body size in bytes (env: RTR_MAX_BODY_BYTES)")
	flags.Duration(requestTimeoutKey, defaultRequestTimeout, "Client request timeout (env: RTR_REQUEST_TIMEOUT)")
	flags.Duration(streamTimeoutKey, defaultStreamTimeout, "Streaming response timeout (env: RTR_STREAM_TIMEOUT)")

	if err := viper.BindPFlags(flags); err != nil {
		panic(fmt.Errorf("bind proxy flags: %w", err))
	}

	return cmd
}

func runProxy(cmd *cobra.Command, _ []string) error {
	natsURL := viper.GetString(natsURLKey)
	streamName := viper.GetString(streamNameKey)
	streamSubj := viper.GetString(streamSubjectKey)
	port := viper.GetString(portKey)
	maxBodyBytes := viper.GetInt64(maxBodyBytesKey)
	requestTimeout := viper.GetDuration(requestTimeoutKey)
	streamTimeout := viper.GetDuration(streamTimeoutKey)

	slog.InfoContext(cmd.Context(), "starting proxy",
		"nats_url", viper.GetString(natsURLKey),
		"stream_name", viper.GetString(streamNameKey),
		"stream_subject", viper.GetString(streamSubjectKey),
		"port", viper.GetString(portKey),
		"max_body_bytes", viper.GetInt64(maxBodyBytesKey),
		"request_timeout", viper.GetDuration(requestTimeoutKey),
		"stream_timeout", viper.GetDuration(streamTimeoutKey),
	)

	qClient := queue.NewClient(cmd.Context(),
		queue.WithNatsURL(natsURL),
		queue.WithStreamName(streamName),
		queue.WithStreamSubject(streamSubj),
	)

	producer := queue.NewProducer(qClient)
	err := producer.Connect(cmd.Context())
	if err != nil {
		return fmt.Errorf("error connecting to producer: %w", err)
	}
	defer producer.Close()

	server := proxy.NewServer(producer, maxBodyBytes, requestTimeout, streamTimeout)
	listenAddr := net.JoinHostPort("", port)
	server.Addr = listenAddr

	if err := server.ListenAndServe(); err != nil {
		return fmt.Errorf("server error: %w", err)
	}
	return nil
}
