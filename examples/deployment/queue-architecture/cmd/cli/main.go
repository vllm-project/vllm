package main

import (
	"context"
	"fmt"
	"log/slog"
	"os"
	"strings"

	"github.com/spf13/cobra"
	"github.com/spf13/viper"
)

func main() {
	rootCmd := buildRootCmd()
	if err := rootCmd.Execute(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func buildRootCmd() *cobra.Command {
	rootCmd := &cobra.Command{
		Use:   "router",
		Short: "LLM Router",
		Long:  `An LLM Router to vLLM.`,
		PersistentPreRunE: func(cmd *cobra.Command, args []string) error {
			return bootstrap(cmd.Context())
		},
	}

	// flags shared between children of the root command
	flags := rootCmd.PersistentFlags()
	flags.String(natsURLKey, "", "NATS server URL (env: RTR_NATS_URL)")
	flags.String(streamNameKey, defaultStreamName, "JetStream stream name (env: RTR_STREAM_NAME)")
	flags.String(streamSubjectKey, defaultStreamSubject, "JetStream subject (env: RTR_STREAM_SUBJECT)")

	cobra.OnInitialize(func() {
		viper.SetEnvPrefix("RTR")
		viper.SetEnvKeyReplacer(strings.NewReplacer("-", "_"))
		viper.AutomaticEnv()

		if err := viper.BindPFlags(flags); err != nil {
			panic(fmt.Errorf("bind flags: %w", err))
		}
	})

	rootCmd.AddCommand(newProxyCmd())
	rootCmd.AddCommand(newSidecarCmd())

	return rootCmd
}

func bootstrap(ctx context.Context) error {
	if err := requireStrings(natsURLKey); err != nil {
		return err
	}

	slog.InfoContext(ctx, "bootstrapping",
		"nats_url", viper.GetString(natsURLKey),
		"stream_name", viper.GetString(streamNameKey),
		"stream_subject", viper.GetString(streamSubjectKey),
	)

	return nil
}
