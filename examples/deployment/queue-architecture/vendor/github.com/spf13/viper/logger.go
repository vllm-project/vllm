package viper

import (
	"context"
	"log/slog"
)

// WithLogger sets a custom logger.
func WithLogger(l *slog.Logger) Option {
	return optionFunc(func(v *Viper) {
		v.logger = l
	})
}

type discardHandler struct{}

func (n *discardHandler) Enabled(_ context.Context, _ slog.Level) bool {
	return false
}

func (n *discardHandler) Handle(_ context.Context, _ slog.Record) error {
	return nil
}

func (n *discardHandler) WithAttrs(_ []slog.Attr) slog.Handler {
	return n
}

func (n *discardHandler) WithGroup(_ string) slog.Handler {
	return n
}
