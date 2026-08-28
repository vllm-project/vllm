package main

import (
	"fmt"
	"strings"

	"github.com/spf13/viper"
)

func requireString(key string) error {
	if viper.GetString(key) == "" {
		envKey := "RTR_" + strings.ToUpper(strings.ReplaceAll(key, "-", "_"))
		return fmt.Errorf("%q is required (flag, %s, or config file)", key, envKey)
	}
	return nil
}

func requireStrings(keys ...string) error {
	for _, key := range keys {
		if err := requireString(key); err != nil {
			return err
		}
	}
	return nil
}
