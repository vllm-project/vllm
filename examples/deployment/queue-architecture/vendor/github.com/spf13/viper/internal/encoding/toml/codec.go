package toml

import (
	"github.com/pelletier/go-toml/v2"
)

// Codec implements the encoding.Encoder and encoding.Decoder interfaces for TOML encoding.
type Codec struct{}

func (Codec) Encode(v map[string]any) ([]byte, error) {
	return toml.Marshal(v)
}

func (Codec) Decode(b []byte, v map[string]any) error {
	return toml.Unmarshal(b, &v)
}
