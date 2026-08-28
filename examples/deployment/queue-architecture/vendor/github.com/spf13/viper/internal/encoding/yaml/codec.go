package yaml

import "go.yaml.in/yaml/v3"

// Codec implements the encoding.Encoder and encoding.Decoder interfaces for YAML encoding.
type Codec struct{}

func (Codec) Encode(v map[string]any) ([]byte, error) {
	return yaml.Marshal(v)
}

func (Codec) Decode(b []byte, v map[string]any) error {
	return yaml.Unmarshal(b, &v)
}
