package dotenv

import (
	"bytes"
	"fmt"
	"sort"
	"strings"

	"github.com/subosito/gotenv"
)

const keyDelimiter = "_"

// Codec implements the encoding.Encoder and encoding.Decoder interfaces for encoding data containing environment variables
// (commonly called as dotenv format).
type Codec struct{}

func (Codec) Encode(v map[string]any) ([]byte, error) {
	flattened := map[string]any{}

	flattened = flattenAndMergeMap(flattened, v, "", keyDelimiter)

	keys := make([]string, 0, len(flattened))

	for key := range flattened {
		keys = append(keys, key)
	}

	sort.Strings(keys)

	var buf bytes.Buffer

	for _, key := range keys {
		_, err := buf.WriteString(fmt.Sprintf("%v=%v\n", strings.ToUpper(key), flattened[key]))
		if err != nil {
			return nil, err
		}
	}

	return buf.Bytes(), nil
}

func (Codec) Decode(b []byte, v map[string]any) error {
	var buf bytes.Buffer

	_, err := buf.Write(b)
	if err != nil {
		return err
	}

	env, err := gotenv.StrictParse(&buf)
	if err != nil {
		return err
	}

	for key, value := range env {
		v[key] = value
	}

	return nil
}
