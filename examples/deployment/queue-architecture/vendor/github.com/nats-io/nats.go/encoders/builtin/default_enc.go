// Copyright 2012-2023 The NATS Authors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package builtin

import (
	"bytes"
	"fmt"
	"reflect"
	"strconv"
	"unsafe"
)

// DefaultEncoder implementation for EncodedConn.
// This encoder will leave []byte and string untouched, but will attempt to
// turn numbers into appropriate strings that can be decoded. It will also
// properly encoded and decode bools. If will encode a struct, but if you want
// to properly handle structures you should use JsonEncoder.
//
// Deprecated: Encoded connections are no longer supported.
type DefaultEncoder struct {
	// Empty
}

var trueB = []byte("true")
var falseB = []byte("false")
var nilB = []byte("")

// Encode
//
// Deprecated: Encoded connections are no longer supported.
func (je *DefaultEncoder) Encode(subject string, v any) ([]byte, error) {
	switch arg := v.(type) {
	case string:
		bytes := *(*[]byte)(unsafe.Pointer(&arg))
		return bytes, nil
	case []byte:
		return arg, nil
	case bool:
		if arg {
			return trueB, nil
		} else {
			return falseB, nil
		}
	case nil:
		return nilB, nil
	default:
		var buf bytes.Buffer
		fmt.Fprintf(&buf, "%+v", arg)
		return buf.Bytes(), nil
	}
}

// Decode
//
// Deprecated: Encoded connections are no longer supported.
func (je *DefaultEncoder) Decode(subject string, data []byte, vPtr any) error {
	// Figure out what it's pointing to...
	sData := *(*string)(unsafe.Pointer(&data))
	switch arg := vPtr.(type) {
	case *string:
		*arg = sData
		return nil
	case *[]byte:
		*arg = data
		return nil
	case *int:
		n, err := strconv.ParseInt(sData, 10, 64)
		if err != nil {
			return err
		}
		*arg = int(n)
		return nil
	case *int32:
		n, err := strconv.ParseInt(sData, 10, 64)
		if err != nil {
			return err
		}
		*arg = int32(n)
		return nil
	case *int64:
		n, err := strconv.ParseInt(sData, 10, 64)
		if err != nil {
			return err
		}
		*arg = int64(n)
		return nil
	case *float32:
		n, err := strconv.ParseFloat(sData, 32)
		if err != nil {
			return err
		}
		*arg = float32(n)
		return nil
	case *float64:
		n, err := strconv.ParseFloat(sData, 64)
		if err != nil {
			return err
		}
		*arg = float64(n)
		return nil
	case *bool:
		b, err := strconv.ParseBool(sData)
		if err != nil {
			return err
		}
		*arg = b
		return nil
	default:
		vt := reflect.TypeOf(arg).Elem()
		return fmt.Errorf("nats: Default Encoder can't decode to type %s", vt)
	}
}
