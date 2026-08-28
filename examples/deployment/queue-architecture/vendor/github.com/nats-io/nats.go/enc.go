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

package nats

import (
	"errors"
	"fmt"
	"reflect"
	"sync"
	"time"

	// Default Encoders
	"github.com/nats-io/nats.go/encoders/builtin"
)

//lint:file-ignore SA1019 Ignore deprecation warnings for EncodedConn

// Encoder interface is for all register encoders
//
// Deprecated: Encoded connections are no longer supported.
type Encoder interface {
	Encode(subject string, v any) ([]byte, error)
	Decode(subject string, data []byte, vPtr any) error
}

var encMap map[string]Encoder
var encLock sync.Mutex

// Indexed names into the Registered Encoders.
const (
	JSON_ENCODER    = "json"
	GOB_ENCODER     = "gob"
	DEFAULT_ENCODER = "default"
)

func init() {
	encMap = make(map[string]Encoder)
	// Register json, gob and default encoder
	RegisterEncoder(JSON_ENCODER, &builtin.JsonEncoder{})
	RegisterEncoder(GOB_ENCODER, &builtin.GobEncoder{})
	RegisterEncoder(DEFAULT_ENCODER, &builtin.DefaultEncoder{})
}

// EncodedConn are the preferred way to interface with NATS. They wrap a bare connection to
// a nats server and have an extendable encoder system that will encode and decode messages
// from raw Go types.
//
// Deprecated: Encoded connections are no longer supported.
type EncodedConn struct {
	Conn *Conn
	Enc  Encoder
}

// NewEncodedConn will wrap an existing Connection and utilize the appropriate registered
// encoder.
//
// Deprecated: Encoded connections are no longer supported.
func NewEncodedConn(c *Conn, encType string) (*EncodedConn, error) {
	if c == nil {
		return nil, errors.New("nats: Nil Connection")
	}
	if c.IsClosed() {
		return nil, ErrConnectionClosed
	}
	ec := &EncodedConn{Conn: c, Enc: EncoderForType(encType)}
	if ec.Enc == nil {
		return nil, fmt.Errorf("no encoder registered for '%s'", encType)
	}
	return ec, nil
}

// RegisterEncoder will register the encType with the given Encoder. Useful for customization.
//
// Deprecated: Encoded connections are no longer supported.
func RegisterEncoder(encType string, enc Encoder) {
	encLock.Lock()
	defer encLock.Unlock()
	encMap[encType] = enc
}

// EncoderForType will return the registered Encoder for the encType.
//
// Deprecated: Encoded connections are no longer supported.
func EncoderForType(encType string) Encoder {
	encLock.Lock()
	defer encLock.Unlock()
	return encMap[encType]
}

// Publish publishes the data argument to the given subject. The data argument
// will be encoded using the associated encoder.
//
// Deprecated: Encoded connections are no longer supported.
func (c *EncodedConn) Publish(subject string, v any) error {
	b, err := c.Enc.Encode(subject, v)
	if err != nil {
		return err
	}
	return c.Conn.publish(subject, _EMPTY_, false, nil, b)
}

// PublishRequest will perform a Publish() expecting a response on the
// reply subject. Use Request() for automatically waiting for a response
// inline.
//
// Deprecated: Encoded connections are no longer supported.
func (c *EncodedConn) PublishRequest(subject, reply string, v any) error {
	b, err := c.Enc.Encode(subject, v)
	if err != nil {
		return err
	}
	return c.Conn.publish(subject, reply, true, nil, b)
}

// Request will create an Inbox and perform a Request() call
// with the Inbox reply for the data v. A response will be
// decoded into the vPtr Response.
//
// Deprecated: Encoded connections are no longer supported.
func (c *EncodedConn) Request(subject string, v any, vPtr any, timeout time.Duration) error {
	b, err := c.Enc.Encode(subject, v)
	if err != nil {
		return err
	}
	m, err := c.Conn.Request(subject, b, timeout)
	if err != nil {
		return err
	}
	if reflect.TypeOf(vPtr) == emptyMsgType {
		mPtr := vPtr.(*Msg)
		*mPtr = *m
	} else {
		err = c.Enc.Decode(m.Subject, m.Data, vPtr)
	}
	return err
}

// Handler is a specific callback used for Subscribe. It is generalized to
// an any, but we will discover its format and arguments at runtime
// and perform the correct callback, including demarshaling encoded data
// back into the appropriate struct based on the signature of the Handler.
//
// Handlers are expected to have one of four signatures.
//
//	type person struct {
//		Name string `json:"name,omitempty"`
//		Age  uint   `json:"age,omitempty"`
//	}
//
//	handler := func(m *Msg)
//	handler := func(p *person)
//	handler := func(subject string, o *obj)
//	handler := func(subject, reply string, o *obj)
//
// These forms allow a callback to request a raw Msg ptr, where the processing
// of the message from the wire is untouched. Process a JSON representation
// and demarshal it into the given struct, e.g. person.
// There are also variants where the callback wants either the subject, or the
// subject and the reply subject.
//
// Deprecated: Encoded connections are no longer supported.
type Handler any

// Dissect the cb Handler's signature
func argInfo(cb Handler) (reflect.Type, int) {
	cbType := reflect.TypeOf(cb)
	if cbType.Kind() != reflect.Func {
		panic("nats: Handler needs to be a func")
	}
	numArgs := cbType.NumIn()
	if numArgs == 0 {
		return nil, numArgs
	}
	return cbType.In(numArgs - 1), numArgs
}

var emptyMsgType = reflect.TypeOf(&Msg{})

// Subscribe will create a subscription on the given subject and process incoming
// messages using the specified Handler. The Handler should be a func that matches
// a signature from the description of Handler from above.
//
// Deprecated: Encoded connections are no longer supported.
func (c *EncodedConn) Subscribe(subject string, cb Handler) (*Subscription, error) {
	return c.subscribe(subject, _EMPTY_, cb)
}

// QueueSubscribe will create a queue subscription on the given subject and process
// incoming messages using the specified Handler. The Handler should be a func that
// matches a signature from the description of Handler from above.
//
// Deprecated: Encoded connections are no longer supported.
func (c *EncodedConn) QueueSubscribe(subject, queue string, cb Handler) (*Subscription, error) {
	return c.subscribe(subject, queue, cb)
}

// Internal implementation that all public functions will use.
func (c *EncodedConn) subscribe(subject, queue string, cb Handler) (*Subscription, error) {
	if cb == nil {
		return nil, errors.New("nats: Handler required for EncodedConn Subscription")
	}
	argType, numArgs := argInfo(cb)
	if argType == nil {
		return nil, errors.New("nats: Handler requires at least one argument")
	}

	cbValue := reflect.ValueOf(cb)
	wantsRaw := (argType == emptyMsgType)

	natsCB := func(m *Msg) {
		var oV []reflect.Value
		if wantsRaw {
			oV = []reflect.Value{reflect.ValueOf(m)}
		} else {
			var oPtr reflect.Value
			if argType.Kind() != reflect.Ptr {
				oPtr = reflect.New(argType)
			} else {
				oPtr = reflect.New(argType.Elem())
			}
			if err := c.Enc.Decode(m.Subject, m.Data, oPtr.Interface()); err != nil {
				if c.Conn.Opts.AsyncErrorCB != nil {
					c.Conn.ach.push(func() {
						c.Conn.Opts.AsyncErrorCB(c.Conn, m.Sub, errors.New("nats: Got an error trying to unmarshal: "+err.Error()))
					})
				}
				return
			}
			if argType.Kind() != reflect.Ptr {
				oPtr = reflect.Indirect(oPtr)
			}

			// Callback Arity
			switch numArgs {
			case 1:
				oV = []reflect.Value{oPtr}
			case 2:
				subV := reflect.ValueOf(m.Subject)
				oV = []reflect.Value{subV, oPtr}
			case 3:
				subV := reflect.ValueOf(m.Subject)
				replyV := reflect.ValueOf(m.Reply)
				oV = []reflect.Value{subV, replyV, oPtr}
			}

		}
		cbValue.Call(oV)
	}

	return c.Conn.subscribe(subject, queue, natsCB, nil, nil, false, nil)
}

// FlushTimeout allows a Flush operation to have an associated timeout.
//
// Deprecated: Encoded connections are no longer supported.
func (c *EncodedConn) FlushTimeout(timeout time.Duration) (err error) {
	return c.Conn.FlushTimeout(timeout)
}

// Flush will perform a round trip to the server and return when it
// receives the internal reply.
//
// Deprecated: Encoded connections are no longer supported.
func (c *EncodedConn) Flush() error {
	return c.Conn.Flush()
}

// Close will close the connection to the server. This call will release
// all blocking calls, such as Flush(), etc.
//
// Deprecated: Encoded connections are no longer supported.
func (c *EncodedConn) Close() {
	c.Conn.Close()
}

// Drain will put a connection into a drain state. All subscriptions will
// immediately be put into a drain state. Upon completion, the publishers
// will be drained and can not publish any additional messages. Upon draining
// of the publishers, the connection will be closed. Use the ClosedCB()
// option to know when the connection has moved from draining to closed.
//
// Deprecated: Encoded connections are no longer supported.
func (c *EncodedConn) Drain() error {
	return c.Conn.Drain()
}

// LastError reports the last error encountered via the Connection.
//
// Deprecated: Encoded connections are no longer supported.
func (c *EncodedConn) LastError() error {
	return c.Conn.LastError()
}
