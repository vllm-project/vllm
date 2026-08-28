// Copyright 2013-2023 The NATS Authors
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
	"reflect"
)

// This allows the functionality for network channels by binding send and receive Go chans
// to subjects and optionally queue groups.
// Data will be encoded and decoded via the EncodedConn and its associated encoders.

// BindSendChan binds a channel for send operations to NATS.
//
// Deprecated: Encoded connections are no longer supported.
func (c *EncodedConn) BindSendChan(subject string, channel any) error {
	chVal := reflect.ValueOf(channel)
	if chVal.Kind() != reflect.Chan {
		return ErrChanArg
	}
	go chPublish(c, chVal, subject)
	return nil
}

// Publish all values that arrive on the channel until it is closed or we
// encounter an error.
func chPublish(c *EncodedConn, chVal reflect.Value, subject string) {
	for {
		val, ok := chVal.Recv()
		if !ok {
			// Channel has most likely been closed.
			return
		}
		if e := c.Publish(subject, val.Interface()); e != nil {
			// Do this under lock.
			c.Conn.mu.Lock()
			defer c.Conn.mu.Unlock()

			if c.Conn.Opts.AsyncErrorCB != nil {
				// FIXME(dlc) - Not sure this is the right thing to do.
				// FIXME(ivan) - If the connection is not yet closed, try to schedule the callback
				if c.Conn.isClosed() {
					go c.Conn.Opts.AsyncErrorCB(c.Conn, nil, e)
				} else {
					c.Conn.ach.push(func() { c.Conn.Opts.AsyncErrorCB(c.Conn, nil, e) })
				}
			}
			return
		}
	}
}

// BindRecvChan binds a channel for receive operations from NATS.
//
// Deprecated: Encoded connections are no longer supported.
func (c *EncodedConn) BindRecvChan(subject string, channel any) (*Subscription, error) {
	return c.bindRecvChan(subject, _EMPTY_, channel)
}

// BindRecvQueueChan binds a channel for queue-based receive operations from NATS.
//
// Deprecated: Encoded connections are no longer supported.
func (c *EncodedConn) BindRecvQueueChan(subject, queue string, channel any) (*Subscription, error) {
	return c.bindRecvChan(subject, queue, channel)
}

// Internal function to bind receive operations for a channel.
func (c *EncodedConn) bindRecvChan(subject, queue string, channel any) (*Subscription, error) {
	chVal := reflect.ValueOf(channel)
	if chVal.Kind() != reflect.Chan {
		return nil, ErrChanArg
	}
	argType := chVal.Type().Elem()

	cb := func(m *Msg) {
		var oPtr reflect.Value
		if argType.Kind() != reflect.Ptr {
			oPtr = reflect.New(argType)
		} else {
			oPtr = reflect.New(argType.Elem())
		}
		if err := c.Enc.Decode(m.Subject, m.Data, oPtr.Interface()); err != nil {
			c.Conn.err = errors.New("nats: Got an error trying to unmarshal: " + err.Error())
			if c.Conn.Opts.AsyncErrorCB != nil {
				c.Conn.ach.push(func() { c.Conn.Opts.AsyncErrorCB(c.Conn, m.Sub, c.Conn.err) })
			}
			return
		}
		if argType.Kind() != reflect.Ptr {
			oPtr = reflect.Indirect(oPtr)
		}
		// This is a bit hacky, but in this instance we may be trying to send to a closed channel.
		// and the user does not know when it is safe to close the channel.
		defer func() {
			// If we have panicked, recover and close the subscription.
			if r := recover(); r != nil {
				m.Sub.Unsubscribe()
			}
		}()
		// Actually do the send to the channel.
		chVal.Send(oPtr)
	}

	return c.Conn.subscribe(subject, queue, cb, nil, nil, false, nil)
}
