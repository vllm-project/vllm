// Copyright 2012-2024 The NATS Authors
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

//go:build go1.23

package nats

import (
	"errors"
	"iter"
	"time"
)

// Msgs returns an iter.Seq2[*Msg, error] that can be used to iterate over
// messages. It can only be used with a subscription that has been created with
// SubscribeSync or QueueSubscribeSync, otherwise it will return an error on the
// first iteration.
//
// The iterator will block until a message is available. The
// subscription will not be closed when the iterator is done.
func (sub *Subscription) Msgs() iter.Seq2[*Msg, error] {
	return func(yield func(*Msg, error) bool) {
		for {
			msg, err := sub.nextMsgNoTimeout()
			if err != nil {
				yield(nil, err)
				return
			}
			if !yield(msg, nil) {
				return
			}

		}
	}
}

// MsgsTimeout returns an iter.Seq2[*Msg, error] that can be used to iterate
// over messages. It can only be used with a subscription that has been created
// with SubscribeSync or QueueSubscribeSync, otherwise it will return an error
// on the first iteration.
//
// The iterator will block until a message is available or the timeout is
// reached. If the timeout is reached, the iterator will return nats.ErrTimeout
// but it will not be closed.
func (sub *Subscription) MsgsTimeout(timeout time.Duration) iter.Seq2[*Msg, error] {
	return func(yield func(*Msg, error) bool) {
		for {
			msg, err := sub.NextMsg(timeout)
			if err != nil {
				if !yield(nil, err) {
					return
				}
				if !errors.Is(err, ErrTimeout) {
					return
				}
				continue
			}
			if !yield(msg, nil) {
				return
			}
		}
	}
}
