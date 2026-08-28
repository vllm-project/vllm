// Copyright 2023 The NATS Authors
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

//go:build internal_testing

// Functions in this file are only available when building nats.go with the
// internal_testing build tag. They are used by the nats.go test suite.
package nats

// AddMsgFilter adds a message filter for the given subject
// to the connection. The filter will be called for each
// message received on the subject. If the filter returns
// nil, the message will be dropped.
func (nc *Conn) AddMsgFilter(subject string, filter msgFilter) {
	nc.subsMu.Lock()
	defer nc.subsMu.Unlock()

	if nc.filters == nil {
		nc.filters = make(map[string]msgFilter)
	}
	nc.filters[subject] = filter
}

// RemoveMsgFilter removes a message filter for the given subject.
func (nc *Conn) RemoveMsgFilter(subject string) {
	nc.subsMu.Lock()
	defer nc.subsMu.Unlock()

	if nc.filters != nil {
		delete(nc.filters, subject)
		if len(nc.filters) == 0 {
			nc.filters = nil
		}
	}
}

// IsJSControlMessage returns true if the message is a JetStream control message.
func IsJSControlMessage(msg *Msg) (bool, int) {
	return isJSControlMessage(msg)
}

// CloseTCPConn closes the underlying TCP connection.
// It can be used to simulate a disconnect.
func (nc *Conn) CloseTCPConn() {
	nc.mu.Lock()
	defer nc.mu.Unlock()
	nc.conn.Close()
}
