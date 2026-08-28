// Copyright 2022-2025 The NATS Authors
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

package jetstream

import (
	"context"
	"encoding/json"
	"strings"
)

type (
	apiResponse struct {
		Type  string    `json:"type"`
		Error *APIError `json:"error,omitempty"`
	}

	// apiPaged includes variables used to create paged responses from the JSON API
	apiPaged struct {
		Total  int `json:"total"`
		Offset int `json:"offset"`
		Limit  int `json:"limit"`
	}
)

// Request API subjects for JetStream.
const (
	// DefaultAPIPrefix is the default prefix for the JetStream API.
	DefaultAPIPrefix = "$JS.API."

	// jsDomainT is used to create JetStream API prefix by specifying only Domain
	jsDomainT = "$JS.%s.API."

	// jsExtDomainT is used to create a StreamSource External APIPrefix
	jsExtDomainT = "$JS.%s.API"

	// apiAccountInfo is for obtaining general information about JetStream.
	apiAccountInfo = "INFO"

	// apiConsumerCreateT is used to create consumers.
	apiConsumerCreateT = "CONSUMER.CREATE.%s.%s"

	// apiConsumerCreateWithFilterSubjectT is used to create consumers with a filter subject.
	// it accepts stream name, consumer name and filter subject
	apiConsumerCreateWithFilterSubjectT = "CONSUMER.CREATE.%s.%s.%s"

	// apiConsumerInfoT is used to retrieve consumer information.
	apiConsumerInfoT = "CONSUMER.INFO.%s.%s"

	// apiRequestNextT is the prefix for the request next message(s) for a consumer in worker/pull mode.
	apiRequestNextT = "CONSUMER.MSG.NEXT.%s.%s"

	// apiConsumerDeleteT is used to delete consumers.
	apiConsumerDeleteT = "CONSUMER.DELETE.%s.%s"

	// apiConsumerPauseT is used to pause a consumer.
	apiConsumerPauseT = "CONSUMER.PAUSE.%s.%s"

	// apiConsumerListT is used to return all detailed consumer information
	apiConsumerListT = "CONSUMER.LIST.%s"

	// apiConsumerNamesT is used to return a list with all consumer names for the stream.
	apiConsumerNamesT = "CONSUMER.NAMES.%s"

	// apiStreams can lookup a stream by subject.
	apiStreams = "STREAM.NAMES"

	// apiStreamCreateT is the endpoint to create new streams.
	apiStreamCreateT = "STREAM.CREATE.%s"

	// apiStreamInfoT is the endpoint to get information on a stream.
	apiStreamInfoT = "STREAM.INFO.%s"

	// apiStreamUpdateT is the endpoint to update existing streams.
	apiStreamUpdateT = "STREAM.UPDATE.%s"

	// apiStreamDeleteT is the endpoint to delete streams.
	apiStreamDeleteT = "STREAM.DELETE.%s"

	// apiStreamPurgeT is the endpoint to purge streams.
	apiStreamPurgeT = "STREAM.PURGE.%s"

	// apiStreamListT is the endpoint that will return all detailed stream information
	apiStreamListT = "STREAM.LIST"

	// apiMsgGetT is the endpoint to get a message.
	apiMsgGetT = "STREAM.MSG.GET.%s"

	// apiDirectMsgGetT is the endpoint to perform a direct get of a message.
	apiDirectMsgGetT = "DIRECT.GET.%s"

	// apiDirectMsgGetLastBySubjectT is the endpoint to perform a direct get of a message by subject.
	apiDirectMsgGetLastBySubjectT = "DIRECT.GET.%s.%s"

	// apiMsgDeleteT is the endpoint to remove a message.
	apiMsgDeleteT = "STREAM.MSG.DELETE.%s"

	// apiConsumerUnpinT is the endpoint to unpin a consumer.
	apiConsumerUnpinT = "CONSUMER.UNPIN.%s.%s"

	// apiConsumerResetT is the endpoint to reset a consumer's delivery state.
	apiConsumerResetT = "CONSUMER.RESET.%s.%s"
)

func (js *jetStream) apiRequestJSON(ctx context.Context, subject string, resp any, data ...[]byte) (*jetStreamMsg, error) {
	jsMsg, err := js.apiRequest(ctx, subject, data...)
	if err != nil {
		return nil, err
	}
	if err := json.Unmarshal(jsMsg.Data(), resp); err != nil {
		return nil, err
	}
	return jsMsg, nil
}

// a RequestWithContext with tracing via TraceCB
func (js *jetStream) apiRequest(ctx context.Context, subj string, data ...[]byte) (*jetStreamMsg, error) {
	subj = js.apiSubject(subj)
	var req []byte
	if len(data) > 0 {
		req = data[0]
	}
	if js.opts.ClientTrace != nil {
		ctrace := js.opts.ClientTrace
		if ctrace.RequestSent != nil {
			ctrace.RequestSent(subj, req)
		}
	}
	resp, err := js.conn.RequestWithContext(ctx, subj, req)
	if err != nil {
		return nil, err
	}
	if js.opts.ClientTrace != nil {
		ctrace := js.opts.ClientTrace
		if ctrace.ResponseReceived != nil {
			ctrace.ResponseReceived(subj, resp.Data, resp.Header)
		}
	}

	return js.toJSMsg(resp), nil
}

func (js *jetStream) apiSubject(subj string) string {
	if js.opts.apiPrefix == "" {
		return subj
	}
	var b strings.Builder
	b.WriteString(js.opts.apiPrefix)
	b.WriteString(subj)
	return b.String()
}
