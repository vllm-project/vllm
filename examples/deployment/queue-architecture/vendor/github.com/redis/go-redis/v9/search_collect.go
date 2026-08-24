package redis

import (
	"fmt"
	"strings"
)

// ----------------------
// FT.AGGREGATE COLLECT reducer
// ----------------------
//
// COLLECT is a GROUPBY reducer for FT.AGGREGATE (Redis 8.8+, gated behind
// search-enable-unstable-features). Within each group it projects a chosen
// set of fields from every row, optionally deduplicates, sorts, and limits
// them, and emits the result as an array of per-entry maps under the reducer
// alias.
//
// COLLECT is not a standalone command; it is a REDUCE clause inside
// FT.AGGREGATE. The helpers below assemble the reducer token list and compute
// its argument count, so callers do not have to hand-write FIELDS/SORTBY/LIMIT
// tokens or remember to @-prefix every name.

// FTAggregateCollect describes a COLLECT reducer. It is rendered into a
// standard FTAggregateReducer via NewCollectReducer, or appended to a builder
// via AggregateBuilder.Collect.
//
// Field and sort names may be supplied with or without a leading "@"; each is
// normalized to a single "@<name>" on the wire. Output map keys returned by
// the server are the bare names (see AggregateRow.Collect).
type FTAggregateCollect struct {
	// FieldsAll emits FIELDS *, projecting every field present in the
	// pipeline at the COLLECT stage. It is not a whole-document fetch; pair
	// it with an upstream LOAD * to collect complete documents. FieldsAll
	// takes precedence over Fields when both are set.
	FieldsAll bool

	// Fields is the explicit list of fields to project (FIELDS <n> @f ...).
	// Ignored when FieldsAll is true. Exactly one of FieldsAll or a non-empty
	// Fields must be set.
	Fields []string

	// Distinct emits DISTINCT, deduplicating entries with identical projected
	// fields.
	//
	// NOTE: DISTINCT is specified by the product but not yet implemented by
	// the server. Sending it currently produces a server error. The option is
	// kept for forward compatibility; leave it false unless the target server
	// supports it.
	Distinct bool

	// SortBy orders entries within each group. Direction defaults to ASC when
	// neither Asc nor Desc is set. With Limit, SORTBY acts as a top-N
	// selection. Reuses FTAggregateSortBy for consistency with the rest of the
	// aggregate API.
	SortBy []FTAggregateSortBy

	// Limit returns at most Count entries per group after skipping Offset.
	// nil means no LIMIT clause (distinct from LIMIT 0 0).
	Limit *FTAggregateCollectLimit

	// As sets the reducer output column name (AS <alias>). It is emitted
	// outside the reducer argument count.
	As string
}

// FTAggregateCollectLimit is the LIMIT <offset> <count> clause of a COLLECT
// reducer. Numeric bounds are enforced by the server, not the client.
type FTAggregateCollectLimit struct {
	Offset int
	Count  int
}

// ensureAtPrefix normalizes a field or sort name to exactly one leading "@",
// collapsing any number of leading "@" (including none) to a single prefix.
func ensureAtPrefix(name string) string {
	return "@" + strings.TrimLeft(name, "@")
}

// buildCollectArgs renders a FTAggregateCollect into the reducer argument
// token list (everything after "REDUCE COLLECT <narg>", excluding AS <alias>).
// The serializer computes <narg> as len(args), which matches the COLLECT
// contract: narg counts every FIELDS/DISTINCT/SORTBY/LIMIT token.
func buildCollectArgs(o FTAggregateCollect) ([]interface{}, error) {
	args := make([]interface{}, 0, 8)

	// FIELDS (required): either * or a counted list of @-names.
	switch {
	case o.FieldsAll:
		args = append(args, "FIELDS", "*")
	case len(o.Fields) > 0:
		args = append(args, "FIELDS", len(o.Fields))
		for _, f := range o.Fields {
			if strings.TrimLeft(f, "@") == "" {
				return nil, fmt.Errorf("redis: FT.AGGREGATE COLLECT: empty field name in Fields")
			}
			args = append(args, ensureAtPrefix(f))
		}
	default:
		return nil, fmt.Errorf("redis: FT.AGGREGATE COLLECT requires FieldsAll or a non-empty Fields list")
	}

	// DISTINCT (optional, forward-compatible).
	if o.Distinct {
		args = append(args, "DISTINCT")
	}

	// SORTBY (optional). sort_narg counts each field plus its optional
	// direction token.
	if len(o.SortBy) > 0 {
		sortTokens := make([]interface{}, 0, len(o.SortBy)*2)
		for _, s := range o.SortBy {
			if strings.TrimLeft(s.FieldName, "@") == "" {
				return nil, fmt.Errorf("redis: FT.AGGREGATE COLLECT: empty field name in SortBy")
			}
			if s.Asc && s.Desc {
				return nil, fmt.Errorf("redis: FT.AGGREGATE COLLECT: ASC and DESC are mutually exclusive")
			}
			sortTokens = append(sortTokens, ensureAtPrefix(s.FieldName))
			switch {
			case s.Desc:
				sortTokens = append(sortTokens, "DESC")
			case s.Asc:
				sortTokens = append(sortTokens, "ASC")
				// neither set: ASC is the server default; emit nothing.
			}
		}
		args = append(args, "SORTBY", len(sortTokens))
		args = append(args, sortTokens...)
	}

	// LIMIT (optional).
	if o.Limit != nil {
		args = append(args, "LIMIT", o.Limit.Offset, o.Limit.Count)
	}

	return args, nil
}

// NewCollectReducer builds a COLLECT FTAggregateReducer for use with
// FTAggregateOptions.GroupBy[i].Reduce. It normalizes field/sort names and
// computes the argument count automatically.
//
// It returns an error only for local API misuse: a missing FIELDS selector, an
// empty field name (in Fields or SortBy), or a SortBy entry with both Asc and
// Desc set. Numeric bounds and the unstable-features gate are enforced by the
// server and surface unchanged through the command reply.
func NewCollectReducer(o FTAggregateCollect) (FTAggregateReducer, error) {
	args, err := buildCollectArgs(o)
	if err != nil {
		return FTAggregateReducer{}, err
	}
	return FTAggregateReducer{Reducer: SearchCollect, Args: args, As: o.As}, nil
}

// ----------------------
// COLLECT response decoding
// ----------------------

// CollectEntry is a single collected row: a sparse map of bare field name to
// value. A field absent from a row is omitted from its entry (no NULL
// placeholder), so entries in the same column may have different key sets.
type CollectEntry = map[string]interface{}

// CollectColumn is the value stored under a COLLECT reducer alias: the ordered
// list of collected entries for a group.
type CollectColumn = []CollectEntry

// Collect decodes the COLLECT reducer column stored under alias in this row
// into a uniform CollectColumn, hiding the RESP2/RESP3 representation
// difference (RESP3 entries are maps; RESP2 entries are flat key/value
// arrays).
//
// It returns (nil, nil) when the alias is absent from the row. Entry order is
// preserved as returned by the server; it is meaningful only when the COLLECT
// reducer was given a SORTBY.
func (r AggregateRow) Collect(alias string) (CollectColumn, error) {
	v, ok := r.Fields[alias]
	if !ok {
		return nil, nil
	}
	return parseCollectValue(v)
}

// parseCollectValue decodes a raw COLLECT alias value (an array of entries)
// into a CollectColumn.
func parseCollectValue(v interface{}) (CollectColumn, error) {
	if v == nil {
		return nil, nil
	}
	arr, ok := v.([]interface{})
	if !ok {
		return nil, fmt.Errorf("redis: COLLECT value has type %T, want array of entries", v)
	}
	out := make(CollectColumn, 0, len(arr))
	for i, e := range arr {
		entry, err := parseCollectEntry(e)
		if err != nil {
			return nil, fmt.Errorf("redis: COLLECT entry %d: %w", i, err)
		}
		out = append(out, entry)
	}
	return out, nil
}

// parseCollectEntry decodes a single collected entry from either the RESP3
// map form or the RESP2 flat key/value array form into a CollectEntry. Keys
// are passed through as-is: the server already returns them without the "@"
// prefix.
func parseCollectEntry(e interface{}) (CollectEntry, error) {
	switch m := e.(type) {
	case map[interface{}]interface{}: // RESP3
		out := make(CollectEntry, len(m))
		for k, val := range m {
			out[fmt.Sprint(k)] = val
		}
		return out, nil
	case map[string]interface{}: // already string-keyed
		return m, nil
	case []interface{}: // RESP2 flat [field, value, field, value, ...]
		if len(m)%2 != 0 {
			return nil, fmt.Errorf("odd-length key/value array of length %d", len(m))
		}
		out := make(CollectEntry, len(m)/2)
		for i := 0; i < len(m); i += 2 {
			key, ok := m[i].(string)
			if !ok {
				key = fmt.Sprint(m[i])
			}
			out[key] = m[i+1]
		}
		return out, nil
	default:
		return nil, fmt.Errorf("unexpected type %T, want map or key/value array", e)
	}
}
