package redis

import (
	"context"
	"fmt"
)

// ----------------------
// Search Module Builders
// ----------------------

// SearchBuilder provides a fluent API for FT.SEARCH
// (see original FTSearchOptions for all options).
// EXPERIMENTAL: this API is subject to change, use with caution.
type SearchBuilder struct {
	c       *Client
	ctx     context.Context
	index   string
	query   string
	options *FTSearchOptions
}

// NewSearchBuilder creates a new SearchBuilder for FT.SEARCH commands.
// EXPERIMENTAL: this API is subject to change, use with caution.
func (c *Client) NewSearchBuilder(ctx context.Context, index, query string) *SearchBuilder {
	b := &SearchBuilder{c: c, ctx: ctx, index: index, query: query, options: &FTSearchOptions{LimitOffset: -1}}
	return b
}

// WithScores includes WITHSCORES.
func (b *SearchBuilder) WithScores() *SearchBuilder {
	b.options.WithScores = true
	return b
}

// NoContent includes NOCONTENT.
func (b *SearchBuilder) NoContent() *SearchBuilder { b.options.NoContent = true; return b }

// Verbatim includes VERBATIM.
func (b *SearchBuilder) Verbatim() *SearchBuilder { b.options.Verbatim = true; return b }

// NoStopWords includes NOSTOPWORDS.
func (b *SearchBuilder) NoStopWords() *SearchBuilder { b.options.NoStopWords = true; return b }

// WithPayloads includes WITHPAYLOADS.
func (b *SearchBuilder) WithPayloads() *SearchBuilder {
	b.options.WithPayloads = true
	return b
}

// WithSortKeys includes WITHSORTKEYS.
func (b *SearchBuilder) WithSortKeys() *SearchBuilder {
	b.options.WithSortKeys = true
	return b
}

// Filter adds a FILTER clause: FILTER <field> <min> <max>.
func (b *SearchBuilder) Filter(field string, min, max interface{}) *SearchBuilder {
	b.options.Filters = append(b.options.Filters, FTSearchFilter{
		FieldName: field,
		Min:       min,
		Max:       max,
	})
	return b
}

// GeoFilter adds a GEOFILTER clause: GEOFILTER <field> <lon> <lat> <radius> <unit>.
func (b *SearchBuilder) GeoFilter(field string, lon, lat, radius float64, unit string) *SearchBuilder {
	b.options.GeoFilter = append(b.options.GeoFilter, FTSearchGeoFilter{
		FieldName: field,
		Longitude: lon,
		Latitude:  lat,
		Radius:    radius,
		Unit:      unit,
	})
	return b
}

// InKeys restricts the search to the given keys.
func (b *SearchBuilder) InKeys(keys ...interface{}) *SearchBuilder {
	b.options.InKeys = append(b.options.InKeys, keys...)
	return b
}

// InFields restricts the search to the given fields.
func (b *SearchBuilder) InFields(fields ...interface{}) *SearchBuilder {
	b.options.InFields = append(b.options.InFields, fields...)
	return b
}

// ReturnFields adds simple RETURN <n> <field>...
func (b *SearchBuilder) ReturnFields(fields ...string) *SearchBuilder {
	for _, f := range fields {
		b.options.Return = append(b.options.Return, FTSearchReturn{FieldName: f})
	}
	return b
}

// ReturnAs adds RETURN <field> AS <alias>.
func (b *SearchBuilder) ReturnAs(field, alias string) *SearchBuilder {
	b.options.Return = append(b.options.Return, FTSearchReturn{FieldName: field, As: alias})
	return b
}

// Slop adds SLOP <n>.
func (b *SearchBuilder) Slop(slop int) *SearchBuilder {
	b.options.Slop = slop
	return b
}

// Timeout adds TIMEOUT <ms>.
func (b *SearchBuilder) Timeout(timeout int) *SearchBuilder {
	b.options.Timeout = timeout
	return b
}

// InOrder includes INORDER.
func (b *SearchBuilder) InOrder() *SearchBuilder {
	b.options.InOrder = true
	return b
}

// Language sets LANGUAGE <lang>.
func (b *SearchBuilder) Language(lang string) *SearchBuilder {
	b.options.Language = lang
	return b
}

// Expander sets EXPANDER <expander>.
func (b *SearchBuilder) Expander(expander string) *SearchBuilder {
	b.options.Expander = expander
	return b
}

// Scorer sets SCORER <scorer>.
func (b *SearchBuilder) Scorer(scorer string) *SearchBuilder {
	b.options.Scorer = scorer
	return b
}

// ExplainScore includes EXPLAINSCORE.
func (b *SearchBuilder) ExplainScore() *SearchBuilder {
	b.options.ExplainScore = true
	return b
}

// Payload sets PAYLOAD <payload>.
func (b *SearchBuilder) Payload(payload string) *SearchBuilder {
	b.options.Payload = payload
	return b
}

// SortBy adds SORTBY <field> ASC|DESC.
func (b *SearchBuilder) SortBy(field string, asc bool) *SearchBuilder {
	b.options.SortBy = append(b.options.SortBy, FTSearchSortBy{
		FieldName: field,
		Asc:       asc,
		Desc:      !asc,
	})
	return b
}

// WithSortByCount includes WITHCOUNT (when used with SortBy).
func (b *SearchBuilder) WithSortByCount() *SearchBuilder {
	b.options.SortByWithCount = true
	return b
}

// Param adds a single PARAMS <k> <v>.
func (b *SearchBuilder) Param(key string, value interface{}) *SearchBuilder {
	if b.options.Params == nil {
		b.options.Params = make(map[string]interface{}, 1)
	}
	b.options.Params[key] = value
	return b
}

// ParamsMap adds multiple PARAMS at once.
func (b *SearchBuilder) ParamsMap(p map[string]interface{}) *SearchBuilder {
	if b.options.Params == nil {
		b.options.Params = make(map[string]interface{}, len(p))
	}
	for k, v := range p {
		b.options.Params[k] = v
	}
	return b
}

// Dialect sets DIALECT <version>.
func (b *SearchBuilder) Dialect(version int) *SearchBuilder {
	b.options.DialectVersion = version
	return b
}

// Limit sets OFFSET and COUNT. CountOnly uses LIMIT 0 0.
func (b *SearchBuilder) Limit(offset, count int) *SearchBuilder {
	b.options.LimitOffset = offset
	b.options.Limit = count
	return b
}
func (b *SearchBuilder) CountOnly() *SearchBuilder { b.options.CountOnly = true; return b }

// Run executes FT.SEARCH and returns a typed result.
func (b *SearchBuilder) Run() (FTSearchResult, error) {
	cmd := b.c.FTSearchWithArgs(b.ctx, b.index, b.query, b.options)
	return cmd.Result()
}

// ----------------------
// AggregateBuilder for FT.AGGREGATE
// ----------------------

type AggregateBuilder struct {
	c       *Client
	ctx     context.Context
	index   string
	query   string
	options *FTAggregateOptions
	err     error
}

// NewAggregateBuilder creates a new AggregateBuilder for FT.AGGREGATE commands.
// EXPERIMENTAL: this API is subject to change, use with caution.
func (c *Client) NewAggregateBuilder(ctx context.Context, index, query string) *AggregateBuilder {
	return &AggregateBuilder{c: c, ctx: ctx, index: index, query: query, options: &FTAggregateOptions{LimitOffset: -1}}
}

// setErr records the first error produced while building the pipeline.
// Subsequent errors are ignored; the first error is returned from Run.
func (b *AggregateBuilder) setErr(err error) {
	if b.err == nil {
		b.err = err
	}
}

// Verbatim includes VERBATIM.
func (b *AggregateBuilder) Verbatim() *AggregateBuilder { b.options.Verbatim = true; return b }

// AddScores includes ADDSCORES.
func (b *AggregateBuilder) AddScores() *AggregateBuilder { b.options.AddScores = true; return b }

// Scorer sets SCORER <scorer>.
func (b *AggregateBuilder) Scorer(s string) *AggregateBuilder {
	b.options.Scorer = s
	return b
}

// LoadAll includes LOAD * (mutually exclusive with Load).
func (b *AggregateBuilder) LoadAll() *AggregateBuilder {
	b.options.LoadAll = true
	return b
}

// Load adds a LOAD <field> [AS alias] step.
// You can call it multiple times; each call becomes a separate LOAD clause
// at its position in the pipeline.
func (b *AggregateBuilder) Load(field string, alias ...string) *AggregateBuilder {
	l := &FTAggregateLoad{Field: field}
	if len(alias) > 0 {
		l.As = alias[0]
	}
	b.options.Steps = append(b.options.Steps, FTAggregateStep{Load: l})
	return b
}

// Timeout sets TIMEOUT <ms>.
func (b *AggregateBuilder) Timeout(ms int) *AggregateBuilder {
	b.options.Timeout = ms
	return b
}

// Apply adds an APPLY <field> [AS alias] step.
func (b *AggregateBuilder) Apply(field string, alias ...string) *AggregateBuilder {
	a := &FTAggregateApply{Field: field}
	if len(alias) > 0 {
		a.As = alias[0]
	}
	b.options.Steps = append(b.options.Steps, FTAggregateStep{Apply: a})
	return b
}

// GroupBy adds a new GROUPBY <fields...> step.
func (b *AggregateBuilder) GroupBy(fields ...interface{}) *AggregateBuilder {
	b.options.Steps = append(b.options.Steps, FTAggregateStep{
		GroupBy: &FTAggregateGroupBy{Fields: fields},
	})
	return b
}

// Reduce adds a REDUCE <fn> [<#args> <args...>] clause to the last step,
// which must be a GROUPBY. If it is not, Run will return an error.
func (b *AggregateBuilder) Reduce(fn SearchAggregator, args ...interface{}) *AggregateBuilder {
	n := len(b.options.Steps)
	if n == 0 || b.options.Steps[n-1].GroupBy == nil {
		b.setErr(fmt.Errorf("FT.AGGREGATE: Reduce must follow a GroupBy step"))
		return b
	}
	g := b.options.Steps[n-1].GroupBy
	g.Reduce = append(g.Reduce, FTAggregateReducer{Reducer: fn, Args: args})
	return b
}

// ReduceAs does the same but also sets an alias: REDUCE <fn> … AS <alias>.
// The last step must be a GROUPBY; otherwise Run will return an error.
func (b *AggregateBuilder) ReduceAs(fn SearchAggregator, alias string, args ...interface{}) *AggregateBuilder {
	n := len(b.options.Steps)
	if n == 0 || b.options.Steps[n-1].GroupBy == nil {
		b.setErr(fmt.Errorf("FT.AGGREGATE: ReduceAs must follow a GroupBy step"))
		return b
	}
	g := b.options.Steps[n-1].GroupBy
	g.Reduce = append(g.Reduce, FTAggregateReducer{Reducer: fn, Args: args, As: alias})
	return b
}

// Collect adds a REDUCE COLLECT clause to the last step, which must be a
// GROUPBY. The COLLECT options (FIELDS/DISTINCT/SORTBY/LIMIT/AS) are rendered
// and the argument count is computed automatically; field and sort names are
// normalized to a single "@" prefix. Set FTAggregateCollect.As to alias the
// output column.
//
// If the last step is not a GROUPBY, or the options are invalid (no FIELDS
// selector), Run returns the recorded error without issuing the command.
// COLLECT requires Redis 8.8+ with unstable features enabled.
func (b *AggregateBuilder) Collect(o FTAggregateCollect) *AggregateBuilder {
	n := len(b.options.Steps)
	if n == 0 || b.options.Steps[n-1].GroupBy == nil {
		b.setErr(fmt.Errorf("FT.AGGREGATE: Collect must follow a GroupBy step"))
		return b
	}
	reducer, err := NewCollectReducer(o)
	if err != nil {
		b.setErr(err)
		return b
	}
	g := b.options.Steps[n-1].GroupBy
	g.Reduce = append(g.Reduce, reducer)
	return b
}

// SortBy adds SORTBY <field> ASC|DESC. Consecutive SortBy calls (with no
// other step in between) are merged into a single SORTBY clause so fields
// act as tiebreakers. A SortBy call after a non-SortBy step starts a new
// SORTBY step.
//
// Note: this is a semantics change from earlier experimental versions of
// the builder, where SortBy always accumulated into a single SORTBY clause
// regardless of position in the pipeline.
func (b *AggregateBuilder) SortBy(field string, asc bool) *AggregateBuilder {
	sb := FTAggregateSortBy{FieldName: field, Asc: asc, Desc: !asc}
	if n := len(b.options.Steps); n > 0 && b.options.Steps[n-1].SortBy != nil {
		b.options.Steps[n-1].SortBy.Fields = append(b.options.Steps[n-1].SortBy.Fields, sb)
		return b
	}
	b.options.Steps = append(b.options.Steps, FTAggregateStep{
		SortBy: &FTAggregateSortByStep{Fields: []FTAggregateSortBy{sb}},
	})
	return b
}

// SortByMax sets MAX <n> on the last SORTBY step. The last step must be a
// SORTBY; otherwise Run will return an error.
func (b *AggregateBuilder) SortByMax(max int) *AggregateBuilder {
	n := len(b.options.Steps)
	if n == 0 || b.options.Steps[n-1].SortBy == nil {
		b.setErr(fmt.Errorf("FT.AGGREGATE: SortByMax must follow a SortBy step"))
		return b
	}
	b.options.Steps[n-1].SortBy.Max = max
	return b
}

// Filter sets FILTER <expr>.
func (b *AggregateBuilder) Filter(expr string) *AggregateBuilder {
	b.options.Filter = expr
	return b
}

// WithCursor enables WITHCURSOR [COUNT <n>] [MAXIDLE <ms>].
func (b *AggregateBuilder) WithCursor(count, maxIdle int) *AggregateBuilder {
	b.options.WithCursor = true
	if b.options.WithCursorOptions == nil {
		b.options.WithCursorOptions = &FTAggregateWithCursor{}
	}
	b.options.WithCursorOptions.Count = count
	b.options.WithCursorOptions.MaxIdle = maxIdle
	return b
}

// Params adds PARAMS <k v> pairs.
func (b *AggregateBuilder) Params(p map[string]interface{}) *AggregateBuilder {
	if b.options.Params == nil {
		b.options.Params = make(map[string]interface{}, len(p))
	}
	for k, v := range p {
		b.options.Params[k] = v
	}
	return b
}

// Dialect sets DIALECT <version>.
func (b *AggregateBuilder) Dialect(version int) *AggregateBuilder {
	b.options.DialectVersion = version
	return b
}

// Run executes FT.AGGREGATE and returns a typed result. If the builder
// recorded a validation error while constructing the pipeline (for example,
// calling SortByMax when the last step is not a SortBy), that error is
// returned without issuing the command.
func (b *AggregateBuilder) Run() (*FTAggregateResult, error) {
	if b.err != nil {
		return nil, b.err
	}
	cmd := b.c.FTAggregateWithArgs(b.ctx, b.index, b.query, b.options)
	return cmd.Result()
}

// ----------------------
// CreateIndexBuilder for FT.CREATE
// ----------------------
// CreateIndexBuilder is builder for FT.CREATE
// EXPERIMENTAL: this API is subject to change, use with caution.
type CreateIndexBuilder struct {
	c       *Client
	ctx     context.Context
	index   string
	options *FTCreateOptions
	schema  []*FieldSchema
}

// NewCreateIndexBuilder creates a new CreateIndexBuilder for FT.CREATE commands.
// EXPERIMENTAL: this API is subject to change, use with caution.
func (c *Client) NewCreateIndexBuilder(ctx context.Context, index string) *CreateIndexBuilder {
	return &CreateIndexBuilder{c: c, ctx: ctx, index: index, options: &FTCreateOptions{}}
}

// OnHash sets ON HASH.
func (b *CreateIndexBuilder) OnHash() *CreateIndexBuilder { b.options.OnHash = true; return b }

// OnJSON sets ON JSON.
func (b *CreateIndexBuilder) OnJSON() *CreateIndexBuilder { b.options.OnJSON = true; return b }

// Prefix sets PREFIX.
func (b *CreateIndexBuilder) Prefix(prefixes ...interface{}) *CreateIndexBuilder {
	b.options.Prefix = prefixes
	return b
}

// Filter sets FILTER.
func (b *CreateIndexBuilder) Filter(filter string) *CreateIndexBuilder {
	b.options.Filter = filter
	return b
}

// DefaultLanguage sets LANGUAGE.
func (b *CreateIndexBuilder) DefaultLanguage(lang string) *CreateIndexBuilder {
	b.options.DefaultLanguage = lang
	return b
}

// LanguageField sets LANGUAGE_FIELD.
func (b *CreateIndexBuilder) LanguageField(field string) *CreateIndexBuilder {
	b.options.LanguageField = field
	return b
}

// Score sets SCORE.
func (b *CreateIndexBuilder) Score(score float64) *CreateIndexBuilder {
	b.options.Score = score
	return b
}

// ScoreField sets SCORE_FIELD.
func (b *CreateIndexBuilder) ScoreField(field string) *CreateIndexBuilder {
	b.options.ScoreField = field
	return b
}

// PayloadField sets PAYLOAD_FIELD.
func (b *CreateIndexBuilder) PayloadField(field string) *CreateIndexBuilder {
	b.options.PayloadField = field
	return b
}

// NoOffsets includes NOOFFSETS.
func (b *CreateIndexBuilder) NoOffsets() *CreateIndexBuilder { b.options.NoOffsets = true; return b }

// Temporary sets TEMPORARY seconds.
func (b *CreateIndexBuilder) Temporary(sec int) *CreateIndexBuilder {
	b.options.Temporary = sec
	return b
}

// NoHL includes NOHL.
func (b *CreateIndexBuilder) NoHL() *CreateIndexBuilder { b.options.NoHL = true; return b }

// NoFields includes NOFIELDS.
func (b *CreateIndexBuilder) NoFields() *CreateIndexBuilder { b.options.NoFields = true; return b }

// NoFreqs includes NOFREQS.
func (b *CreateIndexBuilder) NoFreqs() *CreateIndexBuilder { b.options.NoFreqs = true; return b }

// StopWords sets STOPWORDS.
func (b *CreateIndexBuilder) StopWords(words ...interface{}) *CreateIndexBuilder {
	b.options.StopWords = words
	return b
}

// SkipInitialScan includes SKIPINITIALSCAN.
func (b *CreateIndexBuilder) SkipInitialScan() *CreateIndexBuilder {
	b.options.SkipInitialScan = true
	return b
}

// Schema adds a FieldSchema.
func (b *CreateIndexBuilder) Schema(field *FieldSchema) *CreateIndexBuilder {
	b.schema = append(b.schema, field)
	return b
}

// Run executes FT.CREATE and returns the status.
func (b *CreateIndexBuilder) Run() (string, error) {
	cmd := b.c.FTCreate(b.ctx, b.index, b.options, b.schema...)
	return cmd.Result()
}

// ----------------------
// DropIndexBuilder for FT.DROPINDEX
// ----------------------
// DropIndexBuilder is a builder for FT.DROPINDEX
// EXPERIMENTAL: this API is subject to change, use with caution.
type DropIndexBuilder struct {
	c       *Client
	ctx     context.Context
	index   string
	options *FTDropIndexOptions
}

// NewDropIndexBuilder creates a new DropIndexBuilder for FT.DROPINDEX commands.
// EXPERIMENTAL: this API is subject to change, use with caution.
func (c *Client) NewDropIndexBuilder(ctx context.Context, index string) *DropIndexBuilder {
	return &DropIndexBuilder{c: c, ctx: ctx, index: index}
}

// DeleteRuncs includes DD.
func (b *DropIndexBuilder) DeleteDocs() *DropIndexBuilder { b.options.DeleteDocs = true; return b }

// Run executes FT.DROPINDEX.
func (b *DropIndexBuilder) Run() (string, error) {
	cmd := b.c.FTDropIndexWithArgs(b.ctx, b.index, b.options)
	return cmd.Result()
}

// ----------------------
// AliasBuilder for FT.ALIAS* commands
// ----------------------
// AliasBuilder is builder for FT.ALIAS* commands
// EXPERIMENTAL: this API is subject to change, use with caution.
type AliasBuilder struct {
	c      *Client
	ctx    context.Context
	alias  string
	index  string
	action string // add|del|update
}

// NewAliasBuilder creates a new AliasBuilder for FT.ALIAS* commands.
// EXPERIMENTAL: this API is subject to change, use with caution.
func (c *Client) NewAliasBuilder(ctx context.Context, alias string) *AliasBuilder {
	return &AliasBuilder{c: c, ctx: ctx, alias: alias}
}

// Action sets the action for the alias builder.
func (b *AliasBuilder) Action(action string) *AliasBuilder {
	b.action = action
	return b
}

// Add sets the action to "add" and requires an index.
func (b *AliasBuilder) Add(index string) *AliasBuilder {
	b.action = "add"
	b.index = index
	return b
}

// Del sets the action to "del".
func (b *AliasBuilder) Del() *AliasBuilder {
	b.action = "del"
	return b
}

// Update sets the action to "update" and requires an index.
func (b *AliasBuilder) Update(index string) *AliasBuilder {
	b.action = "update"
	b.index = index
	return b
}

// Run executes the configured alias command.
func (b *AliasBuilder) Run() (string, error) {
	switch b.action {
	case "add":
		cmd := b.c.FTAliasAdd(b.ctx, b.index, b.alias)
		return cmd.Result()
	case "del":
		cmd := b.c.FTAliasDel(b.ctx, b.alias)
		return cmd.Result()
	case "update":
		cmd := b.c.FTAliasUpdate(b.ctx, b.index, b.alias)
		return cmd.Result()
	}
	return "", nil
}

// ----------------------
// ExplainBuilder for FT.EXPLAIN
// ----------------------
// ExplainBuilder is builder for FT.EXPLAIN
// EXPERIMENTAL: this API is subject to change, use with caution.
type ExplainBuilder struct {
	c       *Client
	ctx     context.Context
	index   string
	query   string
	options *FTExplainOptions
}

// NewExplainBuilder creates a new ExplainBuilder for FT.EXPLAIN commands.
// EXPERIMENTAL: this API is subject to change, use with caution.
func (c *Client) NewExplainBuilder(ctx context.Context, index, query string) *ExplainBuilder {
	return &ExplainBuilder{c: c, ctx: ctx, index: index, query: query, options: &FTExplainOptions{}}
}

// Dialect sets dialect for EXPLAINCLI.
func (b *ExplainBuilder) Dialect(d string) *ExplainBuilder { b.options.Dialect = d; return b }

// Run executes FT.EXPLAIN and returns the plan.
func (b *ExplainBuilder) Run() (string, error) {
	cmd := b.c.FTExplainWithArgs(b.ctx, b.index, b.query, b.options)
	return cmd.Result()
}

// ----------------------
// InfoBuilder for FT.INFO
// ----------------------

type FTInfoBuilder struct {
	c     *Client
	ctx   context.Context
	index string
}

// NewSearchInfoBuilder creates a new FTInfoBuilder for FT.INFO commands.
func (c *Client) NewSearchInfoBuilder(ctx context.Context, index string) *FTInfoBuilder {
	return &FTInfoBuilder{c: c, ctx: ctx, index: index}
}

// Run executes FT.INFO and returns detailed info.
func (b *FTInfoBuilder) Run() (FTInfoResult, error) {
	cmd := b.c.FTInfo(b.ctx, b.index)
	return cmd.Result()
}

// ----------------------
// SpellCheckBuilder for FT.SPELLCHECK
// ----------------------
// SpellCheckBuilder is builder for FT.SPELLCHECK
// EXPERIMENTAL: this API is subject to change, use with caution.
type SpellCheckBuilder struct {
	c       *Client
	ctx     context.Context
	index   string
	query   string
	options *FTSpellCheckOptions
}

// NewSpellCheckBuilder creates a new SpellCheckBuilder for FT.SPELLCHECK commands.
// EXPERIMENTAL: this API is subject to change, use with caution.
func (c *Client) NewSpellCheckBuilder(ctx context.Context, index, query string) *SpellCheckBuilder {
	return &SpellCheckBuilder{c: c, ctx: ctx, index: index, query: query, options: &FTSpellCheckOptions{}}
}

// Distance sets MAXDISTANCE.
func (b *SpellCheckBuilder) Distance(d int) *SpellCheckBuilder { b.options.Distance = d; return b }

// Terms sets INCLUDE or EXCLUDE terms.
func (b *SpellCheckBuilder) Terms(include bool, dictionary string, terms ...interface{}) *SpellCheckBuilder {
	if b.options.Terms == nil {
		b.options.Terms = &FTSpellCheckTerms{}
	}
	if include {
		b.options.Terms.Inclusion = "INCLUDE"
	} else {
		b.options.Terms.Inclusion = "EXCLUDE"
	}
	b.options.Terms.Dictionary = dictionary
	b.options.Terms.Terms = terms
	return b
}

// Dialect sets dialect version.
func (b *SpellCheckBuilder) Dialect(d int) *SpellCheckBuilder { b.options.Dialect = d; return b }

// Run executes FT.SPELLCHECK and returns suggestions.
func (b *SpellCheckBuilder) Run() ([]SpellCheckResult, error) {
	cmd := b.c.FTSpellCheckWithArgs(b.ctx, b.index, b.query, b.options)
	return cmd.Result()
}

// ----------------------
// DictBuilder for FT.DICT* commands
// ----------------------
// DictBuilder is builder for FT.DICT* commands
// EXPERIMENTAL: this API is subject to change, use with caution.
type DictBuilder struct {
	c      *Client
	ctx    context.Context
	dict   string
	terms  []interface{}
	action string // add|del|dump
}

// NewDictBuilder creates a new DictBuilder for FT.DICT* commands.
// EXPERIMENTAL: this API is subject to change, use with caution.
func (c *Client) NewDictBuilder(ctx context.Context, dict string) *DictBuilder {
	return &DictBuilder{c: c, ctx: ctx, dict: dict}
}

// Action sets the action for the dictionary builder.
func (b *DictBuilder) Action(action string) *DictBuilder {
	b.action = action
	return b
}

// Add sets the action to "add" and requires terms.
func (b *DictBuilder) Add(terms ...interface{}) *DictBuilder {
	b.action = "add"
	b.terms = terms
	return b
}

// Del sets the action to "del" and requires terms.
func (b *DictBuilder) Del(terms ...interface{}) *DictBuilder {
	b.action = "del"
	b.terms = terms
	return b
}

// Dump sets the action to "dump".
func (b *DictBuilder) Dump() *DictBuilder {
	b.action = "dump"
	return b
}

// Run executes the configured dictionary command.
func (b *DictBuilder) Run() (interface{}, error) {
	switch b.action {
	case "add":
		cmd := b.c.FTDictAdd(b.ctx, b.dict, b.terms...)
		return cmd.Result()
	case "del":
		cmd := b.c.FTDictDel(b.ctx, b.dict, b.terms...)
		return cmd.Result()
	case "dump":
		cmd := b.c.FTDictDump(b.ctx, b.dict)
		return cmd.Result()
	}
	return nil, nil
}

// ----------------------
// TagValsBuilder for FT.TAGVALS
// ----------------------
// TagValsBuilder is builder for FT.TAGVALS
// EXPERIMENTAL: this API is subject to change, use with caution.
type TagValsBuilder struct {
	c     *Client
	ctx   context.Context
	index string
	field string
}

// NewTagValsBuilder creates a new TagValsBuilder for FT.TAGVALS commands.
// EXPERIMENTAL: this API is subject to change, use with caution.
func (c *Client) NewTagValsBuilder(ctx context.Context, index, field string) *TagValsBuilder {
	return &TagValsBuilder{c: c, ctx: ctx, index: index, field: field}
}

// Run executes FT.TAGVALS and returns tag values.
func (b *TagValsBuilder) Run() ([]string, error) {
	cmd := b.c.FTTagVals(b.ctx, b.index, b.field)
	return cmd.Result()
}

// ----------------------
// CursorBuilder for FT.CURSOR*
// ----------------------
// CursorBuilder is builder for FT.CURSOR* commands
// EXPERIMENTAL: this API is subject to change, use with caution.
type CursorBuilder struct {
	c        *Client
	ctx      context.Context
	index    string
	cursorId int64
	count    int
	action   string // read|del
}

// NewCursorBuilder creates a new CursorBuilder for FT.CURSOR* commands.
// EXPERIMENTAL: this API is subject to change, use with caution.
func (c *Client) NewCursorBuilder(ctx context.Context, index string, cursorId int64) *CursorBuilder {
	return &CursorBuilder{c: c, ctx: ctx, index: index, cursorId: cursorId}
}

// Action sets the action for the cursor builder.
func (b *CursorBuilder) Action(action string) *CursorBuilder {
	b.action = action
	return b
}

// Read sets the action to "read".
func (b *CursorBuilder) Read() *CursorBuilder {
	b.action = "read"
	return b
}

// Del sets the action to "del".
func (b *CursorBuilder) Del() *CursorBuilder {
	b.action = "del"
	return b
}

// Count for READ.
func (b *CursorBuilder) Count(count int) *CursorBuilder { b.count = count; return b }

// Run executes the cursor command.
func (b *CursorBuilder) Run() (interface{}, error) {
	switch b.action {
	case "read":
		cmd := b.c.FTCursorRead(b.ctx, b.index, int(b.cursorId), b.count)
		return cmd.Result()
	case "del":
		cmd := b.c.FTCursorDel(b.ctx, b.index, int(b.cursorId))
		return cmd.Result()
	}
	return nil, nil
}

// ----------------------
// SynUpdateBuilder for FT.SYNUPDATE
// ----------------------
// SyncUpdateBuilder is builder for FT.SYNCUPDATE
// EXPERIMENTAL: this API is subject to change, use with caution.
type SynUpdateBuilder struct {
	c       *Client
	ctx     context.Context
	index   string
	groupId interface{}
	options *FTSynUpdateOptions
	terms   []interface{}
}

// NewSynUpdateBuilder creates a new SynUpdateBuilder for FT.SYNUPDATE commands.
// EXPERIMENTAL: this API is subject to change, use with caution.
func (c *Client) NewSynUpdateBuilder(ctx context.Context, index string, groupId interface{}) *SynUpdateBuilder {
	return &SynUpdateBuilder{c: c, ctx: ctx, index: index, groupId: groupId, options: &FTSynUpdateOptions{}}
}

// SkipInitialScan includes SKIPINITIALSCAN.
func (b *SynUpdateBuilder) SkipInitialScan() *SynUpdateBuilder {
	b.options.SkipInitialScan = true
	return b
}

// Terms adds synonyms to the group.
func (b *SynUpdateBuilder) Terms(terms ...interface{}) *SynUpdateBuilder { b.terms = terms; return b }

// Run executes FT.SYNUPDATE.
func (b *SynUpdateBuilder) Run() (string, error) {
	cmd := b.c.FTSynUpdateWithArgs(b.ctx, b.index, b.groupId, b.options, b.terms)
	return cmd.Result()
}
