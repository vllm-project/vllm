package redis

import (
	"context"
	"fmt"
	"maps"
	"slices"
	"strconv"
	"strings"

	"github.com/redis/go-redis/v9/internal"
	"github.com/redis/go-redis/v9/internal/proto"
)

type SearchCmdable interface {
	FT_List(ctx context.Context) *StringSliceCmd
	FTAggregate(ctx context.Context, index string, query string) *MapStringInterfaceCmd
	FTAggregateWithArgs(ctx context.Context, index string, query string, options *FTAggregateOptions) *AggregateCmd
	FTAliasAdd(ctx context.Context, index string, alias string) *StatusCmd
	FTAliasDel(ctx context.Context, alias string) *StatusCmd
	FTAliasList(ctx context.Context, index string) *StringSliceCmd
	FTAliasUpdate(ctx context.Context, index string, alias string) *StatusCmd
	FTAlter(ctx context.Context, index string, skipInitialScan bool, definition []interface{}) *StatusCmd
	FTConfigGet(ctx context.Context, option string) *MapMapStringInterfaceCmd
	FTConfigSet(ctx context.Context, option string, value interface{}) *StatusCmd
	FTCreate(ctx context.Context, index string, options *FTCreateOptions, schema ...*FieldSchema) *StatusCmd
	FTCursorDel(ctx context.Context, index string, cursorId int) *StatusCmd
	FTCursorRead(ctx context.Context, index string, cursorId int, count int) *MapStringInterfaceCmd
	FTDictAdd(ctx context.Context, dict string, term ...interface{}) *IntCmd
	FTDictDel(ctx context.Context, dict string, term ...interface{}) *IntCmd
	FTDictDump(ctx context.Context, dict string) *StringSliceCmd
	FTDropIndex(ctx context.Context, index string) *StatusCmd
	FTDropIndexWithArgs(ctx context.Context, index string, options *FTDropIndexOptions) *StatusCmd
	FTExplain(ctx context.Context, index string, query string) *StringCmd
	FTExplainWithArgs(ctx context.Context, index string, query string, options *FTExplainOptions) *StringCmd
	FTHybrid(ctx context.Context, index string, searchExpr string, vectorField string, vectorData Vector) *FTHybridCmd
	FTHybridWithArgs(ctx context.Context, index string, options *FTHybridOptions) *FTHybridCmd
	FTInfo(ctx context.Context, index string) *FTInfoCmd
	FTSpellCheck(ctx context.Context, index string, query string) *FTSpellCheckCmd
	FTSpellCheckWithArgs(ctx context.Context, index string, query string, options *FTSpellCheckOptions) *FTSpellCheckCmd
	FTSearch(ctx context.Context, index string, query string) *FTSearchCmd
	FTSearchWithArgs(ctx context.Context, index string, query string, options *FTSearchOptions) *FTSearchCmd
	FTSynDump(ctx context.Context, index string) *FTSynDumpCmd
	FTSynUpdate(ctx context.Context, index string, synGroupId interface{}, terms []interface{}) *StatusCmd
	FTSynUpdateWithArgs(ctx context.Context, index string, synGroupId interface{}, options *FTSynUpdateOptions, terms []interface{}) *StatusCmd
	FTTagVals(ctx context.Context, index string, field string) *StringSliceCmd
}

type FTCreateOptions struct {
	OnHash          bool
	OnJSON          bool
	Prefix          []interface{}
	Filter          string
	DefaultLanguage string
	LanguageField   string
	Score           float64
	ScoreField      string
	PayloadField    string
	MaxTextFields   int
	NoOffsets       bool
	Temporary       int
	NoHL            bool
	NoFields        bool
	NoFreqs         bool
	StopWords       []interface{}
	SkipInitialScan bool
}

type FieldSchema struct {
	FieldName         string
	As                string
	FieldType         SearchFieldType
	Sortable          bool
	UNF               bool
	NoStem            bool
	NoIndex           bool
	PhoneticMatcher   string
	Weight            float64
	Separator         string
	CaseSensitive     bool
	WithSuffixtrie    bool
	VectorArgs        *FTVectorArgs
	GeoShapeFieldType string
	IndexEmpty        bool
	IndexMissing      bool
}

type FTVectorArgs struct {
	FlatOptions   *FTFlatOptions
	HNSWOptions   *FTHNSWOptions
	VamanaOptions *FTVamanaOptions
}

type FTFlatOptions struct {
	Type            string
	Dim             int
	DistanceMetric  string
	InitialCapacity int
	BlockSize       int
}

type FTHNSWOptions struct {
	Type                   string
	Dim                    int
	DistanceMetric         string
	InitialCapacity        int
	MaxEdgesPerNode        int
	MaxAllowedEdgesPerNode int
	EFRunTime              int
	Epsilon                float64
	// Rerank toggles the exact re-scoring pass over approximate candidates on
	// disk-backed HNSW indexes (Redis 8.10+), where the server requires it to
	// be set explicitly. Rerank=true emits RERANK TRUE on its own; to emit
	// RERANK FALSE, set HasRerank=true with Rerank=false, so that an explicit
	// false can be distinguished from unset (omitted).
	Rerank    bool
	HasRerank bool
}

type FTVamanaOptions struct {
	Type                   string
	Dim                    int
	DistanceMetric         string
	Compression            string
	ConstructionWindowSize int
	GraphMaxDegree         int
	SearchWindowSize       int
	Epsilon                float64
	TrainingThreshold      int
	ReduceDim              int
}

type FTDropIndexOptions struct {
	DeleteDocs bool
}

type SpellCheckTerms struct {
	Include    bool
	Exclude    bool
	Dictionary string
}

type FTExplainOptions struct {
	// Dialect 1,3 and 4 are deprecated since redis 8.0
	Dialect string
}

type FTSynUpdateOptions struct {
	SkipInitialScan bool
}

type SearchAggregator int

const (
	SearchInvalid = SearchAggregator(iota)
	SearchAvg
	SearchSum
	SearchMin
	SearchMax
	SearchCount
	SearchCountDistinct
	SearchCountDistinctish
	SearchStdDev
	SearchQuantile
	SearchToList
	SearchFirstValue
	SearchRandomSample
	// SearchCollect is the COLLECT reducer for FT.AGGREGATE. Within each
	// GROUPBY group it projects a chosen set of fields from every row and
	// emits them as an array of per-entry maps under the reducer alias.
	// Requires Redis 8.8+ with unstable features enabled
	// (CONFIG SET search-enable-unstable-features yes).
	SearchCollect
)

func (a SearchAggregator) String() string {
	switch a {
	case SearchInvalid:
		return ""
	case SearchAvg:
		return "AVG"
	case SearchSum:
		return "SUM"
	case SearchMin:
		return "MIN"
	case SearchMax:
		return "MAX"
	case SearchCount:
		return "COUNT"
	case SearchCountDistinct:
		return "COUNT_DISTINCT"
	case SearchCountDistinctish:
		return "COUNT_DISTINCTISH"
	case SearchStdDev:
		return "STDDEV"
	case SearchQuantile:
		return "QUANTILE"
	case SearchToList:
		return "TOLIST"
	case SearchFirstValue:
		return "FIRST_VALUE"
	case SearchRandomSample:
		return "RANDOM_SAMPLE"
	case SearchCollect:
		return "COLLECT"
	default:
		return ""
	}
}

type SearchFieldType int

const (
	SearchFieldTypeInvalid = SearchFieldType(iota)
	SearchFieldTypeNumeric
	SearchFieldTypeTag
	SearchFieldTypeText
	SearchFieldTypeGeo
	SearchFieldTypeVector
	SearchFieldTypeGeoShape
)

func (t SearchFieldType) String() string {
	switch t {
	case SearchFieldTypeInvalid:
		return ""
	case SearchFieldTypeNumeric:
		return "NUMERIC"
	case SearchFieldTypeTag:
		return "TAG"
	case SearchFieldTypeText:
		return "TEXT"
	case SearchFieldTypeGeo:
		return "GEO"
	case SearchFieldTypeVector:
		return "VECTOR"
	case SearchFieldTypeGeoShape:
		return "GEOSHAPE"
	default:
		return "TEXT"
	}
}

// Each AggregateReducer have different args.
// Please follow https://redis.io/docs/interact/search-and-query/search/aggregations/#supported-groupby-reducers for more information.
type FTAggregateReducer struct {
	Reducer SearchAggregator
	Args    []interface{}
	As      string
}

type FTAggregateGroupBy struct {
	Fields []interface{}
	Reduce []FTAggregateReducer
}

type FTAggregateSortBy struct {
	FieldName string
	Asc       bool
	Desc      bool
}

type FTAggregateApply struct {
	Field string
	As    string
}

type FTAggregateLoad struct {
	Field string
	As    string
}

type FTAggregateWithCursor struct {
	Count   int
	MaxIdle int
}

// FTAggregateSortByStep represents a SORTBY operation with optional MAX.
// Used inside FTAggregateStep to place SORTBY at an arbitrary position in
// the aggregation pipeline.
type FTAggregateSortByStep struct {
	Fields []FTAggregateSortBy
	Max    int // 0 means no MAX
}

// FTAggregateStep represents a single operation in the aggregation pipeline.
// LOAD, APPLY, SORTBY and GROUPBY can all appear multiple times in any order.
// Exactly one of the fields should be set per step.
type FTAggregateStep struct {
	Load    *FTAggregateLoad
	Apply   *FTAggregateApply
	GroupBy *FTAggregateGroupBy
	SortBy  *FTAggregateSortByStep
}

type FTAggregateOptions struct {
	Verbatim bool
	LoadAll  bool
	Timeout  int
	// Scorer is used to set scoring function, if not set passed, a default will be used.
	// The default scorer depends on the Redis version:
	// - `BM25` for Redis >= 8
	// - `TFIDF` for Redis < 8
	Scorer string
	// AddScores is available in Redis CE 8
	AddScores bool

	// Steps is the ordered sequence of aggregation pipeline operations.
	// It can contain LOAD, APPLY, GROUPBY and SORTBY in any order, multiple times.
	// Steps cannot be combined with the deprecated Load, Apply, GroupBy, SortBy
	// and SortByMax fields: doing so returns an error.
	Steps []FTAggregateStep

	LimitOffset       int
	Limit             int
	Filter            string
	WithCursor        bool
	WithCursorOptions *FTAggregateWithCursor
	Params            map[string]interface{}
	// Dialect 1,3 and 4 are deprecated since redis 8.0
	DialectVersion int

	// Deprecated: Use Steps instead.
	Load []FTAggregateLoad
	// Deprecated: Use Steps instead.
	GroupBy []FTAggregateGroupBy
	// Deprecated: Use Steps instead.
	SortBy []FTAggregateSortBy
	// Deprecated: Use Steps instead.
	SortByMax int
	// Deprecated: Use Steps instead.
	Apply []FTAggregateApply
}

type FTSearchFilter struct {
	FieldName interface{}
	Min       interface{}
	Max       interface{}
}

type FTSearchGeoFilter struct {
	FieldName string
	Longitude float64
	Latitude  float64
	Radius    float64
	Unit      string
}

type FTSearchReturn struct {
	FieldName string
	As        string
}

type FTSearchSortBy struct {
	FieldName string
	Asc       bool
	Desc      bool
}

// FTSearchOptions hold options that can be passed to the FT.SEARCH command.
// More information about the options can be found
// in the documentation for FT.SEARCH https://redis.io/docs/latest/commands/ft.search/
type FTSearchOptions struct {
	NoContent    bool
	Verbatim     bool
	NoStopWords  bool
	WithScores   bool
	WithPayloads bool
	WithSortKeys bool
	Filters      []FTSearchFilter
	GeoFilter    []FTSearchGeoFilter
	InKeys       []interface{}
	InFields     []interface{}
	Return       []FTSearchReturn
	Slop         int
	Timeout      int
	InOrder      bool
	Language     string
	Expander     string
	// Scorer is used to set scoring function, if not set passed, a default will be used.
	// The default scorer depends on the Redis version:
	// - `BM25` for Redis >= 8
	// - `TFIDF` for Redis < 8
	Scorer          string
	ExplainScore    bool
	Payload         string
	SortBy          []FTSearchSortBy
	SortByWithCount bool
	LimitOffset     int
	Limit           int
	// CountOnly sets LIMIT 0 0 to get the count - number of documents in the result set without actually returning the result set.
	// When using this option, the Limit and LimitOffset options are ignored.
	CountOnly bool
	Params    map[string]interface{}
	// Dialect 1,3 and 4 are deprecated since redis 8.0
	DialectVersion int
}

// FTHybridCombineMethod represents the fusion method for combining search and vector results
type FTHybridCombineMethod string

const (
	FTHybridCombineRRF      FTHybridCombineMethod = "RRF"
	FTHybridCombineLinear   FTHybridCombineMethod = "LINEAR"
	FTHybridCombineFunction FTHybridCombineMethod = "FUNCTION"
)

// FTHybridSearchExpression represents a search expression in hybrid search
type FTHybridSearchExpression struct {
	Query        string
	Scorer       string
	ScorerParams []interface{}
	YieldScoreAs string
}

type FTHybridVectorMethod = string

const (
	KNN   FTHybridCombineMethod = "KNN"
	RANGE FTHybridCombineMethod = "RANGE"
)

// FTHybridVectorExpression represents a vector expression in hybrid search
type FTHybridVectorExpression struct {
	VectorField string
	VectorData  Vector
	// VectorParamName optionally specifies the parameter name used to pass the
	// vector data via the PARAMS mechanism.
	// Vector data is always passed via PARAMS because inline vector blobs are no
	// longer supported by Redis. When left empty, the library generates a unique
	// parameter name automatically (e.g. "__vector_param_0") without mutating
	// FTHybridOptions.Params and without colliding with any explicit names.
	// The vector blob is passed as: VSIM @field $VectorParamName ... PARAMS ... VectorParamName <blob>
	VectorParamName string
	Method          FTHybridVectorMethod
	MethodParams    []interface{}
	// ShardKRatio controls how many results each shard returns relative to the
	// requested KNN K, trading recall for latency in Redis cluster setups.
	// Valid range: 0.1 - 1.0. The zero value means "unset" and falls back to
	// the server default of 1.0 (no per-shard reduction). Has no effect on
	// standalone Redis, and only applies to the KNN method. Requires Redis 8.8+.
	// See https://redis.io/docs/latest/develop/ai/search-and-query/query/vector-search/
	ShardKRatio  float64
	Filter       string
	YieldScoreAs string
}

// FTHybridCombineOptions represents options for result fusion
type FTHybridCombineOptions struct {
	Method       FTHybridCombineMethod
	Count        int
	Window       int     // For RRF
	Constant     float64 // For RRF
	Alpha        float64 // For LINEAR
	Beta         float64 // For LINEAR
	YieldScoreAs string
}

// FTHybridGroupBy represents GROUP BY functionality
type FTHybridGroupBy struct {
	Count        int
	Fields       []string
	ReduceFunc   string
	ReduceCount  int
	ReduceParams []interface{}
}

// FTHybridApply represents APPLY functionality
type FTHybridApply struct {
	Expression string
	AsField    string
}

// FTHybridWithCursor represents cursor configuration for hybrid search
type FTHybridWithCursor struct {
	Count   int // Number of results to return per cursor read
	MaxIdle int // Maximum idle time in milliseconds before cursor is automatically deleted
}

// FTHybridOptions hold options that can be passed to the FT.HYBRID command
type FTHybridOptions struct {
	CountExpressions  int                        // Number of search/vector expressions
	SearchExpressions []FTHybridSearchExpression // Multiple search expressions
	VectorExpressions []FTHybridVectorExpression // Multiple vector expressions
	Combine           *FTHybridCombineOptions    // Fusion step options
	Load              []string                   // Projected fields
	GroupBy           *FTHybridGroupBy           // Aggregation grouping
	Apply             []FTHybridApply            // Field transformations
	SortBy            []FTSearchSortBy           // Reuse from FTSearch
	Filter            string                     // Post-filter expression
	LimitOffset       int                        // Result limiting
	Limit             int
	Params            map[string]interface{} // Parameter substitution
	ExplainScore      bool                   // Include score explanations
	Timeout           int                    // Runtime timeout
	WithCursor        bool                   // Enable cursor support for large result sets
	WithCursorOptions *FTHybridWithCursor    // Cursor configuration options
}

type FTSynDumpResult struct {
	Term     string
	Synonyms []string
}

type FTSynDumpCmd struct {
	baseCmd
	val []FTSynDumpResult
}

// FTAggregateResult represents the result of an aggregate operation
// NOTE: For RESP3 Total is not reliable (before Redis 8.8)
type FTAggregateResult struct {
	Total int
	Rows  []AggregateRow
	// Warnings holds server warnings for a partial result (search-on-timeout
	// return/return-strict). RESP3 only; the fail policy returns an error instead.
	Warnings []string
}

type AggregateRow struct {
	Fields map[string]interface{}
}

type AggregateCmd struct {
	baseCmd
	val *FTAggregateResult
}

type FTInfoResult struct {
	IndexErrors              IndexErrors
	Attributes               []FTAttribute
	BytesPerRecordAvg        string
	Cleaning                 int
	CursorStats              CursorStats
	DialectStats             map[string]int
	DocTableSizeMB           float64
	FieldStatistics          []FieldStatistic
	GCStats                  GCStats
	GeoshapesSzMB            float64
	HashIndexingFailures     int
	IndexDefinition          IndexDefinition
	IndexName                string
	IndexOptions             []string
	Indexing                 int
	InvertedSzMB             float64
	KeyTableSizeMB           float64
	MaxDocID                 int
	NumDocs                  int
	NumRecords               int
	NumTerms                 int
	NumberOfUses             int
	OffsetBitsPerRecordAvg   string
	OffsetVectorsSzMB        float64
	OffsetsPerTermAvg        string
	PercentIndexed           float64
	RecordsPerDocAvg         string
	SortableValuesSizeMB     float64
	TagOverheadSzMB          float64
	TextOverheadSzMB         float64
	TotalIndexMemorySzMB     float64
	TotalIndexingTime        int
	TotalInvertedIndexBlocks int
	VectorIndexSzMB          float64
}

type IndexErrors struct {
	IndexingFailures     int
	LastIndexingError    string
	LastIndexingErrorKey string
}

type FTAttribute struct {
	Identifier      string
	Attribute       string
	Type            string
	Weight          float64
	Sortable        bool
	NoStem          bool
	NoIndex         bool
	UNF             bool
	PhoneticMatcher string
	CaseSensitive   bool
	WithSuffixtrie  bool

	// Vector specific attributes
	Algorithm      string
	DataType       string
	Dim            int
	DistanceMetric string
	M              int
	EFConstruction int
}

type CursorStats struct {
	GlobalIdle    int
	GlobalTotal   int
	IndexCapacity int
	IndexTotal    int
}

type FieldStatistic struct {
	Identifier  string
	Attribute   string
	IndexErrors IndexErrors
}

type GCStats struct {
	BytesCollected       int
	TotalMsRun           int
	TotalCycles          int
	AverageCycleTimeMs   string
	LastRunTimeMs        int
	GCNumericTreesMissed int
	GCBlocksDenied       int
}

type IndexDefinition struct {
	KeyType      string
	Prefixes     []string
	DefaultScore float64
}

type FTSpellCheckOptions struct {
	Distance int
	Terms    *FTSpellCheckTerms
	// Dialect 1,3 and 4 are deprecated since redis 8.0
	Dialect int
}

type FTSpellCheckTerms struct {
	Inclusion  string // Either "INCLUDE" or "EXCLUDE"
	Dictionary string
	Terms      []interface{}
}

type SpellCheckResult struct {
	Term        string
	Suggestions []SpellCheckSuggestion
}

type SpellCheckSuggestion struct {
	Score      float64
	Suggestion string
}

type FTSearchResult struct {
	Total int
	Docs  []Document
	// Warnings holds server warnings for a partial result (search-on-timeout
	// return/return-strict). RESP3 only; the fail policy returns an error instead.
	Warnings []string
}

type Document struct {
	ID      string
	Score   *float64
	Payload *string
	SortKey *string
	Fields  map[string]string
	Error   error
}

type AggregateQuery []interface{}

// FT_List - Lists all the existing indexes in the database.
// For more information, please refer to the Redis documentation:
// [FT._LIST]: (https://redis.io/commands/ft._list/)
func (c cmdable) FT_List(ctx context.Context) *StringSliceCmd {
	cmd := NewStringSliceCmd(ctx, "FT._LIST")
	_ = c(ctx, cmd)
	return cmd
}

// FTAggregate - Performs a search query on an index and applies a series of aggregate transformations to the result.
// The 'index' parameter specifies the index to search, and the 'query' parameter specifies the search query.
// For more information, please refer to the Redis documentation:
// [FT.AGGREGATE]: (https://redis.io/commands/ft.aggregate/)
func (c cmdable) FTAggregate(ctx context.Context, index string, query string) *MapStringInterfaceCmd {
	args := []interface{}{"FT.AGGREGATE", index, query}
	cmd := NewMapStringInterfaceCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// validateFTAggregateOptions validates mutually exclusive combinations of
// FTAggregateOptions fields before any command arguments are constructed.
func validateFTAggregateOptions(options *FTAggregateOptions) error {
	if len(options.Steps) > 0 {
		if options.Load != nil || options.Apply != nil || options.GroupBy != nil ||
			options.SortBy != nil || options.SortByMax != 0 {
			return fmt.Errorf("FT.AGGREGATE: Steps cannot be combined with the deprecated Load, Apply, GroupBy, SortBy and SortByMax fields")
		}
		if options.LoadAll {
			for _, step := range options.Steps {
				if step.Load != nil {
					return fmt.Errorf("FT.AGGREGATE: LOADALL and LOAD are mutually exclusive")
				}
			}
		}
	}
	if options.LoadAll && options.Load != nil {
		return fmt.Errorf("FT.AGGREGATE: LOADALL and LOAD are mutually exclusive")
	}
	return nil
}

// appendFTAggregateStep appends the Redis command arguments for a single
// aggregation pipeline step. Each step must set exactly one of Load, Apply,
// GroupBy or SortBy.
func appendFTAggregateStep(args []interface{}, step FTAggregateStep) ([]interface{}, error) {
	set := 0
	if step.Load != nil {
		set++
	}
	if step.Apply != nil {
		set++
	}
	if step.GroupBy != nil {
		set++
	}
	if step.SortBy != nil {
		set++
	}
	if set != 1 {
		return args, fmt.Errorf("FT.AGGREGATE: each step must set exactly one of Load, Apply, GroupBy, SortBy (got %d)", set)
	}

	switch {
	case step.Load != nil:
		args = append(args, "LOAD")
		countIdx := len(args)
		args = append(args, 0)
		count := 0
		args = append(args, step.Load.Field)
		count++
		if step.Load.As != "" {
			args = append(args, "AS", step.Load.As)
			count += 2
		}
		args[countIdx] = count
	case step.Apply != nil:
		args = append(args, "APPLY", step.Apply.Field)
		if step.Apply.As != "" {
			args = append(args, "AS", step.Apply.As)
		}
	case step.GroupBy != nil:
		args = append(args, "GROUPBY", len(step.GroupBy.Fields))
		args = append(args, step.GroupBy.Fields...)
		for _, reducer := range step.GroupBy.Reduce {
			args = append(args, "REDUCE", reducer.Reducer.String())
			if reducer.Args != nil {
				args = append(args, len(reducer.Args))
				args = append(args, reducer.Args...)
			} else {
				args = append(args, 0)
			}
			if reducer.As != "" {
				args = append(args, "AS", reducer.As)
			}
		}
	case step.SortBy != nil:
		args = append(args, "SORTBY")
		sortByOptions := []interface{}{}
		for _, sortBy := range step.SortBy.Fields {
			if sortBy.Asc && sortBy.Desc {
				return args, fmt.Errorf("FT.AGGREGATE: ASC and DESC are mutually exclusive")
			}
			sortByOptions = append(sortByOptions, sortBy.FieldName)
			if sortBy.Asc {
				sortByOptions = append(sortByOptions, "ASC")
			}
			if sortBy.Desc {
				sortByOptions = append(sortByOptions, "DESC")
			}
		}
		args = append(args, len(sortByOptions))
		args = append(args, sortByOptions...)
		if step.SortBy.Max > 0 {
			args = append(args, "MAX", step.SortBy.Max)
		}
	}
	return args, nil
}

func FTAggregateQuery(query string, options *FTAggregateOptions) (AggregateQuery, error) {
	queryArgs := []interface{}{query}
	if options != nil {
		if err := validateFTAggregateOptions(options); err != nil {
			return nil, err
		}
		if options.Verbatim {
			queryArgs = append(queryArgs, "VERBATIM")
		}

		if options.Scorer != "" {
			queryArgs = append(queryArgs, "SCORER", options.Scorer)
		}

		if options.AddScores {
			queryArgs = append(queryArgs, "ADDSCORES")
		}

		if options.LoadAll {
			queryArgs = append(queryArgs, "LOAD", "*")
		}
		if len(options.Steps) == 0 && options.Load != nil {
			queryArgs = append(queryArgs, "LOAD", len(options.Load))
			index, count := len(queryArgs)-1, 0
			for _, load := range options.Load {
				queryArgs = append(queryArgs, load.Field)
				count++
				if load.As != "" {
					queryArgs = append(queryArgs, "AS", load.As)
					count += 2
				}
			}
			queryArgs[index] = count
		}

		if options.Timeout > 0 {
			queryArgs = append(queryArgs, "TIMEOUT", options.Timeout)
		}

		if len(options.Steps) > 0 {
			for _, step := range options.Steps {
				var err error
				queryArgs, err = appendFTAggregateStep(queryArgs, step)
				if err != nil {
					return nil, err
				}
			}
		} else {
			for _, apply := range options.Apply {
				queryArgs = append(queryArgs, "APPLY", apply.Field)
				if apply.As != "" {
					queryArgs = append(queryArgs, "AS", apply.As)
				}
			}

			if options.GroupBy != nil {
				for _, groupBy := range options.GroupBy {
					queryArgs = append(queryArgs, "GROUPBY", len(groupBy.Fields))
					queryArgs = append(queryArgs, groupBy.Fields...)

					for _, reducer := range groupBy.Reduce {
						queryArgs = append(queryArgs, "REDUCE")
						queryArgs = append(queryArgs, reducer.Reducer.String())
						if reducer.Args != nil {
							queryArgs = append(queryArgs, len(reducer.Args))
							queryArgs = append(queryArgs, reducer.Args...)
						} else {
							queryArgs = append(queryArgs, 0)
						}
						if reducer.As != "" {
							queryArgs = append(queryArgs, "AS", reducer.As)
						}
					}
				}
			}
			if options.SortBy != nil {
				queryArgs = append(queryArgs, "SORTBY")
				sortByOptions := []interface{}{}
				for _, sortBy := range options.SortBy {
					sortByOptions = append(sortByOptions, sortBy.FieldName)
					if sortBy.Asc && sortBy.Desc {
						return nil, fmt.Errorf("FT.AGGREGATE: ASC and DESC are mutually exclusive")
					}
					if sortBy.Asc {
						sortByOptions = append(sortByOptions, "ASC")
					}
					if sortBy.Desc {
						sortByOptions = append(sortByOptions, "DESC")
					}
				}
				queryArgs = append(queryArgs, len(sortByOptions))
				queryArgs = append(queryArgs, sortByOptions...)
			}
			if options.SortByMax > 0 {
				queryArgs = append(queryArgs, "MAX", options.SortByMax)
			}
		}
		if options.LimitOffset >= 0 && options.Limit > 0 {
			queryArgs = append(queryArgs, "LIMIT", options.LimitOffset, options.Limit)
		}
		if options.Filter != "" {
			queryArgs = append(queryArgs, "FILTER", options.Filter)
		}
		if options.WithCursor {
			queryArgs = append(queryArgs, "WITHCURSOR")
			if options.WithCursorOptions != nil {
				if options.WithCursorOptions.Count > 0 {
					queryArgs = append(queryArgs, "COUNT", options.WithCursorOptions.Count)
				}
				if options.WithCursorOptions.MaxIdle > 0 {
					queryArgs = append(queryArgs, "MAXIDLE", options.WithCursorOptions.MaxIdle)
				}
			}
		}
		if options.Params != nil {
			queryArgs = append(queryArgs, "PARAMS", len(options.Params)*2)
			for key, value := range options.Params {
				queryArgs = append(queryArgs, key, value)
			}
		}

		if options.DialectVersion > 0 {
			queryArgs = append(queryArgs, "DIALECT", options.DialectVersion)
		} else {
			queryArgs = append(queryArgs, "DIALECT", 2)
		}
	}
	return queryArgs, nil
}

func ProcessAggregateResult(data []interface{}) (*FTAggregateResult, error) {
	if len(data) == 0 {
		return nil, fmt.Errorf("no data returned")
	}

	total, ok := data[0].(int64)
	if !ok {
		return nil, fmt.Errorf("invalid total format")
	}

	rows := make([]AggregateRow, 0, len(data)-1)
	for _, row := range data[1:] {
		fields, ok := row.([]interface{})
		if !ok {
			return nil, fmt.Errorf("invalid row format")
		}

		rowMap := make(map[string]interface{})
		for i := 0; i < len(fields); i += 2 {
			key, ok := fields[i].(string)
			if !ok {
				return nil, fmt.Errorf("invalid field key format")
			}
			value := fields[i+1]
			rowMap[key] = value
		}
		rows = append(rows, AggregateRow{Fields: rowMap})
	}

	result := &FTAggregateResult{
		Total: int(total),
		Rows:  rows,
	}
	return result, nil
}

func NewAggregateCmd(ctx context.Context, args ...interface{}) *AggregateCmd {
	return &AggregateCmd{
		baseCmd: baseCmd{
			ctx:     ctx,
			args:    args,
			cmdType: CmdTypeAggregate,
		},
	}
}

func (cmd *AggregateCmd) SetVal(val *FTAggregateResult) {
	cmd.val = val
}

func (cmd *AggregateCmd) Val() *FTAggregateResult {
	cmd.await()
	return cmd.val
}

func (cmd *AggregateCmd) Result() (*FTAggregateResult, error) {
	cmd.await()
	return cmd.val, cmd.err
}

func (cmd *AggregateCmd) RawVal() interface{} {
	cmd.await()
	return cmd.rawVal
}

func (cmd *AggregateCmd) RawResult() (interface{}, error) {
	cmd.await()
	return cmd.rawVal, cmd.err
}

func (cmd *AggregateCmd) String() string {
	cmd.await()
	return cmdString(cmd, cmd.val)
}

func (cmd *AggregateCmd) readReply(rd *proto.Reader) (err error) {
	readType, err := rd.PeekReplyType()
	if err != nil {
		return err
	}

	// RESP3 returns a map, RESP2 returns an array
	if readType == proto.RespMap {
		// Read raw response first for backwards compatibility
		cmd.rawVal, err = rd.ReadReply()
		if err != nil {
			return err
		}
		// Parse the raw response into structured result
		if mapVal, ok := cmd.rawVal.(map[interface{}]interface{}); ok {
			cmd.val, err = parseFTAggregateMapRESP3(mapVal)
		} else {
			return fmt.Errorf("unexpected RESP3 response type: %T", cmd.rawVal)
		}
		return err
	}

	// RESP2 format or error response - use ReadReply to handle errors properly
	data, err := rd.ReadReply()
	if err != nil {
		return err
	}
	cmd.rawVal = data // Store raw value for debugging
	if dataSlice, ok := data.([]interface{}); ok {
		cmd.val, err = ProcessAggregateResult(dataSlice)
		return err
	}
	return fmt.Errorf("unexpected response type: %T", data)
}

// parseFTAggregateMapRESP3 parses the RESP3 format response from FT.AGGREGATE.
// It takes a map[interface{}]interface{} which is the raw response from ReadReply().
// RESP3 format:
//
//	%5
//	  $10 attributes => *0
//	  $13 total_results => :N
//	  $6 format => $6 STRING
//	  $7 results => *N (array of maps with extra_attributes, values)
//	  $7 warning => *N (array of strings)
func parseFTAggregateMapRESP3(data map[interface{}]interface{}) (*FTAggregateResult, error) {
	result := &FTAggregateResult{
		Rows: make([]AggregateRow, 0),
	}

	for k, v := range data {
		key, ok := k.(string)
		if !ok {
			continue
		}

		switch key {
		case "total_results":
			result.Total = internal.ToInteger(v)
		case "results":
			if resultsData, ok := v.([]interface{}); ok {
				rows, err := parseFTAggregateResultsMapRESP3(resultsData)
				if err != nil {
					return nil, err
				}
				result.Rows = rows
			}
		case "warning":
			if warningsData, ok := v.([]interface{}); ok {
				result.Warnings = make([]string, 0, len(warningsData))
				for _, w := range warningsData {
					if ws, ok := w.(string); ok {
						result.Warnings = append(result.Warnings, ws)
					}
				}
			}
			// Ignore "attributes", "format", and other fields as per the spec
		}
	}

	return result, nil
}

// parseFTAggregateResultsMapRESP3 parses the results array from RESP3 FT.AGGREGATE response.
func parseFTAggregateResultsMapRESP3(resultsData []interface{}) ([]AggregateRow, error) {
	rows := make([]AggregateRow, 0, len(resultsData))
	for _, item := range resultsData {
		if itemMap, ok := item.(map[interface{}]interface{}); ok {
			row, err := parseFTAggregateRowMapRESP3(itemMap)
			if err != nil {
				return nil, err
			}
			rows = append(rows, row)
		}
	}
	return rows, nil
}

// parseFTAggregateRowMapRESP3 parses a single row from RESP3 FT.AGGREGATE response.
func parseFTAggregateRowMapRESP3(itemMap map[interface{}]interface{}) (AggregateRow, error) {
	row := AggregateRow{
		Fields: make(map[string]interface{}),
	}

	for k, v := range itemMap {
		key, ok := k.(string)
		if !ok {
			continue
		}

		switch key {
		case "extra_attributes":
			if extraAttrs, ok := v.(map[interface{}]interface{}); ok {
				for ek, ev := range extraAttrs {
					if ekStr, ok := ek.(string); ok {
						row.Fields[ekStr] = ev
					}
				}
			}
			// Ignore "values" and other fields as per the spec
		}
	}

	return row, nil
}

func (cmd *AggregateCmd) Clone() Cmder {
	var val *FTAggregateResult
	if cmd.val != nil {
		val = &FTAggregateResult{
			Total: cmd.val.Total,
		}
		if cmd.val.Rows != nil {
			val.Rows = make([]AggregateRow, len(cmd.val.Rows))
			for i, row := range cmd.val.Rows {
				val.Rows[i] = AggregateRow{}
				if row.Fields != nil {
					val.Rows[i].Fields = make(map[string]interface{}, len(row.Fields))
					for k, v := range row.Fields {
						val.Rows[i].Fields[k] = v
					}
				}
			}
		}
		if cmd.val.Warnings != nil {
			val.Warnings = make([]string, len(cmd.val.Warnings))
			copy(val.Warnings, cmd.val.Warnings)
		}
	}
	return &AggregateCmd{
		baseCmd: cmd.cloneBaseCmd(),
		val:     val,
	}
}

// FTAggregateWithArgs - Performs a search query on an index and applies a series of aggregate transformations to the result.
// The 'index' parameter specifies the index to search, and the 'query' parameter specifies the search query.
// This function also allows for specifying additional options such as: Verbatim, LoadAll, Load, Timeout, GroupBy, SortBy, SortByMax, Apply, LimitOffset, Limit, Filter, WithCursor, Params, and DialectVersion.
// For more information, please refer to the Redis documentation:
// [FT.AGGREGATE]: (https://redis.io/commands/ft.aggregate/)
func (c cmdable) FTAggregateWithArgs(ctx context.Context, index string, query string, options *FTAggregateOptions) *AggregateCmd {
	args := []interface{}{"FT.AGGREGATE", index, query}
	if options != nil {
		if err := validateFTAggregateOptions(options); err != nil {
			cmd := NewAggregateCmd(ctx, args...)
			cmd.SetErr(err)
			return cmd
		}
		if options.Verbatim {
			args = append(args, "VERBATIM")
		}
		if options.Scorer != "" {
			args = append(args, "SCORER", options.Scorer)
		}
		if options.AddScores {
			args = append(args, "ADDSCORES")
		}
		if options.LoadAll {
			args = append(args, "LOAD", "*")
		}
		if len(options.Steps) == 0 && options.Load != nil {
			args = append(args, "LOAD", len(options.Load))
			index, count := len(args)-1, 0
			for _, load := range options.Load {
				args = append(args, load.Field)
				count++
				if load.As != "" {
					args = append(args, "AS", load.As)
					count += 2
				}
			}
			args[index] = count
		}
		if options.Timeout > 0 {
			args = append(args, "TIMEOUT", options.Timeout)
		}
		if len(options.Steps) > 0 {
			for _, step := range options.Steps {
				var err error
				args, err = appendFTAggregateStep(args, step)
				if err != nil {
					cmd := NewAggregateCmd(ctx, args...)
					cmd.SetErr(err)
					return cmd
				}
			}
		} else {
			for _, apply := range options.Apply {
				args = append(args, "APPLY", apply.Field)
				if apply.As != "" {
					args = append(args, "AS", apply.As)
				}
			}
			if options.GroupBy != nil {
				for _, groupBy := range options.GroupBy {
					args = append(args, "GROUPBY", len(groupBy.Fields))
					args = append(args, groupBy.Fields...)

					for _, reducer := range groupBy.Reduce {
						args = append(args, "REDUCE")
						args = append(args, reducer.Reducer.String())
						if reducer.Args != nil {
							args = append(args, len(reducer.Args))
							args = append(args, reducer.Args...)
						} else {
							args = append(args, 0)
						}
						if reducer.As != "" {
							args = append(args, "AS", reducer.As)
						}
					}
				}
			}
			if options.SortBy != nil {
				args = append(args, "SORTBY")
				sortByOptions := []interface{}{}
				for _, sortBy := range options.SortBy {
					sortByOptions = append(sortByOptions, sortBy.FieldName)
					if sortBy.Asc && sortBy.Desc {
						cmd := NewAggregateCmd(ctx, args...)
						cmd.SetErr(fmt.Errorf("FT.AGGREGATE: ASC and DESC are mutually exclusive"))
						return cmd
					}
					if sortBy.Asc {
						sortByOptions = append(sortByOptions, "ASC")
					}
					if sortBy.Desc {
						sortByOptions = append(sortByOptions, "DESC")
					}
				}
				args = append(args, len(sortByOptions))
				args = append(args, sortByOptions...)
			}
			if options.SortByMax > 0 {
				args = append(args, "MAX", options.SortByMax)
			}
		}
		if options.LimitOffset >= 0 && options.Limit > 0 {
			args = append(args, "LIMIT", options.LimitOffset, options.Limit)
		}
		if options.Filter != "" {
			args = append(args, "FILTER", options.Filter)
		}
		if options.WithCursor {
			args = append(args, "WITHCURSOR")
			if options.WithCursorOptions != nil {
				if options.WithCursorOptions.Count > 0 {
					args = append(args, "COUNT", options.WithCursorOptions.Count)
				}
				if options.WithCursorOptions.MaxIdle > 0 {
					args = append(args, "MAXIDLE", options.WithCursorOptions.MaxIdle)
				}
			}
		}
		if options.Params != nil {
			args = append(args, "PARAMS", len(options.Params)*2)
			for key, value := range options.Params {
				args = append(args, key, value)
			}
		}
		if options.DialectVersion > 0 {
			args = append(args, "DIALECT", options.DialectVersion)
		} else {
			args = append(args, "DIALECT", 2)
		}
	}

	cmd := NewAggregateCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// FTAliasAdd - Adds an alias to an index.
// The 'index' parameter specifies the index to which the alias is added, and the 'alias' parameter specifies the alias.
// For more information, please refer to the Redis documentation:
// [FT.ALIASADD]: (https://redis.io/commands/ft.aliasadd/)
func (c cmdable) FTAliasAdd(ctx context.Context, index string, alias string) *StatusCmd {
	args := []interface{}{"FT.ALIASADD", alias, index}
	cmd := NewStatusCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// FTAliasDel - Removes an alias from an index.
// The 'alias' parameter specifies the alias to be removed.
// For more information, please refer to the Redis documentation:
// [FT.ALIASDEL]: (https://redis.io/commands/ft.aliasdel/)
func (c cmdable) FTAliasDel(ctx context.Context, alias string) *StatusCmd {
	cmd := NewStatusCmd(ctx, "FT.ALIASDEL", alias)
	_ = c(ctx, cmd)
	return cmd
}

// FTAliasList - Lists all aliases associated with an index.
// The 'index' parameter specifies the index whose aliases are listed; it must
// be the name of an index created with FT.CREATE, not an alias.
// The reply is an unordered collection of alias names, already deduplicated
// by the server; an index with no aliases yields an empty result, not an
// error. Available since Redis 8.10.
// For more information, please refer to the Redis documentation:
// [FT.ALIASLIST]: (https://redis.io/commands/ft.aliaslist/)
func (c cmdable) FTAliasList(ctx context.Context, index string) *StringSliceCmd {
	cmd := NewStringSliceCmd(ctx, "FT.ALIASLIST", index)
	_ = c(ctx, cmd)
	return cmd
}

// FTAliasUpdate - Updates an alias to an index.
// The 'index' parameter specifies the index to which the alias is updated, and the 'alias' parameter specifies the alias.
// If the alias already exists for a different index, it updates the alias to point to the specified index instead.
// For more information, please refer to the Redis documentation:
// [FT.ALIASUPDATE]: (https://redis.io/commands/ft.aliasupdate/)
func (c cmdable) FTAliasUpdate(ctx context.Context, index string, alias string) *StatusCmd {
	cmd := NewStatusCmd(ctx, "FT.ALIASUPDATE", alias, index)
	_ = c(ctx, cmd)
	return cmd
}

// FTAlter - Alters the definition of an existing index.
// The 'index' parameter specifies the index to alter, and the 'skipInitialScan' parameter specifies whether to skip the initial scan.
// The 'definition' parameter specifies the new definition for the index.
// For more information, please refer to the Redis documentation:
// [FT.ALTER]: (https://redis.io/commands/ft.alter/)
func (c cmdable) FTAlter(ctx context.Context, index string, skipInitialScan bool, definition []interface{}) *StatusCmd {
	args := []interface{}{"FT.ALTER", index}
	if skipInitialScan {
		args = append(args, "SKIPINITIALSCAN")
	}
	args = append(args, "SCHEMA", "ADD")
	args = append(args, definition...)
	cmd := NewStatusCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// Retrieves the value of a RediSearch configuration parameter.
// The 'option' parameter specifies the configuration parameter to retrieve.
// For more information, please refer to the Redis [FT.CONFIG GET] documentation.
//
// Deprecated: FTConfigGet is deprecated in Redis 8.
// All configuration will be done with the CONFIG GET command.
// For more information check [Client.ConfigGet] and [CONFIG GET Documentation]
//
// [CONFIG GET Documentation]: https://redis.io/commands/config-get/
// [FT.CONFIG GET]: https://redis.io/commands/ft.config-get/
func (c cmdable) FTConfigGet(ctx context.Context, option string) *MapMapStringInterfaceCmd {
	cmd := NewMapMapStringInterfaceCmd(ctx, "FT.CONFIG", "GET", option)
	_ = c(ctx, cmd)
	return cmd
}

// Sets the value of a RediSearch configuration parameter.
// The 'option' parameter specifies the configuration parameter to set, and the 'value' parameter specifies the new value.
// For more information, please refer to the Redis [FT.CONFIG SET] documentation.
//
// Deprecated: FTConfigSet is deprecated in Redis 8.
// All configuration will be done with the CONFIG SET command.
// For more information check [Client.ConfigSet] and [CONFIG SET Documentation]
//
// [CONFIG SET Documentation]: https://redis.io/commands/config-set/
// [FT.CONFIG SET]: https://redis.io/commands/ft.config-set/
func (c cmdable) FTConfigSet(ctx context.Context, option string, value interface{}) *StatusCmd {
	cmd := NewStatusCmd(ctx, "FT.CONFIG", "SET", option, value)
	_ = c(ctx, cmd)
	return cmd
}

// FTCreate - Creates a new index with the given options and schema.
// The 'index' parameter specifies the name of the index to create.
// The 'options' parameter specifies various options for the index, such as:
// whether to index hashes or JSONs, prefixes, filters, default language, score, score field, payload field, etc.
// The 'schema' parameter specifies the schema for the index, which includes the field name, field type, etc.
// For more information, please refer to the Redis documentation:
// [FT.CREATE]: (https://redis.io/commands/ft.create/)
func (c cmdable) FTCreate(ctx context.Context, index string, options *FTCreateOptions, schema ...*FieldSchema) *StatusCmd {
	args := []interface{}{"FT.CREATE", index}
	if options != nil {
		if options.OnHash && !options.OnJSON {
			args = append(args, "ON", "HASH")
		}
		if options.OnJSON && !options.OnHash {
			args = append(args, "ON", "JSON")
		}
		if options.OnHash && options.OnJSON {
			cmd := NewStatusCmd(ctx, args...)
			cmd.SetErr(fmt.Errorf("FT.CREATE: ON HASH and ON JSON are mutually exclusive"))
			return cmd
		}
		if options.Prefix != nil {
			args = append(args, "PREFIX", len(options.Prefix))
			args = append(args, options.Prefix...)
		}
		if options.Filter != "" {
			args = append(args, "FILTER", options.Filter)
		}
		if options.DefaultLanguage != "" {
			args = append(args, "LANGUAGE", options.DefaultLanguage)
		}
		if options.LanguageField != "" {
			args = append(args, "LANGUAGE_FIELD", options.LanguageField)
		}
		if options.Score > 0 {
			args = append(args, "SCORE", options.Score)
		}
		if options.ScoreField != "" {
			args = append(args, "SCORE_FIELD", options.ScoreField)
		}
		if options.PayloadField != "" {
			args = append(args, "PAYLOAD_FIELD", options.PayloadField)
		}
		if options.MaxTextFields > 0 {
			args = append(args, "MAXTEXTFIELDS", options.MaxTextFields)
		}
		if options.NoOffsets {
			args = append(args, "NOOFFSETS")
		}
		if options.Temporary > 0 {
			args = append(args, "TEMPORARY", options.Temporary)
		}
		if options.NoHL {
			args = append(args, "NOHL")
		}
		if options.NoFields {
			args = append(args, "NOFIELDS")
		}
		if options.NoFreqs {
			args = append(args, "NOFREQS")
		}
		if options.StopWords != nil {
			args = append(args, "STOPWORDS", len(options.StopWords))
			args = append(args, options.StopWords...)
		}
		if options.SkipInitialScan {
			args = append(args, "SKIPINITIALSCAN")
		}
	}
	if schema == nil {
		cmd := NewStatusCmd(ctx, args...)
		cmd.SetErr(fmt.Errorf("FT.CREATE: SCHEMA is required"))
		return cmd
	}
	args = append(args, "SCHEMA")
	for _, schema := range schema {
		if schema.FieldName == "" || schema.FieldType == SearchFieldTypeInvalid {
			cmd := NewStatusCmd(ctx, args...)
			cmd.SetErr(fmt.Errorf("FT.CREATE: SCHEMA FieldName and FieldType are required"))
			return cmd
		}
		args = append(args, schema.FieldName)
		if schema.As != "" {
			args = append(args, "AS", schema.As)
		}
		args = append(args, schema.FieldType.String())
		if schema.VectorArgs != nil {
			if schema.FieldType != SearchFieldTypeVector {
				cmd := NewStatusCmd(ctx, args...)
				cmd.SetErr(fmt.Errorf("FT.CREATE: SCHEMA FieldType VECTOR is required for VectorArgs"))
				return cmd
			}
			// Check mutual exclusivity of vector options
			optionCount := 0
			if schema.VectorArgs.FlatOptions != nil {
				optionCount++
			}
			if schema.VectorArgs.HNSWOptions != nil {
				optionCount++
			}
			if schema.VectorArgs.VamanaOptions != nil {
				optionCount++
			}
			if optionCount != 1 {
				cmd := NewStatusCmd(ctx, args...)
				cmd.SetErr(fmt.Errorf("FT.CREATE: SCHEMA VectorArgs must have exactly one of FlatOptions, HNSWOptions, or VamanaOptions"))
				return cmd
			}
			if schema.VectorArgs.FlatOptions != nil {
				args = append(args, "FLAT")
				if schema.VectorArgs.FlatOptions.Type == "" || schema.VectorArgs.FlatOptions.Dim == 0 || schema.VectorArgs.FlatOptions.DistanceMetric == "" {
					cmd := NewStatusCmd(ctx, args...)
					cmd.SetErr(fmt.Errorf("FT.CREATE: Type, Dim and DistanceMetric are required for VECTOR FLAT"))
					return cmd
				}
				flatArgs := []interface{}{
					"TYPE", schema.VectorArgs.FlatOptions.Type,
					"DIM", schema.VectorArgs.FlatOptions.Dim,
					"DISTANCE_METRIC", schema.VectorArgs.FlatOptions.DistanceMetric,
				}
				if schema.VectorArgs.FlatOptions.InitialCapacity > 0 {
					flatArgs = append(flatArgs, "INITIAL_CAP", schema.VectorArgs.FlatOptions.InitialCapacity)
				}
				if schema.VectorArgs.FlatOptions.BlockSize > 0 {
					flatArgs = append(flatArgs, "BLOCK_SIZE", schema.VectorArgs.FlatOptions.BlockSize)
				}
				args = append(args, len(flatArgs))
				args = append(args, flatArgs...)
			}
			if schema.VectorArgs.HNSWOptions != nil {
				args = append(args, "HNSW")
				if schema.VectorArgs.HNSWOptions.Type == "" || schema.VectorArgs.HNSWOptions.Dim == 0 || schema.VectorArgs.HNSWOptions.DistanceMetric == "" {
					cmd := NewStatusCmd(ctx, args...)
					cmd.SetErr(fmt.Errorf("FT.CREATE: Type, Dim and DistanceMetric are required for VECTOR HNSW"))
					return cmd
				}
				hnswArgs := []interface{}{
					"TYPE", schema.VectorArgs.HNSWOptions.Type,
					"DIM", schema.VectorArgs.HNSWOptions.Dim,
					"DISTANCE_METRIC", schema.VectorArgs.HNSWOptions.DistanceMetric,
				}
				if schema.VectorArgs.HNSWOptions.InitialCapacity > 0 {
					hnswArgs = append(hnswArgs, "INITIAL_CAP", schema.VectorArgs.HNSWOptions.InitialCapacity)
				}
				if schema.VectorArgs.HNSWOptions.MaxEdgesPerNode > 0 {
					hnswArgs = append(hnswArgs, "M", schema.VectorArgs.HNSWOptions.MaxEdgesPerNode)
				}
				if schema.VectorArgs.HNSWOptions.MaxAllowedEdgesPerNode > 0 {
					hnswArgs = append(hnswArgs, "EF_CONSTRUCTION", schema.VectorArgs.HNSWOptions.MaxAllowedEdgesPerNode)
				}
				if schema.VectorArgs.HNSWOptions.EFRunTime > 0 {
					hnswArgs = append(hnswArgs, "EF_RUNTIME", schema.VectorArgs.HNSWOptions.EFRunTime)
				}
				if schema.VectorArgs.HNSWOptions.Epsilon > 0 {
					hnswArgs = append(hnswArgs, "EPSILON", schema.VectorArgs.HNSWOptions.Epsilon)
				}
				if schema.VectorArgs.HNSWOptions.Rerank || schema.VectorArgs.HNSWOptions.HasRerank {
					rerank := "FALSE"
					if schema.VectorArgs.HNSWOptions.Rerank {
						rerank = "TRUE"
					}
					hnswArgs = append(hnswArgs, "RERANK", rerank)
				}
				args = append(args, len(hnswArgs))
				args = append(args, hnswArgs...)
			}
			if schema.VectorArgs.VamanaOptions != nil {
				args = append(args, "SVS-VAMANA")
				if schema.VectorArgs.VamanaOptions.Type == "" || schema.VectorArgs.VamanaOptions.Dim == 0 || schema.VectorArgs.VamanaOptions.DistanceMetric == "" {
					cmd := NewStatusCmd(ctx, args...)
					cmd.SetErr(fmt.Errorf("FT.CREATE: Type, Dim and DistanceMetric are required for VECTOR VAMANA"))
					return cmd
				}
				vamanaArgs := []interface{}{
					"TYPE", schema.VectorArgs.VamanaOptions.Type,
					"DIM", schema.VectorArgs.VamanaOptions.Dim,
					"DISTANCE_METRIC", schema.VectorArgs.VamanaOptions.DistanceMetric,
				}
				if schema.VectorArgs.VamanaOptions.Compression != "" {
					vamanaArgs = append(vamanaArgs, "COMPRESSION", schema.VectorArgs.VamanaOptions.Compression)
				}
				if schema.VectorArgs.VamanaOptions.ConstructionWindowSize > 0 {
					vamanaArgs = append(vamanaArgs, "CONSTRUCTION_WINDOW_SIZE", schema.VectorArgs.VamanaOptions.ConstructionWindowSize)
				}
				if schema.VectorArgs.VamanaOptions.GraphMaxDegree > 0 {
					vamanaArgs = append(vamanaArgs, "GRAPH_MAX_DEGREE", schema.VectorArgs.VamanaOptions.GraphMaxDegree)
				}
				if schema.VectorArgs.VamanaOptions.SearchWindowSize > 0 {
					vamanaArgs = append(vamanaArgs, "SEARCH_WINDOW_SIZE", schema.VectorArgs.VamanaOptions.SearchWindowSize)
				}
				if schema.VectorArgs.VamanaOptions.Epsilon > 0 {
					vamanaArgs = append(vamanaArgs, "EPSILON", schema.VectorArgs.VamanaOptions.Epsilon)
				}
				if schema.VectorArgs.VamanaOptions.TrainingThreshold > 0 {
					vamanaArgs = append(vamanaArgs, "TRAINING_THRESHOLD", schema.VectorArgs.VamanaOptions.TrainingThreshold)
				}
				if schema.VectorArgs.VamanaOptions.ReduceDim > 0 {
					vamanaArgs = append(vamanaArgs, "REDUCE", schema.VectorArgs.VamanaOptions.ReduceDim)
				}
				args = append(args, len(vamanaArgs))
				args = append(args, vamanaArgs...)
			}
		}
		if schema.GeoShapeFieldType != "" {
			if schema.FieldType != SearchFieldTypeGeoShape {
				cmd := NewStatusCmd(ctx, args...)
				cmd.SetErr(fmt.Errorf("FT.CREATE: SCHEMA FieldType GEOSHAPE is required for GeoShapeFieldType"))
				return cmd
			}
			args = append(args, schema.GeoShapeFieldType)
		}
		if schema.NoStem {
			args = append(args, "NOSTEM")
		}
		if schema.Sortable {
			args = append(args, "SORTABLE")
		}
		if schema.UNF {
			args = append(args, "UNF")
		}
		if schema.NoIndex {
			args = append(args, "NOINDEX")
		}
		if schema.PhoneticMatcher != "" {
			args = append(args, "PHONETIC", schema.PhoneticMatcher)
		}
		if schema.Weight > 0 {
			args = append(args, "WEIGHT", schema.Weight)
		}
		if schema.Separator != "" {
			args = append(args, "SEPARATOR", schema.Separator)
		}
		if schema.CaseSensitive {
			args = append(args, "CASESENSITIVE")
		}
		if schema.WithSuffixtrie {
			args = append(args, "WITHSUFFIXTRIE")
		}
		if schema.IndexEmpty {
			args = append(args, "INDEXEMPTY")
		}
		if schema.IndexMissing {
			args = append(args, "INDEXMISSING")
		}
	}
	cmd := NewStatusCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// FTCursorDel - Deletes a cursor from an existing index.
// The 'index' parameter specifies the index from which to delete the cursor, and the 'cursorId' parameter specifies the ID of the cursor to delete.
// For more information, please refer to the Redis documentation:
// [FT.CURSOR DEL]: (https://redis.io/commands/ft.cursor-del/)
func (c cmdable) FTCursorDel(ctx context.Context, index string, cursorId int) *StatusCmd {
	cmd := NewStatusCmd(ctx, "FT.CURSOR", "DEL", index, cursorId)
	_ = c(ctx, cmd)
	return cmd
}

// FTCursorRead - Reads the next results from an existing cursor.
// The 'index' parameter specifies the index from which to read the cursor, the 'cursorId' parameter specifies the ID of the cursor to read, and the 'count' parameter specifies the number of results to read.
// For more information, please refer to the Redis documentation:
// [FT.CURSOR READ]: (https://redis.io/commands/ft.cursor-read/)
func (c cmdable) FTCursorRead(ctx context.Context, index string, cursorId int, count int) *MapStringInterfaceCmd {
	args := []interface{}{"FT.CURSOR", "READ", index, cursorId}
	if count > 0 {
		args = append(args, "COUNT", count)
	}
	cmd := NewMapStringInterfaceCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// FTDictAdd - Adds terms to a dictionary.
// The 'dict' parameter specifies the dictionary to which to add the terms, and the 'term' parameter specifies the terms to add.
// For more information, please refer to the Redis documentation:
// [FT.DICTADD]: (https://redis.io/commands/ft.dictadd/)
func (c cmdable) FTDictAdd(ctx context.Context, dict string, term ...interface{}) *IntCmd {
	args := []interface{}{"FT.DICTADD", dict}
	args = append(args, term...)
	cmd := NewIntCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// FTDictDel - Deletes terms from a dictionary.
// The 'dict' parameter specifies the dictionary from which to delete the terms, and the 'term' parameter specifies the terms to delete.
// For more information, please refer to the Redis documentation:
// [FT.DICTDEL]: (https://redis.io/commands/ft.dictdel/)
func (c cmdable) FTDictDel(ctx context.Context, dict string, term ...interface{}) *IntCmd {
	args := []interface{}{"FT.DICTDEL", dict}
	args = append(args, term...)
	cmd := NewIntCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// FTDictDump - Returns all terms in the specified dictionary.
// The 'dict' parameter specifies the dictionary from which to return the terms.
// For more information, please refer to the Redis documentation:
// [FT.DICTDUMP]: (https://redis.io/commands/ft.dictdump/)
func (c cmdable) FTDictDump(ctx context.Context, dict string) *StringSliceCmd {
	cmd := NewStringSliceCmd(ctx, "FT.DICTDUMP", dict)
	_ = c(ctx, cmd)
	return cmd
}

// FTDropIndex - Deletes an index.
// The 'index' parameter specifies the index to delete.
// For more information, please refer to the Redis documentation:
// [FT.DROPINDEX]: (https://redis.io/commands/ft.dropindex/)
func (c cmdable) FTDropIndex(ctx context.Context, index string) *StatusCmd {
	args := []interface{}{"FT.DROPINDEX", index}
	cmd := NewStatusCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// FTDropIndexWithArgs - Deletes an index with options.
// The 'index' parameter specifies the index to delete, and the 'options' parameter specifies the DeleteDocs option for docs deletion.
// For more information, please refer to the Redis documentation:
// [FT.DROPINDEX]: (https://redis.io/commands/ft.dropindex/)
func (c cmdable) FTDropIndexWithArgs(ctx context.Context, index string, options *FTDropIndexOptions) *StatusCmd {
	args := []interface{}{"FT.DROPINDEX", index}
	if options != nil {
		if options.DeleteDocs {
			args = append(args, "DD")
		}
	}
	cmd := NewStatusCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// FTExplain - Returns the execution plan for a complex query.
// The 'index' parameter specifies the index to query, and the 'query' parameter specifies the query string.
// For more information, please refer to the Redis documentation:
// [FT.EXPLAIN]: (https://redis.io/commands/ft.explain/)
func (c cmdable) FTExplain(ctx context.Context, index string, query string) *StringCmd {
	cmd := NewStringCmd(ctx, "FT.EXPLAIN", index, query)
	_ = c(ctx, cmd)
	return cmd
}

// FTExplainWithArgs - Returns the execution plan for a complex query with options.
// The 'index' parameter specifies the index to query, the 'query' parameter specifies the query string, and the 'options' parameter specifies the Dialect for the query.
// For more information, please refer to the Redis documentation:
// [FT.EXPLAIN]: (https://redis.io/commands/ft.explain/)
func (c cmdable) FTExplainWithArgs(ctx context.Context, index string, query string, options *FTExplainOptions) *StringCmd {
	args := []interface{}{"FT.EXPLAIN", index, query}
	if options.Dialect != "" {
		args = append(args, "DIALECT", options.Dialect)
	} else {
		args = append(args, "DIALECT", 2)
	}
	cmd := NewStringCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// FTExplainCli - Returns the execution plan for a complex query. [Not Implemented]
// For more information, see https://redis.io/commands/ft.explaincli/
func (c cmdable) FTExplainCli(ctx context.Context, key, path string) error {
	return fmt.Errorf("FTExplainCli is not implemented")
}

// parseFTAttributeFromMap parses an FTAttribute from a RESP3 map format
func parseFTAttributeFromMap(attrMap map[interface{}]interface{}) FTAttribute {
	att := FTAttribute{}
	for k, v := range attrMap {
		key := internal.ToLower(internal.ToString(k))
		switch key {
		case "attribute":
			att.Attribute = internal.ToString(v)
		case "identifier":
			att.Identifier = internal.ToString(v)
		case "type":
			att.Type = internal.ToString(v)
		case "weight":
			att.Weight = internal.ToFloat(v)
		case "phonetic":
			att.PhoneticMatcher = internal.ToString(v)
		case "algorithm":
			att.Algorithm = internal.ToString(v)
		case "data_type":
			att.DataType = internal.ToString(v)
		case "dim":
			att.Dim = internal.ToInteger(v)
		case "distance_metric":
			att.DistanceMetric = internal.ToString(v)
		case "m":
			att.M = internal.ToInteger(v)
		case "ef_construction":
			att.EFConstruction = internal.ToInteger(v)
		case "flags":
			// flags is an array of strings like ["SORTABLE", "NOSTEM"]
			if flags, ok := v.([]interface{}); ok {
				for _, flag := range flags {
					flagStr := internal.ToLower(internal.ToString(flag))
					switch flagStr {
					case "nostem":
						att.NoStem = true
					case "sortable":
						att.Sortable = true
					case "noindex":
						att.NoIndex = true
					case "unf":
						att.UNF = true
					case "case_sensitive":
						att.CaseSensitive = true
					case "withsuffixtrie":
						att.WithSuffixtrie = true
					}
				}
			}
		}
	}
	return att
}

// getMapStringKey extracts a string value from a map with interface{} keys
func getMapStringKey(m map[interface{}]interface{}, key string) interface{} {
	if v, ok := m[key]; ok {
		return v
	}
	return nil
}

// parseIndexErrorsRESP3 parses Index Errors from RESP3 map format
func parseIndexErrorsRESP3(m map[interface{}]interface{}) IndexErrors {
	return IndexErrors{
		IndexingFailures:     internal.ToInteger(getMapStringKey(m, "indexing failures")),
		LastIndexingError:    internal.ToString(getMapStringKey(m, "last indexing error")),
		LastIndexingErrorKey: internal.ToString(getMapStringKey(m, "last indexing error key")),
	}
}

// parseCursorStatsRESP3 parses cursor_stats from RESP3 map format
func parseCursorStatsRESP3(m map[interface{}]interface{}) CursorStats {
	return CursorStats{
		GlobalIdle:    internal.ToInteger(getMapStringKey(m, "global_idle")),
		GlobalTotal:   internal.ToInteger(getMapStringKey(m, "global_total")),
		IndexCapacity: internal.ToInteger(getMapStringKey(m, "index_capacity")),
		IndexTotal:    internal.ToInteger(getMapStringKey(m, "index_total")),
	}
}

// parseGCStatsRESP3 parses gc_stats from RESP3 map format
func parseGCStatsRESP3(m map[interface{}]interface{}) GCStats {
	// Handle average_cycle_time_ms which can be a float64 (including NaN) or string
	avgCycleTime := ""
	if v := getMapStringKey(m, "average_cycle_time_ms"); v != nil {
		switch val := v.(type) {
		case string:
			// Normalize to lowercase for consistency with RESP2
			avgCycleTime = strings.ToLower(val)
		case float64:
			avgCycleTime = internal.FormatFloat(val)
		}
	}

	return GCStats{
		BytesCollected:       ftInfoNumInt(getMapStringKey(m, "bytes_collected")),
		TotalMsRun:           ftInfoNumInt(getMapStringKey(m, "total_ms_run")),
		TotalCycles:          ftInfoNumInt(getMapStringKey(m, "total_cycles")),
		AverageCycleTimeMs:   avgCycleTime,
		LastRunTimeMs:        ftInfoNumInt(getMapStringKey(m, "last_run_time_ms")),
		GCNumericTreesMissed: ftInfoNumInt(getMapStringKey(m, "gc_numeric_trees_missed")),
		GCBlocksDenied:       ftInfoNumInt(getMapStringKey(m, "gc_blocks_denied")),
	}
}

// parseIndexDefinitionRESP3 parses index_definition from RESP3 map format
func parseIndexDefinitionRESP3(m map[interface{}]interface{}) IndexDefinition {
	def := IndexDefinition{
		KeyType:      internal.ToString(getMapStringKey(m, "key_type")),
		DefaultScore: internal.ToFloat(getMapStringKey(m, "default_score")),
	}
	if prefixes, ok := getMapStringKey(m, "prefixes").([]interface{}); ok {
		def.Prefixes = internal.ToStringSlice(prefixes)
	}
	return def
}

// parseDialectStatsRESP3 parses dialect_stats from RESP3 map format
func parseDialectStatsRESP3(m map[interface{}]interface{}) map[string]int {
	result := make(map[string]int)
	for k, v := range m {
		if kStr, ok := k.(string); ok {
			result[kStr] = internal.ToInteger(v)
		}
	}
	return result
}

// ftInfoNumString stringifies a value that RediSearch emits via REPLY_KVNUM
// (RedisModule_ReplyWithDouble): a bulk string in RESP2 but a native double
// in RESP3. Used for FTInfoResult fields whose public type is string.
// Special float values (NaN, +Inf, -Inf) are normalized to lowercase to match
// the RESP2 wire format.
func ftInfoNumString(val interface{}) string {
	switch v := val.(type) {
	case string:
		return v
	case float64:
		return internal.FormatFloat(v)
	case float32:
		return internal.FormatFloat(float64(v))
	case int64:
		return strconv.FormatInt(v, 10)
	case int:
		return strconv.Itoa(v)
	default:
		return ""
	}
}

// ftInfoNumInt converts a value that RediSearch emits via REPLY_KVNUM to int.
// In RESP2 the value is a bulk string; in RESP3 it is a native double, even
// for logically-integer fields (counters, byte sizes). This helper exists so
// the internal.ToInteger helper can remain strict about float-to-int coercion
// while still letting the RediSearch parsers read those values correctly.
func ftInfoNumInt(val interface{}) int {
	switch v := val.(type) {
	case float64:
		return int(v)
	case float32:
		return int(v)
	default:
		return internal.ToInteger(v)
	}
}

func parseFTInfo(data map[string]interface{}) (FTInfoResult, error) {
	var ftInfo FTInfoResult

	// Parse Index Errors - handle both RESP2 (array) and RESP3 (map) formats
	if indexErrors, ok := data["Index Errors"].([]interface{}); ok {
		// RESP2 format: array with key-value pairs
		ftInfo.IndexErrors = IndexErrors{
			IndexingFailures:     internal.ToInteger(indexErrors[1]),
			LastIndexingError:    internal.ToString(indexErrors[3]),
			LastIndexingErrorKey: internal.ToString(indexErrors[5]),
		}
	} else if indexErrors, ok := data["Index Errors"].(map[interface{}]interface{}); ok {
		// RESP3 format: map
		ftInfo.IndexErrors = parseIndexErrorsRESP3(indexErrors)
	}

	if attributes, ok := data["attributes"].([]interface{}); ok {
		for _, attr := range attributes {
			att := FTAttribute{}
			// Handle RESP2 format: attribute is []interface{}
			if attrSlice, ok := attr.([]interface{}); ok {
				attrLen := len(attrSlice)
				for i := 0; i < attrLen; i++ {
					if internal.ToLower(internal.ToString(attrSlice[i])) == "attribute" && i+1 < attrLen {
						att.Attribute = internal.ToString(attrSlice[i+1])
						i++
						continue
					}
					if internal.ToLower(internal.ToString(attrSlice[i])) == "identifier" && i+1 < attrLen {
						att.Identifier = internal.ToString(attrSlice[i+1])
						i++
						continue
					}
					if internal.ToLower(internal.ToString(attrSlice[i])) == "type" && i+1 < attrLen {
						att.Type = internal.ToString(attrSlice[i+1])
						i++
						continue
					}
					if internal.ToLower(internal.ToString(attrSlice[i])) == "weight" && i+1 < attrLen {
						att.Weight = internal.ToFloat(attrSlice[i+1])
						i++
						continue
					}
					if internal.ToLower(internal.ToString(attrSlice[i])) == "nostem" {
						att.NoStem = true
						continue
					}
					if internal.ToLower(internal.ToString(attrSlice[i])) == "sortable" {
						att.Sortable = true
						continue
					}
					if internal.ToLower(internal.ToString(attrSlice[i])) == "noindex" {
						att.NoIndex = true
						continue
					}
					if internal.ToLower(internal.ToString(attrSlice[i])) == "unf" {
						att.UNF = true
						continue
					}
					if internal.ToLower(internal.ToString(attrSlice[i])) == "phonetic" && i+1 < attrLen {
						att.PhoneticMatcher = internal.ToString(attrSlice[i+1])
						continue
					}
					if internal.ToLower(internal.ToString(attrSlice[i])) == "case_sensitive" {
						att.CaseSensitive = true
						continue
					}
					if internal.ToLower(internal.ToString(attrSlice[i])) == "withsuffixtrie" {
						att.WithSuffixtrie = true
						continue
					}

					// vector specific attributes
					if internal.ToLower(internal.ToString(attrSlice[i])) == "algorithm" && i+1 < attrLen {
						att.Algorithm = internal.ToString(attrSlice[i+1])
						i++
						continue
					}
					if internal.ToLower(internal.ToString(attrSlice[i])) == "data_type" && i+1 < attrLen {
						att.DataType = internal.ToString(attrSlice[i+1])
						i++
						continue
					}
					if internal.ToLower(internal.ToString(attrSlice[i])) == "dim" && i+1 < attrLen {
						att.Dim = internal.ToInteger(attrSlice[i+1])
						i++
						continue
					}
					if internal.ToLower(internal.ToString(attrSlice[i])) == "distance_metric" && i+1 < attrLen {
						att.DistanceMetric = internal.ToString(attrSlice[i+1])
						i++
						continue
					}
					if internal.ToLower(internal.ToString(attrSlice[i])) == "m" && i+1 < attrLen {
						att.M = internal.ToInteger(attrSlice[i+1])
						i++
						continue
					}
					if internal.ToLower(internal.ToString(attrSlice[i])) == "ef_construction" && i+1 < attrLen {
						att.EFConstruction = internal.ToInteger(attrSlice[i+1])
						i++
						continue
					}
				}
				ftInfo.Attributes = append(ftInfo.Attributes, att)
			} else if attrMap, ok := attr.(map[interface{}]interface{}); ok {
				// Handle RESP3 format: attribute is map[interface{}]interface{}
				att = parseFTAttributeFromMap(attrMap)
				ftInfo.Attributes = append(ftInfo.Attributes, att)
			}
		}
	}

	ftInfo.BytesPerRecordAvg = ftInfoNumString(data["bytes_per_record_avg"])
	ftInfo.Cleaning = internal.ToInteger(data["cleaning"])

	// Parse cursor_stats - handle both RESP2 (array) and RESP3 (map) formats
	if cursorStats, ok := data["cursor_stats"].([]interface{}); ok {
		// RESP2 format
		ftInfo.CursorStats = CursorStats{
			GlobalIdle:    internal.ToInteger(cursorStats[1]),
			GlobalTotal:   internal.ToInteger(cursorStats[3]),
			IndexCapacity: internal.ToInteger(cursorStats[5]),
			IndexTotal:    internal.ToInteger(cursorStats[7]),
		}
	} else if cursorStats, ok := data["cursor_stats"].(map[interface{}]interface{}); ok {
		// RESP3 format
		ftInfo.CursorStats = parseCursorStatsRESP3(cursorStats)
	}

	// Parse dialect_stats - handle both RESP2 (array) and RESP3 (map) formats
	if dialectStats, ok := data["dialect_stats"].([]interface{}); ok {
		// RESP2 format
		ftInfo.DialectStats = make(map[string]int)
		for i := 0; i < len(dialectStats); i += 2 {
			ftInfo.DialectStats[internal.ToString(dialectStats[i])] = internal.ToInteger(dialectStats[i+1])
		}
	} else if dialectStats, ok := data["dialect_stats"].(map[interface{}]interface{}); ok {
		// RESP3 format
		ftInfo.DialectStats = parseDialectStatsRESP3(dialectStats)
	}

	ftInfo.DocTableSizeMB = internal.ToFloat(data["doc_table_size_mb"])

	// Parse field statistics - handle both RESP2 and RESP3 formats
	if fieldStats, ok := data["field statistics"].([]interface{}); ok {
		for _, stat := range fieldStats {
			if statMap, ok := stat.([]interface{}); ok {
				// RESP2 format
				ftInfo.FieldStatistics = append(ftInfo.FieldStatistics, FieldStatistic{
					Identifier: internal.ToString(statMap[1]),
					Attribute:  internal.ToString(statMap[3]),
					IndexErrors: IndexErrors{
						IndexingFailures:     internal.ToInteger(statMap[5].([]interface{})[1]),
						LastIndexingError:    internal.ToString(statMap[5].([]interface{})[3]),
						LastIndexingErrorKey: internal.ToString(statMap[5].([]interface{})[5]),
					},
				})
			} else if statMap, ok := stat.(map[interface{}]interface{}); ok {
				// RESP3 format
				fs := FieldStatistic{
					Identifier: internal.ToString(getMapStringKey(statMap, "identifier")),
					Attribute:  internal.ToString(getMapStringKey(statMap, "attribute")),
				}
				if indexErrors, ok := getMapStringKey(statMap, "Index Errors").(map[interface{}]interface{}); ok {
					fs.IndexErrors = parseIndexErrorsRESP3(indexErrors)
				}
				ftInfo.FieldStatistics = append(ftInfo.FieldStatistics, fs)
			}
		}
	}

	// Parse gc_stats - handle both RESP2 (array) and RESP3 (map) formats
	if gcStats, ok := data["gc_stats"].([]interface{}); ok {
		// RESP2 format
		ftInfo.GCStats = GCStats{}
		for i := 0; i < len(gcStats); i += 2 {
			if internal.ToLower(internal.ToString(gcStats[i])) == "bytes_collected" {
				ftInfo.GCStats.BytesCollected = internal.ToInteger(gcStats[i+1])
				continue
			}
			if internal.ToLower(internal.ToString(gcStats[i])) == "total_ms_run" {
				ftInfo.GCStats.TotalMsRun = internal.ToInteger(gcStats[i+1])
				continue
			}
			if internal.ToLower(internal.ToString(gcStats[i])) == "total_cycles" {
				ftInfo.GCStats.TotalCycles = internal.ToInteger(gcStats[i+1])
				continue
			}
			if internal.ToLower(internal.ToString(gcStats[i])) == "average_cycle_time_ms" {
				ftInfo.GCStats.AverageCycleTimeMs = internal.ToString(gcStats[i+1])
				continue
			}
			if internal.ToLower(internal.ToString(gcStats[i])) == "last_run_time_ms" {
				ftInfo.GCStats.LastRunTimeMs = internal.ToInteger(gcStats[i+1])
				continue
			}
			if internal.ToLower(internal.ToString(gcStats[i])) == "gc_numeric_trees_missed" {
				ftInfo.GCStats.GCNumericTreesMissed = internal.ToInteger(gcStats[i+1])
				continue
			}
			if internal.ToLower(internal.ToString(gcStats[i])) == "gc_blocks_denied" {
				ftInfo.GCStats.GCBlocksDenied = internal.ToInteger(gcStats[i+1])
				continue
			}
		}
	} else if gcStats, ok := data["gc_stats"].(map[interface{}]interface{}); ok {
		// RESP3 format
		ftInfo.GCStats = parseGCStatsRESP3(gcStats)
	}

	ftInfo.GeoshapesSzMB = internal.ToFloat(data["geoshapes_sz_mb"])
	ftInfo.HashIndexingFailures = internal.ToInteger(data["hash_indexing_failures"])

	// Parse index_definition - handle both RESP2 (array) and RESP3 (map) formats
	if indexDef, ok := data["index_definition"].([]interface{}); ok {
		// RESP2 format
		ftInfo.IndexDefinition = IndexDefinition{
			KeyType:      internal.ToString(indexDef[1]),
			Prefixes:     internal.ToStringSlice(indexDef[3]),
			DefaultScore: internal.ToFloat(indexDef[5]),
		}
	} else if indexDef, ok := data["index_definition"].(map[interface{}]interface{}); ok {
		// RESP3 format
		ftInfo.IndexDefinition = parseIndexDefinitionRESP3(indexDef)
	}

	ftInfo.IndexName = internal.ToString(data["index_name"])
	if indexOptions, ok := data["index_options"].([]interface{}); ok {
		ftInfo.IndexOptions = internal.ToStringSlice(indexOptions)
	}
	ftInfo.Indexing = internal.ToInteger(data["indexing"])
	ftInfo.InvertedSzMB = internal.ToFloat(data["inverted_sz_mb"])
	ftInfo.KeyTableSizeMB = internal.ToFloat(data["key_table_size_mb"])
	ftInfo.MaxDocID = internal.ToInteger(data["max_doc_id"])
	ftInfo.NumDocs = internal.ToInteger(data["num_docs"])
	ftInfo.NumRecords = internal.ToInteger(data["num_records"])
	ftInfo.NumTerms = internal.ToInteger(data["num_terms"])
	ftInfo.NumberOfUses = internal.ToInteger(data["number_of_uses"])
	ftInfo.OffsetBitsPerRecordAvg = ftInfoNumString(data["offset_bits_per_record_avg"])
	ftInfo.OffsetVectorsSzMB = internal.ToFloat(data["offset_vectors_sz_mb"])
	ftInfo.OffsetsPerTermAvg = ftInfoNumString(data["offsets_per_term_avg"])
	ftInfo.PercentIndexed = internal.ToFloat(data["percent_indexed"])
	ftInfo.RecordsPerDocAvg = ftInfoNumString(data["records_per_doc_avg"])
	ftInfo.SortableValuesSizeMB = internal.ToFloat(data["sortable_values_size_mb"])
	ftInfo.TagOverheadSzMB = internal.ToFloat(data["tag_overhead_sz_mb"])
	ftInfo.TextOverheadSzMB = internal.ToFloat(data["text_overhead_sz_mb"])
	ftInfo.TotalIndexMemorySzMB = internal.ToFloat(data["total_index_memory_sz_mb"])
	ftInfo.TotalIndexingTime = ftInfoNumInt(data["total_indexing_time"])
	ftInfo.TotalInvertedIndexBlocks = internal.ToInteger(data["total_inverted_index_blocks"])
	ftInfo.VectorIndexSzMB = internal.ToFloat(data["vector_index_sz_mb"])

	return ftInfo, nil
}

type FTInfoCmd struct {
	baseCmd
	val FTInfoResult
}

func newFTInfoCmd(ctx context.Context, args ...interface{}) *FTInfoCmd {
	return &FTInfoCmd{
		baseCmd: baseCmd{
			ctx:     ctx,
			args:    args,
			cmdType: CmdTypeFTInfo,
		},
	}
}

func (cmd *FTInfoCmd) String() string {
	cmd.await()
	return cmdString(cmd, cmd.val)
}

func (cmd *FTInfoCmd) SetVal(val FTInfoResult) {
	cmd.val = val
}

func (cmd *FTInfoCmd) Result() (FTInfoResult, error) {
	cmd.await()
	return cmd.val, cmd.err
}

func (cmd *FTInfoCmd) Val() FTInfoResult {
	cmd.await()
	return cmd.val
}

func (cmd *FTInfoCmd) RawVal() interface{} {
	cmd.await()
	return cmd.rawVal
}

func (cmd *FTInfoCmd) RawResult() (interface{}, error) {
	cmd.await()
	return cmd.rawVal, cmd.err
}

func (cmd *FTInfoCmd) readReply(rd *proto.Reader) (err error) {
	readType, err := rd.PeekReplyType()
	if err != nil {
		return err
	}

	// RESP3 returns a map, RESP2 returns an array
	if readType == proto.RespMap {
		// Read raw response first for backwards compatibility
		cmd.rawVal, err = rd.ReadReply()
		if err != nil {
			return err
		}

		// Convert map[interface{}]interface{} to map[string]interface{}
		rawMap, ok := cmd.rawVal.(map[interface{}]interface{})
		if !ok {
			return fmt.Errorf("unexpected RESP3 response type: %T", cmd.rawVal)
		}

		data := make(map[string]interface{}, len(rawMap))
		for k, v := range rawMap {
			if kStr, ok := k.(string); ok {
				data[kStr] = v
			}
		}

		cmd.val, err = parseFTInfo(data)
		return err
	}

	// RESP2 format - read as map
	n, err := rd.ReadMapLen()
	if err != nil {
		return err
	}

	data := make(map[string]interface{}, n)
	for i := 0; i < n; i++ {
		k, err := rd.ReadString()
		if err != nil {
			return err
		}
		v, err := rd.ReadReply()
		if err != nil {
			if err == Nil {
				data[k] = Nil
				continue
			}
			if err, ok := err.(proto.RedisError); ok {
				data[k] = err
				continue
			}
			return err
		}
		data[k] = v
	}
	cmd.val, err = parseFTInfo(data)
	return err
}

func (cmd *FTInfoCmd) Clone() Cmder {
	val := FTInfoResult{
		IndexErrors:              cmd.val.IndexErrors,
		BytesPerRecordAvg:        cmd.val.BytesPerRecordAvg,
		Cleaning:                 cmd.val.Cleaning,
		CursorStats:              cmd.val.CursorStats,
		DocTableSizeMB:           cmd.val.DocTableSizeMB,
		GCStats:                  cmd.val.GCStats,
		GeoshapesSzMB:            cmd.val.GeoshapesSzMB,
		HashIndexingFailures:     cmd.val.HashIndexingFailures,
		IndexDefinition:          cmd.val.IndexDefinition,
		IndexName:                cmd.val.IndexName,
		Indexing:                 cmd.val.Indexing,
		InvertedSzMB:             cmd.val.InvertedSzMB,
		KeyTableSizeMB:           cmd.val.KeyTableSizeMB,
		MaxDocID:                 cmd.val.MaxDocID,
		NumDocs:                  cmd.val.NumDocs,
		NumRecords:               cmd.val.NumRecords,
		NumTerms:                 cmd.val.NumTerms,
		NumberOfUses:             cmd.val.NumberOfUses,
		OffsetBitsPerRecordAvg:   cmd.val.OffsetBitsPerRecordAvg,
		OffsetVectorsSzMB:        cmd.val.OffsetVectorsSzMB,
		OffsetsPerTermAvg:        cmd.val.OffsetsPerTermAvg,
		PercentIndexed:           cmd.val.PercentIndexed,
		RecordsPerDocAvg:         cmd.val.RecordsPerDocAvg,
		SortableValuesSizeMB:     cmd.val.SortableValuesSizeMB,
		TagOverheadSzMB:          cmd.val.TagOverheadSzMB,
		TextOverheadSzMB:         cmd.val.TextOverheadSzMB,
		TotalIndexMemorySzMB:     cmd.val.TotalIndexMemorySzMB,
		TotalIndexingTime:        cmd.val.TotalIndexingTime,
		TotalInvertedIndexBlocks: cmd.val.TotalInvertedIndexBlocks,
		VectorIndexSzMB:          cmd.val.VectorIndexSzMB,
	}
	// Clone slices and maps
	if cmd.val.Attributes != nil {
		val.Attributes = slices.Clone(cmd.val.Attributes)
	}
	if cmd.val.DialectStats != nil {
		val.DialectStats = maps.Clone(cmd.val.DialectStats)
	}
	if cmd.val.FieldStatistics != nil {
		val.FieldStatistics = slices.Clone(cmd.val.FieldStatistics)
	}
	if cmd.val.IndexOptions != nil {
		val.IndexOptions = slices.Clone(cmd.val.IndexOptions)
	}
	if cmd.val.IndexDefinition.Prefixes != nil {
		val.IndexDefinition.Prefixes = slices.Clone(cmd.val.IndexDefinition.Prefixes)
	}
	return &FTInfoCmd{
		baseCmd: cmd.cloneBaseCmd(),
		val:     val,
	}
}

// FTInfo - Retrieves information about an index.
// The 'index' parameter specifies the index to retrieve information about.
// For more information, please refer to the Redis documentation:
// [FT.INFO]: (https://redis.io/commands/ft.info/)
func (c cmdable) FTInfo(ctx context.Context, index string) *FTInfoCmd {
	cmd := newFTInfoCmd(ctx, "FT.INFO", index)
	_ = c(ctx, cmd)
	return cmd
}

// FTSpellCheck - Checks a query string for spelling errors.
// For more details about spellcheck query please follow:
// https://redis.io/docs/interact/search-and-query/advanced-concepts/spellcheck/
// For more information, please refer to the Redis documentation:
// [FT.SPELLCHECK]: (https://redis.io/commands/ft.spellcheck/)
func (c cmdable) FTSpellCheck(ctx context.Context, index string, query string) *FTSpellCheckCmd {
	args := []interface{}{"FT.SPELLCHECK", index, query}
	cmd := newFTSpellCheckCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// FTSpellCheckWithArgs - Checks a query string for spelling errors with additional options.
// For more details about spellcheck query please follow:
// https://redis.io/docs/interact/search-and-query/advanced-concepts/spellcheck/
// For more information, please refer to the Redis documentation:
// [FT.SPELLCHECK]: (https://redis.io/commands/ft.spellcheck/)
func (c cmdable) FTSpellCheckWithArgs(ctx context.Context, index string, query string, options *FTSpellCheckOptions) *FTSpellCheckCmd {
	args := []interface{}{"FT.SPELLCHECK", index, query}
	if options != nil {
		if options.Distance > 0 {
			args = append(args, "DISTANCE", options.Distance)
		}
		if options.Terms != nil {
			args = append(args, "TERMS", options.Terms.Inclusion, options.Terms.Dictionary)
			args = append(args, options.Terms.Terms...)
		}
		if options.Dialect > 0 {
			args = append(args, "DIALECT", options.Dialect)
		} else {
			args = append(args, "DIALECT", 2)
		}
	}
	cmd := newFTSpellCheckCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

type FTSpellCheckCmd struct {
	baseCmd
	val []SpellCheckResult
}

func newFTSpellCheckCmd(ctx context.Context, args ...interface{}) *FTSpellCheckCmd {
	return &FTSpellCheckCmd{
		baseCmd: baseCmd{
			ctx:     ctx,
			args:    args,
			cmdType: CmdTypeFTSpellCheck,
		},
	}
}

func (cmd *FTSpellCheckCmd) String() string {
	cmd.await()
	return cmdString(cmd, cmd.val)
}

func (cmd *FTSpellCheckCmd) SetVal(val []SpellCheckResult) {
	cmd.val = val
}

func (cmd *FTSpellCheckCmd) Result() ([]SpellCheckResult, error) {
	cmd.await()
	return cmd.val, cmd.err
}

func (cmd *FTSpellCheckCmd) Val() []SpellCheckResult {
	cmd.await()
	return cmd.val
}

func (cmd *FTSpellCheckCmd) RawVal() interface{} {
	cmd.await()
	return cmd.rawVal
}

func (cmd *FTSpellCheckCmd) RawResult() (interface{}, error) {
	cmd.await()
	return cmd.rawVal, cmd.err
}

func (cmd *FTSpellCheckCmd) readReply(rd *proto.Reader) (err error) {
	readType, err := rd.PeekReplyType()
	if err != nil {
		return err
	}

	// RESP3 returns a map, RESP2 returns an array
	if readType == proto.RespMap {
		// Read raw response first for backwards compatibility
		cmd.rawVal, err = rd.ReadReply()
		if err != nil {
			return err
		}

		// Parse the raw response into structured result
		rawMap, ok := cmd.rawVal.(map[interface{}]interface{})
		if !ok {
			return fmt.Errorf("unexpected RESP3 response type: %T", cmd.rawVal)
		}

		cmd.val, err = parseFTSpellCheckRESP3(rawMap)
		return err
	}

	// RESP2 format
	data, err := rd.ReadSlice()
	if err != nil {
		return err
	}
	cmd.val, err = parseFTSpellCheck(data)
	return err
}

// parseFTSpellCheckRESP3 parses the RESP3 format response from FT.SPELLCHECK.
// RESP3 format:
//
//	map{
//	  "results": map{
//	    "misspelled_term": [
//	      map{"suggestion": score},
//	      ...
//	    ],
//	    ...
//	  }
//	}
func parseFTSpellCheckRESP3(data map[interface{}]interface{}) ([]SpellCheckResult, error) {
	results := make([]SpellCheckResult, 0)

	resultsData, ok := data["results"]
	if !ok {
		return results, nil
	}

	resultsMap, ok := resultsData.(map[interface{}]interface{})
	if !ok {
		return nil, fmt.Errorf("invalid results format: expected map, got %T", resultsData)
	}

	for termKey, suggestionsData := range resultsMap {
		term, ok := termKey.(string)
		if !ok {
			continue
		}

		suggestionsArray, ok := suggestionsData.([]interface{})
		if !ok {
			continue
		}

		suggestions := make([]SpellCheckSuggestion, 0, len(suggestionsArray))
		for _, suggestionData := range suggestionsArray {
			suggestionMap, ok := suggestionData.(map[interface{}]interface{})
			if !ok {
				continue
			}

			for suggKey, scoreVal := range suggestionMap {
				suggestion, ok := suggKey.(string)
				if !ok {
					continue
				}

				var score float64
				switch v := scoreVal.(type) {
				case float64:
					score = v
				case int64:
					score = float64(v)
				case string:
					var err error
					score, err = strconv.ParseFloat(v, 64)
					if err != nil {
						continue
					}
				default:
					continue
				}

				suggestions = append(suggestions, SpellCheckSuggestion{
					Score:      score,
					Suggestion: suggestion,
				})
			}
		}

		results = append(results, SpellCheckResult{
			Term:        term,
			Suggestions: suggestions,
		})
	}

	return results, nil
}

func parseFTSpellCheck(data []interface{}) ([]SpellCheckResult, error) {
	results := make([]SpellCheckResult, 0, len(data))

	for _, termData := range data {
		termInfo, ok := termData.([]interface{})
		if !ok || len(termInfo) != 3 {
			return nil, fmt.Errorf("invalid term format")
		}

		term, ok := termInfo[1].(string)
		if !ok {
			return nil, fmt.Errorf("invalid term format")
		}

		suggestionsData, ok := termInfo[2].([]interface{})
		if !ok {
			return nil, fmt.Errorf("invalid suggestions format")
		}

		suggestions := make([]SpellCheckSuggestion, 0, len(suggestionsData))
		for _, suggestionData := range suggestionsData {
			suggestionInfo, ok := suggestionData.([]interface{})
			if !ok || len(suggestionInfo) != 2 {
				return nil, fmt.Errorf("invalid suggestion format")
			}

			scoreStr, ok := suggestionInfo[0].(string)
			if !ok {
				return nil, fmt.Errorf("invalid suggestion score format")
			}
			score, err := strconv.ParseFloat(scoreStr, 64)
			if err != nil {
				return nil, fmt.Errorf("invalid suggestion score value")
			}

			suggestion, ok := suggestionInfo[1].(string)
			if !ok {
				return nil, fmt.Errorf("invalid suggestion format")
			}

			suggestions = append(suggestions, SpellCheckSuggestion{
				Score:      score,
				Suggestion: suggestion,
			})
		}

		results = append(results, SpellCheckResult{
			Term:        term,
			Suggestions: suggestions,
		})
	}

	return results, nil
}

func (cmd *FTSpellCheckCmd) Clone() Cmder {
	var val []SpellCheckResult
	if cmd.val != nil {
		val = make([]SpellCheckResult, len(cmd.val))
		for i, result := range cmd.val {
			val[i] = SpellCheckResult{
				Term: result.Term,
			}
			if result.Suggestions != nil {
				val[i].Suggestions = slices.Clone(result.Suggestions)
			}
		}
	}
	return &FTSpellCheckCmd{
		baseCmd: cmd.cloneBaseCmd(),
		val:     val,
	}
}

func parseFTSearch(data []interface{}, noContent, withScores, withPayloads, withSortKeys bool) (FTSearchResult, error) {
	if len(data) < 1 {
		return FTSearchResult{}, fmt.Errorf("unexpected search result format")
	}

	total, ok := data[0].(int64)
	if !ok {
		return FTSearchResult{}, fmt.Errorf("invalid total results format")
	}

	var results []Document
	for i := 1; i < len(data); {
		docID, ok := data[i].(string)
		if !ok {
			return FTSearchResult{}, fmt.Errorf("invalid document ID format")
		}

		doc := Document{
			ID:     docID,
			Fields: make(map[string]string),
		}
		i++

		if noContent {
			results = append(results, doc)
			continue
		}

		if withScores && i < len(data) {
			if scoreStr, ok := data[i].(string); ok {
				score, err := strconv.ParseFloat(scoreStr, 64)
				if err != nil {
					return FTSearchResult{}, fmt.Errorf("invalid score format")
				}
				doc.Score = &score
				i++
			}
		}

		if withPayloads && i < len(data) {
			if payload, ok := data[i].(string); ok {
				doc.Payload = &payload
				i++
			}
		}

		if withSortKeys && i < len(data) {
			if sortKey, ok := data[i].(string); ok {
				doc.SortKey = &sortKey
				i++
			}
		}

		if i < len(data) {
			fields, ok := data[i].([]interface{})
			if !ok {
				if data[i] == proto.Nil || data[i] == nil {
					doc.Error = proto.Nil
					doc.Fields = map[string]string{}
					fields = []interface{}{}
				} else {
					return FTSearchResult{}, fmt.Errorf("invalid document fields format")
				}
			}

			for j := 0; j < len(fields); j += 2 {
				key, ok := fields[j].(string)
				if !ok {
					return FTSearchResult{}, fmt.Errorf("invalid field key format")
				}
				value, ok := fields[j+1].(string)
				if !ok {
					return FTSearchResult{}, fmt.Errorf("invalid field value format")
				}
				doc.Fields[key] = value
			}
			i++
		}

		results = append(results, doc)
	}
	return FTSearchResult{
		Total: int(total),
		Docs:  results,
	}, nil
}

type FTSearchCmd struct {
	baseCmd
	val     FTSearchResult
	options *FTSearchOptions
}

func newFTSearchCmd(ctx context.Context, options *FTSearchOptions, args ...interface{}) *FTSearchCmd {
	return &FTSearchCmd{
		baseCmd: baseCmd{
			ctx:     ctx,
			args:    args,
			cmdType: CmdTypeFTSearch,
		},
		options: options,
	}
}

func (cmd *FTSearchCmd) String() string {
	cmd.await()
	return cmdString(cmd, cmd.val)
}

func (cmd *FTSearchCmd) SetVal(val FTSearchResult) {
	cmd.val = val
}

func (cmd *FTSearchCmd) Result() (FTSearchResult, error) {
	cmd.await()
	return cmd.val, cmd.err
}

func (cmd *FTSearchCmd) Val() FTSearchResult {
	cmd.await()
	return cmd.val
}

func (cmd *FTSearchCmd) RawVal() interface{} {
	cmd.await()
	return cmd.rawVal
}

func (cmd *FTSearchCmd) RawResult() (interface{}, error) {
	cmd.await()
	return cmd.rawVal, cmd.err
}

func (cmd *FTSearchCmd) readReply(rd *proto.Reader) (err error) {
	readType, err := rd.PeekReplyType()
	if err != nil {
		return err
	}

	// RESP3 returns a map, RESP2 returns an array
	if readType == proto.RespMap {
		// Read raw response first for backwards compatibility
		cmd.rawVal, err = rd.ReadReply()
		if err != nil {
			return err
		}
		// Parse the raw response into structured result
		if mapVal, ok := cmd.rawVal.(map[interface{}]interface{}); ok {
			cmd.val, err = parseFTSearchMapRESP3(mapVal)
		} else {
			return fmt.Errorf("unexpected RESP3 response type: %T", cmd.rawVal)
		}
		return err
	}

	// RESP2 format or error response - use ReadReply to handle errors properly
	data, err := rd.ReadReply()
	if err != nil {
		return err
	}
	if dataSlice, ok := data.([]interface{}); ok {
		cmd.val, err = parseFTSearch(dataSlice, cmd.options.NoContent, cmd.options.WithScores, cmd.options.WithPayloads, cmd.options.WithSortKeys)
		return err
	}
	return fmt.Errorf("unexpected response type: %T", data)
}

// parseFTSearchMapRESP3 parses the RESP3 format response from FT.SEARCH.
// It takes a map[interface{}]interface{} which is the raw response from ReadReply().
// RESP3 format:
//
//	%5
//	  $10 attributes => *0
//	  $13 total_results => :N
//	  $6 format => $6 STRING
//	  $7 results => *N (array of maps with id, score, extra_attributes, values)
//	  $7 warning => *N (array of strings)
func parseFTSearchMapRESP3(data map[interface{}]interface{}) (FTSearchResult, error) {
	var result FTSearchResult
	result.Docs = make([]Document, 0)

	for k, v := range data {
		key, ok := k.(string)
		if !ok {
			continue
		}

		switch key {
		case "total_results":
			result.Total = internal.ToInteger(v)
		case "results":
			if resultsData, ok := v.([]interface{}); ok {
				docs, err := parseFTSearchResultsMapRESP3(resultsData)
				if err != nil {
					return FTSearchResult{}, err
				}
				result.Docs = docs
			}
		case "warning":
			if warningsData, ok := v.([]interface{}); ok {
				result.Warnings = make([]string, 0, len(warningsData))
				for _, w := range warningsData {
					if ws, ok := w.(string); ok {
						result.Warnings = append(result.Warnings, ws)
					}
				}
			}
			// Ignore "attributes", "format", and other fields as per the spec
		}
	}

	return result, nil
}

// parseFTSearchResultsMapRESP3 parses the results array from RESP3 FT.SEARCH response.
func parseFTSearchResultsMapRESP3(resultsData []interface{}) ([]Document, error) {
	docs := make([]Document, 0, len(resultsData))
	for _, item := range resultsData {
		if itemMap, ok := item.(map[interface{}]interface{}); ok {
			doc, err := parseFTSearchDocumentMapRESP3(itemMap)
			if err != nil {
				return nil, err
			}
			docs = append(docs, doc)
		}
	}
	return docs, nil
}

// parseFTSearchDocumentMapRESP3 parses a single document from RESP3 FT.SEARCH response.
func parseFTSearchDocumentMapRESP3(itemMap map[interface{}]interface{}) (Document, error) {
	doc := Document{
		Fields: make(map[string]string),
	}

	for k, v := range itemMap {
		key, ok := k.(string)
		if !ok {
			continue
		}

		switch key {
		case "id":
			if id, ok := v.(string); ok {
				doc.ID = id
			}
		case "score":
			if score, ok := v.(float64); ok {
				doc.Score = &score
			}
		case "payload":
			if payload, ok := v.(string); ok {
				doc.Payload = &payload
			}
		case "sortkey":
			if sortKey, ok := v.(string); ok {
				doc.SortKey = &sortKey
			}
		case "extra_attributes":
			if extraAttrs, ok := v.(map[interface{}]interface{}); ok {
				for ek, ev := range extraAttrs {
					if ekStr, ok := ek.(string); ok {
						if evStr, ok := ev.(string); ok {
							doc.Fields[ekStr] = evStr
						}
					}
				}
			}
			// Ignore "values" and other fields as per the spec
		}
	}

	return doc, nil
}

func (cmd *FTSearchCmd) Clone() Cmder {
	val := FTSearchResult{
		Total: cmd.val.Total,
	}
	if cmd.val.Docs != nil {
		val.Docs = make([]Document, len(cmd.val.Docs))
		for i, doc := range cmd.val.Docs {
			val.Docs[i] = Document{
				ID:      doc.ID,
				Score:   doc.Score,
				Payload: doc.Payload,
				SortKey: doc.SortKey,
			}
			if doc.Fields != nil {
				val.Docs[i].Fields = make(map[string]string, len(doc.Fields))
				for k, v := range doc.Fields {
					val.Docs[i].Fields[k] = v
				}
			}
		}
	}
	if cmd.val.Warnings != nil {
		val.Warnings = make([]string, len(cmd.val.Warnings))
		copy(val.Warnings, cmd.val.Warnings)
	}
	var options *FTSearchOptions
	if cmd.options != nil {
		options = &FTSearchOptions{
			NoContent:       cmd.options.NoContent,
			Verbatim:        cmd.options.Verbatim,
			NoStopWords:     cmd.options.NoStopWords,
			WithScores:      cmd.options.WithScores,
			WithPayloads:    cmd.options.WithPayloads,
			WithSortKeys:    cmd.options.WithSortKeys,
			Slop:            cmd.options.Slop,
			Timeout:         cmd.options.Timeout,
			InOrder:         cmd.options.InOrder,
			Language:        cmd.options.Language,
			Expander:        cmd.options.Expander,
			Scorer:          cmd.options.Scorer,
			ExplainScore:    cmd.options.ExplainScore,
			Payload:         cmd.options.Payload,
			SortByWithCount: cmd.options.SortByWithCount,
			LimitOffset:     cmd.options.LimitOffset,
			Limit:           cmd.options.Limit,
			CountOnly:       cmd.options.CountOnly,
			DialectVersion:  cmd.options.DialectVersion,
		}
		// Clone slices and maps
		if cmd.options.Filters != nil {
			options.Filters = slices.Clone(cmd.options.Filters)
		}
		if cmd.options.GeoFilter != nil {
			options.GeoFilter = slices.Clone(cmd.options.GeoFilter)
		}
		if cmd.options.InKeys != nil {
			options.InKeys = slices.Clone(cmd.options.InKeys)
		}
		if cmd.options.InFields != nil {
			options.InFields = slices.Clone(cmd.options.InFields)
		}
		if cmd.options.Return != nil {
			options.Return = slices.Clone(cmd.options.Return)
		}
		if cmd.options.SortBy != nil {
			options.SortBy = slices.Clone(cmd.options.SortBy)
		}
		if cmd.options.Params != nil {
			options.Params = maps.Clone(cmd.options.Params)
		}
	}
	return &FTSearchCmd{
		baseCmd: cmd.cloneBaseCmd(),
		val:     val,
		options: options,
	}
}

// FTHybridResult represents the result of a hybrid search operation
type FTHybridResult struct {
	TotalResults int
	Results      []map[string]interface{}
	// Warnings holds server warnings for a partial result (search-on-timeout
	// return/return-strict), on RESP2 and RESP3; the fail policy returns an error.
	Warnings      []string
	ExecutionTime float64
}

// FTHybridCursorResult represents cursor result for hybrid search
type FTHybridCursorResult struct {
	SearchCursorID int
	VsimCursorID   int
}

type FTHybridCmd struct {
	baseCmd
	val        FTHybridResult
	cursorVal  *FTHybridCursorResult
	options    *FTHybridOptions
	withCursor bool
}

func newFTHybridCmd(ctx context.Context, options *FTHybridOptions, args ...interface{}) *FTHybridCmd {
	var withCursor bool
	if options != nil && options.WithCursor {
		withCursor = true
	}
	return &FTHybridCmd{
		baseCmd: baseCmd{
			ctx:  ctx,
			args: args,
		},
		options:    options,
		withCursor: withCursor,
	}
}

func (cmd *FTHybridCmd) String() string {
	cmd.await()
	return cmdString(cmd, cmd.val)
}

func (cmd *FTHybridCmd) SetVal(val FTHybridResult) {
	cmd.val = val
}

func (cmd *FTHybridCmd) Result() (FTHybridResult, error) {
	cmd.await()
	return cmd.val, cmd.err
}

func (cmd *FTHybridCmd) CursorResult() (*FTHybridCursorResult, error) {
	cmd.await()
	return cmd.cursorVal, cmd.err
}

func (cmd *FTHybridCmd) Val() FTHybridResult {
	cmd.await()
	return cmd.val
}

func (cmd *FTHybridCmd) CursorVal() *FTHybridCursorResult {
	cmd.await()
	return cmd.cursorVal
}

func (cmd *FTHybridCmd) RawVal() interface{} {
	cmd.await()
	return cmd.rawVal
}

func (cmd *FTHybridCmd) RawResult() (interface{}, error) {
	cmd.await()
	return cmd.rawVal, cmd.err
}

func parseFTHybrid(data []interface{}, withCursor bool) (FTHybridResult, *FTHybridCursorResult, error) {
	// Convert to map
	resultMap := make(map[string]interface{})
	for i := 0; i < len(data); i += 2 {
		if i+1 < len(data) {
			key, ok := data[i].(string)
			if !ok {
				return FTHybridResult{}, nil, fmt.Errorf("invalid key type at index %d", i)
			}
			resultMap[key] = data[i+1]
		}
	}

	// Handle cursor result
	if withCursor {
		searchCursorID, ok1 := resultMap["SEARCH"].(int64)
		vsimCursorID, ok2 := resultMap["VSIM"].(int64)
		if !ok1 || !ok2 {
			return FTHybridResult{}, nil, fmt.Errorf("invalid cursor result format")
		}
		return FTHybridResult{}, &FTHybridCursorResult{
			SearchCursorID: int(searchCursorID),
			VsimCursorID:   int(vsimCursorID),
		}, nil
	}

	// Parse regular result
	totalResults, ok := resultMap["total_results"].(int64)
	if !ok {
		return FTHybridResult{}, nil, fmt.Errorf("invalid total_results format")
	}

	resultsData, ok := resultMap["results"].([]interface{})
	if !ok {
		return FTHybridResult{}, nil, fmt.Errorf("invalid results format")
	}

	// Parse each result item
	results := make([]map[string]interface{}, 0, len(resultsData))
	for _, item := range resultsData {
		// Try parsing as map[string]interface{} first (RESP3 format)
		if itemMap, ok := item.(map[string]interface{}); ok {
			results = append(results, itemMap)
			continue
		}

		// Try parsing as map[interface{}]interface{} (alternative RESP3 format)
		if rawMap, ok := item.(map[interface{}]interface{}); ok {
			itemMap := make(map[string]interface{})
			for k, v := range rawMap {
				if keyStr, ok := k.(string); ok {
					itemMap[keyStr] = v
				}
			}
			results = append(results, itemMap)
			continue
		}

		// Fall back to array format (RESP2 format - key-value pairs)
		itemData, ok := item.([]interface{})
		if !ok {
			return FTHybridResult{}, nil, fmt.Errorf("invalid result item format")
		}

		itemMap := make(map[string]interface{})
		for i := 0; i < len(itemData); i += 2 {
			if i+1 < len(itemData) {
				key, ok := itemData[i].(string)
				if !ok {
					return FTHybridResult{}, nil, fmt.Errorf("invalid item key format")
				}
				itemMap[key] = itemData[i+1]
			}
		}
		results = append(results, itemMap)
	}

	// Optional warnings; accept both "warning" (as FT.SEARCH/FT.AGGREGATE) and "warnings".
	var warnings []string
	warningsData, ok := resultMap["warning"].([]interface{})
	if !ok {
		warningsData, ok = resultMap["warnings"].([]interface{})
	}
	if ok {
		warnings = make([]string, 0, len(warningsData))
		for _, w := range warningsData {
			if ws, ok := w.(string); ok {
				warnings = append(warnings, ws)
			}
		}
	}

	// Parse execution time (optional field)
	var executionTime float64
	if execTimeVal, exists := resultMap["execution_time"]; exists {
		switch v := execTimeVal.(type) {
		case string:
			var err error
			executionTime, err = strconv.ParseFloat(v, 64)
			if err != nil {
				return FTHybridResult{}, nil, fmt.Errorf("invalid execution_time format: %v", err)
			}
		case float64:
			executionTime = v
		case int64:
			executionTime = float64(v)
		}
	}

	return FTHybridResult{
		TotalResults:  int(totalResults),
		Results:       results,
		Warnings:      warnings,
		ExecutionTime: executionTime,
	}, nil, nil
}

func (cmd *FTHybridCmd) readReply(rd *proto.Reader) (err error) {
	data, err := rd.ReadSlice()
	if err != nil {
		return err
	}

	result, cursorResult, err := parseFTHybrid(data, cmd.withCursor)
	if err != nil {
		return err
	}

	if cmd.withCursor {
		cmd.cursorVal = cursorResult
	} else {
		cmd.val = result
	}
	return nil
}

func (cmd *FTHybridCmd) Clone() Cmder {
	val := FTHybridResult{
		TotalResults:  cmd.val.TotalResults,
		ExecutionTime: cmd.val.ExecutionTime,
	}
	if cmd.val.Results != nil {
		val.Results = make([]map[string]interface{}, len(cmd.val.Results))
		for i, result := range cmd.val.Results {
			val.Results[i] = make(map[string]interface{}, len(result))
			for k, v := range result {
				val.Results[i][k] = v
			}
		}
	}
	if cmd.val.Warnings != nil {
		val.Warnings = slices.Clone(cmd.val.Warnings)
	}

	var cursorVal *FTHybridCursorResult
	if cmd.cursorVal != nil {
		cursorVal = &FTHybridCursorResult{
			SearchCursorID: cmd.cursorVal.SearchCursorID,
			VsimCursorID:   cmd.cursorVal.VsimCursorID,
		}
	}

	var options *FTHybridOptions
	if cmd.options != nil {
		options = &FTHybridOptions{
			CountExpressions: cmd.options.CountExpressions,
			Load:             cmd.options.Load,
			Filter:           cmd.options.Filter,
			LimitOffset:      cmd.options.LimitOffset,
			Limit:            cmd.options.Limit,
			ExplainScore:     cmd.options.ExplainScore,
			Timeout:          cmd.options.Timeout,
			WithCursor:       cmd.options.WithCursor,
		}
		// Clone slices and maps
		if cmd.options.SearchExpressions != nil {
			options.SearchExpressions = make([]FTHybridSearchExpression, len(cmd.options.SearchExpressions))
			copy(options.SearchExpressions, cmd.options.SearchExpressions)
		}
		if cmd.options.VectorExpressions != nil {
			options.VectorExpressions = make([]FTHybridVectorExpression, len(cmd.options.VectorExpressions))
			copy(options.VectorExpressions, cmd.options.VectorExpressions)
		}
		if cmd.options.Combine != nil {
			options.Combine = &FTHybridCombineOptions{
				Method:       cmd.options.Combine.Method,
				Count:        cmd.options.Combine.Count,
				Window:       cmd.options.Combine.Window,
				Constant:     cmd.options.Combine.Constant,
				Alpha:        cmd.options.Combine.Alpha,
				Beta:         cmd.options.Combine.Beta,
				YieldScoreAs: cmd.options.Combine.YieldScoreAs,
			}
		}
		if cmd.options.GroupBy != nil {
			options.GroupBy = &FTHybridGroupBy{
				Count:       cmd.options.GroupBy.Count,
				ReduceFunc:  cmd.options.GroupBy.ReduceFunc,
				ReduceCount: cmd.options.GroupBy.ReduceCount,
			}
			if cmd.options.GroupBy.Fields != nil {
				options.GroupBy.Fields = make([]string, len(cmd.options.GroupBy.Fields))
				copy(options.GroupBy.Fields, cmd.options.GroupBy.Fields)
			}
			if cmd.options.GroupBy.ReduceParams != nil {
				options.GroupBy.ReduceParams = make([]interface{}, len(cmd.options.GroupBy.ReduceParams))
				copy(options.GroupBy.ReduceParams, cmd.options.GroupBy.ReduceParams)
			}
		}
		if cmd.options.Apply != nil {
			options.Apply = make([]FTHybridApply, len(cmd.options.Apply))
			copy(options.Apply, cmd.options.Apply)
		}
		if cmd.options.SortBy != nil {
			options.SortBy = make([]FTSearchSortBy, len(cmd.options.SortBy))
			copy(options.SortBy, cmd.options.SortBy)
		}
		if cmd.options.Params != nil {
			options.Params = make(map[string]interface{}, len(cmd.options.Params))
			for k, v := range cmd.options.Params {
				options.Params[k] = v
			}
		}
		if cmd.options.WithCursorOptions != nil {
			options.WithCursorOptions = &FTHybridWithCursor{
				MaxIdle: cmd.options.WithCursorOptions.MaxIdle,
				Count:   cmd.options.WithCursorOptions.Count,
			}
		}
	}

	return &FTHybridCmd{
		baseCmd:    cmd.cloneBaseCmd(),
		val:        val,
		cursorVal:  cursorVal,
		options:    options,
		withCursor: cmd.withCursor,
	}
}

// FTSearch - Executes a search query on an index.
// The 'index' parameter specifies the index to search, and the 'query' parameter specifies the search query.
// For more information, please refer to the Redis documentation about [FT.SEARCH].
//
// [FT.SEARCH]: (https://redis.io/commands/ft.search/)
func (c cmdable) FTSearch(ctx context.Context, index string, query string) *FTSearchCmd {
	args := []interface{}{"FT.SEARCH", index, query}
	cmd := newFTSearchCmd(ctx, &FTSearchOptions{}, args...)
	_ = c(ctx, cmd)
	return cmd
}

type SearchQuery []interface{}

// FTSearchQuery - Executes a search query on an index with additional options.
// The 'index' parameter specifies the index to search, the 'query' parameter specifies the search query,
// and the 'options' parameter specifies additional options for the search.
// For more information, please refer to the Redis documentation about [FT.SEARCH].
//
// [FT.SEARCH]: (https://redis.io/commands/ft.search/)
func FTSearchQuery(query string, options *FTSearchOptions) (SearchQuery, error) {
	queryArgs := []interface{}{query}
	if options != nil {
		if options.NoContent {
			queryArgs = append(queryArgs, "NOCONTENT")
		}
		if options.Verbatim {
			queryArgs = append(queryArgs, "VERBATIM")
		}
		if options.NoStopWords {
			queryArgs = append(queryArgs, "NOSTOPWORDS")
		}
		if options.WithScores {
			queryArgs = append(queryArgs, "WITHSCORES")
		}
		if options.WithPayloads {
			queryArgs = append(queryArgs, "WITHPAYLOADS")
		}
		if options.WithSortKeys {
			queryArgs = append(queryArgs, "WITHSORTKEYS")
		}
		if options.Filters != nil {
			for _, filter := range options.Filters {
				queryArgs = append(queryArgs, "FILTER", filter.FieldName, filter.Min, filter.Max)
			}
		}
		if options.GeoFilter != nil {
			for _, geoFilter := range options.GeoFilter {
				queryArgs = append(queryArgs, "GEOFILTER", geoFilter.FieldName, geoFilter.Longitude, geoFilter.Latitude, geoFilter.Radius, geoFilter.Unit)
			}
		}
		if options.InKeys != nil {
			queryArgs = append(queryArgs, "INKEYS", len(options.InKeys))
			queryArgs = append(queryArgs, options.InKeys...)
		}
		if options.InFields != nil {
			queryArgs = append(queryArgs, "INFIELDS", len(options.InFields))
			queryArgs = append(queryArgs, options.InFields...)
		}
		if options.Return != nil {
			queryArgs = append(queryArgs, "RETURN")
			queryArgsReturn := []interface{}{}
			for _, ret := range options.Return {
				queryArgsReturn = append(queryArgsReturn, ret.FieldName)
				if ret.As != "" {
					queryArgsReturn = append(queryArgsReturn, "AS", ret.As)
				}
			}
			queryArgs = append(queryArgs, len(queryArgsReturn))
			queryArgs = append(queryArgs, queryArgsReturn...)
		}
		if options.Slop > 0 {
			queryArgs = append(queryArgs, "SLOP", options.Slop)
		}
		if options.Timeout > 0 {
			queryArgs = append(queryArgs, "TIMEOUT", options.Timeout)
		}
		if options.InOrder {
			queryArgs = append(queryArgs, "INORDER")
		}
		if options.Language != "" {
			queryArgs = append(queryArgs, "LANGUAGE", options.Language)
		}
		if options.Expander != "" {
			queryArgs = append(queryArgs, "EXPANDER", options.Expander)
		}
		if options.Scorer != "" {
			queryArgs = append(queryArgs, "SCORER", options.Scorer)
		}
		if options.ExplainScore {
			queryArgs = append(queryArgs, "EXPLAINSCORE")
		}
		if options.Payload != "" {
			queryArgs = append(queryArgs, "PAYLOAD", options.Payload)
		}
		if options.SortBy != nil {
			queryArgs = append(queryArgs, "SORTBY")
			for _, sortBy := range options.SortBy {
				queryArgs = append(queryArgs, sortBy.FieldName)
				if sortBy.Asc && sortBy.Desc {
					return nil, fmt.Errorf("FT.SEARCH: ASC and DESC are mutually exclusive")
				}
				if sortBy.Asc {
					queryArgs = append(queryArgs, "ASC")
				}
				if sortBy.Desc {
					queryArgs = append(queryArgs, "DESC")
				}
			}
			if options.SortByWithCount {
				queryArgs = append(queryArgs, "WITHCOUNT")
			}
		}
		if options.LimitOffset >= 0 && options.Limit > 0 {
			queryArgs = append(queryArgs, "LIMIT", options.LimitOffset, options.Limit)
		}
		if options.Params != nil {
			queryArgs = append(queryArgs, "PARAMS", len(options.Params)*2)
			for key, value := range options.Params {
				queryArgs = append(queryArgs, key, value)
			}
		}
		if options.DialectVersion > 0 {
			queryArgs = append(queryArgs, "DIALECT", options.DialectVersion)
		} else {
			queryArgs = append(queryArgs, "DIALECT", 2)
		}
	}
	return queryArgs, nil
}

// FTSearchWithArgs - Executes a search query on an index with additional options.
// The 'index' parameter specifies the index to search, the 'query' parameter specifies the search query,
// and the 'options' parameter specifies additional options for the search.
// For more information, please refer to the Redis documentation about [FT.SEARCH].
//
// [FT.SEARCH]: (https://redis.io/commands/ft.search/)
func (c cmdable) FTSearchWithArgs(ctx context.Context, index string, query string, options *FTSearchOptions) *FTSearchCmd {
	args := []interface{}{"FT.SEARCH", index, query}
	if options != nil {
		if options.NoContent {
			args = append(args, "NOCONTENT")
		}
		if options.Verbatim {
			args = append(args, "VERBATIM")
		}
		if options.NoStopWords {
			args = append(args, "NOSTOPWORDS")
		}
		if options.WithScores {
			args = append(args, "WITHSCORES")
		}
		if options.WithPayloads {
			args = append(args, "WITHPAYLOADS")
		}
		if options.WithSortKeys {
			args = append(args, "WITHSORTKEYS")
		}
		if options.Filters != nil {
			for _, filter := range options.Filters {
				args = append(args, "FILTER", filter.FieldName, filter.Min, filter.Max)
			}
		}
		if options.GeoFilter != nil {
			for _, geoFilter := range options.GeoFilter {
				args = append(args, "GEOFILTER", geoFilter.FieldName, geoFilter.Longitude, geoFilter.Latitude, geoFilter.Radius, geoFilter.Unit)
			}
		}
		if options.InKeys != nil {
			args = append(args, "INKEYS", len(options.InKeys))
			args = append(args, options.InKeys...)
		}
		if options.InFields != nil {
			args = append(args, "INFIELDS", len(options.InFields))
			args = append(args, options.InFields...)
		}
		if options.Return != nil {
			args = append(args, "RETURN")
			argsReturn := []interface{}{}
			for _, ret := range options.Return {
				argsReturn = append(argsReturn, ret.FieldName)
				if ret.As != "" {
					argsReturn = append(argsReturn, "AS", ret.As)
				}
			}
			args = append(args, len(argsReturn))
			args = append(args, argsReturn...)
		}
		if options.Slop > 0 {
			args = append(args, "SLOP", options.Slop)
		}
		if options.Timeout > 0 {
			args = append(args, "TIMEOUT", options.Timeout)
		}
		if options.InOrder {
			args = append(args, "INORDER")
		}
		if options.Language != "" {
			args = append(args, "LANGUAGE", options.Language)
		}
		if options.Expander != "" {
			args = append(args, "EXPANDER", options.Expander)
		}
		if options.Scorer != "" {
			args = append(args, "SCORER", options.Scorer)
		}
		if options.ExplainScore {
			args = append(args, "EXPLAINSCORE")
		}
		if options.Payload != "" {
			args = append(args, "PAYLOAD", options.Payload)
		}
		if options.SortBy != nil {
			args = append(args, "SORTBY")
			for _, sortBy := range options.SortBy {
				args = append(args, sortBy.FieldName)
				if sortBy.Asc && sortBy.Desc {
					cmd := newFTSearchCmd(ctx, options, args...)
					cmd.SetErr(fmt.Errorf("FT.SEARCH: ASC and DESC are mutually exclusive"))
					return cmd
				}
				if sortBy.Asc {
					args = append(args, "ASC")
				}
				if sortBy.Desc {
					args = append(args, "DESC")
				}
			}
			if options.SortByWithCount {
				args = append(args, "WITHCOUNT")
			}
		}
		if options.CountOnly {
			args = append(args, "LIMIT", 0, 0)
		} else {
			if options.LimitOffset >= 0 && options.Limit > 0 || options.LimitOffset > 0 && options.Limit == 0 {
				args = append(args, "LIMIT", options.LimitOffset, options.Limit)
			}
		}
		if options.Params != nil {
			args = append(args, "PARAMS", len(options.Params)*2)
			for key, value := range options.Params {
				args = append(args, key, value)
			}
		}
		if options.DialectVersion > 0 {
			args = append(args, "DIALECT", options.DialectVersion)
		} else {
			args = append(args, "DIALECT", 2)
		}
	}
	cmd := newFTSearchCmd(ctx, options, args...)
	_ = c(ctx, cmd)
	return cmd
}

func NewFTSynDumpCmd(ctx context.Context, args ...interface{}) *FTSynDumpCmd {
	return &FTSynDumpCmd{
		baseCmd: baseCmd{
			ctx:     ctx,
			args:    args,
			cmdType: CmdTypeFTSynDump,
		},
	}
}

func (cmd *FTSynDumpCmd) String() string {
	cmd.await()
	return cmdString(cmd, cmd.val)
}

func (cmd *FTSynDumpCmd) SetVal(val []FTSynDumpResult) {
	cmd.val = val
}

func (cmd *FTSynDumpCmd) Val() []FTSynDumpResult {
	cmd.await()
	return cmd.val
}

func (cmd *FTSynDumpCmd) Result() ([]FTSynDumpResult, error) {
	cmd.await()
	return cmd.val, cmd.err
}

func (cmd *FTSynDumpCmd) RawVal() interface{} {
	cmd.await()
	return cmd.rawVal
}

func (cmd *FTSynDumpCmd) RawResult() (interface{}, error) {
	cmd.await()
	return cmd.rawVal, cmd.err
}

func (cmd *FTSynDumpCmd) readReply(rd *proto.Reader) error {
	readType, err := rd.PeekReplyType()
	if err != nil {
		return err
	}

	// RESP3 returns a map, RESP2 returns an array
	if readType == proto.RespMap {
		// Read raw response first for backwards compatibility
		cmd.rawVal, err = rd.ReadReply()
		if err != nil {
			return err
		}

		// Parse the raw response into structured result
		rawMap, ok := cmd.rawVal.(map[interface{}]interface{})
		if !ok {
			return fmt.Errorf("unexpected RESP3 response type: %T", cmd.rawVal)
		}

		cmd.val, err = parseFTSynDumpRESP3(rawMap)
		return err
	}

	// RESP2 format
	termSynonymPairs, err := rd.ReadSlice()
	if err != nil {
		return err
	}

	var results []FTSynDumpResult
	for i := 0; i < len(termSynonymPairs); i += 2 {
		term, ok := termSynonymPairs[i].(string)
		if !ok {
			return fmt.Errorf("invalid term format")
		}

		synonyms, ok := termSynonymPairs[i+1].([]interface{})
		if !ok {
			return fmt.Errorf("invalid synonyms format")
		}

		synonymList := make([]string, len(synonyms))
		for j, syn := range synonyms {
			synonym, ok := syn.(string)
			if !ok {
				return fmt.Errorf("invalid synonym format")
			}
			synonymList[j] = synonym
		}

		results = append(results, FTSynDumpResult{
			Term:     term,
			Synonyms: synonymList,
		})
	}

	cmd.val = results
	return nil
}

// parseFTSynDumpRESP3 parses the RESP3 format response from FT.SYNDUMP.
// RESP3 format:
//
//	map{
//	  "term1": ["synonym_group_id1", ...],
//	  "term2": ["synonym_group_id2", ...],
//	  ...
//	}
func parseFTSynDumpRESP3(data map[interface{}]interface{}) ([]FTSynDumpResult, error) {
	results := make([]FTSynDumpResult, 0, len(data))

	for termKey, synonymsData := range data {
		term, ok := termKey.(string)
		if !ok {
			continue
		}

		synonymsArray, ok := synonymsData.([]interface{})
		if !ok {
			continue
		}

		synonymList := make([]string, 0, len(synonymsArray))
		for _, syn := range synonymsArray {
			if synonym, ok := syn.(string); ok {
				synonymList = append(synonymList, synonym)
			}
		}

		results = append(results, FTSynDumpResult{
			Term:     term,
			Synonyms: synonymList,
		})
	}

	return results, nil
}

func (cmd *FTSynDumpCmd) Clone() Cmder {
	var val []FTSynDumpResult
	if cmd.val != nil {
		val = make([]FTSynDumpResult, len(cmd.val))
		for i, result := range cmd.val {
			val[i] = FTSynDumpResult{
				Term: result.Term,
			}
			if result.Synonyms != nil {
				val[i].Synonyms = make([]string, len(result.Synonyms))
				copy(val[i].Synonyms, result.Synonyms)
			}
		}
	}
	return &FTSynDumpCmd{
		baseCmd: cmd.cloneBaseCmd(),
		val:     val,
	}
}

// FTSynDump - Dumps the contents of a synonym group.
// The 'index' parameter specifies the index to dump.
// For more information, please refer to the Redis documentation:
// [FT.SYNDUMP]: (https://redis.io/commands/ft.syndump/)
func (c cmdable) FTSynDump(ctx context.Context, index string) *FTSynDumpCmd {
	cmd := NewFTSynDumpCmd(ctx, "FT.SYNDUMP", index)
	_ = c(ctx, cmd)
	return cmd
}

// FTSynUpdate - Creates or updates a synonym group with additional terms.
// The 'index' parameter specifies the index to update, the 'synGroupId' parameter specifies the synonym group id, and the 'terms' parameter specifies the additional terms.
// For more information, please refer to the Redis documentation:
// [FT.SYNUPDATE]: (https://redis.io/commands/ft.synupdate/)
func (c cmdable) FTSynUpdate(ctx context.Context, index string, synGroupId interface{}, terms []interface{}) *StatusCmd {
	args := []interface{}{"FT.SYNUPDATE", index, synGroupId}
	args = append(args, terms...)
	cmd := NewStatusCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// FTSynUpdateWithArgs - Creates or updates a synonym group with additional terms and options.
// The 'index' parameter specifies the index to update, the 'synGroupId' parameter specifies the synonym group id, the 'options' parameter specifies additional options for the update, and the 'terms' parameter specifies the additional terms.
// For more information, please refer to the Redis documentation:
// [FT.SYNUPDATE]: (https://redis.io/commands/ft.synupdate/)
func (c cmdable) FTSynUpdateWithArgs(ctx context.Context, index string, synGroupId interface{}, options *FTSynUpdateOptions, terms []interface{}) *StatusCmd {
	args := []interface{}{"FT.SYNUPDATE", index, synGroupId}
	if options.SkipInitialScan {
		args = append(args, "SKIPINITIALSCAN")
	}
	args = append(args, terms...)
	cmd := NewStatusCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// FTTagVals - Returns all distinct values indexed in a tag field.
// The 'index' parameter specifies the index to check, and the 'field' parameter specifies the tag field to retrieve values from.
// For more information, please refer to the Redis documentation:
// [FT.TAGVALS]: (https://redis.io/commands/ft.tagvals/)
func (c cmdable) FTTagVals(ctx context.Context, index string, field string) *StringSliceCmd {
	cmd := NewStringSliceCmd(ctx, "FT.TAGVALS", index, field)
	_ = c(ctx, cmd)
	return cmd
}

// FTHybrid - Executes a hybrid search combining full-text search and vector similarity
// The 'index' parameter specifies the index to search, 'searchExpr' is the search query,
// 'vectorField' is the name of the vector field, and 'vectorData' is the vector to search with.
// FTHybrid is still experimental, the command behaviour and signature may change
func (c cmdable) FTHybrid(ctx context.Context, index string, searchExpr string, vectorField string, vectorData Vector) *FTHybridCmd {
	options := &FTHybridOptions{
		CountExpressions: 2,
		SearchExpressions: []FTHybridSearchExpression{
			{Query: searchExpr},
		},
		VectorExpressions: []FTHybridVectorExpression{
			{VectorField: vectorField, VectorData: vectorData},
		},
	}
	return c.FTHybridWithArgs(ctx, index, options)
}

func hybridVectorBlob(v Vector) (interface{}, error) {
	if v == nil {
		return nil, fmt.Errorf("FT.HYBRID: vector data is required")
	}

	switch vector := v.(type) {
	case *VectorFP32:
		return hybridVectorBytes(vector.Val)
	case *VectorFloat16:
		return hybridVectorBytes(vector.Val)
	case *VectorBFloat16:
		return hybridVectorBytes(vector.Val)
	case *VectorFloat64:
		return hybridVectorBytes(vector.Val)
	case *VectorInt8:
		return hybridVectorBytes(vector.Val)
	case *VectorUint8:
		return hybridVectorBytes(vector.Val)
	case *VectorValues, *VectorRef:
		return nil, fmt.Errorf("FT.HYBRID: unsupported vector type %T", v)
	default:
		values := v.Value()
		if len(values) < 2 {
			return nil, fmt.Errorf("FT.HYBRID: vector Value must contain a blob at index 1")
		}
		return values[1], nil
	}
}

func hybridVectorBytes(blob []byte) ([]byte, error) {
	if len(blob) == 0 {
		return nil, fmt.Errorf("FT.HYBRID: vector blob is required")
	}
	return blob, nil
}

// generateVectorParamName returns a parameter name that is not already present
// in params. It is used to pass vector data via the PARAMS mechanism when the
// caller does not provide a VectorParamName, since inline vector blobs are no
// longer supported by Redis.
func generateVectorParamName(params map[string]interface{}) string {
	for i := 0; ; i++ {
		name := fmt.Sprintf("__vector_param_%d", i)
		if _, ok := params[name]; !ok {
			return name
		}
	}
}

// FTHybridWithArgs - Executes a hybrid search with advanced options
// FTHybridWithArgs is still experimental, the command behaviour and signature may change
//
// Vector data is always sent through the PARAMS mechanism, because inline vector
// blobs are no longer supported by Redis. For every vector expression whose
// VectorParamName is empty, a unique name is generated (e.g. "__vector_param_0")
// and the corresponding blob is passed via PARAMS.
//
// options.Params is never mutated: the command is built from a local copy that
// combines the caller-provided params with any generated vector parameters. This
// makes it safe to reuse the same *FTHybridOptions across multiple calls. Generated
// names are also reserved against all explicit VectorParamName values, so they never
// collide with explicit names (even those following the "__vector_param_N" pattern).
func (c cmdable) FTHybridWithArgs(ctx context.Context, index string, options *FTHybridOptions) *FTHybridCmd {
	args := []interface{}{"FT.HYBRID", index}

	if options != nil {
		// Add search expressions
		for _, searchExpr := range options.SearchExpressions {
			args = append(args, "SEARCH", searchExpr.Query)

			if searchExpr.Scorer != "" {
				args = append(args, "SCORER", searchExpr.Scorer)
				if len(searchExpr.ScorerParams) > 0 {
					args = append(args, searchExpr.ScorerParams...)
				}
			}

			if searchExpr.YieldScoreAs != "" {
				args = append(args, "YIELD_SCORE_AS", searchExpr.YieldScoreAs)
			}
		}

		// Vector data is always passed via the PARAMS mechanism (inline vector blobs
		// are no longer supported by Redis). When vectors are present, build a local
		// copy of the caller-provided params so options.Params is never mutated, and
		// pre-reserve any explicit VectorParamName values so generated names never
		// collide with them.
		params := options.Params
		if len(options.VectorExpressions) > 0 {
			params = make(map[string]interface{}, len(options.Params)+len(options.VectorExpressions))
			for k, v := range options.Params {
				params[k] = v
			}
			for _, vectorExpr := range options.VectorExpressions {
				if vectorExpr.VectorParamName != "" {
					params[vectorExpr.VectorParamName] = nil
				}
			}
		}
		// Add vector expressions
		for _, vectorExpr := range options.VectorExpressions {
			args = append(args, "VSIM", "@"+vectorExpr.VectorField)

			vectorBlob, err := hybridVectorBlob(vectorExpr.VectorData)
			if err != nil {
				cmd := newFTHybridCmd(ctx, options, args...)
				cmd.SetErr(err)
				return cmd
			}

			// When VectorParamName is not provided, generate a unique name. Generated
			// names are tracked only in the local params map, never written back to
			// options.Params.
			paramName := vectorExpr.VectorParamName
			if paramName == "" {
				paramName = generateVectorParamName(params)
			}
			args = append(args, "$"+paramName)
			params[paramName] = vectorBlob

			if vectorExpr.Method != "" {
				args = append(args, vectorExpr.Method)
				if len(vectorExpr.MethodParams) > 0 {
					// MethodParams should be key-value pairs, count them
					args = append(args, len(vectorExpr.MethodParams))
					args = append(args, vectorExpr.MethodParams...)
				}
			}

			// SHARD_K_RATIO applies to the KNN method only (Redis 8.8+, cluster only).
			// Zero means "unset" and falls back to the server default of 1.0.
			if vectorExpr.ShardKRatio > 0 {
				if vectorExpr.Method != "KNN" {
					cmd := newFTHybridCmd(ctx, options, args...)
					cmd.SetErr(fmt.Errorf("FT.HYBRID: SHARD_K_RATIO requires KNN method"))
					return cmd
				}
				if vectorExpr.ShardKRatio < 0.1 || vectorExpr.ShardKRatio > 1.0 {
					cmd := newFTHybridCmd(ctx, options, args...)
					cmd.SetErr(fmt.Errorf("FT.HYBRID: SHARD_K_RATIO must be between 0.1 and 1.0"))
					return cmd
				}
				args = append(args, "SHARD_K_RATIO", vectorExpr.ShardKRatio)
			}

			if vectorExpr.Filter != "" {
				args = append(args, "FILTER", vectorExpr.Filter)
			}

			if vectorExpr.YieldScoreAs != "" {
				args = append(args, "YIELD_SCORE_AS", vectorExpr.YieldScoreAs)
			}
		}

		// Add combine/fusion options
		if options.Combine != nil {
			// Build combine parameters
			combineParams := []interface{}{}

			switch options.Combine.Method {
			case FTHybridCombineRRF:
				if options.Combine.Window > 0 {
					combineParams = append(combineParams, "WINDOW", options.Combine.Window)
				}
				if options.Combine.Constant > 0 {
					combineParams = append(combineParams, "CONSTANT", options.Combine.Constant)
				}
			case FTHybridCombineLinear:
				if options.Combine.Alpha > 0 {
					combineParams = append(combineParams, "ALPHA", options.Combine.Alpha)
				}
				if options.Combine.Beta > 0 {
					combineParams = append(combineParams, "BETA", options.Combine.Beta)
				}
			}

			if options.Combine.YieldScoreAs != "" {
				combineParams = append(combineParams, "YIELD_SCORE_AS", options.Combine.YieldScoreAs)
			}

			// Add COMBINE with method and parameter count
			args = append(args, "COMBINE", string(options.Combine.Method))
			if len(combineParams) > 0 {
				args = append(args, len(combineParams))
				args = append(args, combineParams...)
			}
		}

		// Add LOAD (projected fields)
		if len(options.Load) > 0 {
			args = append(args, "LOAD", len(options.Load))
			for _, field := range options.Load {
				args = append(args, field)
			}
		}

		// Add GROUPBY
		if options.GroupBy != nil {
			args = append(args, "GROUPBY", options.GroupBy.Count)
			for _, field := range options.GroupBy.Fields {
				args = append(args, field)
			}
			if options.GroupBy.ReduceFunc != "" {
				args = append(args, "REDUCE", options.GroupBy.ReduceFunc, options.GroupBy.ReduceCount)
				args = append(args, options.GroupBy.ReduceParams...)
			}
		}

		// Add APPLY transformations
		for _, apply := range options.Apply {
			args = append(args, "APPLY", apply.Expression, "AS", apply.AsField)
		}

		// Add SORTBY
		if len(options.SortBy) > 0 {
			sortByOptions := []interface{}{}
			for _, sortBy := range options.SortBy {
				sortByOptions = append(sortByOptions, sortBy.FieldName)
				if sortBy.Asc && sortBy.Desc {
					cmd := newFTHybridCmd(ctx, options, args...)
					cmd.SetErr(fmt.Errorf("FT.HYBRID: ASC and DESC are mutually exclusive"))
					return cmd
				}
				if sortBy.Asc {
					sortByOptions = append(sortByOptions, "ASC")
				}
				if sortBy.Desc {
					sortByOptions = append(sortByOptions, "DESC")
				}
			}
			args = append(args, "SORTBY", len(sortByOptions))
			args = append(args, sortByOptions...)
		}

		// Add FILTER (post-filter)
		if options.Filter != "" {
			args = append(args, "FILTER", options.Filter)
		}

		// Add LIMIT
		if options.LimitOffset >= 0 && options.Limit > 0 || options.LimitOffset > 0 && options.Limit == 0 {
			args = append(args, "LIMIT", options.LimitOffset, options.Limit)
		}

		// Add PARAMS
		// Emit from the local params map, which contains the caller-provided params
		// plus any generated vector parameter names. options.Params is left untouched.
		if len(params) > 0 {
			args = append(args, "PARAMS", len(params)*2)
			for key, value := range params {
				// PARAMS entries are passed without a '$' prefix; they are referenced in
				// the query and clauses using "$<name>".
				args = append(args, key, value)
			}
		}

		// Add EXPLAINSCORE
		if options.ExplainScore {
			args = append(args, "EXPLAINSCORE")
		}

		// Add TIMEOUT
		if options.Timeout > 0 {
			args = append(args, "TIMEOUT", options.Timeout)
		}

		// Add WITHCURSOR support
		if options.WithCursor {
			args = append(args, "WITHCURSOR")
			if options.WithCursorOptions != nil {
				if options.WithCursorOptions.Count > 0 {
					args = append(args, "COUNT", options.WithCursorOptions.Count)
				}
				if options.WithCursorOptions.MaxIdle > 0 {
					args = append(args, "MAXIDLE", options.WithCursorOptions.MaxIdle)
				}
			}
		}
	}

	cmd := newFTHybridCmd(ctx, options, args...)
	_ = c(ctx, cmd)
	return cmd
}
