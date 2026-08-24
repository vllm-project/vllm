package routing

import (
	"errors"
	"fmt"
	"math"
	"sync"

	"sync/atomic"

	"github.com/redis/go-redis/v9/internal/util"
	uberAtomic "go.uber.org/atomic"
)

var (
	ErrMaxAggregation = errors.New("redis: no valid results to aggregate for max operation")
	ErrMinAggregation = errors.New("redis: no valid results to aggregate for min operation")
	ErrAndAggregation = errors.New("redis: no valid results to aggregate for logical AND operation")
	ErrOrAggregation  = errors.New("redis: no valid results to aggregate for logical OR operation")
)

// ResponseAggregator defines the interface for aggregating responses from multiple shards.
type ResponseAggregator interface {
	// Add processes a single shard response.
	Add(result interface{}, err error) error

	// AddWithKey processes a single shard response for a specific key (used by keyed aggregators).
	AddWithKey(key string, result interface{}, err error) error

	BatchAdd(map[string]AggregatorResErr) error

	BatchSlice([]AggregatorResErr) error

	// Result returns the final aggregated result and any error.
	Result() (interface{}, error)
}

type AggregatorResErr struct {
	Result interface{}
	Err    error
}

// NewResponseAggregator creates an aggregator based on the response policy.
func NewResponseAggregator(policy ResponsePolicy, cmdName string) ResponseAggregator {
	switch policy {
	case RespDefaultKeyless:
		return &DefaultKeylessAggregator{results: make([]interface{}, 0)}
	case RespDefaultHashSlot:
		return &DefaultKeyedAggregator{results: make(map[string]interface{})}
	case RespAllSucceeded:
		return &AllSucceededAggregator{}
	case RespOneSucceeded:
		return &OneSucceededAggregator{}
	case RespAggSum:
		return &AggSumAggregator{
			// res:
		}
	case RespAggMin:
		return &AggMinAggregator{
			res: util.NewAtomicMin(),
		}
	case RespAggMax:
		return &AggMaxAggregator{
			res: util.NewAtomicMax(),
		}
	case RespAggLogicalAnd:
		andAgg := &AggLogicalAndAggregator{}
		andAgg.res.Store(true)

		return andAgg
	case RespAggLogicalOr:
		return &AggLogicalOrAggregator{}
	case RespSpecial:
		return NewSpecialAggregator(cmdName)
	default:
		return &AllSucceededAggregator{}
	}
}

func NewDefaultAggregator(isKeyed bool) ResponseAggregator {
	if isKeyed {
		return &DefaultKeyedAggregator{
			results: make(map[string]interface{}),
		}
	}
	return &DefaultKeylessAggregator{}
}

// AllSucceededAggregator returns one non-error reply if every shard succeeded,
// propagates the first error otherwise.
type AllSucceededAggregator struct {
	err atomic.Value
	res atomic.Value
}

func (a *AllSucceededAggregator) Add(result interface{}, err error) error {
	if err != nil {
		a.err.CompareAndSwap(nil, err)
		return nil
	}

	if result != nil {
		a.res.CompareAndSwap(nil, result)
	}

	return nil
}

func (a *AllSucceededAggregator) BatchAdd(results map[string]AggregatorResErr) error {
	for _, res := range results {
		err := a.Add(res.Result, res.Err)
		if err != nil {
			return err
		}

		if res.Err != nil {
			return nil
		}
	}

	return nil
}

func (a *AllSucceededAggregator) BatchSlice(results []AggregatorResErr) error {
	for _, res := range results {
		err := a.Add(res.Result, res.Err)
		if err != nil {
			return err
		}

		if res.Err != nil {
			return nil
		}
	}

	return nil
}

func (a *AllSucceededAggregator) Result() (interface{}, error) {
	var err error
	res, e := a.res.Load(), a.err.Load()
	if e != nil {
		err = e.(error)
	}

	return res, err
}

func (a *AllSucceededAggregator) AddWithKey(key string, result interface{}, err error) error {
	return a.Add(result, err)
}

// OneSucceededAggregator returns the first non-error reply,
// if all shards errored, returns any one of those errors.
type OneSucceededAggregator struct {
	err atomic.Value
	res atomic.Value
}

func (a *OneSucceededAggregator) Add(result interface{}, err error) error {
	if err != nil {
		a.err.CompareAndSwap(nil, err)
		return nil
	}

	if result != nil {
		a.res.CompareAndSwap(nil, result)
	}

	return nil
}

func (a *OneSucceededAggregator) BatchAdd(results map[string]AggregatorResErr) error {
	for _, res := range results {
		err := a.Add(res.Result, res.Err)
		if err != nil {
			return err
		}

		if res.Err == nil {
			return nil
		}
	}

	return nil
}

func (a *OneSucceededAggregator) AddWithKey(key string, result interface{}, err error) error {
	return a.Add(result, err)
}

func (a *OneSucceededAggregator) BatchSlice(results []AggregatorResErr) error {
	for _, res := range results {
		err := a.Add(res.Result, res.Err)
		if err != nil {
			return err
		}

		if res.Err == nil {
			return nil
		}
	}

	return nil
}

func (a *OneSucceededAggregator) Result() (interface{}, error) {
	res, e := a.res.Load(), a.err.Load()
	if res == nil {
		return nil, e.(error)
	}

	return res, nil
}

// AggSumAggregator sums numeric replies from all shards.
type AggSumAggregator struct {
	err atomic.Value
	res uberAtomic.Float64
}

func (a *AggSumAggregator) Add(result interface{}, err error) error {
	if err != nil {
		a.err.CompareAndSwap(nil, err)
	}

	if result != nil {
		val, err := toFloat64(result)
		if err != nil {
			a.err.CompareAndSwap(nil, err)
			return err
		}
		a.res.Add(val)
	}

	return nil
}

func (a *AggSumAggregator) BatchAdd(results map[string]AggregatorResErr) error {
	var sum int64

	for _, res := range results {
		if res.Err != nil {
			return a.Add(res.Result, res.Err)
		}

		intRes, err := toInt64(res.Result)
		if err != nil {
			return a.Add(nil, err)
		}

		sum += intRes
	}

	return a.Add(sum, nil)
}

func (a *AggSumAggregator) AddWithKey(key string, result interface{}, err error) error {
	return a.Add(result, err)
}

func (a *AggSumAggregator) BatchSlice(results []AggregatorResErr) error {
	var sum int64

	for _, res := range results {
		if res.Err != nil {
			return a.Add(res.Result, res.Err)
		}

		intRes, err := toInt64(res.Result)
		if err != nil {
			return a.Add(nil, err)
		}

		sum += intRes
	}

	return a.Add(sum, nil)
}

func (a *AggSumAggregator) Result() (interface{}, error) {
	res, err := a.res.Load(), a.err.Load()
	if err != nil {
		return nil, err.(error)
	}

	return res, nil
}

// AggMinAggregator returns the minimum numeric value from all shards.
type AggMinAggregator struct {
	err atomic.Value
	res *util.AtomicMin
}

func (a *AggMinAggregator) Add(result interface{}, err error) error {
	if err != nil {
		a.err.CompareAndSwap(nil, err)
		return nil
	}

	floatVal, e := toFloat64(result)
	if e != nil {
		a.err.CompareAndSwap(nil, err)
		return nil
	}

	a.res.Value(floatVal)

	return nil
}

func (a *AggMinAggregator) BatchAdd(results map[string]AggregatorResErr) error {
	min := int64(math.MaxInt64)

	for _, res := range results {
		if res.Err != nil {
			_ = a.Add(nil, res.Err)
			return nil
		}

		resInt, err := toInt64(res.Result)
		if err != nil {
			_ = a.Add(nil, res.Err)
			return nil
		}

		if resInt < min {
			min = resInt
		}

	}

	return a.Add(min, nil)
}

func (a *AggMinAggregator) AddWithKey(key string, result interface{}, err error) error {
	return a.Add(result, err)
}

func (a *AggMinAggregator) BatchSlice(results []AggregatorResErr) error {
	min := float64(math.MaxFloat64)

	for _, res := range results {
		if res.Err != nil {
			_ = a.Add(nil, res.Err)
			return nil
		}

		floatVal, err := toFloat64(res.Result)
		if err != nil {
			_ = a.Add(nil, res.Err)
			return nil
		}

		if floatVal < min {
			min = floatVal
		}

	}

	return a.Add(min, nil)
}

func (a *AggMinAggregator) Result() (interface{}, error) {
	err := a.err.Load()
	if err != nil {
		return nil, err.(error)
	}

	val, hasVal := a.res.Min()
	if !hasVal {
		return nil, ErrMinAggregation
	}
	return val, nil
}

// AggMaxAggregator returns the maximum numeric value from all shards.
type AggMaxAggregator struct {
	err atomic.Value
	res *util.AtomicMax
}

func (a *AggMaxAggregator) Add(result interface{}, err error) error {
	if err != nil {
		a.err.CompareAndSwap(nil, err)
		return nil
	}

	floatVal, e := toFloat64(result)
	if e != nil {
		a.err.CompareAndSwap(nil, err)
		return nil
	}

	a.res.Value(floatVal)

	return nil
}

func (a *AggMaxAggregator) BatchAdd(results map[string]AggregatorResErr) error {
	max := int64(math.MinInt64)

	for _, res := range results {
		if res.Err != nil {
			_ = a.Add(nil, res.Err)
			return nil
		}

		resInt, err := toInt64(res.Result)
		if err != nil {
			_ = a.Add(nil, res.Err)
			return nil
		}

		if resInt > max {
			max = resInt
		}

	}

	return a.Add(max, nil)
}

func (a *AggMaxAggregator) AddWithKey(key string, result interface{}, err error) error {
	return a.Add(result, err)
}

func (a *AggMaxAggregator) BatchSlice(results []AggregatorResErr) error {
	max := int64(math.MinInt64)

	for _, res := range results {
		if res.Err != nil {
			_ = a.Add(nil, res.Err)
			return nil
		}

		resInt, err := toInt64(res.Result)
		if err != nil {
			_ = a.Add(nil, res.Err)
			return nil
		}

		if resInt > max {
			max = resInt
		}

	}

	return a.Add(max, nil)
}

func (a *AggMaxAggregator) Result() (interface{}, error) {
	err := a.err.Load()
	if err != nil {
		return nil, err.(error)
	}

	val, hasVal := a.res.Max()
	if !hasVal {
		return nil, ErrMaxAggregation
	}
	return val, nil
}

// AggLogicalAndAggregator performs logical AND on boolean values.
type AggLogicalAndAggregator struct {
	err       atomic.Value
	res       atomic.Bool
	hasResult atomic.Bool
}

func (a *AggLogicalAndAggregator) Add(result interface{}, err error) error {
	if err != nil {
		a.err.CompareAndSwap(nil, err)
		return nil
	}

	val, e := toBool(result)
	if e != nil {
		a.err.CompareAndSwap(nil, e)
		return e
	}

	// Atomic AND operation: if val is false, result is always false
	if !val {
		a.res.Store(false)
	}

	a.hasResult.Store(true)

	return nil
}

func (a *AggLogicalAndAggregator) BatchAdd(results map[string]AggregatorResErr) error {
	result := true

	for _, res := range results {
		if res.Err != nil {
			return a.Add(nil, res.Err)
		}

		boolRes, err := toBool(res.Result)
		if err != nil {
			return a.Add(nil, err)
		}

		result = result && boolRes
	}

	return a.Add(result, nil)
}

func (a *AggLogicalAndAggregator) AddWithKey(key string, result interface{}, err error) error {
	return a.Add(result, err)
}

func (a *AggLogicalAndAggregator) BatchSlice(results []AggregatorResErr) error {
	result := true

	for _, res := range results {
		if res.Err != nil {
			return a.Add(nil, res.Err)
		}

		boolRes, err := toBool(res.Result)
		if err != nil {
			return a.Add(nil, err)
		}

		result = result && boolRes
	}

	return a.Add(result, nil)
}

func (a *AggLogicalAndAggregator) Result() (interface{}, error) {
	err := a.err.Load()
	if err != nil {
		return nil, err.(error)
	}

	if !a.hasResult.Load() {
		return nil, ErrAndAggregation
	}
	return a.res.Load(), nil
}

// AggLogicalOrAggregator performs logical OR on boolean values.
type AggLogicalOrAggregator struct {
	err       atomic.Value
	res       atomic.Bool
	hasResult atomic.Bool
}

func (a *AggLogicalOrAggregator) Add(result interface{}, err error) error {
	if err != nil {
		a.err.CompareAndSwap(nil, err)
		return nil
	}

	val, e := toBool(result)
	if e != nil {
		a.err.CompareAndSwap(nil, e)
		return e
	}

	// Atomic OR operation: if val is true, result is always true
	if val {
		a.res.Store(true)
	}

	a.hasResult.Store(true)

	return nil
}

func (a *AggLogicalOrAggregator) BatchAdd(results map[string]AggregatorResErr) error {
	result := false

	for _, res := range results {
		if res.Err != nil {
			return a.Add(nil, res.Err)
		}

		boolRes, err := toBool(res.Result)
		if err != nil {
			return a.Add(nil, err)
		}

		result = result || boolRes
	}

	return a.Add(result, nil)
}

func (a *AggLogicalOrAggregator) AddWithKey(key string, result interface{}, err error) error {
	return a.Add(result, err)
}

func (a *AggLogicalOrAggregator) BatchSlice(results []AggregatorResErr) error {
	result := false

	for _, res := range results {
		if res.Err != nil {
			return a.Add(nil, res.Err)
		}

		boolRes, err := toBool(res.Result)
		if err != nil {
			return a.Add(nil, err)
		}

		result = result || boolRes
	}

	return a.Add(result, nil)
}

func (a *AggLogicalOrAggregator) Result() (interface{}, error) {
	err := a.err.Load()
	if err != nil {
		return nil, err.(error)
	}

	if !a.hasResult.Load() {
		return nil, ErrOrAggregation
	}
	return a.res.Load(), nil
}

func toInt64(val interface{}) (int64, error) {
	if val == nil {
		return 0, nil
	}
	switch v := val.(type) {
	case int64:
		return v, nil
	case int:
		return int64(v), nil
	case int32:
		return int64(v), nil
	case float64:
		if v != math.Trunc(v) {
			return 0, fmt.Errorf("cannot convert float %f to int64", v)
		}
		return int64(v), nil
	default:
		return 0, fmt.Errorf("cannot convert %T to int64", val)
	}
}

func toFloat64(val interface{}) (float64, error) {
	if val == nil {
		return 0, nil
	}

	switch v := val.(type) {
	case float64:
		return v, nil
	case int:
		return float64(v), nil
	case int32:
		return float64(v), nil
	case int64:
		return float64(v), nil
	case float32:
		return float64(v), nil
	default:
		return 0, fmt.Errorf("cannot convert %T to float64", val)
	}
}

func toBool(val interface{}) (bool, error) {
	if val == nil {
		return false, nil
	}
	switch v := val.(type) {
	case bool:
		return v, nil
	case int64:
		return v != 0, nil
	case int:
		return v != 0, nil
	default:
		return false, fmt.Errorf("cannot convert %T to bool", val)
	}
}

// DefaultKeylessAggregator collects all results in an array, order doesn't matter.
type DefaultKeylessAggregator struct {
	mu       sync.Mutex
	results  []interface{}
	firstErr error
}

func (a *DefaultKeylessAggregator) add(result interface{}, err error) error {
	if err != nil && a.firstErr == nil {
		a.firstErr = err
		return nil
	}
	if err == nil {
		a.results = append(a.results, result)
	}
	return nil
}

func (a *DefaultKeylessAggregator) Add(result interface{}, err error) error {
	a.mu.Lock()
	defer a.mu.Unlock()

	return a.add(result, err)
}

func (a *DefaultKeylessAggregator) BatchAdd(results map[string]AggregatorResErr) error {
	a.mu.Lock()
	defer a.mu.Unlock()

	for _, res := range results {
		err := a.add(res.Result, res.Err)
		if err != nil {
			return err
		}

		if res.Err != nil {
			return nil
		}
	}

	return nil
}

func (a *DefaultKeylessAggregator) AddWithKey(key string, result interface{}, err error) error {
	return a.Add(result, err)
}

func (a *DefaultKeylessAggregator) BatchSlice(results []AggregatorResErr) error {
	a.mu.Lock()
	defer a.mu.Unlock()

	for _, res := range results {
		err := a.add(res.Result, res.Err)
		if err != nil {
			return err
		}

		if res.Err != nil {
			return nil
		}
	}

	return nil
}

func (a *DefaultKeylessAggregator) Result() (interface{}, error) {
	a.mu.Lock()
	defer a.mu.Unlock()

	if a.firstErr != nil {
		return nil, a.firstErr
	}
	return a.results, nil
}

// DefaultKeyedAggregator reassembles replies in the exact key order of the original request.
type DefaultKeyedAggregator struct {
	mu       sync.Mutex
	results  map[string]interface{}
	keyOrder []string
	firstErr error
}

func NewDefaultKeyedAggregator(keyOrder []string) *DefaultKeyedAggregator {
	return &DefaultKeyedAggregator{
		results:  make(map[string]interface{}),
		keyOrder: keyOrder,
	}
}

func (a *DefaultKeyedAggregator) add(result interface{}, err error) error {
	if err != nil && a.firstErr == nil {
		a.firstErr = err
		return nil
	}
	// For non-keyed Add, just collect the result without ordering
	if err == nil {
		a.results["__default__"] = result
	}
	return nil
}

func (a *DefaultKeyedAggregator) Add(result interface{}, err error) error {
	a.mu.Lock()
	defer a.mu.Unlock()

	return a.add(result, err)
}

func (a *DefaultKeyedAggregator) BatchAdd(results map[string]AggregatorResErr) error {
	a.mu.Lock()
	defer a.mu.Unlock()

	for _, res := range results {
		err := a.add(res.Result, res.Err)
		if err != nil {
			return err
		}

		if res.Err != nil {
			return nil
		}
	}

	return nil
}

func (a *DefaultKeyedAggregator) addWithKey(key string, result interface{}, err error) error {
	if err != nil && a.firstErr == nil {
		a.firstErr = err
		return nil
	}
	if err == nil {
		a.results[key] = result
	}
	return nil
}

func (a *DefaultKeyedAggregator) AddWithKey(key string, result interface{}, err error) error {
	a.mu.Lock()
	defer a.mu.Unlock()

	return a.addWithKey(key, result, err)
}

func (a *DefaultKeyedAggregator) BatchAddWithKeyOrder(results map[string]AggregatorResErr, keyOrder []string) error {
	a.mu.Lock()
	defer a.mu.Unlock()

	a.keyOrder = keyOrder
	for key, res := range results {
		err := a.addWithKey(key, res.Result, res.Err)
		if err != nil {
			return nil
		}

		if res.Err != nil {
			return nil
		}
	}

	return nil
}

func (a *DefaultKeyedAggregator) SetKeyOrder(keyOrder []string) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.keyOrder = keyOrder
}

func (a *DefaultKeyedAggregator) BatchSlice(results []AggregatorResErr) error {
	a.mu.Lock()
	defer a.mu.Unlock()

	for _, res := range results {
		err := a.add(res.Result, res.Err)
		if err != nil {
			return err
		}

		if res.Err != nil {
			return nil
		}
	}

	return nil
}

func (a *DefaultKeyedAggregator) Result() (interface{}, error) {
	a.mu.Lock()
	defer a.mu.Unlock()

	if a.firstErr != nil {
		return nil, a.firstErr
	}

	// If no explicit key order is set, return results in any order
	if len(a.keyOrder) == 0 {
		orderedResults := make([]interface{}, 0, len(a.results))
		for _, result := range a.results {
			orderedResults = append(orderedResults, result)
		}
		return orderedResults, nil
	}

	// Return results in the exact key order
	orderedResults := make([]interface{}, len(a.keyOrder))
	for i, key := range a.keyOrder {
		if result, exists := a.results[key]; exists {
			orderedResults[i] = result
		}
	}
	return orderedResults, nil
}

// SpecialAggregator provides a registry for command-specific aggregation logic.
type SpecialAggregator struct {
	mu             sync.Mutex
	aggregatorFunc func([]interface{}, []error) (interface{}, error)
	results        []interface{}
	errors         []error
}

func (a *SpecialAggregator) add(result interface{}, err error) error {
	a.results = append(a.results, result)
	a.errors = append(a.errors, err)
	return nil
}

func (a *SpecialAggregator) Add(result interface{}, err error) error {
	a.mu.Lock()
	defer a.mu.Unlock()

	return a.add(result, err)
}

func (a *SpecialAggregator) BatchAdd(results map[string]AggregatorResErr) error {
	a.mu.Lock()
	defer a.mu.Unlock()

	for _, res := range results {
		err := a.add(res.Result, res.Err)
		if err != nil {
			return err
		}

		if res.Err != nil {
			return nil
		}
	}

	return nil
}

func (a *SpecialAggregator) AddWithKey(key string, result interface{}, err error) error {
	return a.Add(result, err)
}

func (a *SpecialAggregator) BatchSlice(results []AggregatorResErr) error {
	a.mu.Lock()
	defer a.mu.Unlock()

	for _, res := range results {
		err := a.add(res.Result, res.Err)
		if err != nil {
			return err
		}

		if res.Err != nil {
			return nil
		}
	}

	return nil
}

func (a *SpecialAggregator) Result() (interface{}, error) {
	a.mu.Lock()
	defer a.mu.Unlock()

	if a.aggregatorFunc != nil {
		return a.aggregatorFunc(a.results, a.errors)
	}
	// Default behavior: return first non-error result or first error
	for i, err := range a.errors {
		if err == nil {
			return a.results[i], nil
		}
	}
	if len(a.errors) > 0 {
		return nil, a.errors[0]
	}
	return nil, nil
}

// SpecialAggregatorRegistry holds custom aggregation functions for specific commands.
var SpecialAggregatorRegistry = make(map[string]func([]interface{}, []error) (interface{}, error))

// RegisterSpecialAggregator registers a custom aggregation function for a command.
func RegisterSpecialAggregator(cmdName string, fn func([]interface{}, []error) (interface{}, error)) {
	SpecialAggregatorRegistry[cmdName] = fn
}

// NewSpecialAggregator creates a special aggregator with command-specific logic if available.
func NewSpecialAggregator(cmdName string) *SpecialAggregator {
	agg := &SpecialAggregator{}
	if fn, exists := SpecialAggregatorRegistry[cmdName]; exists {
		agg.aggregatorFunc = fn
	}
	return agg
}
