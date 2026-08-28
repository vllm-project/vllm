package toml

import (
	"encoding"
	"errors"
	"fmt"
	"io"
	"math"
	"reflect"
	"strconv"
	"strings"
	"sync/atomic"
	"time"

	"github.com/pelletier/go-toml/v2/internal/danger"
	"github.com/pelletier/go-toml/v2/internal/tracker"
	"github.com/pelletier/go-toml/v2/unstable"
)

// Unmarshal deserializes a TOML document into a Go value.
//
// It is a shortcut for Decoder.Decode() with the default options.
func Unmarshal(data []byte, v interface{}) error {
	d := decoder{}
	d.p.Reset(data)
	return d.FromParser(v)
}

// Decoder reads and decode a TOML document from an input stream.
type Decoder struct {
	// input
	r io.Reader

	// global settings
	strict bool

	// toggles unmarshaler interface
	unmarshalerInterface bool
}

// NewDecoder creates a new Decoder that will read from r.
func NewDecoder(r io.Reader) *Decoder {
	return &Decoder{r: r}
}

// DisallowUnknownFields causes the Decoder to return an error when the
// destination is a struct and the input contains a key that does not match a
// non-ignored field.
//
// In that case, the Decoder returns a StrictMissingError that can be used to
// retrieve the individual errors as well as generate a human readable
// description of the missing fields.
func (d *Decoder) DisallowUnknownFields() *Decoder {
	d.strict = true
	return d
}

// EnableUnmarshalerInterface allows to enable unmarshaler interface.
//
// With this feature enabled, types implementing the unstable/Unmarshaler
// interface can be decoded from any structure of the document. It allows types
// that don't have a straightforward TOML representation to provide their own
// decoding logic.
//
// Currently, types can only decode from a single value. Tables and array tables
// are not supported.
//
// *Unstable:* This method does not follow the compatibility guarantees of
// semver. It can be changed or removed without a new major version being
// issued.
func (d *Decoder) EnableUnmarshalerInterface() *Decoder {
	d.unmarshalerInterface = true
	return d
}

// Decode the whole content of r into v.
//
// By default, values in the document that don't exist in the target Go value
// are ignored. See Decoder.DisallowUnknownFields() to change this behavior.
//
// When a TOML local date, time, or date-time is decoded into a time.Time, its
// value is represented in time.Local timezone. Otherwise the appropriate Local*
// structure is used. For time values, precision up to the nanosecond is
// supported by truncating extra digits.
//
// Empty tables decoded in an interface{} create an empty initialized
// map[string]interface{}.
//
// Types implementing the encoding.TextUnmarshaler interface are decoded from a
// TOML string.
//
// When decoding a number, go-toml will return an error if the number is out of
// bounds for the target type (which includes negative numbers when decoding
// into an unsigned int).
//
// If an error occurs while decoding the content of the document, this function
// returns a toml.DecodeError, providing context about the issue. When using
// strict mode and a field is missing, a `toml.StrictMissingError` is
// returned. In any other case, this function returns a standard Go error.
//
// # Type mapping
//
// List of supported TOML types and their associated accepted Go types:
//
//	String           -> string
//	Integer          -> uint*, int*, depending on size
//	Float            -> float*, depending on size
//	Boolean          -> bool
//	Offset Date-Time -> time.Time
//	Local Date-time  -> LocalDateTime, time.Time
//	Local Date       -> LocalDate, time.Time
//	Local Time       -> LocalTime, time.Time
//	Array            -> slice and array, depending on elements types
//	Table            -> map and struct
//	Inline Table     -> same as Table
//	Array of Tables  -> same as Array and Table
func (d *Decoder) Decode(v interface{}) error {
	b, err := io.ReadAll(d.r)
	if err != nil {
		return fmt.Errorf("toml: %w", err)
	}

	dec := decoder{
		strict: strict{
			Enabled: d.strict,
		},
		unmarshalerInterface: d.unmarshalerInterface,
	}
	dec.p.Reset(b)

	return dec.FromParser(v)
}

type decoder struct {
	// Which parser instance in use for this decoding session.
	p unstable.Parser

	// Flag indicating that the current expression is stashed.
	// If set to true, calling nextExpr will not actually pull a new expression
	// but turn off the flag instead.
	stashedExpr bool

	// Skip expressions until a table is found. This is set to true when a
	// table could not be created (missing field in map), so all KV expressions
	// need to be skipped.
	skipUntilTable bool

	// Flag indicating that the current array/slice table should be cleared because
	// it is the first encounter of an array table.
	clearArrayTable bool

	// Tracks position in Go arrays.
	// This is used when decoding [[array tables]] into Go arrays. Given array
	// tables are separate TOML expression, we need to keep track of where we
	// are at in the Go array, as we can't just introspect its size.
	arrayIndexes map[reflect.Value]int

	// Tracks keys that have been seen, with which type.
	seen tracker.SeenTracker

	// Strict mode
	strict strict

	// Flag that enables/disables unmarshaler interface.
	unmarshalerInterface bool

	// Current context for the error.
	errorContext *errorContext
}

type errorContext struct {
	Struct reflect.Type
	Field  []int
}

func (d *decoder) typeMismatchError(toml string, target reflect.Type) error {
	return fmt.Errorf("toml: %s", d.typeMismatchString(toml, target))
}

func (d *decoder) typeMismatchString(toml string, target reflect.Type) string {
	if d.errorContext != nil && d.errorContext.Struct != nil {
		ctx := d.errorContext
		f := ctx.Struct.FieldByIndex(ctx.Field)
		return fmt.Sprintf("cannot decode TOML %s into struct field %s.%s of type %s", toml, ctx.Struct, f.Name, f.Type)
	}
	return fmt.Sprintf("cannot decode TOML %s into a Go value of type %s", toml, target)
}

func (d *decoder) expr() *unstable.Node {
	return d.p.Expression()
}

func (d *decoder) nextExpr() bool {
	if d.stashedExpr {
		d.stashedExpr = false
		return true
	}
	return d.p.NextExpression()
}

func (d *decoder) stashExpr() {
	d.stashedExpr = true
}

func (d *decoder) arrayIndex(shouldAppend bool, v reflect.Value) int {
	if d.arrayIndexes == nil {
		d.arrayIndexes = make(map[reflect.Value]int, 1)
	}

	idx, ok := d.arrayIndexes[v]

	if !ok {
		d.arrayIndexes[v] = 0
	} else if shouldAppend {
		idx++
		d.arrayIndexes[v] = idx
	}

	return idx
}

func (d *decoder) FromParser(v interface{}) error {
	r := reflect.ValueOf(v)
	if r.Kind() != reflect.Ptr {
		return fmt.Errorf("toml: decoding can only be performed into a pointer, not %s", r.Kind())
	}

	if r.IsNil() {
		return fmt.Errorf("toml: decoding pointer target cannot be nil")
	}

	r = r.Elem()
	if r.Kind() == reflect.Interface && r.IsNil() {
		newMap := map[string]interface{}{}
		r.Set(reflect.ValueOf(newMap))
	}

	err := d.fromParser(r)
	if err == nil {
		return d.strict.Error(d.p.Data())
	}

	var e *unstable.ParserError
	if errors.As(err, &e) {
		return wrapDecodeError(d.p.Data(), e)
	}

	return err
}

func (d *decoder) fromParser(root reflect.Value) error {
	for d.nextExpr() {
		err := d.handleRootExpression(d.expr(), root)
		if err != nil {
			return err
		}
	}

	return d.p.Error()
}

/*
Rules for the unmarshal code:

- The stack is used to keep track of which values need to be set where.
- handle* functions <=> switch on a given unstable.Kind.
- unmarshalX* functions need to unmarshal a node of kind X.
- An "object" is either a struct or a map.
*/

func (d *decoder) handleRootExpression(expr *unstable.Node, v reflect.Value) error {
	var x reflect.Value
	var err error
	var first bool // used for to clear array tables on first use

	if !(d.skipUntilTable && expr.Kind == unstable.KeyValue) {
		first, err = d.seen.CheckExpression(expr)
		if err != nil {
			return err
		}
	}

	switch expr.Kind {
	case unstable.KeyValue:
		if d.skipUntilTable {
			return nil
		}
		x, err = d.handleKeyValue(expr, v)
	case unstable.Table:
		d.skipUntilTable = false
		d.strict.EnterTable(expr)
		x, err = d.handleTable(expr.Key(), v)
	case unstable.ArrayTable:
		d.skipUntilTable = false
		d.strict.EnterArrayTable(expr)
		d.clearArrayTable = first
		x, err = d.handleArrayTable(expr.Key(), v)
	default:
		panic(fmt.Errorf("parser should not permit expression of kind %s at document root", expr.Kind))
	}

	if d.skipUntilTable {
		if expr.Kind == unstable.Table || expr.Kind == unstable.ArrayTable {
			d.strict.MissingTable(expr)
		}
	} else if err == nil && x.IsValid() {
		v.Set(x)
	}

	return err
}

func (d *decoder) handleArrayTable(key unstable.Iterator, v reflect.Value) (reflect.Value, error) {
	if key.Next() {
		return d.handleArrayTablePart(key, v)
	}
	return d.handleKeyValues(v)
}

func (d *decoder) handleArrayTableCollectionLast(key unstable.Iterator, v reflect.Value) (reflect.Value, error) {
	switch v.Kind() {
	case reflect.Interface:
		elem := v.Elem()
		if !elem.IsValid() {
			elem = reflect.New(sliceInterfaceType).Elem()
			elem.Set(reflect.MakeSlice(sliceInterfaceType, 0, 16))
		} else if elem.Kind() == reflect.Slice {
			if elem.Type() != sliceInterfaceType {
				elem = reflect.New(sliceInterfaceType).Elem()
				elem.Set(reflect.MakeSlice(sliceInterfaceType, 0, 16))
			} else if !elem.CanSet() {
				nelem := reflect.New(sliceInterfaceType).Elem()
				nelem.Set(reflect.MakeSlice(sliceInterfaceType, elem.Len(), elem.Cap()))
				reflect.Copy(nelem, elem)
				elem = nelem
			}
			if d.clearArrayTable && elem.Len() > 0 {
				elem.SetLen(0)
				d.clearArrayTable = false
			}
		}
		return d.handleArrayTableCollectionLast(key, elem)
	case reflect.Ptr:
		elem := v.Elem()
		if !elem.IsValid() {
			ptr := reflect.New(v.Type().Elem())
			v.Set(ptr)
			elem = ptr.Elem()
		}

		elem, err := d.handleArrayTableCollectionLast(key, elem)
		if err != nil {
			return reflect.Value{}, err
		}
		v.Elem().Set(elem)

		return v, nil
	case reflect.Slice:
		if d.clearArrayTable && v.Len() > 0 {
			v.SetLen(0)
			d.clearArrayTable = false
		}
		elemType := v.Type().Elem()
		var elem reflect.Value
		if elemType.Kind() == reflect.Interface {
			elem = makeMapStringInterface()
		} else {
			elem = reflect.New(elemType).Elem()
		}
		elem2, err := d.handleArrayTable(key, elem)
		if err != nil {
			return reflect.Value{}, err
		}
		if elem2.IsValid() {
			elem = elem2
		}
		return reflect.Append(v, elem), nil
	case reflect.Array:
		idx := d.arrayIndex(true, v)
		if idx >= v.Len() {
			return v, fmt.Errorf("%s at position %d", d.typeMismatchError("array table", v.Type()), idx)
		}
		elem := v.Index(idx)
		_, err := d.handleArrayTable(key, elem)
		return v, err
	default:
		return reflect.Value{}, d.typeMismatchError("array table", v.Type())
	}
}

// When parsing an array table expression, each part of the key needs to be
// evaluated like a normal key, but if it returns a collection, it also needs to
// point to the last element of the collection. Unless it is the last part of
// the key, then it needs to create a new element at the end.
func (d *decoder) handleArrayTableCollection(key unstable.Iterator, v reflect.Value) (reflect.Value, error) {
	if key.IsLast() {
		return d.handleArrayTableCollectionLast(key, v)
	}

	switch v.Kind() {
	case reflect.Ptr:
		elem := v.Elem()
		if !elem.IsValid() {
			ptr := reflect.New(v.Type().Elem())
			v.Set(ptr)
			elem = ptr.Elem()
		}

		elem, err := d.handleArrayTableCollection(key, elem)
		if err != nil {
			return reflect.Value{}, err
		}
		if elem.IsValid() {
			v.Elem().Set(elem)
		}

		return v, nil
	case reflect.Slice:
		elem := v.Index(v.Len() - 1)
		x, err := d.handleArrayTable(key, elem)
		if err != nil || d.skipUntilTable {
			return reflect.Value{}, err
		}
		if x.IsValid() {
			elem.Set(x)
		}

		return v, err
	case reflect.Array:
		idx := d.arrayIndex(false, v)
		if idx >= v.Len() {
			return v, fmt.Errorf("%s at position %d", d.typeMismatchError("array table", v.Type()), idx)
		}
		elem := v.Index(idx)
		_, err := d.handleArrayTable(key, elem)
		return v, err
	}

	return d.handleArrayTable(key, v)
}

func (d *decoder) handleKeyPart(key unstable.Iterator, v reflect.Value, nextFn handlerFn, makeFn valueMakerFn) (reflect.Value, error) {
	var rv reflect.Value

	// First, dispatch over v to make sure it is a valid object.
	// There is no guarantee over what it could be.
	switch v.Kind() {
	case reflect.Ptr:
		elem := v.Elem()
		if !elem.IsValid() {
			v.Set(reflect.New(v.Type().Elem()))
		}
		elem = v.Elem()
		return d.handleKeyPart(key, elem, nextFn, makeFn)
	case reflect.Map:
		vt := v.Type()

		// Create the key for the map element. Convert to key type.
		mk, err := d.keyFromData(vt.Key(), key.Node().Data)
		if err != nil {
			return reflect.Value{}, err
		}

		// If the map does not exist, create it.
		if v.IsNil() {
			vt := v.Type()
			v = reflect.MakeMap(vt)
			rv = v
		}

		mv := v.MapIndex(mk)
		set := false
		if !mv.IsValid() {
			// If there is no value in the map, create a new one according to
			// the map type. If the element type is interface, create either a
			// map[string]interface{} or a []interface{} depending on whether
			// this is the last part of the array table key.

			t := vt.Elem()
			if t.Kind() == reflect.Interface {
				mv = makeFn()
			} else {
				mv = reflect.New(t).Elem()
			}
			set = true
		} else if mv.Kind() == reflect.Interface {
			mv = mv.Elem()
			if !mv.IsValid() {
				mv = makeFn()
			}
			set = true
		} else if !mv.CanAddr() {
			vt := v.Type()
			t := vt.Elem()
			oldmv := mv
			mv = reflect.New(t).Elem()
			mv.Set(oldmv)
			set = true
		}

		x, err := nextFn(key, mv)
		if err != nil {
			return reflect.Value{}, err
		}

		if x.IsValid() {
			mv = x
			set = true
		}

		if set {
			v.SetMapIndex(mk, mv)
		}
	case reflect.Struct:
		path, found := structFieldPath(v, string(key.Node().Data))
		if !found {
			d.skipUntilTable = true
			return reflect.Value{}, nil
		}

		if d.errorContext == nil {
			d.errorContext = new(errorContext)
		}
		t := v.Type()
		d.errorContext.Struct = t
		d.errorContext.Field = path

		f := fieldByIndex(v, path)
		x, err := nextFn(key, f)
		if err != nil || d.skipUntilTable {
			return reflect.Value{}, err
		}
		if x.IsValid() {
			f.Set(x)
		}
		d.errorContext.Field = nil
		d.errorContext.Struct = nil
	case reflect.Interface:
		if v.Elem().IsValid() {
			v = v.Elem()
		} else {
			v = makeMapStringInterface()
		}

		x, err := d.handleKeyPart(key, v, nextFn, makeFn)
		if err != nil {
			return reflect.Value{}, err
		}
		if x.IsValid() {
			v = x
		}
		rv = v
	default:
		panic(fmt.Errorf("unhandled part: %s", v.Kind()))
	}

	return rv, nil
}

// HandleArrayTablePart navigates the Go structure v using the key v. It is
// only used for the prefix (non-last) parts of an array-table. When
// encountering a collection, it should go to the last element.
func (d *decoder) handleArrayTablePart(key unstable.Iterator, v reflect.Value) (reflect.Value, error) {
	var makeFn valueMakerFn
	if key.IsLast() {
		makeFn = makeSliceInterface
	} else {
		makeFn = makeMapStringInterface
	}
	return d.handleKeyPart(key, v, d.handleArrayTableCollection, makeFn)
}

// HandleTable returns a reference when it has checked the next expression but
// cannot handle it.
func (d *decoder) handleTable(key unstable.Iterator, v reflect.Value) (reflect.Value, error) {
	if v.Kind() == reflect.Slice {
		if v.Len() == 0 {
			return reflect.Value{}, unstable.NewParserError(key.Node().Data, "cannot store a table in a slice")
		}
		elem := v.Index(v.Len() - 1)
		x, err := d.handleTable(key, elem)
		if err != nil {
			return reflect.Value{}, err
		}
		if x.IsValid() {
			elem.Set(x)
		}
		return reflect.Value{}, nil
	}
	if key.Next() {
		// Still scoping the key
		return d.handleTablePart(key, v)
	}
	// Done scoping the key.
	// Now handle all the key-value expressions in this table.
	return d.handleKeyValues(v)
}

// Handle root expressions until the end of the document or the next
// non-key-value.
func (d *decoder) handleKeyValues(v reflect.Value) (reflect.Value, error) {
	var rv reflect.Value
	for d.nextExpr() {
		expr := d.expr()
		if expr.Kind != unstable.KeyValue {
			// Stash the expression so that fromParser can just loop and use
			// the right handler.
			// We could just recurse ourselves here, but at least this gives a
			// chance to pop the stack a bit.
			d.stashExpr()
			break
		}

		_, err := d.seen.CheckExpression(expr)
		if err != nil {
			return reflect.Value{}, err
		}

		x, err := d.handleKeyValue(expr, v)
		if err != nil {
			return reflect.Value{}, err
		}
		if x.IsValid() {
			v = x
			rv = x
		}
	}
	return rv, nil
}

type (
	handlerFn    func(key unstable.Iterator, v reflect.Value) (reflect.Value, error)
	valueMakerFn func() reflect.Value
)

func makeMapStringInterface() reflect.Value {
	return reflect.MakeMap(mapStringInterfaceType)
}

func makeSliceInterface() reflect.Value {
	return reflect.MakeSlice(sliceInterfaceType, 0, 16)
}

func (d *decoder) handleTablePart(key unstable.Iterator, v reflect.Value) (reflect.Value, error) {
	return d.handleKeyPart(key, v, d.handleTable, makeMapStringInterface)
}

func (d *decoder) tryTextUnmarshaler(node *unstable.Node, v reflect.Value) (bool, error) {
	// Special case for time, because we allow to unmarshal to it from
	// different kind of AST nodes.
	if v.Type() == timeType {
		return false, nil
	}

	if v.CanAddr() && v.Addr().Type().Implements(textUnmarshalerType) {
		err := v.Addr().Interface().(encoding.TextUnmarshaler).UnmarshalText(node.Data)
		if err != nil {
			return false, unstable.NewParserError(d.p.Raw(node.Raw), "%w", err)
		}

		return true, nil
	}

	return false, nil
}

func (d *decoder) handleValue(value *unstable.Node, v reflect.Value) error {
	for v.Kind() == reflect.Ptr {
		v = initAndDereferencePointer(v)
	}

	if d.unmarshalerInterface {
		if v.CanAddr() && v.Addr().CanInterface() {
			if outi, ok := v.Addr().Interface().(unstable.Unmarshaler); ok {
				return outi.UnmarshalTOML(value)
			}
		}
	}

	ok, err := d.tryTextUnmarshaler(value, v)
	if ok || err != nil {
		return err
	}

	switch value.Kind {
	case unstable.String:
		return d.unmarshalString(value, v)
	case unstable.Integer:
		return d.unmarshalInteger(value, v)
	case unstable.Float:
		return d.unmarshalFloat(value, v)
	case unstable.Bool:
		return d.unmarshalBool(value, v)
	case unstable.DateTime:
		return d.unmarshalDateTime(value, v)
	case unstable.LocalDate:
		return d.unmarshalLocalDate(value, v)
	case unstable.LocalTime:
		return d.unmarshalLocalTime(value, v)
	case unstable.LocalDateTime:
		return d.unmarshalLocalDateTime(value, v)
	case unstable.InlineTable:
		return d.unmarshalInlineTable(value, v)
	case unstable.Array:
		return d.unmarshalArray(value, v)
	default:
		panic(fmt.Errorf("handleValue not implemented for %s", value.Kind))
	}
}

func (d *decoder) unmarshalArray(array *unstable.Node, v reflect.Value) error {
	switch v.Kind() {
	case reflect.Slice:
		if v.IsNil() {
			v.Set(reflect.MakeSlice(v.Type(), 0, 16))
		} else {
			v.SetLen(0)
		}
	case reflect.Array:
		// arrays are always initialized
	case reflect.Interface:
		elem := v.Elem()
		if !elem.IsValid() {
			elem = reflect.New(sliceInterfaceType).Elem()
			elem.Set(reflect.MakeSlice(sliceInterfaceType, 0, 16))
		} else if elem.Kind() == reflect.Slice {
			if elem.Type() != sliceInterfaceType {
				elem = reflect.New(sliceInterfaceType).Elem()
				elem.Set(reflect.MakeSlice(sliceInterfaceType, 0, 16))
			} else if !elem.CanSet() {
				nelem := reflect.New(sliceInterfaceType).Elem()
				nelem.Set(reflect.MakeSlice(sliceInterfaceType, elem.Len(), elem.Cap()))
				reflect.Copy(nelem, elem)
				elem = nelem
			}
		}
		err := d.unmarshalArray(array, elem)
		if err != nil {
			return err
		}
		v.Set(elem)
		return nil
	default:
		// TODO: use newDecodeError, but first the parser needs to fill
		//   array.Data.
		return d.typeMismatchError("array", v.Type())
	}

	elemType := v.Type().Elem()

	it := array.Children()
	idx := 0
	for it.Next() {
		n := it.Node()

		// TODO: optimize
		if v.Kind() == reflect.Slice {
			elem := reflect.New(elemType).Elem()

			err := d.handleValue(n, elem)
			if err != nil {
				return err
			}

			v.Set(reflect.Append(v, elem))
		} else { // array
			if idx >= v.Len() {
				return nil
			}
			elem := v.Index(idx)
			err := d.handleValue(n, elem)
			if err != nil {
				return err
			}
			idx++
		}
	}

	return nil
}

func (d *decoder) unmarshalInlineTable(itable *unstable.Node, v reflect.Value) error {
	// Make sure v is an initialized object.
	switch v.Kind() {
	case reflect.Map:
		if v.IsNil() {
			v.Set(reflect.MakeMap(v.Type()))
		}
	case reflect.Struct:
	// structs are always initialized.
	case reflect.Interface:
		elem := v.Elem()
		if !elem.IsValid() {
			elem = makeMapStringInterface()
			v.Set(elem)
		}
		return d.unmarshalInlineTable(itable, elem)
	default:
		return unstable.NewParserError(d.p.Raw(itable.Raw), "cannot store inline table in Go type %s", v.Kind())
	}

	it := itable.Children()
	for it.Next() {
		n := it.Node()

		x, err := d.handleKeyValue(n, v)
		if err != nil {
			return err
		}
		if x.IsValid() {
			v = x
		}
	}

	return nil
}

func (d *decoder) unmarshalDateTime(value *unstable.Node, v reflect.Value) error {
	dt, err := parseDateTime(value.Data)
	if err != nil {
		return err
	}

	v.Set(reflect.ValueOf(dt))
	return nil
}

func (d *decoder) unmarshalLocalDate(value *unstable.Node, v reflect.Value) error {
	ld, err := parseLocalDate(value.Data)
	if err != nil {
		return err
	}

	if v.Type() == timeType {
		cast := ld.AsTime(time.Local)
		v.Set(reflect.ValueOf(cast))
		return nil
	}

	v.Set(reflect.ValueOf(ld))

	return nil
}

func (d *decoder) unmarshalLocalTime(value *unstable.Node, v reflect.Value) error {
	lt, rest, err := parseLocalTime(value.Data)
	if err != nil {
		return err
	}

	if len(rest) > 0 {
		return unstable.NewParserError(rest, "extra characters at the end of a local time")
	}

	v.Set(reflect.ValueOf(lt))
	return nil
}

func (d *decoder) unmarshalLocalDateTime(value *unstable.Node, v reflect.Value) error {
	ldt, rest, err := parseLocalDateTime(value.Data)
	if err != nil {
		return err
	}

	if len(rest) > 0 {
		return unstable.NewParserError(rest, "extra characters at the end of a local date time")
	}

	if v.Type() == timeType {
		cast := ldt.AsTime(time.Local)

		v.Set(reflect.ValueOf(cast))
		return nil
	}

	v.Set(reflect.ValueOf(ldt))

	return nil
}

func (d *decoder) unmarshalBool(value *unstable.Node, v reflect.Value) error {
	b := value.Data[0] == 't'

	switch v.Kind() {
	case reflect.Bool:
		v.SetBool(b)
	case reflect.Interface:
		v.Set(reflect.ValueOf(b))
	default:
		return unstable.NewParserError(value.Data, "cannot assign boolean to a %t", b)
	}

	return nil
}

func (d *decoder) unmarshalFloat(value *unstable.Node, v reflect.Value) error {
	f, err := parseFloat(value.Data)
	if err != nil {
		return err
	}

	switch v.Kind() {
	case reflect.Float64:
		v.SetFloat(f)
	case reflect.Float32:
		if f > math.MaxFloat32 {
			return unstable.NewParserError(value.Data, "number %f does not fit in a float32", f)
		}
		v.SetFloat(f)
	case reflect.Interface:
		v.Set(reflect.ValueOf(f))
	default:
		return unstable.NewParserError(value.Data, "float cannot be assigned to %s", v.Kind())
	}

	return nil
}

const (
	maxInt = int64(^uint(0) >> 1)
	minInt = -maxInt - 1
)

// Maximum value of uint for decoding. Currently the decoder parses the integer
// into an int64. As a result, on architectures where uint is 64 bits, the
// effective maximum uint we can decode is the maximum of int64. On
// architectures where uint is 32 bits, the maximum value we can decode is
// lower: the maximum of uint32. I didn't find a way to figure out this value at
// compile time, so it is computed during initialization.
var maxUint int64 = math.MaxInt64

func init() {
	m := uint64(^uint(0))
	if m < uint64(maxUint) {
		maxUint = int64(m)
	}
}

func (d *decoder) unmarshalInteger(value *unstable.Node, v reflect.Value) error {
	kind := v.Kind()
	if kind == reflect.Float32 || kind == reflect.Float64 {
		return d.unmarshalFloat(value, v)
	}

	i, err := parseInteger(value.Data)
	if err != nil {
		return err
	}

	var r reflect.Value

	switch kind {
	case reflect.Int64:
		v.SetInt(i)
		return nil
	case reflect.Int32:
		if i < math.MinInt32 || i > math.MaxInt32 {
			return fmt.Errorf("toml: number %d does not fit in an int32", i)
		}

		r = reflect.ValueOf(int32(i))
	case reflect.Int16:
		if i < math.MinInt16 || i > math.MaxInt16 {
			return fmt.Errorf("toml: number %d does not fit in an int16", i)
		}

		r = reflect.ValueOf(int16(i))
	case reflect.Int8:
		if i < math.MinInt8 || i > math.MaxInt8 {
			return fmt.Errorf("toml: number %d does not fit in an int8", i)
		}

		r = reflect.ValueOf(int8(i))
	case reflect.Int:
		if i < minInt || i > maxInt {
			return fmt.Errorf("toml: number %d does not fit in an int", i)
		}

		r = reflect.ValueOf(int(i))
	case reflect.Uint64:
		if i < 0 {
			return fmt.Errorf("toml: negative number %d does not fit in an uint64", i)
		}

		r = reflect.ValueOf(uint64(i))
	case reflect.Uint32:
		if i < 0 || i > math.MaxUint32 {
			return fmt.Errorf("toml: negative number %d does not fit in an uint32", i)
		}

		r = reflect.ValueOf(uint32(i))
	case reflect.Uint16:
		if i < 0 || i > math.MaxUint16 {
			return fmt.Errorf("toml: negative number %d does not fit in an uint16", i)
		}

		r = reflect.ValueOf(uint16(i))
	case reflect.Uint8:
		if i < 0 || i > math.MaxUint8 {
			return fmt.Errorf("toml: negative number %d does not fit in an uint8", i)
		}

		r = reflect.ValueOf(uint8(i))
	case reflect.Uint:
		if i < 0 || i > maxUint {
			return fmt.Errorf("toml: negative number %d does not fit in an uint", i)
		}

		r = reflect.ValueOf(uint(i))
	case reflect.Interface:
		r = reflect.ValueOf(i)
	default:
		return unstable.NewParserError(d.p.Raw(value.Raw), d.typeMismatchString("integer", v.Type()))
	}

	if !r.Type().AssignableTo(v.Type()) {
		r = r.Convert(v.Type())
	}

	v.Set(r)

	return nil
}

func (d *decoder) unmarshalString(value *unstable.Node, v reflect.Value) error {
	switch v.Kind() {
	case reflect.String:
		v.SetString(string(value.Data))
	case reflect.Interface:
		v.Set(reflect.ValueOf(string(value.Data)))
	default:
		return unstable.NewParserError(d.p.Raw(value.Raw), d.typeMismatchString("string", v.Type()))
	}

	return nil
}

func (d *decoder) handleKeyValue(expr *unstable.Node, v reflect.Value) (reflect.Value, error) {
	d.strict.EnterKeyValue(expr)

	v, err := d.handleKeyValueInner(expr.Key(), expr.Value(), v)
	if d.skipUntilTable {
		d.strict.MissingField(expr)
		d.skipUntilTable = false
	}

	d.strict.ExitKeyValue(expr)

	return v, err
}

func (d *decoder) handleKeyValueInner(key unstable.Iterator, value *unstable.Node, v reflect.Value) (reflect.Value, error) {
	if key.Next() {
		// Still scoping the key
		return d.handleKeyValuePart(key, value, v)
	}
	// Done scoping the key.
	// v is whatever Go value we need to fill.
	return reflect.Value{}, d.handleValue(value, v)
}

func (d *decoder) keyFromData(keyType reflect.Type, data []byte) (reflect.Value, error) {
	switch {
	case stringType.AssignableTo(keyType):
		return reflect.ValueOf(string(data)), nil

	case stringType.ConvertibleTo(keyType):
		return reflect.ValueOf(string(data)).Convert(keyType), nil

	case keyType.Implements(textUnmarshalerType):
		mk := reflect.New(keyType.Elem())
		if err := mk.Interface().(encoding.TextUnmarshaler).UnmarshalText(data); err != nil {
			return reflect.Value{}, fmt.Errorf("toml: error unmarshalling key type %s from text: %w", stringType, err)
		}
		return mk, nil

	case reflect.PointerTo(keyType).Implements(textUnmarshalerType):
		mk := reflect.New(keyType)
		if err := mk.Interface().(encoding.TextUnmarshaler).UnmarshalText(data); err != nil {
			return reflect.Value{}, fmt.Errorf("toml: error unmarshalling key type %s from text: %w", stringType, err)
		}
		return mk.Elem(), nil

	case keyType.Kind() == reflect.Int || keyType.Kind() == reflect.Int8 || keyType.Kind() == reflect.Int16 || keyType.Kind() == reflect.Int32 || keyType.Kind() == reflect.Int64:
		key, err := strconv.ParseInt(string(data), 10, 64)
		if err != nil {
			return reflect.Value{}, fmt.Errorf("toml: error parsing key of type %s from integer: %w", stringType, err)
		}
		return reflect.ValueOf(key).Convert(keyType), nil
	case keyType.Kind() == reflect.Uint || keyType.Kind() == reflect.Uint8 || keyType.Kind() == reflect.Uint16 || keyType.Kind() == reflect.Uint32 || keyType.Kind() == reflect.Uint64:
		key, err := strconv.ParseUint(string(data), 10, 64)
		if err != nil {
			return reflect.Value{}, fmt.Errorf("toml: error parsing key of type %s from unsigned integer: %w", stringType, err)
		}
		return reflect.ValueOf(key).Convert(keyType), nil

	case keyType.Kind() == reflect.Float32:
		key, err := strconv.ParseFloat(string(data), 32)
		if err != nil {
			return reflect.Value{}, fmt.Errorf("toml: error parsing key of type %s from float: %w", stringType, err)
		}
		return reflect.ValueOf(float32(key)), nil

	case keyType.Kind() == reflect.Float64:
		key, err := strconv.ParseFloat(string(data), 64)
		if err != nil {
			return reflect.Value{}, fmt.Errorf("toml: error parsing key of type %s from float: %w", stringType, err)
		}
		return reflect.ValueOf(float64(key)), nil
	}
	return reflect.Value{}, fmt.Errorf("toml: cannot convert map key of type %s to expected type %s", stringType, keyType)
}

func (d *decoder) handleKeyValuePart(key unstable.Iterator, value *unstable.Node, v reflect.Value) (reflect.Value, error) {
	// contains the replacement for v
	var rv reflect.Value

	// First, dispatch over v to make sure it is a valid object.
	// There is no guarantee over what it could be.
	switch v.Kind() {
	case reflect.Map:
		vt := v.Type()

		mk, err := d.keyFromData(vt.Key(), key.Node().Data)
		if err != nil {
			return reflect.Value{}, err
		}

		// If the map does not exist, create it.
		if v.IsNil() {
			v = reflect.MakeMap(vt)
			rv = v
		}

		mv := v.MapIndex(mk)
		set := false
		if !mv.IsValid() || key.IsLast() {
			set = true
			mv = reflect.New(v.Type().Elem()).Elem()
		}

		nv, err := d.handleKeyValueInner(key, value, mv)
		if err != nil {
			return reflect.Value{}, err
		}
		if nv.IsValid() {
			mv = nv
			set = true
		}

		if set {
			v.SetMapIndex(mk, mv)
		}
	case reflect.Struct:
		path, found := structFieldPath(v, string(key.Node().Data))
		if !found {
			d.skipUntilTable = true
			break
		}

		if d.errorContext == nil {
			d.errorContext = new(errorContext)
		}
		t := v.Type()
		d.errorContext.Struct = t
		d.errorContext.Field = path

		f := fieldByIndex(v, path)

		if !f.CanAddr() {
			// If the field is not addressable, need to take a slower path and
			// make a copy of the struct itself to a new location.
			nvp := reflect.New(v.Type())
			nvp.Elem().Set(v)
			v = nvp.Elem()
			_, err := d.handleKeyValuePart(key, value, v)
			if err != nil {
				return reflect.Value{}, err
			}
			return nvp.Elem(), nil
		}
		x, err := d.handleKeyValueInner(key, value, f)
		if err != nil {
			return reflect.Value{}, err
		}

		if x.IsValid() {
			f.Set(x)
		}
		d.errorContext.Struct = nil
		d.errorContext.Field = nil
	case reflect.Interface:
		v = v.Elem()

		// Following encoding/json: decoding an object into an
		// interface{}, it needs to always hold a
		// map[string]interface{}. This is for the types to be
		// consistent whether a previous value was set or not.
		if !v.IsValid() || v.Type() != mapStringInterfaceType {
			v = makeMapStringInterface()
		}

		x, err := d.handleKeyValuePart(key, value, v)
		if err != nil {
			return reflect.Value{}, err
		}
		if x.IsValid() {
			v = x
		}
		rv = v
	case reflect.Ptr:
		elem := v.Elem()
		if !elem.IsValid() {
			ptr := reflect.New(v.Type().Elem())
			v.Set(ptr)
			rv = v
			elem = ptr.Elem()
		}

		elem2, err := d.handleKeyValuePart(key, value, elem)
		if err != nil {
			return reflect.Value{}, err
		}
		if elem2.IsValid() {
			elem = elem2
		}
		v.Elem().Set(elem)
	default:
		return reflect.Value{}, fmt.Errorf("unhandled kv part: %s", v.Kind())
	}

	return rv, nil
}

func initAndDereferencePointer(v reflect.Value) reflect.Value {
	var elem reflect.Value
	if v.IsNil() {
		ptr := reflect.New(v.Type().Elem())
		v.Set(ptr)
	}
	elem = v.Elem()
	return elem
}

// Same as reflect.Value.FieldByIndex, but creates pointers if needed.
func fieldByIndex(v reflect.Value, path []int) reflect.Value {
	for _, x := range path {
		v = v.Field(x)

		if v.Kind() == reflect.Ptr {
			if v.IsNil() {
				v.Set(reflect.New(v.Type().Elem()))
			}
			v = v.Elem()
		}
	}
	return v
}

type fieldPathsMap = map[string][]int

var globalFieldPathsCache atomic.Value // map[danger.TypeID]fieldPathsMap

func structFieldPath(v reflect.Value, name string) ([]int, bool) {
	t := v.Type()

	cache, _ := globalFieldPathsCache.Load().(map[danger.TypeID]fieldPathsMap)
	fieldPaths, ok := cache[danger.MakeTypeID(t)]

	if !ok {
		fieldPaths = map[string][]int{}

		forEachField(t, nil, func(name string, path []int) {
			fieldPaths[name] = path
			// extra copy for the case-insensitive match
			fieldPaths[strings.ToLower(name)] = path
		})

		newCache := make(map[danger.TypeID]fieldPathsMap, len(cache)+1)
		newCache[danger.MakeTypeID(t)] = fieldPaths
		for k, v := range cache {
			newCache[k] = v
		}
		globalFieldPathsCache.Store(newCache)
	}

	path, ok := fieldPaths[name]
	if !ok {
		path, ok = fieldPaths[strings.ToLower(name)]
	}
	return path, ok
}

func forEachField(t reflect.Type, path []int, do func(name string, path []int)) {
	n := t.NumField()
	for i := 0; i < n; i++ {
		f := t.Field(i)

		if !f.Anonymous && f.PkgPath != "" {
			// only consider exported fields.
			continue
		}

		fieldPath := append(path, i)
		fieldPath = fieldPath[:len(fieldPath):len(fieldPath)]

		name := f.Tag.Get("toml")
		if name == "-" {
			continue
		}

		if i := strings.IndexByte(name, ','); i >= 0 {
			name = name[:i]
		}

		if f.Anonymous && name == "" {
			t2 := f.Type
			if t2.Kind() == reflect.Ptr {
				t2 = t2.Elem()
			}

			if t2.Kind() == reflect.Struct {
				forEachField(t2, fieldPath, do)
			}
			continue
		}

		if name == "" {
			name = f.Name
		}

		do(name, fieldPath)
	}
}
