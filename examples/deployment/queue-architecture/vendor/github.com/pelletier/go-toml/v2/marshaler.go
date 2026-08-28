package toml

import (
	"bytes"
	"encoding"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"reflect"
	"slices"
	"strconv"
	"strings"
	"time"
	"unicode"

	"github.com/pelletier/go-toml/v2/internal/characters"
)

// Marshal serializes a Go value as a TOML document.
//
// It is a shortcut for Encoder.Encode() with the default options.
func Marshal(v interface{}) ([]byte, error) {
	var buf bytes.Buffer
	enc := NewEncoder(&buf)

	err := enc.Encode(v)
	if err != nil {
		return nil, err
	}

	return buf.Bytes(), nil
}

// Encoder writes a TOML document to an output stream.
type Encoder struct {
	// output
	w io.Writer

	// global settings
	tablesInline       bool
	arraysMultiline    bool
	indentSymbol       string
	indentTables       bool
	marshalJsonNumbers bool
}

// NewEncoder returns a new Encoder that writes to w.
func NewEncoder(w io.Writer) *Encoder {
	return &Encoder{
		w:            w,
		indentSymbol: "  ",
	}
}

// SetTablesInline forces the encoder to emit all tables inline.
//
// This behavior can be controlled on an individual struct field basis with the
// inline tag:
//
//	MyField `toml:",inline"`
func (enc *Encoder) SetTablesInline(inline bool) *Encoder {
	enc.tablesInline = inline
	return enc
}

// SetArraysMultiline forces the encoder to emit all arrays with one element per
// line.
//
// This behavior can be controlled on an individual struct field basis with the multiline tag:
//
//	MyField `multiline:"true"`
func (enc *Encoder) SetArraysMultiline(multiline bool) *Encoder {
	enc.arraysMultiline = multiline
	return enc
}

// SetIndentSymbol defines the string that should be used for indentation. The
// provided string is repeated for each indentation level. Defaults to two
// spaces.
func (enc *Encoder) SetIndentSymbol(s string) *Encoder {
	enc.indentSymbol = s
	return enc
}

// SetIndentTables forces the encoder to intent tables and array tables.
func (enc *Encoder) SetIndentTables(indent bool) *Encoder {
	enc.indentTables = indent
	return enc
}

// SetMarshalJsonNumbers forces the encoder to serialize `json.Number` as a
// float or integer instead of relying on TextMarshaler to emit a string.
//
// *Unstable:* This method does not follow the compatibility guarantees of
// semver. It can be changed or removed without a new major version being
// issued.
func (enc *Encoder) SetMarshalJsonNumbers(indent bool) *Encoder {
	enc.marshalJsonNumbers = indent
	return enc
}

// Encode writes a TOML representation of v to the stream.
//
// If v cannot be represented to TOML it returns an error.
//
// # Encoding rules
//
// A top level slice containing only maps or structs is encoded as [[table
// array]].
//
// All slices not matching rule 1 are encoded as [array]. As a result, any map
// or struct they contain is encoded as an {inline table}.
//
// Nil interfaces and nil pointers are not supported.
//
// Keys in key-values always have one part.
//
// Intermediate tables are always printed.
//
// By default, strings are encoded as literal string, unless they contain either
// a newline character or a single quote. In that case they are emitted as
// quoted strings.
//
// Unsigned integers larger than math.MaxInt64 cannot be encoded. Doing so
// results in an error. This rule exists because the TOML specification only
// requires parsers to support at least the 64 bits integer range. Allowing
// larger numbers would create non-standard TOML documents, which may not be
// readable (at best) by other implementations. To encode such numbers, a
// solution is a custom type that implements encoding.TextMarshaler.
//
// When encoding structs, fields are encoded in order of definition, with their
// exact name.
//
// Tables and array tables are separated by empty lines. However, consecutive
// subtables definitions are not. For example:
//
//	[top1]
//
//	[top2]
//	[top2.child1]
//
//	[[array]]
//
//	[[array]]
//	[array.child2]
//
// # Struct tags
//
// The encoding of each public struct field can be customized by the format
// string in the "toml" key of the struct field's tag. This follows
// encoding/json's convention. The format string starts with the name of the
// field, optionally followed by a comma-separated list of options. The name may
// be empty in order to provide options without overriding the default name.
//
// The "multiline" option emits strings as quoted multi-line TOML strings. It
// has no effect on fields that would not be encoded as strings.
//
// The "inline" option turns fields that would be emitted as tables into inline
// tables instead. It has no effect on other fields.
//
// The "omitempty" option prevents empty values or groups from being emitted.
//
// The "commented" option prefixes the value and all its children with a comment
// symbol.
//
// In addition to the "toml" tag struct tag, a "comment" tag can be used to emit
// a TOML comment before the value being annotated. Comments are ignored inside
// inline tables. For array tables, the comment is only present before the first
// element of the array.
func (enc *Encoder) Encode(v interface{}) error {
	var (
		b   []byte
		ctx encoderCtx
	)

	ctx.inline = enc.tablesInline

	if v == nil {
		return fmt.Errorf("toml: cannot encode a nil interface")
	}

	b, err := enc.encode(b, ctx, reflect.ValueOf(v))
	if err != nil {
		return err
	}

	_, err = enc.w.Write(b)
	if err != nil {
		return fmt.Errorf("toml: cannot write: %w", err)
	}

	return nil
}

type valueOptions struct {
	multiline bool
	omitempty bool
	commented bool
	comment   string
}

type encoderCtx struct {
	// Current top-level key.
	parentKey []string

	// Key that should be used for a KV.
	key string
	// Extra flag to account for the empty string
	hasKey bool

	// Set to true to indicate that the encoder is inside a KV, so that all
	// tables need to be inlined.
	insideKv bool

	// Set to true to skip the first table header in an array table.
	skipTableHeader bool

	// Should the next table be encoded as inline
	inline bool

	// Indentation level
	indent int

	// Prefix the current value with a comment.
	commented bool

	// Options coming from struct tags
	options valueOptions
}

func (ctx *encoderCtx) shiftKey() {
	if ctx.hasKey {
		ctx.parentKey = append(ctx.parentKey, ctx.key)
		ctx.clearKey()
	}
}

func (ctx *encoderCtx) setKey(k string) {
	ctx.key = k
	ctx.hasKey = true
}

func (ctx *encoderCtx) clearKey() {
	ctx.key = ""
	ctx.hasKey = false
}

func (ctx *encoderCtx) isRoot() bool {
	return len(ctx.parentKey) == 0 && !ctx.hasKey
}

func (enc *Encoder) encode(b []byte, ctx encoderCtx, v reflect.Value) ([]byte, error) {
	i := v.Interface()

	switch x := i.(type) {
	case time.Time:
		if x.Nanosecond() > 0 {
			return x.AppendFormat(b, time.RFC3339Nano), nil
		}
		return x.AppendFormat(b, time.RFC3339), nil
	case LocalTime:
		return append(b, x.String()...), nil
	case LocalDate:
		return append(b, x.String()...), nil
	case LocalDateTime:
		return append(b, x.String()...), nil
	case json.Number:
		if enc.marshalJsonNumbers {
			if x == "" { /// Useful zero value.
				return append(b, "0"...), nil
			} else if v, err := x.Int64(); err == nil {
				return enc.encode(b, ctx, reflect.ValueOf(v))
			} else if f, err := x.Float64(); err == nil {
				return enc.encode(b, ctx, reflect.ValueOf(f))
			} else {
				return nil, fmt.Errorf("toml: unable to convert %q to int64 or float64", x)
			}
		}
	}

	hasTextMarshaler := v.Type().Implements(textMarshalerType)
	if hasTextMarshaler || (v.CanAddr() && reflect.PointerTo(v.Type()).Implements(textMarshalerType)) {
		if !hasTextMarshaler {
			v = v.Addr()
		}

		if ctx.isRoot() {
			return nil, fmt.Errorf("toml: type %s implementing the TextMarshaler interface cannot be a root element", v.Type())
		}

		text, err := v.Interface().(encoding.TextMarshaler).MarshalText()
		if err != nil {
			return nil, err
		}

		b = enc.encodeString(b, string(text), ctx.options)

		return b, nil
	}

	switch v.Kind() {
	// containers
	case reflect.Map:
		return enc.encodeMap(b, ctx, v)
	case reflect.Struct:
		return enc.encodeStruct(b, ctx, v)
	case reflect.Slice, reflect.Array:
		return enc.encodeSlice(b, ctx, v)
	case reflect.Interface:
		if v.IsNil() {
			return nil, fmt.Errorf("toml: encoding a nil interface is not supported")
		}

		return enc.encode(b, ctx, v.Elem())
	case reflect.Ptr:
		if v.IsNil() {
			return enc.encode(b, ctx, reflect.Zero(v.Type().Elem()))
		}

		return enc.encode(b, ctx, v.Elem())

	// values
	case reflect.String:
		b = enc.encodeString(b, v.String(), ctx.options)
	case reflect.Float32:
		f := v.Float()

		if math.IsNaN(f) {
			b = append(b, "nan"...)
		} else if f > math.MaxFloat32 {
			b = append(b, "inf"...)
		} else if f < -math.MaxFloat32 {
			b = append(b, "-inf"...)
		} else if math.Trunc(f) == f {
			b = strconv.AppendFloat(b, f, 'f', 1, 32)
		} else {
			b = strconv.AppendFloat(b, f, 'f', -1, 32)
		}
	case reflect.Float64:
		f := v.Float()
		if math.IsNaN(f) {
			b = append(b, "nan"...)
		} else if f > math.MaxFloat64 {
			b = append(b, "inf"...)
		} else if f < -math.MaxFloat64 {
			b = append(b, "-inf"...)
		} else if math.Trunc(f) == f {
			b = strconv.AppendFloat(b, f, 'f', 1, 64)
		} else {
			b = strconv.AppendFloat(b, f, 'f', -1, 64)
		}
	case reflect.Bool:
		if v.Bool() {
			b = append(b, "true"...)
		} else {
			b = append(b, "false"...)
		}
	case reflect.Uint64, reflect.Uint32, reflect.Uint16, reflect.Uint8, reflect.Uint:
		x := v.Uint()
		if x > uint64(math.MaxInt64) {
			return nil, fmt.Errorf("toml: not encoding uint (%d) greater than max int64 (%d)", x, int64(math.MaxInt64))
		}
		b = strconv.AppendUint(b, x, 10)
	case reflect.Int64, reflect.Int32, reflect.Int16, reflect.Int8, reflect.Int:
		b = strconv.AppendInt(b, v.Int(), 10)
	default:
		return nil, fmt.Errorf("toml: cannot encode value of type %s", v.Kind())
	}

	return b, nil
}

func isNil(v reflect.Value) bool {
	switch v.Kind() {
	case reflect.Ptr, reflect.Interface, reflect.Map:
		return v.IsNil()
	default:
		return false
	}
}

func shouldOmitEmpty(options valueOptions, v reflect.Value) bool {
	return options.omitempty && isEmptyValue(v)
}

func (enc *Encoder) encodeKv(b []byte, ctx encoderCtx, options valueOptions, v reflect.Value) ([]byte, error) {
	var err error

	if !ctx.inline {
		b = enc.encodeComment(ctx.indent, options.comment, b)
		b = enc.commented(ctx.commented, b)
		b = enc.indent(ctx.indent, b)
	}

	b = enc.encodeKey(b, ctx.key)
	b = append(b, " = "...)

	// create a copy of the context because the value of a KV shouldn't
	// modify the global context.
	subctx := ctx
	subctx.insideKv = true
	subctx.shiftKey()
	subctx.options = options

	b, err = enc.encode(b, subctx, v)
	if err != nil {
		return nil, err
	}

	return b, nil
}

func (enc *Encoder) commented(commented bool, b []byte) []byte {
	if commented {
		return append(b, "# "...)
	}
	return b
}

func isEmptyValue(v reflect.Value) bool {
	switch v.Kind() {
	case reflect.Struct:
		return isEmptyStruct(v)
	case reflect.Array, reflect.Map, reflect.Slice, reflect.String:
		return v.Len() == 0
	case reflect.Bool:
		return !v.Bool()
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return v.Int() == 0
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
		return v.Uint() == 0
	case reflect.Float32, reflect.Float64:
		return v.Float() == 0
	case reflect.Interface, reflect.Ptr:
		return v.IsNil()
	}
	return false
}

func isEmptyStruct(v reflect.Value) bool {
	// TODO: merge with walkStruct and cache.
	typ := v.Type()
	for i := 0; i < typ.NumField(); i++ {
		fieldType := typ.Field(i)

		// only consider exported fields
		if fieldType.PkgPath != "" {
			continue
		}

		tag := fieldType.Tag.Get("toml")

		// special field name to skip field
		if tag == "-" {
			continue
		}

		f := v.Field(i)

		if !isEmptyValue(f) {
			return false
		}
	}

	return true
}

const literalQuote = '\''

func (enc *Encoder) encodeString(b []byte, v string, options valueOptions) []byte {
	if needsQuoting(v) {
		return enc.encodeQuotedString(options.multiline, b, v)
	}

	return enc.encodeLiteralString(b, v)
}

func needsQuoting(v string) bool {
	// TODO: vectorize
	for _, b := range []byte(v) {
		if b == '\'' || b == '\r' || b == '\n' || characters.InvalidAscii(b) {
			return true
		}
	}
	return false
}

// caller should have checked that the string does not contain new lines or ' .
func (enc *Encoder) encodeLiteralString(b []byte, v string) []byte {
	b = append(b, literalQuote)
	b = append(b, v...)
	b = append(b, literalQuote)

	return b
}

func (enc *Encoder) encodeQuotedString(multiline bool, b []byte, v string) []byte {
	stringQuote := `"`

	if multiline {
		stringQuote = `"""`
	}

	b = append(b, stringQuote...)
	if multiline {
		b = append(b, '\n')
	}

	const (
		hextable = "0123456789ABCDEF"
		// U+0000 to U+0008, U+000A to U+001F, U+007F
		nul = 0x0
		bs  = 0x8
		lf  = 0xa
		us  = 0x1f
		del = 0x7f
	)

	for _, r := range []byte(v) {
		switch r {
		case '\\':
			b = append(b, `\\`...)
		case '"':
			b = append(b, `\"`...)
		case '\b':
			b = append(b, `\b`...)
		case '\f':
			b = append(b, `\f`...)
		case '\n':
			if multiline {
				b = append(b, r)
			} else {
				b = append(b, `\n`...)
			}
		case '\r':
			b = append(b, `\r`...)
		case '\t':
			b = append(b, `\t`...)
		default:
			switch {
			case r >= nul && r <= bs, r >= lf && r <= us, r == del:
				b = append(b, `\u00`...)
				b = append(b, hextable[r>>4])
				b = append(b, hextable[r&0x0f])
			default:
				b = append(b, r)
			}
		}
	}

	b = append(b, stringQuote...)

	return b
}

// caller should have checked that the string is in A-Z / a-z / 0-9 / - / _ .
func (enc *Encoder) encodeUnquotedKey(b []byte, v string) []byte {
	return append(b, v...)
}

func (enc *Encoder) encodeTableHeader(ctx encoderCtx, b []byte) ([]byte, error) {
	if len(ctx.parentKey) == 0 {
		return b, nil
	}

	b = enc.encodeComment(ctx.indent, ctx.options.comment, b)

	b = enc.commented(ctx.commented, b)

	b = enc.indent(ctx.indent, b)

	b = append(b, '[')

	b = enc.encodeKey(b, ctx.parentKey[0])

	for _, k := range ctx.parentKey[1:] {
		b = append(b, '.')
		b = enc.encodeKey(b, k)
	}

	b = append(b, "]\n"...)

	return b, nil
}

//nolint:cyclop
func (enc *Encoder) encodeKey(b []byte, k string) []byte {
	needsQuotation := false
	cannotUseLiteral := false

	if len(k) == 0 {
		return append(b, "''"...)
	}

	for _, c := range k {
		if (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '-' || c == '_' {
			continue
		}

		if c == literalQuote {
			cannotUseLiteral = true
		}

		needsQuotation = true
	}

	if needsQuotation && needsQuoting(k) {
		cannotUseLiteral = true
	}

	switch {
	case cannotUseLiteral:
		return enc.encodeQuotedString(false, b, k)
	case needsQuotation:
		return enc.encodeLiteralString(b, k)
	default:
		return enc.encodeUnquotedKey(b, k)
	}
}

func (enc *Encoder) keyToString(k reflect.Value) (string, error) {
	keyType := k.Type()
	switch {
	case keyType.Kind() == reflect.String:
		return k.String(), nil

	case keyType.Implements(textMarshalerType):
		keyB, err := k.Interface().(encoding.TextMarshaler).MarshalText()
		if err != nil {
			return "", fmt.Errorf("toml: error marshalling key %v from text: %w", k, err)
		}
		return string(keyB), nil

	case keyType.Kind() == reflect.Int || keyType.Kind() == reflect.Int8 || keyType.Kind() == reflect.Int16 || keyType.Kind() == reflect.Int32 || keyType.Kind() == reflect.Int64:
		return strconv.FormatInt(k.Int(), 10), nil

	case keyType.Kind() == reflect.Uint || keyType.Kind() == reflect.Uint8 || keyType.Kind() == reflect.Uint16 || keyType.Kind() == reflect.Uint32 || keyType.Kind() == reflect.Uint64:
		return strconv.FormatUint(k.Uint(), 10), nil

	case keyType.Kind() == reflect.Float32:
		return strconv.FormatFloat(k.Float(), 'f', -1, 32), nil

	case keyType.Kind() == reflect.Float64:
		return strconv.FormatFloat(k.Float(), 'f', -1, 64), nil
	}
	return "", fmt.Errorf("toml: type %s is not supported as a map key", keyType.Kind())
}

func (enc *Encoder) encodeMap(b []byte, ctx encoderCtx, v reflect.Value) ([]byte, error) {
	var (
		t                 table
		emptyValueOptions valueOptions
	)

	iter := v.MapRange()
	for iter.Next() {
		v := iter.Value()

		if isNil(v) {
			continue
		}

		k, err := enc.keyToString(iter.Key())
		if err != nil {
			return nil, err
		}

		if willConvertToTableOrArrayTable(ctx, v) {
			t.pushTable(k, v, emptyValueOptions)
		} else {
			t.pushKV(k, v, emptyValueOptions)
		}
	}

	sortEntriesByKey(t.kvs)
	sortEntriesByKey(t.tables)

	return enc.encodeTable(b, ctx, t)
}

func sortEntriesByKey(e []entry) {
	slices.SortFunc(e, func(a, b entry) int {
		return strings.Compare(a.Key, b.Key)
	})
}

type entry struct {
	Key     string
	Value   reflect.Value
	Options valueOptions
}

type table struct {
	kvs    []entry
	tables []entry
}

func (t *table) pushKV(k string, v reflect.Value, options valueOptions) {
	for _, e := range t.kvs {
		if e.Key == k {
			return
		}
	}

	t.kvs = append(t.kvs, entry{Key: k, Value: v, Options: options})
}

func (t *table) pushTable(k string, v reflect.Value, options valueOptions) {
	for _, e := range t.tables {
		if e.Key == k {
			return
		}
	}
	t.tables = append(t.tables, entry{Key: k, Value: v, Options: options})
}

func walkStruct(ctx encoderCtx, t *table, v reflect.Value) {
	// TODO: cache this
	typ := v.Type()
	for i := 0; i < typ.NumField(); i++ {
		fieldType := typ.Field(i)

		// only consider exported fields
		if fieldType.PkgPath != "" {
			continue
		}

		tag := fieldType.Tag.Get("toml")

		// special field name to skip field
		if tag == "-" {
			continue
		}

		k, opts := parseTag(tag)
		if !isValidName(k) {
			k = ""
		}

		f := v.Field(i)

		if k == "" {
			if fieldType.Anonymous {
				if fieldType.Type.Kind() == reflect.Struct {
					walkStruct(ctx, t, f)
				} else if fieldType.Type.Kind() == reflect.Ptr && !f.IsNil() && f.Elem().Kind() == reflect.Struct {
					walkStruct(ctx, t, f.Elem())
				}
				continue
			} else {
				k = fieldType.Name
			}
		}

		if isNil(f) {
			continue
		}

		options := valueOptions{
			multiline: opts.multiline,
			omitempty: opts.omitempty,
			commented: opts.commented,
			comment:   fieldType.Tag.Get("comment"),
		}

		if opts.inline || !willConvertToTableOrArrayTable(ctx, f) {
			t.pushKV(k, f, options)
		} else {
			t.pushTable(k, f, options)
		}
	}
}

func (enc *Encoder) encodeStruct(b []byte, ctx encoderCtx, v reflect.Value) ([]byte, error) {
	var t table

	walkStruct(ctx, &t, v)

	return enc.encodeTable(b, ctx, t)
}

func (enc *Encoder) encodeComment(indent int, comment string, b []byte) []byte {
	for len(comment) > 0 {
		var line string
		idx := strings.IndexByte(comment, '\n')
		if idx >= 0 {
			line = comment[:idx]
			comment = comment[idx+1:]
		} else {
			line = comment
			comment = ""
		}
		b = enc.indent(indent, b)
		b = append(b, "# "...)
		b = append(b, line...)
		b = append(b, '\n')
	}
	return b
}

func isValidName(s string) bool {
	if s == "" {
		return false
	}
	for _, c := range s {
		switch {
		case strings.ContainsRune("!#$%&()*+-./:;<=>?@[]^_{|}~ ", c):
			// Backslash and quote chars are reserved, but
			// otherwise any punctuation chars are allowed
			// in a tag name.
		case !unicode.IsLetter(c) && !unicode.IsDigit(c):
			return false
		}
	}
	return true
}

type tagOptions struct {
	multiline bool
	inline    bool
	omitempty bool
	commented bool
}

func parseTag(tag string) (string, tagOptions) {
	opts := tagOptions{}

	idx := strings.Index(tag, ",")
	if idx == -1 {
		return tag, opts
	}

	raw := tag[idx+1:]
	tag = string(tag[:idx])
	for raw != "" {
		var o string
		i := strings.Index(raw, ",")
		if i >= 0 {
			o, raw = raw[:i], raw[i+1:]
		} else {
			o, raw = raw, ""
		}
		switch o {
		case "multiline":
			opts.multiline = true
		case "inline":
			opts.inline = true
		case "omitempty":
			opts.omitempty = true
		case "commented":
			opts.commented = true
		}
	}

	return tag, opts
}

func (enc *Encoder) encodeTable(b []byte, ctx encoderCtx, t table) ([]byte, error) {
	var err error

	ctx.shiftKey()

	if ctx.insideKv || (ctx.inline && !ctx.isRoot()) {
		return enc.encodeTableInline(b, ctx, t)
	}

	if !ctx.skipTableHeader {
		b, err = enc.encodeTableHeader(ctx, b)
		if err != nil {
			return nil, err
		}

		if enc.indentTables && len(ctx.parentKey) > 0 {
			ctx.indent++
		}
	}
	ctx.skipTableHeader = false

	hasNonEmptyKV := false
	for _, kv := range t.kvs {
		if shouldOmitEmpty(kv.Options, kv.Value) {
			continue
		}
		hasNonEmptyKV = true

		ctx.setKey(kv.Key)
		ctx2 := ctx
		ctx2.commented = kv.Options.commented || ctx2.commented

		b, err = enc.encodeKv(b, ctx2, kv.Options, kv.Value)
		if err != nil {
			return nil, err
		}

		b = append(b, '\n')
	}

	first := true
	for _, table := range t.tables {
		if shouldOmitEmpty(table.Options, table.Value) {
			continue
		}
		if first {
			first = false
			if hasNonEmptyKV {
				b = append(b, '\n')
			}
		} else {
			b = append(b, "\n"...)
		}

		ctx.setKey(table.Key)

		ctx.options = table.Options
		ctx2 := ctx
		ctx2.commented = ctx2.commented || ctx.options.commented

		b, err = enc.encode(b, ctx2, table.Value)
		if err != nil {
			return nil, err
		}
	}

	return b, nil
}

func (enc *Encoder) encodeTableInline(b []byte, ctx encoderCtx, t table) ([]byte, error) {
	var err error

	b = append(b, '{')

	first := true
	for _, kv := range t.kvs {
		if shouldOmitEmpty(kv.Options, kv.Value) {
			continue
		}

		if first {
			first = false
		} else {
			b = append(b, `, `...)
		}

		ctx.setKey(kv.Key)

		b, err = enc.encodeKv(b, ctx, kv.Options, kv.Value)
		if err != nil {
			return nil, err
		}
	}

	if len(t.tables) > 0 {
		panic("inline table cannot contain nested tables, only key-values")
	}

	b = append(b, "}"...)

	return b, nil
}

func willConvertToTable(ctx encoderCtx, v reflect.Value) bool {
	if !v.IsValid() {
		return false
	}
	if v.Type() == timeType || v.Type().Implements(textMarshalerType) || (v.Kind() != reflect.Ptr && v.CanAddr() && reflect.PointerTo(v.Type()).Implements(textMarshalerType)) {
		return false
	}

	t := v.Type()
	switch t.Kind() {
	case reflect.Map, reflect.Struct:
		return !ctx.inline
	case reflect.Interface:
		return willConvertToTable(ctx, v.Elem())
	case reflect.Ptr:
		if v.IsNil() {
			return false
		}

		return willConvertToTable(ctx, v.Elem())
	default:
		return false
	}
}

func willConvertToTableOrArrayTable(ctx encoderCtx, v reflect.Value) bool {
	if ctx.insideKv {
		return false
	}
	t := v.Type()

	if t.Kind() == reflect.Interface {
		return willConvertToTableOrArrayTable(ctx, v.Elem())
	}

	if t.Kind() == reflect.Slice || t.Kind() == reflect.Array {
		if v.Len() == 0 {
			// An empty slice should be a kv = [].
			return false
		}

		for i := 0; i < v.Len(); i++ {
			t := willConvertToTable(ctx, v.Index(i))

			if !t {
				return false
			}
		}

		return true
	}

	return willConvertToTable(ctx, v)
}

func (enc *Encoder) encodeSlice(b []byte, ctx encoderCtx, v reflect.Value) ([]byte, error) {
	if v.Len() == 0 {
		b = append(b, "[]"...)

		return b, nil
	}

	if willConvertToTableOrArrayTable(ctx, v) {
		return enc.encodeSliceAsArrayTable(b, ctx, v)
	}

	return enc.encodeSliceAsArray(b, ctx, v)
}

// caller should have checked that v is a slice that only contains values that
// encode into tables.
func (enc *Encoder) encodeSliceAsArrayTable(b []byte, ctx encoderCtx, v reflect.Value) ([]byte, error) {
	ctx.shiftKey()

	scratch := make([]byte, 0, 64)

	scratch = enc.commented(ctx.commented, scratch)

	if enc.indentTables {
		scratch = enc.indent(ctx.indent, scratch)
	}

	scratch = append(scratch, "[["...)

	for i, k := range ctx.parentKey {
		if i > 0 {
			scratch = append(scratch, '.')
		}

		scratch = enc.encodeKey(scratch, k)
	}

	scratch = append(scratch, "]]\n"...)
	ctx.skipTableHeader = true

	b = enc.encodeComment(ctx.indent, ctx.options.comment, b)

	if enc.indentTables {
		ctx.indent++
	}

	for i := 0; i < v.Len(); i++ {
		if i != 0 {
			b = append(b, "\n"...)
		}

		b = append(b, scratch...)

		var err error
		b, err = enc.encode(b, ctx, v.Index(i))
		if err != nil {
			return nil, err
		}
	}

	return b, nil
}

func (enc *Encoder) encodeSliceAsArray(b []byte, ctx encoderCtx, v reflect.Value) ([]byte, error) {
	multiline := ctx.options.multiline || enc.arraysMultiline
	separator := ", "

	b = append(b, '[')

	subCtx := ctx
	subCtx.options = valueOptions{}

	if multiline {
		separator = ",\n"

		b = append(b, '\n')

		subCtx.indent++
	}

	var err error
	first := true

	for i := 0; i < v.Len(); i++ {
		if first {
			first = false
		} else {
			b = append(b, separator...)
		}

		if multiline {
			b = enc.indent(subCtx.indent, b)
		}

		b, err = enc.encode(b, subCtx, v.Index(i))
		if err != nil {
			return nil, err
		}
	}

	if multiline {
		b = append(b, '\n')
		b = enc.indent(ctx.indent, b)
	}

	b = append(b, ']')

	return b, nil
}

func (enc *Encoder) indent(level int, b []byte) []byte {
	for i := 0; i < level; i++ {
		b = append(b, enc.indentSymbol...)
	}

	return b
}
