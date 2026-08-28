package unstable

import (
	"bytes"
	"fmt"
	"unicode"

	"github.com/pelletier/go-toml/v2/internal/characters"
	"github.com/pelletier/go-toml/v2/internal/danger"
)

// ParserError describes an error relative to the content of the document.
//
// It cannot outlive the instance of Parser it refers to, and may cause panics
// if the parser is reset.
type ParserError struct {
	Highlight []byte
	Message   string
	Key       []string // optional
}

// Error is the implementation of the error interface.
func (e *ParserError) Error() string {
	return e.Message
}

// NewParserError is a convenience function to create a ParserError
//
// Warning: Highlight needs to be a subslice of Parser.data, so only slices
// returned by Parser.Raw are valid candidates.
func NewParserError(highlight []byte, format string, args ...interface{}) error {
	return &ParserError{
		Highlight: highlight,
		Message:   fmt.Errorf(format, args...).Error(),
	}
}

// Parser scans over a TOML-encoded document and generates an iterative AST.
//
// To prime the Parser, first reset it with the contents of a TOML document.
// Then, process all top-level expressions sequentially. See Example.
//
// Don't forget to check Error() after you're done parsing.
//
// Each top-level expression needs to be fully processed before calling
// NextExpression() again. Otherwise, calls to various Node methods may panic if
// the parser has moved on the next expression.
//
// For performance reasons, go-toml doesn't make a copy of the input bytes to
// the parser. Make sure to copy all the bytes you need to outlive the slice
// given to the parser.
type Parser struct {
	data    []byte
	builder builder
	ref     reference
	left    []byte
	err     error
	first   bool

	KeepComments bool
}

// Data returns the slice provided to the last call to Reset.
func (p *Parser) Data() []byte {
	return p.data
}

// Range returns a range description that corresponds to a given slice of the
// input. If the argument is not a subslice of the parser input, this function
// panics.
func (p *Parser) Range(b []byte) Range {
	return Range{
		Offset: uint32(danger.SubsliceOffset(p.data, b)),
		Length: uint32(len(b)),
	}
}

// Raw returns the slice corresponding to the bytes in the given range.
func (p *Parser) Raw(raw Range) []byte {
	return p.data[raw.Offset : raw.Offset+raw.Length]
}

// Reset brings the parser to its initial state for a given input. It wipes an
// reuses internal storage to reduce allocation.
func (p *Parser) Reset(b []byte) {
	p.builder.Reset()
	p.ref = invalidReference
	p.data = b
	p.left = b
	p.err = nil
	p.first = true
}

// NextExpression parses the next top-level expression. If an expression was
// successfully parsed, it returns true. If the parser is at the end of the
// document or an error occurred, it returns false.
//
// Retrieve the parsed expression with Expression().
func (p *Parser) NextExpression() bool {
	if len(p.left) == 0 || p.err != nil {
		return false
	}

	p.builder.Reset()
	p.ref = invalidReference

	for {
		if len(p.left) == 0 || p.err != nil {
			return false
		}

		if !p.first {
			p.left, p.err = p.parseNewline(p.left)
		}

		if len(p.left) == 0 || p.err != nil {
			return false
		}

		p.ref, p.left, p.err = p.parseExpression(p.left)

		if p.err != nil {
			return false
		}

		p.first = false

		if p.ref.Valid() {
			return true
		}
	}
}

// Expression returns a pointer to the node representing the last successfully
// parsed expression.
func (p *Parser) Expression() *Node {
	return p.builder.NodeAt(p.ref)
}

// Error returns any error that has occurred during parsing.
func (p *Parser) Error() error {
	return p.err
}

// Position describes a position in the input.
type Position struct {
	// Number of bytes from the beginning of the input.
	Offset int
	// Line number, starting at 1.
	Line int
	// Column number, starting at 1.
	Column int
}

// Shape describes the position of a range in the input.
type Shape struct {
	Start Position
	End   Position
}

func (p *Parser) position(b []byte) Position {
	offset := danger.SubsliceOffset(p.data, b)

	lead := p.data[:offset]

	return Position{
		Offset: offset,
		Line:   bytes.Count(lead, []byte{'\n'}) + 1,
		Column: len(lead) - bytes.LastIndex(lead, []byte{'\n'}),
	}
}

// Shape returns the shape of the given range in the input.  Will
// panic if the range is not a subslice of the input.
func (p *Parser) Shape(r Range) Shape {
	raw := p.Raw(r)
	return Shape{
		Start: p.position(raw),
		End:   p.position(raw[r.Length:]),
	}
}

func (p *Parser) parseNewline(b []byte) ([]byte, error) {
	if b[0] == '\n' {
		return b[1:], nil
	}

	if b[0] == '\r' {
		_, rest, err := scanWindowsNewline(b)
		return rest, err
	}

	return nil, NewParserError(b[0:1], "expected newline but got %#U", b[0])
}

func (p *Parser) parseComment(b []byte) (reference, []byte, error) {
	ref := invalidReference
	data, rest, err := scanComment(b)
	if p.KeepComments && err == nil {
		ref = p.builder.Push(Node{
			Kind: Comment,
			Raw:  p.Range(data),
			Data: data,
		})
	}
	return ref, rest, err
}

func (p *Parser) parseExpression(b []byte) (reference, []byte, error) {
	// expression =  ws [ comment ]
	// expression =/ ws keyval ws [ comment ]
	// expression =/ ws table ws [ comment ]
	ref := invalidReference

	b = p.parseWhitespace(b)

	if len(b) == 0 {
		return ref, b, nil
	}

	if b[0] == '#' {
		ref, rest, err := p.parseComment(b)
		return ref, rest, err
	}

	if b[0] == '\n' || b[0] == '\r' {
		return ref, b, nil
	}

	var err error
	if b[0] == '[' {
		ref, b, err = p.parseTable(b)
	} else {
		ref, b, err = p.parseKeyval(b)
	}

	if err != nil {
		return ref, nil, err
	}

	b = p.parseWhitespace(b)

	if len(b) > 0 && b[0] == '#' {
		cref, rest, err := p.parseComment(b)
		if cref != invalidReference {
			p.builder.Chain(ref, cref)
		}
		return ref, rest, err
	}

	return ref, b, nil
}

func (p *Parser) parseTable(b []byte) (reference, []byte, error) {
	// table = std-table / array-table
	if len(b) > 1 && b[1] == '[' {
		return p.parseArrayTable(b)
	}

	return p.parseStdTable(b)
}

func (p *Parser) parseArrayTable(b []byte) (reference, []byte, error) {
	// array-table = array-table-open key array-table-close
	// array-table-open  = %x5B.5B ws  ; [[ Double left square bracket
	// array-table-close = ws %x5D.5D  ; ]] Double right square bracket
	ref := p.builder.Push(Node{
		Kind: ArrayTable,
	})

	b = b[2:]
	b = p.parseWhitespace(b)

	k, b, err := p.parseKey(b)
	if err != nil {
		return ref, nil, err
	}

	p.builder.AttachChild(ref, k)
	b = p.parseWhitespace(b)

	b, err = expect(']', b)
	if err != nil {
		return ref, nil, err
	}

	b, err = expect(']', b)

	return ref, b, err
}

func (p *Parser) parseStdTable(b []byte) (reference, []byte, error) {
	// std-table = std-table-open key std-table-close
	// std-table-open  = %x5B ws     ; [ Left square bracket
	// std-table-close = ws %x5D     ; ] Right square bracket
	ref := p.builder.Push(Node{
		Kind: Table,
	})

	b = b[1:]
	b = p.parseWhitespace(b)

	key, b, err := p.parseKey(b)
	if err != nil {
		return ref, nil, err
	}

	p.builder.AttachChild(ref, key)

	b = p.parseWhitespace(b)

	b, err = expect(']', b)

	return ref, b, err
}

func (p *Parser) parseKeyval(b []byte) (reference, []byte, error) {
	// keyval = key keyval-sep val
	ref := p.builder.Push(Node{
		Kind: KeyValue,
	})

	key, b, err := p.parseKey(b)
	if err != nil {
		return invalidReference, nil, err
	}

	// keyval-sep = ws %x3D ws ; =

	b = p.parseWhitespace(b)

	if len(b) == 0 {
		return invalidReference, nil, NewParserError(b, "expected = after a key, but the document ends there")
	}

	b, err = expect('=', b)
	if err != nil {
		return invalidReference, nil, err
	}

	b = p.parseWhitespace(b)

	valRef, b, err := p.parseVal(b)
	if err != nil {
		return ref, b, err
	}

	p.builder.Chain(valRef, key)
	p.builder.AttachChild(ref, valRef)

	return ref, b, err
}

//nolint:cyclop,funlen
func (p *Parser) parseVal(b []byte) (reference, []byte, error) {
	// val = string / boolean / array / inline-table / date-time / float / integer
	ref := invalidReference

	if len(b) == 0 {
		return ref, nil, NewParserError(b, "expected value, not eof")
	}

	var err error
	c := b[0]

	switch c {
	case '"':
		var raw []byte
		var v []byte
		if scanFollowsMultilineBasicStringDelimiter(b) {
			raw, v, b, err = p.parseMultilineBasicString(b)
		} else {
			raw, v, b, err = p.parseBasicString(b)
		}

		if err == nil {
			ref = p.builder.Push(Node{
				Kind: String,
				Raw:  p.Range(raw),
				Data: v,
			})
		}

		return ref, b, err
	case '\'':
		var raw []byte
		var v []byte
		if scanFollowsMultilineLiteralStringDelimiter(b) {
			raw, v, b, err = p.parseMultilineLiteralString(b)
		} else {
			raw, v, b, err = p.parseLiteralString(b)
		}

		if err == nil {
			ref = p.builder.Push(Node{
				Kind: String,
				Raw:  p.Range(raw),
				Data: v,
			})
		}

		return ref, b, err
	case 't':
		if !scanFollowsTrue(b) {
			return ref, nil, NewParserError(atmost(b, 4), "expected 'true'")
		}

		ref = p.builder.Push(Node{
			Kind: Bool,
			Data: b[:4],
		})

		return ref, b[4:], nil
	case 'f':
		if !scanFollowsFalse(b) {
			return ref, nil, NewParserError(atmost(b, 5), "expected 'false'")
		}

		ref = p.builder.Push(Node{
			Kind: Bool,
			Data: b[:5],
		})

		return ref, b[5:], nil
	case '[':
		return p.parseValArray(b)
	case '{':
		return p.parseInlineTable(b)
	default:
		return p.parseIntOrFloatOrDateTime(b)
	}
}

func atmost(b []byte, n int) []byte {
	if n >= len(b) {
		return b
	}

	return b[:n]
}

func (p *Parser) parseLiteralString(b []byte) ([]byte, []byte, []byte, error) {
	v, rest, err := scanLiteralString(b)
	if err != nil {
		return nil, nil, nil, err
	}

	return v, v[1 : len(v)-1], rest, nil
}

func (p *Parser) parseInlineTable(b []byte) (reference, []byte, error) {
	// inline-table = inline-table-open [ inline-table-keyvals ] inline-table-close
	// inline-table-open  = %x7B ws     ; {
	// inline-table-close = ws %x7D     ; }
	// inline-table-sep   = ws %x2C ws  ; , Comma
	// inline-table-keyvals = keyval [ inline-table-sep inline-table-keyvals ]
	parent := p.builder.Push(Node{
		Kind: InlineTable,
		Raw:  p.Range(b[:1]),
	})

	first := true

	var child reference

	b = b[1:]

	var err error

	for len(b) > 0 {
		previousB := b
		b = p.parseWhitespace(b)

		if len(b) == 0 {
			return parent, nil, NewParserError(previousB[:1], "inline table is incomplete")
		}

		if b[0] == '}' {
			break
		}

		if !first {
			b, err = expect(',', b)
			if err != nil {
				return parent, nil, err
			}
			b = p.parseWhitespace(b)
		}

		var kv reference

		kv, b, err = p.parseKeyval(b)
		if err != nil {
			return parent, nil, err
		}

		if first {
			p.builder.AttachChild(parent, kv)
		} else {
			p.builder.Chain(child, kv)
		}
		child = kv

		first = false
	}

	rest, err := expect('}', b)

	return parent, rest, err
}

//nolint:funlen,cyclop
func (p *Parser) parseValArray(b []byte) (reference, []byte, error) {
	// array = array-open [ array-values ] ws-comment-newline array-close
	// array-open =  %x5B ; [
	// array-close = %x5D ; ]
	// array-values =  ws-comment-newline val ws-comment-newline array-sep array-values
	// array-values =/ ws-comment-newline val ws-comment-newline [ array-sep ]
	// array-sep = %x2C  ; , Comma
	// ws-comment-newline = *( wschar / [ comment ] newline )
	arrayStart := b
	b = b[1:]

	parent := p.builder.Push(Node{
		Kind: Array,
	})

	// First indicates whether the parser is looking for the first element
	// (non-comment) of the array.
	first := true

	lastChild := invalidReference

	addChild := func(valueRef reference) {
		if lastChild == invalidReference {
			p.builder.AttachChild(parent, valueRef)
		} else {
			p.builder.Chain(lastChild, valueRef)
		}
		lastChild = valueRef
	}

	var err error
	for len(b) > 0 {
		cref := invalidReference
		cref, b, err = p.parseOptionalWhitespaceCommentNewline(b)
		if err != nil {
			return parent, nil, err
		}

		if cref != invalidReference {
			addChild(cref)
		}

		if len(b) == 0 {
			return parent, nil, NewParserError(arrayStart[:1], "array is incomplete")
		}

		if b[0] == ']' {
			break
		}

		if b[0] == ',' {
			if first {
				return parent, nil, NewParserError(b[0:1], "array cannot start with comma")
			}
			b = b[1:]

			cref, b, err = p.parseOptionalWhitespaceCommentNewline(b)
			if err != nil {
				return parent, nil, err
			}
			if cref != invalidReference {
				addChild(cref)
			}
		} else if !first {
			return parent, nil, NewParserError(b[0:1], "array elements must be separated by commas")
		}

		// TOML allows trailing commas in arrays.
		if len(b) > 0 && b[0] == ']' {
			break
		}

		var valueRef reference
		valueRef, b, err = p.parseVal(b)
		if err != nil {
			return parent, nil, err
		}

		addChild(valueRef)

		cref, b, err = p.parseOptionalWhitespaceCommentNewline(b)
		if err != nil {
			return parent, nil, err
		}
		if cref != invalidReference {
			addChild(cref)
		}

		first = false
	}

	rest, err := expect(']', b)

	return parent, rest, err
}

func (p *Parser) parseOptionalWhitespaceCommentNewline(b []byte) (reference, []byte, error) {
	rootCommentRef := invalidReference
	latestCommentRef := invalidReference

	addComment := func(ref reference) {
		if rootCommentRef == invalidReference {
			rootCommentRef = ref
		} else if latestCommentRef == invalidReference {
			p.builder.AttachChild(rootCommentRef, ref)
			latestCommentRef = ref
		} else {
			p.builder.Chain(latestCommentRef, ref)
			latestCommentRef = ref
		}
	}

	for len(b) > 0 {
		var err error
		b = p.parseWhitespace(b)

		if len(b) > 0 && b[0] == '#' {
			var ref reference
			ref, b, err = p.parseComment(b)
			if err != nil {
				return invalidReference, nil, err
			}
			if ref != invalidReference {
				addComment(ref)
			}
		}

		if len(b) == 0 {
			break
		}

		if b[0] == '\n' || b[0] == '\r' {
			b, err = p.parseNewline(b)
			if err != nil {
				return invalidReference, nil, err
			}
		} else {
			break
		}
	}

	return rootCommentRef, b, nil
}

func (p *Parser) parseMultilineLiteralString(b []byte) ([]byte, []byte, []byte, error) {
	token, rest, err := scanMultilineLiteralString(b)
	if err != nil {
		return nil, nil, nil, err
	}

	i := 3

	// skip the immediate new line
	if token[i] == '\n' {
		i++
	} else if token[i] == '\r' && token[i+1] == '\n' {
		i += 2
	}

	return token, token[i : len(token)-3], rest, err
}

//nolint:funlen,gocognit,cyclop
func (p *Parser) parseMultilineBasicString(b []byte) ([]byte, []byte, []byte, error) {
	// ml-basic-string = ml-basic-string-delim [ newline ] ml-basic-body
	// ml-basic-string-delim
	// ml-basic-string-delim = 3quotation-mark
	// ml-basic-body = *mlb-content *( mlb-quotes 1*mlb-content ) [ mlb-quotes ]
	//
	// mlb-content = mlb-char / newline / mlb-escaped-nl
	// mlb-char = mlb-unescaped / escaped
	// mlb-quotes = 1*2quotation-mark
	// mlb-unescaped = wschar / %x21 / %x23-5B / %x5D-7E / non-ascii
	// mlb-escaped-nl = escape ws newline *( wschar / newline )
	token, escaped, rest, err := scanMultilineBasicString(b)
	if err != nil {
		return nil, nil, nil, err
	}

	i := 3

	// skip the immediate new line
	if token[i] == '\n' {
		i++
	} else if token[i] == '\r' && token[i+1] == '\n' {
		i += 2
	}

	// fast path
	startIdx := i
	endIdx := len(token) - len(`"""`)

	if !escaped {
		str := token[startIdx:endIdx]
		verr := characters.Utf8TomlValidAlreadyEscaped(str)
		if verr.Zero() {
			return token, str, rest, nil
		}
		return nil, nil, nil, NewParserError(str[verr.Index:verr.Index+verr.Size], "invalid UTF-8")
	}

	var builder bytes.Buffer

	// The scanner ensures that the token starts and ends with quotes and that
	// escapes are balanced.
	for i < len(token)-3 {
		c := token[i]

		//nolint:nestif
		if c == '\\' {
			// When the last non-whitespace character on a line is an unescaped \,
			// it will be trimmed along with all whitespace (including newlines) up
			// to the next non-whitespace character or closing delimiter.

			isLastNonWhitespaceOnLine := false
			j := 1
		findEOLLoop:
			for ; j < len(token)-3-i; j++ {
				switch token[i+j] {
				case ' ', '\t':
					continue
				case '\r':
					if token[i+j+1] == '\n' {
						continue
					}
				case '\n':
					isLastNonWhitespaceOnLine = true
				}
				break findEOLLoop
			}
			if isLastNonWhitespaceOnLine {
				i += j
				for ; i < len(token)-3; i++ {
					c := token[i]
					if !(c == '\n' || c == '\r' || c == ' ' || c == '\t') {
						i--
						break
					}
				}
				i++
				continue
			}

			// handle escaping
			i++
			c = token[i]

			switch c {
			case '"', '\\':
				builder.WriteByte(c)
			case 'b':
				builder.WriteByte('\b')
			case 'f':
				builder.WriteByte('\f')
			case 'n':
				builder.WriteByte('\n')
			case 'r':
				builder.WriteByte('\r')
			case 't':
				builder.WriteByte('\t')
			case 'e':
				builder.WriteByte(0x1B)
			case 'u':
				x, err := hexToRune(atmost(token[i+1:], 4), 4)
				if err != nil {
					return nil, nil, nil, err
				}
				builder.WriteRune(x)
				i += 4
			case 'U':
				x, err := hexToRune(atmost(token[i+1:], 8), 8)
				if err != nil {
					return nil, nil, nil, err
				}

				builder.WriteRune(x)
				i += 8
			default:
				return nil, nil, nil, NewParserError(token[i:i+1], "invalid escaped character %#U", c)
			}
			i++
		} else {
			size := characters.Utf8ValidNext(token[i:])
			if size == 0 {
				return nil, nil, nil, NewParserError(token[i:i+1], "invalid character %#U", c)
			}
			builder.Write(token[i : i+size])
			i += size
		}
	}

	return token, builder.Bytes(), rest, nil
}

func (p *Parser) parseKey(b []byte) (reference, []byte, error) {
	// key = simple-key / dotted-key
	// simple-key = quoted-key / unquoted-key
	//
	// unquoted-key = 1*( ALPHA / DIGIT / %x2D / %x5F ) ; A-Z / a-z / 0-9 / - / _
	// quoted-key = basic-string / literal-string
	// dotted-key = simple-key 1*( dot-sep simple-key )
	//
	// dot-sep   = ws %x2E ws  ; . Period
	raw, key, b, err := p.parseSimpleKey(b)
	if err != nil {
		return invalidReference, nil, err
	}

	ref := p.builder.Push(Node{
		Kind: Key,
		Raw:  p.Range(raw),
		Data: key,
	})

	for {
		b = p.parseWhitespace(b)
		if len(b) > 0 && b[0] == '.' {
			b = p.parseWhitespace(b[1:])

			raw, key, b, err = p.parseSimpleKey(b)
			if err != nil {
				return ref, nil, err
			}

			p.builder.PushAndChain(Node{
				Kind: Key,
				Raw:  p.Range(raw),
				Data: key,
			})
		} else {
			break
		}
	}

	return ref, b, nil
}

func (p *Parser) parseSimpleKey(b []byte) (raw, key, rest []byte, err error) {
	if len(b) == 0 {
		return nil, nil, nil, NewParserError(b, "expected key but found none")
	}

	// simple-key = quoted-key / unquoted-key
	// unquoted-key = 1*( ALPHA / DIGIT / %x2D / %x5F ) ; A-Z / a-z / 0-9 / - / _
	// quoted-key = basic-string / literal-string
	switch {
	case b[0] == '\'':
		return p.parseLiteralString(b)
	case b[0] == '"':
		return p.parseBasicString(b)
	case isUnquotedKeyChar(b[0]):
		key, rest = scanUnquotedKey(b)
		return key, key, rest, nil
	default:
		return nil, nil, nil, NewParserError(b[0:1], "invalid character at start of key: %c", b[0])
	}
}

//nolint:funlen,cyclop
func (p *Parser) parseBasicString(b []byte) ([]byte, []byte, []byte, error) {
	// basic-string = quotation-mark *basic-char quotation-mark
	// quotation-mark = %x22            ; "
	// basic-char = basic-unescaped / escaped
	// basic-unescaped = wschar / %x21 / %x23-5B / %x5D-7E / non-ascii
	// escaped = escape escape-seq-char
	// escape-seq-char =  %x22         ; "    quotation mark  U+0022
	// escape-seq-char =/ %x5C         ; \    reverse solidus U+005C
	// escape-seq-char =/ %x62         ; b    backspace       U+0008
	// escape-seq-char =/ %x66         ; f    form feed       U+000C
	// escape-seq-char =/ %x6E         ; n    line feed       U+000A
	// escape-seq-char =/ %x72         ; r    carriage return U+000D
	// escape-seq-char =/ %x74         ; t    tab             U+0009
	// escape-seq-char =/ %x75 4HEXDIG ; uXXXX                U+XXXX
	// escape-seq-char =/ %x55 8HEXDIG ; UXXXXXXXX            U+XXXXXXXX
	token, escaped, rest, err := scanBasicString(b)
	if err != nil {
		return nil, nil, nil, err
	}

	startIdx := len(`"`)
	endIdx := len(token) - len(`"`)

	// Fast path. If there is no escape sequence, the string should just be
	// an UTF-8 encoded string, which is the same as Go. In that case,
	// validate the string and return a direct reference to the buffer.
	if !escaped {
		str := token[startIdx:endIdx]
		verr := characters.Utf8TomlValidAlreadyEscaped(str)
		if verr.Zero() {
			return token, str, rest, nil
		}
		return nil, nil, nil, NewParserError(str[verr.Index:verr.Index+verr.Size], "invalid UTF-8")
	}

	i := startIdx

	var builder bytes.Buffer

	// The scanner ensures that the token starts and ends with quotes and that
	// escapes are balanced.
	for i < len(token)-1 {
		c := token[i]
		if c == '\\' {
			i++
			c = token[i]

			switch c {
			case '"', '\\':
				builder.WriteByte(c)
			case 'b':
				builder.WriteByte('\b')
			case 'f':
				builder.WriteByte('\f')
			case 'n':
				builder.WriteByte('\n')
			case 'r':
				builder.WriteByte('\r')
			case 't':
				builder.WriteByte('\t')
			case 'e':
				builder.WriteByte(0x1B)
			case 'u':
				x, err := hexToRune(token[i+1:len(token)-1], 4)
				if err != nil {
					return nil, nil, nil, err
				}

				builder.WriteRune(x)
				i += 4
			case 'U':
				x, err := hexToRune(token[i+1:len(token)-1], 8)
				if err != nil {
					return nil, nil, nil, err
				}

				builder.WriteRune(x)
				i += 8
			default:
				return nil, nil, nil, NewParserError(token[i:i+1], "invalid escaped character %#U", c)
			}
			i++
		} else {
			size := characters.Utf8ValidNext(token[i:])
			if size == 0 {
				return nil, nil, nil, NewParserError(token[i:i+1], "invalid character %#U", c)
			}
			builder.Write(token[i : i+size])
			i += size
		}
	}

	return token, builder.Bytes(), rest, nil
}

func hexToRune(b []byte, length int) (rune, error) {
	if len(b) < length {
		return -1, NewParserError(b, "unicode point needs %d character, not %d", length, len(b))
	}
	b = b[:length]

	var r uint32
	for i, c := range b {
		d := uint32(0)
		switch {
		case '0' <= c && c <= '9':
			d = uint32(c - '0')
		case 'a' <= c && c <= 'f':
			d = uint32(c - 'a' + 10)
		case 'A' <= c && c <= 'F':
			d = uint32(c - 'A' + 10)
		default:
			return -1, NewParserError(b[i:i+1], "non-hex character")
		}
		r = r*16 + d
	}

	if r > unicode.MaxRune || 0xD800 <= r && r < 0xE000 {
		return -1, NewParserError(b, "escape sequence is invalid Unicode code point")
	}

	return rune(r), nil
}

func (p *Parser) parseWhitespace(b []byte) []byte {
	// ws = *wschar
	// wschar =  %x20  ; Space
	// wschar =/ %x09  ; Horizontal tab
	_, rest := scanWhitespace(b)

	return rest
}

//nolint:cyclop
func (p *Parser) parseIntOrFloatOrDateTime(b []byte) (reference, []byte, error) {
	switch b[0] {
	case 'i':
		if !scanFollowsInf(b) {
			return invalidReference, nil, NewParserError(atmost(b, 3), "expected 'inf'")
		}

		return p.builder.Push(Node{
			Kind: Float,
			Data: b[:3],
			Raw:  p.Range(b[:3]),
		}), b[3:], nil
	case 'n':
		if !scanFollowsNan(b) {
			return invalidReference, nil, NewParserError(atmost(b, 3), "expected 'nan'")
		}

		return p.builder.Push(Node{
			Kind: Float,
			Data: b[:3],
			Raw:  p.Range(b[:3]),
		}), b[3:], nil
	case '+', '-':
		return p.scanIntOrFloat(b)
	}

	if len(b) < 3 {
		return p.scanIntOrFloat(b)
	}

	s := 5
	if len(b) < s {
		s = len(b)
	}

	for idx, c := range b[:s] {
		if isDigit(c) {
			continue
		}

		if idx == 2 && c == ':' || (idx == 4 && c == '-') {
			return p.scanDateTime(b)
		}

		break
	}

	return p.scanIntOrFloat(b)
}

func (p *Parser) scanDateTime(b []byte) (reference, []byte, error) {
	// scans for contiguous characters in [0-9T:Z.+-], and up to one space if
	// followed by a digit.
	hasDate := false
	hasTime := false
	hasTz := false
	seenSpace := false

	i := 0
byteLoop:
	for ; i < len(b); i++ {
		c := b[i]

		switch {
		case isDigit(c):
		case c == '-':
			hasDate = true
			const minOffsetOfTz = 8
			if i >= minOffsetOfTz {
				hasTz = true
			}
		case c == 'T' || c == 't' || c == ':' || c == '.':
			hasTime = true
		case c == '+' || c == '-' || c == 'Z' || c == 'z':
			hasTz = true
		case c == ' ':
			if !seenSpace && i+1 < len(b) && isDigit(b[i+1]) {
				i += 2
				// Avoid reaching past the end of the document in case the time
				// is malformed. See TestIssue585.
				if i >= len(b) {
					i--
				}
				seenSpace = true
				hasTime = true
			} else {
				break byteLoop
			}
		default:
			break byteLoop
		}
	}

	var kind Kind

	if hasTime {
		if hasDate {
			if hasTz {
				kind = DateTime
			} else {
				kind = LocalDateTime
			}
		} else {
			kind = LocalTime
		}
	} else {
		kind = LocalDate
	}

	return p.builder.Push(Node{
		Kind: kind,
		Data: b[:i],
	}), b[i:], nil
}

//nolint:funlen,gocognit,cyclop
func (p *Parser) scanIntOrFloat(b []byte) (reference, []byte, error) {
	i := 0

	if len(b) > 2 && b[0] == '0' && b[1] != '.' && b[1] != 'e' && b[1] != 'E' {
		var isValidRune validRuneFn

		switch b[1] {
		case 'x':
			isValidRune = isValidHexRune
		case 'o':
			isValidRune = isValidOctalRune
		case 'b':
			isValidRune = isValidBinaryRune
		default:
			i++
		}

		if isValidRune != nil {
			i += 2
			for ; i < len(b); i++ {
				if !isValidRune(b[i]) {
					break
				}
			}
		}

		return p.builder.Push(Node{
			Kind: Integer,
			Data: b[:i],
			Raw:  p.Range(b[:i]),
		}), b[i:], nil
	}

	isFloat := false

	for ; i < len(b); i++ {
		c := b[i]

		if c >= '0' && c <= '9' || c == '+' || c == '-' || c == '_' {
			continue
		}

		if c == '.' || c == 'e' || c == 'E' {
			isFloat = true

			continue
		}

		if c == 'i' {
			if scanFollowsInf(b[i:]) {
				return p.builder.Push(Node{
					Kind: Float,
					Data: b[:i+3],
					Raw:  p.Range(b[:i+3]),
				}), b[i+3:], nil
			}

			return invalidReference, nil, NewParserError(b[i:i+1], "unexpected character 'i' while scanning for a number")
		}

		if c == 'n' {
			if scanFollowsNan(b[i:]) {
				return p.builder.Push(Node{
					Kind: Float,
					Data: b[:i+3],
					Raw:  p.Range(b[:i+3]),
				}), b[i+3:], nil
			}

			return invalidReference, nil, NewParserError(b[i:i+1], "unexpected character 'n' while scanning for a number")
		}

		break
	}

	if i == 0 {
		return invalidReference, b, NewParserError(b, "incomplete number")
	}

	kind := Integer

	if isFloat {
		kind = Float
	}

	return p.builder.Push(Node{
		Kind: kind,
		Data: b[:i],
		Raw:  p.Range(b[:i]),
	}), b[i:], nil
}

func isDigit(r byte) bool {
	return r >= '0' && r <= '9'
}

type validRuneFn func(r byte) bool

func isValidHexRune(r byte) bool {
	return r >= 'a' && r <= 'f' ||
		r >= 'A' && r <= 'F' ||
		r >= '0' && r <= '9' ||
		r == '_'
}

func isValidOctalRune(r byte) bool {
	return r >= '0' && r <= '7' || r == '_'
}

func isValidBinaryRune(r byte) bool {
	return r == '0' || r == '1' || r == '_'
}

func expect(x byte, b []byte) ([]byte, error) {
	if len(b) == 0 {
		return nil, NewParserError(b, "expected character %c but the document ended here", x)
	}

	if b[0] != x {
		return nil, NewParserError(b[0:1], "expected character %c", x)
	}

	return b[1:], nil
}
