package redis

import (
	"context"
	"encoding/json"
	"strings"

	"github.com/redis/go-redis/v9/internal/proto"
	"github.com/redis/go-redis/v9/internal/util"
)

// -------------------------------------------

type JSONCmdable interface {
	JSONArrAppend(ctx context.Context, key, path string, values ...interface{}) *IntSliceCmd
	JSONArrIndex(ctx context.Context, key, path string, value ...interface{}) *IntSliceCmd
	JSONArrIndexWithArgs(ctx context.Context, key, path string, options *JSONArrIndexArgs, value ...interface{}) *IntSliceCmd
	JSONArrInsert(ctx context.Context, key, path string, index int64, values ...interface{}) *IntSliceCmd
	JSONArrLen(ctx context.Context, key, path string) *IntSliceCmd
	JSONArrPop(ctx context.Context, key, path string, index int) *StringSliceCmd
	JSONArrTrim(ctx context.Context, key, path string) *IntSliceCmd
	JSONArrTrimWithArgs(ctx context.Context, key, path string, options *JSONArrTrimArgs) *IntSliceCmd
	JSONClear(ctx context.Context, key, path string) *IntCmd
	JSONDebugMemory(ctx context.Context, key, path string) *IntCmd
	JSONDel(ctx context.Context, key, path string) *IntCmd
	JSONForget(ctx context.Context, key, path string) *IntCmd
	JSONGet(ctx context.Context, key string, paths ...string) *JSONCmd
	JSONGetWithArgs(ctx context.Context, key string, options *JSONGetArgs, paths ...string) *JSONCmd
	JSONMerge(ctx context.Context, key, path string, value string) *StatusCmd
	JSONMSetArgs(ctx context.Context, docs []JSONSetArgs) *StatusCmd
	JSONMSet(ctx context.Context, params ...interface{}) *StatusCmd
	JSONMGet(ctx context.Context, path string, keys ...string) *JSONSliceCmd
	JSONNumIncrBy(ctx context.Context, key, path string, value float64) *JSONCmd
	JSONObjKeys(ctx context.Context, key, path string) *SliceCmd
	JSONObjLen(ctx context.Context, key, path string) *IntPointerSliceCmd
	JSONSet(ctx context.Context, key, path string, value interface{}) *StatusCmd
	JSONSetMode(ctx context.Context, key, path string, value interface{}, mode string) *StatusCmd
	JSONSetWithArgs(ctx context.Context, key, path string, value interface{}, options *JSONSetArgsOptions) *StatusCmd
	JSONStrAppend(ctx context.Context, key, path, value string) *IntPointerSliceCmd
	JSONStrLen(ctx context.Context, key, path string) *IntPointerSliceCmd
	JSONToggle(ctx context.Context, key, path string) *IntPointerSliceCmd
	JSONType(ctx context.Context, key, path string) *JSONSliceCmd
}

type JSONSetArgs struct {
	Key   string
	Path  string
	Value interface{}
}

type JSONArrIndexArgs struct {
	Start int
	Stop  *int
}

type JSONArrTrimArgs struct {
	Start int
	Stop  *int
}

// FPHAType is the floating-point type used for storing FP homogeneous arrays
// in JSON.SET (Redis 8.8+).
type FPHAType string

const (
	FPHATypeBF16 FPHAType = "BF16"
	FPHATypeFP16 FPHAType = "FP16"
	FPHATypeFP32 FPHAType = "FP32"
	FPHATypeFP64 FPHAType = "FP64"
)

// JSONSetArgsOptions are the optional arguments for JSONSetWithArgs.
// Mode is "NX" or "XX" (case-insensitive). FPHA, when set, forces Redis to
// store all FP homogeneous arrays using the specified floating-point type.
type JSONSetArgsOptions struct {
	Mode string
	FPHA FPHAType
}

type JSONCmd struct {
	baseCmd
	val      string
	expanded interface{}
}

var _ Cmder = (*JSONCmd)(nil)

func newJSONCmd(ctx context.Context, args ...interface{}) *JSONCmd {
	return &JSONCmd{
		baseCmd: baseCmd{
			ctx:     ctx,
			args:    args,
			cmdType: CmdTypeJSON,
		},
	}
}

func (cmd *JSONCmd) String() string {
	cmd.await()
	return cmdString(cmd, cmd.val)
}

func (cmd *JSONCmd) SetVal(val string) {
	cmd.val = val
}

// Val returns the result of the JSON.GET command as a string.
func (cmd *JSONCmd) Val() string {
	cmd.await()
	if len(cmd.val) == 0 && cmd.expanded != nil {
		val, err := json.Marshal(cmd.expanded)
		if err != nil {
			cmd.SetErr(err)
			return ""
		}
		return string(val)

	} else {
		return cmd.val
	}
}

func (cmd *JSONCmd) Result() (string, error) {
	cmd.await()
	return cmd.Val(), cmd.Err()
}

// Expanded returns the result of the JSON.GET command as unmarshalled JSON.
func (cmd *JSONCmd) Expanded() (interface{}, error) {
	cmd.await()
	if len(cmd.val) != 0 && cmd.expanded == nil {
		err := json.Unmarshal([]byte(cmd.val), &cmd.expanded)
		if err != nil {
			return nil, err
		}
	}

	return cmd.expanded, nil
}

func (cmd *JSONCmd) readReply(rd *proto.Reader) error {
	// nil response from JSON.(M)GET (cmd.baseCmd.err will be "redis: nil")
	// This happens when the key doesn't exist.
	// Use rawErr() (not Err()): readReply runs inside the batch's Exec, before
	// the autopipeline batch's done channel is closed, so Err()->await() would
	// deadlock on the very Exec that is calling readReply.
	if cmd.baseCmd.rawErr() == Nil {
		cmd.val = ""
		return Nil
	}

	// Handle other base command errors
	if cmd.baseCmd.rawErr() != nil {
		return cmd.baseCmd.rawErr()
	}

	if readType, err := rd.PeekReplyType(); err != nil {
		return err
	} else if readType == proto.RespArray {

		size, err := rd.ReadArrayLen()
		if err != nil {
			return err
		}

		// Empty array means no results found for JSON path, but key exists
		// This should return "[]", not an error
		if size == 0 {
			cmd.val = "[]"
			return nil
		}

		expanded := make([]interface{}, size)

		for i := 0; i < size; i++ {
			if expanded[i], err = rd.ReadReply(); err != nil {
				return err
			}
		}
		cmd.expanded = expanded

	} else {
		if str, err := rd.ReadString(); err != nil && err != Nil {
			return err
		} else if str == "" || err == Nil {
			cmd.val = ""
			return Nil
		} else {
			cmd.val = str
		}
	}

	return nil
}

func (cmd *JSONCmd) Clone() Cmder {
	return &JSONCmd{
		baseCmd:  cmd.cloneBaseCmd(),
		val:      cmd.val,
		expanded: cmd.expanded, // interface{} can be shared as it should be immutable after parsing
	}
}

// -------------------------------------------

type JSONSliceCmd struct {
	baseCmd
	val []interface{}
}

func NewJSONSliceCmd(ctx context.Context, args ...interface{}) *JSONSliceCmd {
	return &JSONSliceCmd{
		baseCmd: baseCmd{
			ctx:     ctx,
			args:    args,
			cmdType: CmdTypeJSONSlice,
		},
	}
}

func (cmd *JSONSliceCmd) String() string {
	cmd.await()
	return cmdString(cmd, cmd.val)
}

func (cmd *JSONSliceCmd) SetVal(val []interface{}) {
	cmd.val = val
}

func (cmd *JSONSliceCmd) Val() []interface{} {
	cmd.await()
	return cmd.val
}

func (cmd *JSONSliceCmd) Result() ([]interface{}, error) {
	cmd.await()
	return cmd.val, cmd.err
}

func (cmd *JSONSliceCmd) readReply(rd *proto.Reader) error {
	// rawErr(), not Err(): readReply runs inside Exec before the batch's done
	// channel closes, so Err()->await() would deadlock (see JSONCmd.readReply).
	if cmd.baseCmd.rawErr() == Nil {
		cmd.val = nil
		return Nil
	}

	if readType, err := rd.PeekReplyType(); err != nil {
		return err
	} else if readType == proto.RespArray {
		response, err := rd.ReadReply()
		if err != nil {
			return err
		} else {
			cmd.val = response.([]interface{})
		}

	} else {
		n, err := rd.ReadArrayLen()
		if err != nil {
			return err
		}
		cmd.val = make([]interface{}, n)
		for i := 0; i < len(cmd.val); i++ {
			switch s, err := rd.ReadString(); {
			case err == Nil:
				cmd.val[i] = ""
			case err != nil:
				return err
			default:
				cmd.val[i] = s
			}
		}
	}
	return nil
}

func (cmd *JSONSliceCmd) Clone() Cmder {
	var val []interface{}
	if cmd.val != nil {
		val = make([]interface{}, len(cmd.val))
		copy(val, cmd.val)
	}
	return &JSONSliceCmd{
		baseCmd: cmd.cloneBaseCmd(),
		val:     val,
	}
}

/*******************************************************************************
*
* IntPointerSliceCmd
* used to represent a RedisJSON response where the result is either an integer or nil
*
*******************************************************************************/

type IntPointerSliceCmd struct {
	baseCmd
	val []*int64
}

// NewIntPointerSliceCmd initialises an IntPointerSliceCmd
func NewIntPointerSliceCmd(ctx context.Context, args ...interface{}) *IntPointerSliceCmd {
	return &IntPointerSliceCmd{
		baseCmd: baseCmd{
			ctx:     ctx,
			args:    args,
			cmdType: CmdTypeIntPointerSlice,
		},
	}
}

func (cmd *IntPointerSliceCmd) String() string {
	cmd.await()
	return cmdString(cmd, cmd.val)
}

func (cmd *IntPointerSliceCmd) SetVal(val []*int64) {
	cmd.val = val
}

func (cmd *IntPointerSliceCmd) Val() []*int64 {
	cmd.await()
	return cmd.val
}

func (cmd *IntPointerSliceCmd) Result() ([]*int64, error) {
	cmd.await()
	return cmd.val, cmd.err
}

func (cmd *IntPointerSliceCmd) readReply(rd *proto.Reader) error {
	n, err := rd.ReadArrayLen()
	if err != nil {
		return err
	}
	cmd.val = make([]*int64, n)

	for i := 0; i < len(cmd.val); i++ {
		val, err := rd.ReadInt()
		if err != nil && err != Nil {
			return err
		} else if err != Nil {
			cmd.val[i] = &val
		}
	}

	return nil
}

func (cmd *IntPointerSliceCmd) Clone() Cmder {
	var val []*int64
	if cmd.val != nil {
		val = make([]*int64, len(cmd.val))
		copy(val, cmd.val)
	}
	return &IntPointerSliceCmd{
		baseCmd: cmd.cloneBaseCmd(),
		val:     val,
	}
}

//------------------------------------------------------------------------------

// JSONArrAppend adds the provided JSON values to the end of the array at the given path.
// For more information, see https://redis.io/commands/json.arrappend
func (c cmdable) JSONArrAppend(ctx context.Context, key, path string, values ...interface{}) *IntSliceCmd {
	args := []interface{}{"JSON.ARRAPPEND", key, path}
	args = append(args, values...)
	cmd := NewIntSliceCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// JSONArrIndex searches for the first occurrence of the provided JSON value in the array at the given path.
// For more information, see https://redis.io/commands/json.arrindex
func (c cmdable) JSONArrIndex(ctx context.Context, key, path string, value ...interface{}) *IntSliceCmd {
	args := []interface{}{"JSON.ARRINDEX", key, path}
	args = append(args, value...)
	cmd := NewIntSliceCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// JSONArrIndexWithArgs searches for the first occurrence of a JSON value in an array while allowing the start and
// stop options to be provided.
// For more information, see https://redis.io/commands/json.arrindex
func (c cmdable) JSONArrIndexWithArgs(ctx context.Context, key, path string, options *JSONArrIndexArgs, value ...interface{}) *IntSliceCmd {
	args := []interface{}{"JSON.ARRINDEX", key, path}
	args = append(args, value...)

	if options != nil {
		args = append(args, options.Start)
		if options.Stop != nil {
			args = append(args, *options.Stop)
		}
	}
	cmd := NewIntSliceCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// JSONArrInsert inserts the JSON values into the array at the specified path before the index (shifts to the right).
// For more information, see https://redis.io/commands/json.arrinsert
func (c cmdable) JSONArrInsert(ctx context.Context, key, path string, index int64, values ...interface{}) *IntSliceCmd {
	args := []interface{}{"JSON.ARRINSERT", key, path, index}
	args = append(args, values...)
	cmd := NewIntSliceCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// JSONArrLen reports the length of the JSON array at the specified path in the given key.
// For more information, see https://redis.io/commands/json.arrlen
func (c cmdable) JSONArrLen(ctx context.Context, key, path string) *IntSliceCmd {
	args := []interface{}{"JSON.ARRLEN", key, path}
	cmd := NewIntSliceCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// JSONArrPop removes and returns an element from the specified index in the array.
// For more information, see https://redis.io/commands/json.arrpop
func (c cmdable) JSONArrPop(ctx context.Context, key, path string, index int) *StringSliceCmd {
	args := []interface{}{"JSON.ARRPOP", key, path, index}
	cmd := NewStringSliceCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// JSONArrTrim trims an array to contain only the specified inclusive range of elements.
// For more information, see https://redis.io/commands/json.arrtrim
func (c cmdable) JSONArrTrim(ctx context.Context, key, path string) *IntSliceCmd {
	args := []interface{}{"JSON.ARRTRIM", key, path}
	cmd := NewIntSliceCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// JSONArrTrimWithArgs trims an array to contain only the specified inclusive range of elements.
// For more information, see https://redis.io/commands/json.arrtrim
func (c cmdable) JSONArrTrimWithArgs(ctx context.Context, key, path string, options *JSONArrTrimArgs) *IntSliceCmd {
	args := []interface{}{"JSON.ARRTRIM", key, path}

	if options != nil {
		args = append(args, options.Start)

		if options.Stop != nil {
			args = append(args, *options.Stop)
		}
	}
	cmd := NewIntSliceCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// JSONClear clears container values (arrays/objects) and sets numeric values to 0.
// For more information, see https://redis.io/commands/json.clear
func (c cmdable) JSONClear(ctx context.Context, key, path string) *IntCmd {
	args := []interface{}{"JSON.CLEAR", key, path}
	cmd := NewIntCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// JSONDebugMemory reports a value's memory usage in bytes (unimplemented)
// For more information, see https://redis.io/commands/json.debug-memory
func (c cmdable) JSONDebugMemory(ctx context.Context, key, path string) *IntCmd {
	panic("not implemented")
}

// JSONDel deletes a value.
// For more information, see https://redis.io/commands/json.del
func (c cmdable) JSONDel(ctx context.Context, key, path string) *IntCmd {
	args := []interface{}{"JSON.DEL", key, path}
	cmd := NewIntCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// JSONForget deletes a value.
// For more information, see https://redis.io/commands/json.forget
func (c cmdable) JSONForget(ctx context.Context, key, path string) *IntCmd {
	args := []interface{}{"JSON.FORGET", key, path}
	cmd := NewIntCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// JSONGet returns the value at path in JSON serialized form. JSON.GET returns an
// array of strings. This function parses out the wrapping array but leaves the
// internal strings unprocessed by default (see Val())
// For more information - https://redis.io/commands/json.get/
func (c cmdable) JSONGet(ctx context.Context, key string, paths ...string) *JSONCmd {
	args := make([]interface{}, len(paths)+2)
	args[0] = "JSON.GET"
	args[1] = key
	for n, path := range paths {
		args[n+2] = path
	}
	cmd := newJSONCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

type JSONGetArgs struct {
	Indent  string
	Newline string
	Space   string
}

// JSONGetWithArgs - Retrieves the value of a key from a JSON document.
// This function also allows for specifying additional options such as:
// Indention, NewLine and Space
// For more information - https://redis.io/commands/json.get/
func (c cmdable) JSONGetWithArgs(ctx context.Context, key string, options *JSONGetArgs, paths ...string) *JSONCmd {
	args := []interface{}{"JSON.GET", key}
	if options != nil {
		if options.Indent != "" {
			args = append(args, "INDENT", options.Indent)
		}
		if options.Newline != "" {
			args = append(args, "NEWLINE", options.Newline)
		}
		if options.Space != "" {
			args = append(args, "SPACE", options.Space)
		}
		for _, path := range paths {
			args = append(args, path)
		}
	}
	cmd := newJSONCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// JSONMerge merges a given JSON value into matching paths.
// For more information, see https://redis.io/commands/json.merge
func (c cmdable) JSONMerge(ctx context.Context, key, path string, value string) *StatusCmd {
	args := []interface{}{"JSON.MERGE", key, path, value}
	cmd := NewStatusCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// JSONMGet returns the values at the specified path from multiple key arguments.
// Note - the arguments are reversed when compared with `JSON.MGET` as we want
// to follow the pattern of having the last argument be variable.
// For more information, see https://redis.io/commands/json.mget
func (c cmdable) JSONMGet(ctx context.Context, path string, keys ...string) *JSONSliceCmd {
	args := make([]interface{}, len(keys)+1)
	args[0] = "JSON.MGET"
	for n, key := range keys {
		args[n+1] = key
	}
	args = append(args, path)
	cmd := NewJSONSliceCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// JSONMSetArgs sets or updates one or more JSON values according to the specified key-path-value triplets.
// For more information, see https://redis.io/commands/json.mset
func (c cmdable) JSONMSetArgs(ctx context.Context, docs []JSONSetArgs) *StatusCmd {
	args := []interface{}{"JSON.MSET"}
	for _, doc := range docs {
		args = append(args, doc.Key, doc.Path, doc.Value)
	}
	cmd := NewStatusCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) JSONMSet(ctx context.Context, params ...interface{}) *StatusCmd {
	args := []interface{}{"JSON.MSET"}
	args = append(args, params...)
	cmd := NewStatusCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// JSONNumIncrBy increments the number value stored at the specified path by the provided number.
// For more information, see https://redis.io/docs/latest/commands/json.numincrby/
func (c cmdable) JSONNumIncrBy(ctx context.Context, key, path string, value float64) *JSONCmd {
	args := []interface{}{"JSON.NUMINCRBY", key, path, value}
	cmd := newJSONCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// JSONObjKeys returns the keys in the object that's referenced by the specified path.
// For more information, see https://redis.io/commands/json.objkeys
func (c cmdable) JSONObjKeys(ctx context.Context, key, path string) *SliceCmd {
	args := []interface{}{"JSON.OBJKEYS", key, path}
	cmd := NewSliceCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// JSONObjLen reports the number of keys in the JSON object at the specified path in the given key.
// For more information, see https://redis.io/commands/json.objlen
func (c cmdable) JSONObjLen(ctx context.Context, key, path string) *IntPointerSliceCmd {
	args := []interface{}{"JSON.OBJLEN", key, path}
	cmd := NewIntPointerSliceCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// JSONSet sets the JSON value at the given path in the given key. The value must be something that
// can be marshaled to JSON (using encoding/JSON) unless the argument is a string or a []byte when we assume that
// it can be passed directly as JSON.
// For more information, see https://redis.io/commands/json.set
func (c cmdable) JSONSet(ctx context.Context, key, path string, value interface{}) *StatusCmd {
	return c.JSONSetMode(ctx, key, path, value, "")
}

// JSONSetMode sets the JSON value at the given path in the given key and allows the mode to be set
// (the mode value must be "XX" or "NX"). The value must be something that can be marshaled to JSON (using encoding/JSON) unless
// the argument is a string or []byte when we assume that it can be passed directly as JSON.
// For more information, see https://redis.io/commands/json.set
func (c cmdable) JSONSetMode(ctx context.Context, key, path string, value interface{}, mode string) *StatusCmd {
	return c.JSONSetWithArgs(ctx, key, path, value, &JSONSetArgsOptions{Mode: mode})
}

// JSONSetWithArgs sets the JSON value at the given path in the given key with optional arguments
// for setting mode (NX/XX) and the FPHA (Floating-Point Homogeneous Array) type used for storing
// FP arrays. The value must be something that can be marshaled to JSON (using encoding/JSON) unless
// the argument is a string or []byte when we assume that it can be passed directly as JSON.
// For more information, see https://redis.io/commands/json.set
func (c cmdable) JSONSetWithArgs(ctx context.Context, key, path string, value interface{}, options *JSONSetArgsOptions) *StatusCmd {
	var bytes []byte
	var err error
	switch v := value.(type) {
	case string:
		bytes = []byte(v)
	case []byte:
		bytes = v
	default:
		bytes, err = json.Marshal(v)
	}
	args := []interface{}{"JSON.SET", key, path, util.BytesToString(bytes)}
	if options != nil {
		if options.Mode != "" {
			switch strings.ToUpper(options.Mode) {
			case "XX", "NX":
				args = append(args, strings.ToUpper(options.Mode))
			default:
				panic("redis: JSON.SET mode must be NX or XX")
			}
		}
		if options.FPHA != "" {
			args = append(args, "FPHA", string(options.FPHA))
		}
	}
	cmd := NewStatusCmd(ctx, args...)
	if err != nil {
		cmd.SetErr(err)
	} else {
		_ = c(ctx, cmd)
	}
	return cmd
}

// JSONStrAppend appends the JSON-string values to the string at the specified path.
// For more information, see https://redis.io/commands/json.strappend
func (c cmdable) JSONStrAppend(ctx context.Context, key, path, value string) *IntPointerSliceCmd {
	args := []interface{}{"JSON.STRAPPEND", key, path, value}
	cmd := NewIntPointerSliceCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// JSONStrLen reports the length of the JSON String at the specified path in the given key.
// For more information, see https://redis.io/commands/json.strlen
func (c cmdable) JSONStrLen(ctx context.Context, key, path string) *IntPointerSliceCmd {
	args := []interface{}{"JSON.STRLEN", key, path}
	cmd := NewIntPointerSliceCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// JSONToggle toggles a Boolean value stored at the specified path.
// For more information, see https://redis.io/commands/json.toggle
func (c cmdable) JSONToggle(ctx context.Context, key, path string) *IntPointerSliceCmd {
	args := []interface{}{"JSON.TOGGLE", key, path}
	cmd := NewIntPointerSliceCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// JSONType reports the type of JSON value at the specified path.
// For more information, see https://redis.io/commands/json.type
func (c cmdable) JSONType(ctx context.Context, key, path string) *JSONSliceCmd {
	args := []interface{}{"JSON.TYPE", key, path}
	cmd := NewJSONSliceCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}
