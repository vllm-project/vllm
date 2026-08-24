package redis

import (
	"context"
	"encoding"
	"errors"
	"fmt"
	"io"
	"net"
	"reflect"
	"runtime"
	"strings"
	"time"

	"github.com/redis/go-redis/v9/internal"
)

// KeepTTL is a Redis KEEPTTL option to keep existing TTL, it requires your redis-server version >= 6.0,
// otherwise you will receive an error: (error) ERR syntax error.
// For example:
//
//	rdb.Set(ctx, key, value, redis.KeepTTL)
const KeepTTL = -1

func usePrecise(dur time.Duration) bool {
	return dur < time.Second || dur%time.Second != 0
}

func formatMs(ctx context.Context, dur time.Duration) int64 {
	if dur > 0 && dur < time.Millisecond {
		internal.Logger.Printf(
			ctx,
			"specified duration is %s, but minimal supported value is %s - truncating to 1ms",
			dur, time.Millisecond,
		)
		return 1
	}
	return int64(dur / time.Millisecond)
}

func formatSec(ctx context.Context, dur time.Duration) int64 {
	if dur > 0 && dur < time.Second {
		internal.Logger.Printf(
			ctx,
			"specified duration is %s, but minimal supported value is %s - truncating to 1s",
			dur, time.Second,
		)
		return 1
	}
	return int64(dur / time.Second)
}

func appendArgs(dst, src []interface{}) []interface{} {
	if len(src) == 1 {
		return appendArg(dst, src[0])
	}

	if cap(dst) < len(dst)+len(src) {
		newDst := make([]interface{}, len(dst), len(dst)+len(src))
		copy(newDst, dst)
		dst = newDst
	}
	dst = append(dst, src...)
	return dst
}

func appendArg(dst []interface{}, arg interface{}) []interface{} {
	switch arg := arg.(type) {
	case []string:
		for _, s := range arg {
			dst = append(dst, s)
		}
		return dst
	case []interface{}:
		dst = append(dst, arg...)
		return dst
	case map[string]interface{}:
		for k, v := range arg {
			dst = append(dst, k, v)
		}
		return dst
	case map[string]string:
		for k, v := range arg {
			dst = append(dst, k, v)
		}
		return dst
	case time.Time, time.Duration, encoding.BinaryMarshaler, net.IP:
		return append(dst, arg)
	case nil:
		return dst
	default:
		// scan struct field
		v := reflect.ValueOf(arg)
		if v.Type().Kind() == reflect.Ptr {
			if v.IsNil() {
				// error: arg is not a valid object
				return dst
			}
			v = v.Elem()
		}

		if v.Type().Kind() == reflect.Struct {
			return appendStructField(dst, v)
		}

		return append(dst, arg)
	}
}

// appendStructField appends the field and value held by the structure v to dst, and returns the appended dst.
func appendStructField(dst []interface{}, v reflect.Value) []interface{} {
	typ := v.Type()
	for i := 0; i < typ.NumField(); i++ {
		tag := typ.Field(i).Tag.Get("redis")
		if tag == "" || tag == "-" {
			continue
		}
		name, opt, _ := strings.Cut(tag, ",")
		if name == "" {
			continue
		}

		field := v.Field(i)

		// miss field
		if omitEmpty(opt) && isEmptyValue(field) {
			continue
		}

		if field.CanInterface() {
			dst = append(dst, name, field.Interface())
		}
	}

	return dst
}

func omitEmpty(opt string) bool {
	for opt != "" {
		var name string
		name, opt, _ = strings.Cut(opt, ",")
		if name == "omitempty" {
			return true
		}
	}
	return false
}

func isEmptyValue(v reflect.Value) bool {
	switch v.Kind() {
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
	case reflect.Interface, reflect.Pointer:
		return v.IsNil()
	case reflect.Struct:
		if v.Type() == reflect.TypeOf(time.Time{}) {
			return v.IsZero()
		}
		// Only supports the struct time.Time,
		// subsequent iterations will follow the func Scan support decoder.
	}
	return false
}

type Cmdable interface {
	Pipeline() Pipeliner
	Pipelined(ctx context.Context, fn func(Pipeliner) error) ([]Cmder, error)

	TxPipelined(ctx context.Context, fn func(Pipeliner) error) ([]Cmder, error)
	TxPipeline() Pipeliner

	Command(ctx context.Context) *CommandsInfoCmd
	CommandList(ctx context.Context, filter *FilterBy) *StringSliceCmd
	CommandGetKeys(ctx context.Context, commands ...interface{}) *StringSliceCmd
	CommandGetKeysAndFlags(ctx context.Context, commands ...interface{}) *KeyFlagsCmd
	ClientGetName(ctx context.Context) *StringCmd
	Echo(ctx context.Context, message interface{}) *StringCmd
	Ping(ctx context.Context) *StatusCmd
	Quit(ctx context.Context) *StatusCmd
	Unlink(ctx context.Context, keys ...string) *IntCmd

	BgRewriteAOF(ctx context.Context) *StatusCmd
	BgSave(ctx context.Context) *StatusCmd
	ClientKill(ctx context.Context, ipPort string) *StatusCmd
	ClientKillByFilter(ctx context.Context, keys ...string) *IntCmd
	ClientList(ctx context.Context) *StringCmd
	ClientInfo(ctx context.Context) *ClientInfoCmd
	ClientPause(ctx context.Context, dur time.Duration) *BoolCmd
	ClientUnpause(ctx context.Context) *BoolCmd
	ClientID(ctx context.Context) *IntCmd
	ClientUnblock(ctx context.Context, id int64) *IntCmd
	ClientUnblockWithError(ctx context.Context, id int64) *IntCmd
	ClientMaintNotifications(ctx context.Context, enabled bool, endpointType string) *StatusCmd
	ClientTracking(ctx context.Context, on bool, opt *ClientTrackingOptions) *StatusCmd
	ClientTrackingOn(ctx context.Context, opt *ClientTrackingOptions) *StatusCmd
	ClientTrackingOff(ctx context.Context) *StatusCmd
	ConfigGet(ctx context.Context, parameter string) *MapStringStringCmd
	ConfigResetStat(ctx context.Context) *StatusCmd
	ConfigSet(ctx context.Context, parameter, value string) *StatusCmd
	ConfigRewrite(ctx context.Context) *StatusCmd
	DBSize(ctx context.Context) *IntCmd
	FlushAll(ctx context.Context) *StatusCmd
	FlushAllAsync(ctx context.Context) *StatusCmd
	FlushDB(ctx context.Context) *StatusCmd
	FlushDBAsync(ctx context.Context) *StatusCmd
	Info(ctx context.Context, section ...string) *StringCmd
	InfoMap(ctx context.Context, section ...string) *InfoCmd
	LastSave(ctx context.Context) *IntCmd
	Save(ctx context.Context) *StatusCmd
	Shutdown(ctx context.Context) *StatusCmd
	ShutdownSave(ctx context.Context) *StatusCmd
	ShutdownNoSave(ctx context.Context) *StatusCmd
	SlaveOf(ctx context.Context, host, port string) *StatusCmd
	ReplicaOf(ctx context.Context, host, port string) *StatusCmd
	SlowLogGet(ctx context.Context, num int64) *SlowLogCmd
	SlowLogLen(ctx context.Context) *IntCmd
	SlowLogReset(ctx context.Context) *StatusCmd
	Time(ctx context.Context) *TimeCmd
	DebugObject(ctx context.Context, key string) *StringCmd
	MemoryUsage(ctx context.Context, key string, samples ...int) *IntCmd
	Latency(ctx context.Context) *LatencyCmd
	LatencyReset(ctx context.Context, events ...interface{}) *StatusCmd

	ModuleLoadex(ctx context.Context, conf *ModuleLoadexConfig) *StringCmd

	ACLCmdable
	ArrayCmdable
	BitMapCmdable
	ClusterCmdable
	GenericCmdable
	GeoCmdable
	HashCmdable
	HyperLogLogCmdable
	ListCmdable
	ProbabilisticCmdable
	PubSubCmdable
	ScriptingFunctionsCmdable
	SearchCmdable
	SetCmdable
	SortedSetCmdable
	StringCmdable
	StreamCmdable
	TimeseriesCmdable
	JSONCmdable
	VectorSetCmdable
}

type StatefulCmdable interface {
	Cmdable
	Auth(ctx context.Context, password string) *StatusCmd
	AuthACL(ctx context.Context, username, password string) *StatusCmd
	Select(ctx context.Context, index int) *StatusCmd
	SwapDB(ctx context.Context, index1, index2 int) *StatusCmd
	ClientSetName(ctx context.Context, name string) *BoolCmd
	ClientSetInfo(ctx context.Context, info LibraryInfo) *StatusCmd
	Hello(ctx context.Context, ver int, username, password, clientName string) *MapStringInterfaceCmd
}

var (
	_ Cmdable = (*Client)(nil)
	_ Cmdable = (*Tx)(nil)
	_ Cmdable = (*Ring)(nil)
	_ Cmdable = (*ClusterClient)(nil)
	_ Cmdable = (*Pipeline)(nil)
)

type cmdable func(ctx context.Context, cmd Cmder) error

type statefulCmdable func(ctx context.Context, cmd Cmder) error

//------------------------------------------------------------------------------

func (c statefulCmdable) Auth(ctx context.Context, password string) *StatusCmd {
	cmd := NewStatusCmd(ctx, "auth", password)
	_ = c(ctx, cmd)
	return cmd
}

// AuthACL Perform an AUTH command, using the given user and pass.
// Should be used to authenticate the current connection with one of the connections defined in the ACL list
// when connecting to a Redis 6.0 instance, or greater, that is using the Redis ACL system.
func (c statefulCmdable) AuthACL(ctx context.Context, username, password string) *StatusCmd {
	cmd := NewStatusCmd(ctx, "auth", username, password)
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) Wait(ctx context.Context, numSlaves int, timeout time.Duration) *IntCmd {
	cmd := NewIntCmd(ctx, "wait", numSlaves, int(timeout/time.Millisecond))
	cmd.setReadTimeout(timeout)
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) WaitAOF(ctx context.Context, numLocal, numSlaves int, timeout time.Duration) *IntSliceCmd {
	cmd := NewIntSliceCmd(ctx, "waitAOF", numLocal, numSlaves, int(timeout/time.Millisecond))
	cmd.setReadTimeout(timeout)
	_ = c(ctx, cmd)
	return cmd
}

func (c statefulCmdable) Select(ctx context.Context, index int) *StatusCmd {
	cmd := NewStatusCmd(ctx, "select", index)
	_ = c(ctx, cmd)
	return cmd
}

func (c statefulCmdable) SwapDB(ctx context.Context, index1, index2 int) *StatusCmd {
	cmd := NewStatusCmd(ctx, "swapdb", index1, index2)
	_ = c(ctx, cmd)
	return cmd
}

// ClientSetName assigns a name to the connection.
func (c statefulCmdable) ClientSetName(ctx context.Context, name string) *BoolCmd {
	cmd := NewBoolCmd(ctx, "client", "setname", name)
	_ = c(ctx, cmd)
	return cmd
}

// ClientSetInfo sends a CLIENT SETINFO command with the provided info.
func (c statefulCmdable) ClientSetInfo(ctx context.Context, info LibraryInfo) *StatusCmd {
	err := info.Validate()
	if err != nil {
		panic(err.Error())
	}

	var cmd *StatusCmd
	if info.LibName != nil {
		libName := fmt.Sprintf("go-redis(%s,%s)", *info.LibName, internal.ReplaceSpaces(runtime.Version()))
		cmd = NewStatusCmd(ctx, "client", "setinfo", "LIB-NAME", libName)
	} else {
		cmd = NewStatusCmd(ctx, "client", "setinfo", "LIB-VER", *info.LibVer)
	}

	_ = c(ctx, cmd)
	return cmd
}

// Validate checks if only one field in the struct is non-nil.
func (info LibraryInfo) Validate() error {
	if info.LibName != nil && info.LibVer != nil {
		return errors.New("both LibName and LibVer cannot be set at the same time")
	}
	if info.LibName == nil && info.LibVer == nil {
		return errors.New("at least one of LibName and LibVer should be set")
	}
	return nil
}

// Hello sets the resp protocol used.
func (c statefulCmdable) Hello(ctx context.Context,
	ver int, username, password, clientName string,
) *MapStringInterfaceCmd {
	args := make([]interface{}, 0, 7)
	args = append(args, "hello", ver)
	if password != "" {
		if username != "" {
			args = append(args, "auth", username, password)
		} else {
			args = append(args, "auth", "default", password)
		}
	}
	if clientName != "" {
		args = append(args, "setname", clientName)
	}
	cmd := NewMapStringInterfaceCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

//------------------------------------------------------------------------------

func (c cmdable) Command(ctx context.Context) *CommandsInfoCmd {
	cmd := NewCommandsInfoCmd(ctx, "command")
	_ = c(ctx, cmd)
	return cmd
}

// FilterBy is used for the `CommandList` command parameter.
type FilterBy struct {
	Module  string
	ACLCat  string
	Pattern string
}

func (c cmdable) CommandList(ctx context.Context, filter *FilterBy) *StringSliceCmd {
	args := make([]interface{}, 0, 5)
	args = append(args, "command", "list")
	if filter != nil {
		if filter.Module != "" {
			args = append(args, "filterby", "module", filter.Module)
		} else if filter.ACLCat != "" {
			args = append(args, "filterby", "aclcat", filter.ACLCat)
		} else if filter.Pattern != "" {
			args = append(args, "filterby", "pattern", filter.Pattern)
		}
	}
	cmd := NewStringSliceCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) CommandGetKeys(ctx context.Context, commands ...interface{}) *StringSliceCmd {
	args := make([]interface{}, 2+len(commands))
	args[0] = "command"
	args[1] = "getkeys"
	copy(args[2:], commands)
	cmd := NewStringSliceCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) CommandGetKeysAndFlags(ctx context.Context, commands ...interface{}) *KeyFlagsCmd {
	args := make([]interface{}, 2+len(commands))
	args[0] = "command"
	args[1] = "getkeysandflags"
	copy(args[2:], commands)
	cmd := NewKeyFlagsCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// ClientGetName returns the name of the connection.
func (c cmdable) ClientGetName(ctx context.Context) *StringCmd {
	cmd := NewStringCmd(ctx, "client", "getname")
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) Echo(ctx context.Context, message interface{}) *StringCmd {
	cmd := NewStringCmd(ctx, "echo", message)
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) Ping(ctx context.Context) *StatusCmd {
	cmd := NewStatusCmd(ctx, "ping")
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) Do(ctx context.Context, args ...interface{}) *Cmd {
	cmd := NewCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// DoRaw executes a command and returns the raw RESP protocol bytes without parsing.
func (c cmdable) DoRaw(ctx context.Context, args ...interface{}) *RawCmd {
	cmd := NewRawCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// DoRawWriteTo executes a command and streams raw RESP bytes directly to w without intermediate allocations.
func (c cmdable) DoRawWriteTo(ctx context.Context, w io.Writer, args ...interface{}) *RawWriteToCmd {
	cmd := NewRawWriteToCmd(ctx, w, args...)
	_ = c(ctx, cmd)
	return cmd
}

// Quit closes the connection.
//
// Deprecated: Just close the connection instead as of Redis 7.2.0.
func (c cmdable) Quit(_ context.Context) *StatusCmd {
	panic("not implemented")
}

//------------------------------------------------------------------------------

func (c cmdable) BgRewriteAOF(ctx context.Context) *StatusCmd {
	cmd := NewStatusCmd(ctx, "bgrewriteaof")
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) BgSave(ctx context.Context) *StatusCmd {
	cmd := NewStatusCmd(ctx, "bgsave")
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) ClientKill(ctx context.Context, ipPort string) *StatusCmd {
	cmd := NewStatusCmd(ctx, "client", "kill", ipPort)
	_ = c(ctx, cmd)
	return cmd
}

// ClientKillByFilter is new style syntax, while the ClientKill is old
//
//	CLIENT KILL <option> [value] ... <option> [value]
func (c cmdable) ClientKillByFilter(ctx context.Context, keys ...string) *IntCmd {
	args := make([]interface{}, 2+len(keys))
	args[0] = "client"
	args[1] = "kill"
	for i, key := range keys {
		args[2+i] = key
	}
	cmd := NewIntCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) ClientList(ctx context.Context) *StringCmd {
	cmd := NewStringCmd(ctx, "client", "list")
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) ClientPause(ctx context.Context, dur time.Duration) *BoolCmd {
	cmd := NewBoolCmd(ctx, "client", "pause", formatMs(ctx, dur))
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) ClientUnpause(ctx context.Context) *BoolCmd {
	cmd := NewBoolCmd(ctx, "client", "unpause")
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) ClientID(ctx context.Context) *IntCmd {
	cmd := NewIntCmd(ctx, "client", "id")
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) ClientUnblock(ctx context.Context, id int64) *IntCmd {
	cmd := NewIntCmd(ctx, "client", "unblock", id)
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) ClientUnblockWithError(ctx context.Context, id int64) *IntCmd {
	cmd := NewIntCmd(ctx, "client", "unblock", id, "error")
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) ClientInfo(ctx context.Context) *ClientInfoCmd {
	cmd := NewClientInfoCmd(ctx, "client", "info")
	_ = c(ctx, cmd)
	return cmd
}

// ClientMaintNotifications enables or disables maintenance notifications for maintenance upgrades.
// When enabled, the client will receive push notifications about Redis maintenance events.
func (c cmdable) ClientMaintNotifications(ctx context.Context, enabled bool, endpointType string) *StatusCmd {
	args := []interface{}{"client", "maint_notifications"}
	if enabled {
		if endpointType == "" {
			endpointType = "none"
		}
		args = append(args, "on", "moving-endpoint-type", endpointType)
	} else {
		args = append(args, "off")
	}
	cmd := NewStatusCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// ClientTrackingOptions configures CLIENT TRACKING ON. See
// https://redis.io/commands/client-tracking/ for semantics.
type ClientTrackingOptions struct {
	Redirect int64
	Bcast    bool
	Prefixes []string
	OptIn    bool
	OptOut   bool
	NoLoop   bool
}

// ClientTracking enables or disables server-assisted client-side caching for
// the ONE connection that happens to serve this command. On a pooled client
// that connection is arbitrary, so this is only meaningful on a dedicated
// connection (see Client.Conn). When on is false, opt is ignored. Invalid
// option combinations are reported via the returned command's Err and nothing
// is sent to the server.
//
// Must not be combined with the built-in client-side cache: on a client
// configured with Options.ClientSideCache or ClientSideCacheConfig this
// command is rejected, because changing a pool connection's tracking state
// would silently break the cache's invalidation.
func (c cmdable) ClientTracking(ctx context.Context, on bool, opt *ClientTrackingOptions) *StatusCmd {
	if !on {
		return c.ClientTrackingOff(ctx)
	}
	return c.ClientTrackingOn(ctx, opt)
}

// ClientTrackingOn enables tracking on the serving connection. See
// ClientTracking for the pooled-client and built-in-CSC caveats.
func (c cmdable) ClientTrackingOn(ctx context.Context, opt *ClientTrackingOptions) *StatusCmd {
	args := []interface{}{"client", "tracking", "on"}
	if opt != nil {
		if err := validateClientTrackingOptions(opt); err != nil {
			cmd := NewStatusCmd(ctx, args...)
			cmd.SetErr(err)
			return cmd
		}
		args = appendClientTrackingOptions(args, opt)
	}
	cmd := NewStatusCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// ClientTrackingOff disables tracking on the serving connection. See
// ClientTracking for the pooled-client and built-in-CSC caveats.
func (c cmdable) ClientTrackingOff(ctx context.Context) *StatusCmd {
	cmd := NewStatusCmd(ctx, "client", "tracking", "off")
	_ = c(ctx, cmd)
	return cmd
}

func validateClientTrackingOptions(opt *ClientTrackingOptions) error {
	if opt.OptIn && opt.OptOut {
		return errors.New("redis: CLIENT TRACKING OPTIN and OPTOUT are mutually exclusive")
	}
	if opt.Bcast && (opt.OptIn || opt.OptOut) {
		return errors.New("redis: CLIENT TRACKING BCAST cannot be combined with OPTIN or OPTOUT")
	}
	if len(opt.Prefixes) > 0 && !opt.Bcast {
		return errors.New("redis: CLIENT TRACKING PREFIX requires BCAST")
	}
	return nil
}

func appendClientTrackingOptions(args []interface{}, opt *ClientTrackingOptions) []interface{} {
	if opt.Redirect != 0 {
		args = append(args, "redirect", opt.Redirect)
	}
	if opt.Bcast {
		args = append(args, "bcast")
	}
	for _, p := range opt.Prefixes {
		args = append(args, "prefix", p)
	}
	if opt.OptIn {
		args = append(args, "optin")
	}
	if opt.OptOut {
		args = append(args, "optout")
	}
	if opt.NoLoop {
		args = append(args, "noloop")
	}
	return args
}

// ------------------------------------------------------------------------------------------------

func (c cmdable) ConfigGet(ctx context.Context, parameter string) *MapStringStringCmd {
	cmd := NewMapStringStringCmd(ctx, "config", "get", parameter)
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) ConfigResetStat(ctx context.Context) *StatusCmd {
	cmd := NewStatusCmd(ctx, "config", "resetstat")
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) ConfigSet(ctx context.Context, parameter, value string) *StatusCmd {
	cmd := NewStatusCmd(ctx, "config", "set", parameter, value)
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) ConfigRewrite(ctx context.Context) *StatusCmd {
	cmd := NewStatusCmd(ctx, "config", "rewrite")
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) DBSize(ctx context.Context) *IntCmd {
	cmd := NewIntCmd(ctx, "dbsize")
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) FlushAll(ctx context.Context) *StatusCmd {
	cmd := NewStatusCmd(ctx, "flushall")
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) FlushAllAsync(ctx context.Context) *StatusCmd {
	cmd := NewStatusCmd(ctx, "flushall", "async")
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) FlushDB(ctx context.Context) *StatusCmd {
	cmd := NewStatusCmd(ctx, "flushdb")
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) FlushDBAsync(ctx context.Context) *StatusCmd {
	cmd := NewStatusCmd(ctx, "flushdb", "async")
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) Info(ctx context.Context, sections ...string) *StringCmd {
	args := make([]interface{}, 1+len(sections))
	args[0] = "info"
	for i, section := range sections {
		args[i+1] = section
	}
	cmd := NewStringCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) InfoMap(ctx context.Context, sections ...string) *InfoCmd {
	args := make([]interface{}, 1+len(sections))
	args[0] = "info"
	for i, section := range sections {
		args[i+1] = section
	}
	cmd := NewInfoCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) LastSave(ctx context.Context) *IntCmd {
	cmd := NewIntCmd(ctx, "lastsave")
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) Save(ctx context.Context) *StatusCmd {
	cmd := NewStatusCmd(ctx, "save")
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) shutdown(ctx context.Context, modifier string) *StatusCmd {
	var args []interface{}
	if modifier == "" {
		args = []interface{}{"shutdown"}
	} else {
		args = []interface{}{"shutdown", modifier}
	}
	cmd := NewStatusCmd(ctx, args...)
	_ = c(ctx, cmd)
	if err := cmd.Err(); err != nil {
		if err == io.EOF {
			// Server quit as expected.
			cmd.err = nil
		}
	} else {
		// Server did not quit. String reply contains the reason.
		cmd.err = errors.New(cmd.val)
		cmd.val = ""
	}
	return cmd
}

func (c cmdable) Shutdown(ctx context.Context) *StatusCmd {
	return c.shutdown(ctx, "")
}

func (c cmdable) ShutdownSave(ctx context.Context) *StatusCmd {
	return c.shutdown(ctx, "save")
}

func (c cmdable) ShutdownNoSave(ctx context.Context) *StatusCmd {
	return c.shutdown(ctx, "nosave")
}

// SlaveOf sets a Redis server as a replica of another, or promotes it to being a master.
//
// Deprecated: Use ReplicaOf instead as of Redis 5.0.0.
func (c cmdable) SlaveOf(ctx context.Context, host, port string) *StatusCmd {
	cmd := NewStatusCmd(ctx, "slaveof", host, port)
	_ = c(ctx, cmd)
	return cmd
}

// ReplicaOf sets a Redis server as a replica of another, or promotes it to being a master.
func (c cmdable) ReplicaOf(ctx context.Context, host, port string) *StatusCmd {
	cmd := NewStatusCmd(ctx, "replicaof", host, port)
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) SlowLogGet(ctx context.Context, num int64) *SlowLogCmd {
	cmd := NewSlowLogCmd(ctx, "slowlog", "get", num)
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) SlowLogLen(ctx context.Context) *IntCmd {
	cmd := NewIntCmd(ctx, "slowlog", "len")
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) SlowLogReset(ctx context.Context) *StatusCmd {
	cmd := NewStatusCmd(ctx, "slowlog", "reset")
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) Latency(ctx context.Context) *LatencyCmd {
	cmd := NewLatencyCmd(ctx, "latency", "latest")
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) LatencyReset(ctx context.Context, events ...interface{}) *StatusCmd {
	args := make([]interface{}, 2+len(events))
	args[0] = "latency"
	args[1] = "reset"
	copy(args[2:], events)
	cmd := NewStatusCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) Sync(_ context.Context) {
	panic("not implemented")
}

func (c cmdable) Time(ctx context.Context) *TimeCmd {
	cmd := NewTimeCmd(ctx, "time")
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) DebugObject(ctx context.Context, key string) *StringCmd {
	cmd := NewStringCmd(ctx, "debug", "object", key)
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) MemoryUsage(ctx context.Context, key string, samples ...int) *IntCmd {
	args := []interface{}{"memory", "usage", key}
	if len(samples) > 0 {
		if len(samples) != 1 {
			cmd := NewIntCmd(ctx)
			cmd.SetErr(errors.New("MemoryUsage expects single sample count"))
			return cmd
		}
		args = append(args, "SAMPLES", samples[0])
	}
	cmd := NewIntCmd(ctx, args...)
	cmd.SetFirstKeyPos(2)
	_ = c(ctx, cmd)
	return cmd
}

//------------------------------------------------------------------------------

// ModuleLoadexConfig struct is used to specify the arguments for the MODULE LOADEX command of redis.
// `MODULE LOADEX path [CONFIG name value [CONFIG name value ...]] [ARGS args [args ...]]`
type ModuleLoadexConfig struct {
	Path string
	Conf map[string]interface{}
	Args []interface{}
}

func (c *ModuleLoadexConfig) toArgs() []interface{} {
	args := make([]interface{}, 3, 3+len(c.Conf)*3+len(c.Args)*2)
	args[0] = "MODULE"
	args[1] = "LOADEX"
	args[2] = c.Path
	for k, v := range c.Conf {
		args = append(args, "CONFIG", k, v)
	}
	for _, arg := range c.Args {
		args = append(args, "ARGS", arg)
	}
	return args
}

// ModuleLoadex Redis `MODULE LOADEX path [CONFIG name value [CONFIG name value ...]] [ARGS args [args ...]]` command.
func (c cmdable) ModuleLoadex(ctx context.Context, conf *ModuleLoadexConfig) *StringCmd {
	if conf == nil {
		cmd := NewStringCmd(ctx)
		cmd.SetErr(errors.New("redis: ModuleLoadex nil config"))
		return cmd
	}
	cmd := NewStringCmd(ctx, conf.toArgs()...)
	_ = c(ctx, cmd)
	return cmd
}

/*
Monitor - represents a Redis MONITOR command, allowing the user to capture
and process all commands sent to a Redis server. This mimics the behavior of
MONITOR in the redis-cli.

Notes:
- Using MONITOR blocks the connection to the server for itself. It needs a dedicated connection
- The user should create a channel of type string
- This runs concurrently in the background. Trigger via the Start and Stop functions
See further: Redis MONITOR command: https://redis.io/commands/monitor
*/
func (c cmdable) Monitor(ctx context.Context, ch chan string) *MonitorCmd {
	cmd := newMonitorCmd(ctx, ch)
	_ = c(ctx, cmd)
	return cmd
}
