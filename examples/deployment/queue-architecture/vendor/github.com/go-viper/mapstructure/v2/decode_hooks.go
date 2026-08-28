package mapstructure

import (
	"encoding"
	"errors"
	"fmt"
	"net"
	"net/netip"
	"net/url"
	"reflect"
	"strconv"
	"strings"
	"time"
)

// typedDecodeHook takes a raw DecodeHookFunc (an any) and turns
// it into the proper DecodeHookFunc type, such as DecodeHookFuncType.
func typedDecodeHook(h DecodeHookFunc) DecodeHookFunc {
	// Create variables here so we can reference them with the reflect pkg
	var f1 DecodeHookFuncType
	var f2 DecodeHookFuncKind
	var f3 DecodeHookFuncValue

	// Fill in the variables into this interface and the rest is done
	// automatically using the reflect package.
	potential := []any{f1, f2, f3}

	v := reflect.ValueOf(h)
	vt := v.Type()
	for _, raw := range potential {
		pt := reflect.ValueOf(raw).Type()
		if vt.ConvertibleTo(pt) {
			return v.Convert(pt).Interface()
		}
	}

	return nil
}

// cachedDecodeHook takes a raw DecodeHookFunc (an any) and turns
// it into a closure to be used directly
// if the type fails to convert we return a closure always erroring to keep the previous behaviour
func cachedDecodeHook(raw DecodeHookFunc) func(from reflect.Value, to reflect.Value) (any, error) {
	switch f := typedDecodeHook(raw).(type) {
	case DecodeHookFuncType:
		return func(from reflect.Value, to reflect.Value) (any, error) {
			return f(from.Type(), to.Type(), from.Interface())
		}
	case DecodeHookFuncKind:
		return func(from reflect.Value, to reflect.Value) (any, error) {
			return f(from.Kind(), to.Kind(), from.Interface())
		}
	case DecodeHookFuncValue:
		return func(from reflect.Value, to reflect.Value) (any, error) {
			return f(from, to)
		}
	default:
		return func(from reflect.Value, to reflect.Value) (any, error) {
			return nil, errors.New("invalid decode hook signature")
		}
	}
}

// DecodeHookExec executes the given decode hook. This should be used
// since it'll naturally degrade to the older backwards compatible DecodeHookFunc
// that took reflect.Kind instead of reflect.Type.
func DecodeHookExec(
	raw DecodeHookFunc,
	from reflect.Value, to reflect.Value,
) (any, error) {
	switch f := typedDecodeHook(raw).(type) {
	case DecodeHookFuncType:
		return f(from.Type(), to.Type(), from.Interface())
	case DecodeHookFuncKind:
		return f(from.Kind(), to.Kind(), from.Interface())
	case DecodeHookFuncValue:
		return f(from, to)
	default:
		return nil, errors.New("invalid decode hook signature")
	}
}

// ComposeDecodeHookFunc creates a single DecodeHookFunc that
// automatically composes multiple DecodeHookFuncs.
//
// The composed funcs are called in order, with the result of the
// previous transformation.
func ComposeDecodeHookFunc(fs ...DecodeHookFunc) DecodeHookFunc {
	cached := make([]func(from reflect.Value, to reflect.Value) (any, error), 0, len(fs))
	for _, f := range fs {
		cached = append(cached, cachedDecodeHook(f))
	}
	return func(f reflect.Value, t reflect.Value) (any, error) {
		var err error
		data := f.Interface()

		newFrom := f
		for _, c := range cached {
			data, err = c(newFrom, t)
			if err != nil {
				return nil, err
			}
			if v, ok := data.(reflect.Value); ok {
				newFrom = v
			} else {
				newFrom = reflect.ValueOf(data)
			}
		}

		return data, nil
	}
}

// OrComposeDecodeHookFunc executes all input hook functions until one of them returns no error. In that case its value is returned.
// If all hooks return an error, OrComposeDecodeHookFunc returns an error concatenating all error messages.
func OrComposeDecodeHookFunc(ff ...DecodeHookFunc) DecodeHookFunc {
	cached := make([]func(from reflect.Value, to reflect.Value) (any, error), 0, len(ff))
	for _, f := range ff {
		cached = append(cached, cachedDecodeHook(f))
	}
	return func(a, b reflect.Value) (any, error) {
		var allErrs string
		var out any
		var err error

		for _, c := range cached {
			out, err = c(a, b)
			if err != nil {
				allErrs += err.Error() + "\n"
				continue
			}

			return out, nil
		}

		return nil, errors.New(allErrs)
	}
}

// StringToSliceHookFunc returns a DecodeHookFunc that converts
// string to []string by splitting on the given sep.
func StringToSliceHookFunc(sep string) DecodeHookFunc {
	return func(
		f reflect.Type,
		t reflect.Type,
		data any,
	) (any, error) {
		if f.Kind() != reflect.String {
			return data, nil
		}
		if t != reflect.SliceOf(f) {
			return data, nil
		}

		raw := data.(string)
		if raw == "" {
			return []string{}, nil
		}

		return strings.Split(raw, sep), nil
	}
}

// StringToWeakSliceHookFunc brings back the old (pre-v2) behavior of [StringToSliceHookFunc].
//
// As of mapstructure v2.0.0 [StringToSliceHookFunc] checks if the return type is a string slice.
// This function removes that check.
func StringToWeakSliceHookFunc(sep string) DecodeHookFunc {
	return func(
		f reflect.Type,
		t reflect.Type,
		data any,
	) (any, error) {
		if f.Kind() != reflect.String || t.Kind() != reflect.Slice {
			return data, nil
		}

		raw := data.(string)
		if raw == "" {
			return []string{}, nil
		}

		return strings.Split(raw, sep), nil
	}
}

// StringToTimeDurationHookFunc returns a DecodeHookFunc that converts
// strings to time.Duration.
func StringToTimeDurationHookFunc() DecodeHookFunc {
	return func(
		f reflect.Type,
		t reflect.Type,
		data any,
	) (any, error) {
		if f.Kind() != reflect.String {
			return data, nil
		}
		if t != reflect.TypeOf(time.Duration(5)) {
			return data, nil
		}

		// Convert it by parsing
		d, err := time.ParseDuration(data.(string))

		return d, wrapTimeParseDurationError(err)
	}
}

// StringToTimeLocationHookFunc returns a DecodeHookFunc that converts
// strings to *time.Location.
func StringToTimeLocationHookFunc() DecodeHookFunc {
	return func(
		f reflect.Type,
		t reflect.Type,
		data any,
	) (any, error) {
		if f.Kind() != reflect.String {
			return data, nil
		}
		if t != reflect.TypeOf(time.Local) {
			return data, nil
		}
		d, err := time.LoadLocation(data.(string))

		return d, wrapTimeParseLocationError(err)
	}
}

// StringToURLHookFunc returns a DecodeHookFunc that converts
// strings to *url.URL.
func StringToURLHookFunc() DecodeHookFunc {
	return func(
		f reflect.Type,
		t reflect.Type,
		data any,
	) (any, error) {
		if f.Kind() != reflect.String {
			return data, nil
		}
		if t != reflect.TypeOf(&url.URL{}) {
			return data, nil
		}

		// Convert it by parsing
		u, err := url.Parse(data.(string))

		return u, wrapUrlError(err)
	}
}

// StringToIPHookFunc returns a DecodeHookFunc that converts
// strings to net.IP
func StringToIPHookFunc() DecodeHookFunc {
	return func(
		f reflect.Type,
		t reflect.Type,
		data any,
	) (any, error) {
		if f.Kind() != reflect.String {
			return data, nil
		}
		if t != reflect.TypeOf(net.IP{}) {
			return data, nil
		}

		// Convert it by parsing
		ip := net.ParseIP(data.(string))
		if ip == nil {
			return net.IP{}, fmt.Errorf("failed parsing ip")
		}

		return ip, nil
	}
}

// StringToIPNetHookFunc returns a DecodeHookFunc that converts
// strings to net.IPNet
func StringToIPNetHookFunc() DecodeHookFunc {
	return func(
		f reflect.Type,
		t reflect.Type,
		data any,
	) (any, error) {
		if f.Kind() != reflect.String {
			return data, nil
		}
		if t != reflect.TypeOf(net.IPNet{}) {
			return data, nil
		}

		// Convert it by parsing
		_, net, err := net.ParseCIDR(data.(string))
		return net, wrapNetParseError(err)
	}
}

// StringToTimeHookFunc returns a DecodeHookFunc that converts
// strings to time.Time.
func StringToTimeHookFunc(layout string) DecodeHookFunc {
	return func(
		f reflect.Type,
		t reflect.Type,
		data any,
	) (any, error) {
		if f.Kind() != reflect.String {
			return data, nil
		}
		if t != reflect.TypeOf(time.Time{}) {
			return data, nil
		}

		// Convert it by parsing
		ti, err := time.Parse(layout, data.(string))

		return ti, wrapTimeParseError(err)
	}
}

// WeaklyTypedHook is a DecodeHookFunc which adds support for weak typing to
// the decoder.
//
// Note that this is significantly different from the WeaklyTypedInput option
// of the DecoderConfig.
func WeaklyTypedHook(
	f reflect.Kind,
	t reflect.Kind,
	data any,
) (any, error) {
	dataVal := reflect.ValueOf(data)
	switch t {
	case reflect.String:
		switch f {
		case reflect.Bool:
			if dataVal.Bool() {
				return "1", nil
			}
			return "0", nil
		case reflect.Float32:
			return strconv.FormatFloat(dataVal.Float(), 'f', -1, 64), nil
		case reflect.Int:
			return strconv.FormatInt(dataVal.Int(), 10), nil
		case reflect.Slice:
			dataType := dataVal.Type()
			elemKind := dataType.Elem().Kind()
			if elemKind == reflect.Uint8 {
				return string(dataVal.Interface().([]uint8)), nil
			}
		case reflect.Uint:
			return strconv.FormatUint(dataVal.Uint(), 10), nil
		}
	}

	return data, nil
}

func RecursiveStructToMapHookFunc() DecodeHookFunc {
	return func(f reflect.Value, t reflect.Value) (any, error) {
		if f.Kind() != reflect.Struct {
			return f.Interface(), nil
		}

		var i any = struct{}{}
		if t.Type() != reflect.TypeOf(&i).Elem() {
			return f.Interface(), nil
		}

		m := make(map[string]any)
		t.Set(reflect.ValueOf(m))

		return f.Interface(), nil
	}
}

// TextUnmarshallerHookFunc returns a DecodeHookFunc that applies
// strings to the UnmarshalText function, when the target type
// implements the encoding.TextUnmarshaler interface
func TextUnmarshallerHookFunc() DecodeHookFuncType {
	return func(
		f reflect.Type,
		t reflect.Type,
		data any,
	) (any, error) {
		if f.Kind() != reflect.String {
			return data, nil
		}
		result := reflect.New(t).Interface()
		unmarshaller, ok := result.(encoding.TextUnmarshaler)
		if !ok {
			return data, nil
		}
		str, ok := data.(string)
		if !ok {
			str = reflect.Indirect(reflect.ValueOf(&data)).Elem().String()
		}
		if err := unmarshaller.UnmarshalText([]byte(str)); err != nil {
			return nil, err
		}
		return result, nil
	}
}

// StringToNetIPAddrHookFunc returns a DecodeHookFunc that converts
// strings to netip.Addr.
func StringToNetIPAddrHookFunc() DecodeHookFunc {
	return func(
		f reflect.Type,
		t reflect.Type,
		data any,
	) (any, error) {
		if f.Kind() != reflect.String {
			return data, nil
		}
		if t != reflect.TypeOf(netip.Addr{}) {
			return data, nil
		}

		// Convert it by parsing
		addr, err := netip.ParseAddr(data.(string))

		return addr, wrapNetIPParseAddrError(err)
	}
}

// StringToNetIPAddrPortHookFunc returns a DecodeHookFunc that converts
// strings to netip.AddrPort.
func StringToNetIPAddrPortHookFunc() DecodeHookFunc {
	return func(
		f reflect.Type,
		t reflect.Type,
		data any,
	) (any, error) {
		if f.Kind() != reflect.String {
			return data, nil
		}
		if t != reflect.TypeOf(netip.AddrPort{}) {
			return data, nil
		}

		// Convert it by parsing
		addrPort, err := netip.ParseAddrPort(data.(string))

		return addrPort, wrapNetIPParseAddrPortError(err)
	}
}

// StringToNetIPPrefixHookFunc returns a DecodeHookFunc that converts
// strings to netip.Prefix.
func StringToNetIPPrefixHookFunc() DecodeHookFunc {
	return func(
		f reflect.Type,
		t reflect.Type,
		data any,
	) (any, error) {
		if f.Kind() != reflect.String {
			return data, nil
		}
		if t != reflect.TypeOf(netip.Prefix{}) {
			return data, nil
		}

		// Convert it by parsing
		prefix, err := netip.ParsePrefix(data.(string))

		return prefix, wrapNetIPParsePrefixError(err)
	}
}

// StringToBasicTypeHookFunc returns a DecodeHookFunc that converts
// strings to basic types.
// int8, uint8, int16, uint16, int32, uint32, int64, uint64, int, uint, float32, float64, bool, byte, rune, complex64, complex128
func StringToBasicTypeHookFunc() DecodeHookFunc {
	return ComposeDecodeHookFunc(
		StringToInt8HookFunc(),
		StringToUint8HookFunc(),
		StringToInt16HookFunc(),
		StringToUint16HookFunc(),
		StringToInt32HookFunc(),
		StringToUint32HookFunc(),
		StringToInt64HookFunc(),
		StringToUint64HookFunc(),
		StringToIntHookFunc(),
		StringToUintHookFunc(),
		StringToFloat32HookFunc(),
		StringToFloat64HookFunc(),
		StringToBoolHookFunc(),
		// byte and rune are aliases for uint8 and int32 respectively
		// StringToByteHookFunc(),
		// StringToRuneHookFunc(),
		StringToComplex64HookFunc(),
		StringToComplex128HookFunc(),
	)
}

// StringToInt8HookFunc returns a DecodeHookFunc that converts
// strings to int8.
func StringToInt8HookFunc() DecodeHookFunc {
	return func(f reflect.Type, t reflect.Type, data any) (any, error) {
		if f.Kind() != reflect.String || t.Kind() != reflect.Int8 {
			return data, nil
		}

		// Convert it by parsing
		i64, err := strconv.ParseInt(data.(string), 0, 8)
		return int8(i64), wrapStrconvNumError(err)
	}
}

// StringToUint8HookFunc returns a DecodeHookFunc that converts
// strings to uint8.
func StringToUint8HookFunc() DecodeHookFunc {
	return func(f reflect.Type, t reflect.Type, data any) (any, error) {
		if f.Kind() != reflect.String || t.Kind() != reflect.Uint8 {
			return data, nil
		}

		// Convert it by parsing
		u64, err := strconv.ParseUint(data.(string), 0, 8)
		return uint8(u64), wrapStrconvNumError(err)
	}
}

// StringToInt16HookFunc returns a DecodeHookFunc that converts
// strings to int16.
func StringToInt16HookFunc() DecodeHookFunc {
	return func(f reflect.Type, t reflect.Type, data any) (any, error) {
		if f.Kind() != reflect.String || t.Kind() != reflect.Int16 {
			return data, nil
		}

		// Convert it by parsing
		i64, err := strconv.ParseInt(data.(string), 0, 16)
		return int16(i64), wrapStrconvNumError(err)
	}
}

// StringToUint16HookFunc returns a DecodeHookFunc that converts
// strings to uint16.
func StringToUint16HookFunc() DecodeHookFunc {
	return func(f reflect.Type, t reflect.Type, data any) (any, error) {
		if f.Kind() != reflect.String || t.Kind() != reflect.Uint16 {
			return data, nil
		}

		// Convert it by parsing
		u64, err := strconv.ParseUint(data.(string), 0, 16)
		return uint16(u64), wrapStrconvNumError(err)
	}
}

// StringToInt32HookFunc returns a DecodeHookFunc that converts
// strings to int32.
func StringToInt32HookFunc() DecodeHookFunc {
	return func(f reflect.Type, t reflect.Type, data any) (any, error) {
		if f.Kind() != reflect.String || t.Kind() != reflect.Int32 {
			return data, nil
		}

		// Convert it by parsing
		i64, err := strconv.ParseInt(data.(string), 0, 32)
		return int32(i64), wrapStrconvNumError(err)
	}
}

// StringToUint32HookFunc returns a DecodeHookFunc that converts
// strings to uint32.
func StringToUint32HookFunc() DecodeHookFunc {
	return func(f reflect.Type, t reflect.Type, data any) (any, error) {
		if f.Kind() != reflect.String || t.Kind() != reflect.Uint32 {
			return data, nil
		}

		// Convert it by parsing
		u64, err := strconv.ParseUint(data.(string), 0, 32)
		return uint32(u64), wrapStrconvNumError(err)
	}
}

// StringToInt64HookFunc returns a DecodeHookFunc that converts
// strings to int64.
func StringToInt64HookFunc() DecodeHookFunc {
	return func(f reflect.Type, t reflect.Type, data any) (any, error) {
		if f.Kind() != reflect.String || t.Kind() != reflect.Int64 {
			return data, nil
		}

		// Convert it by parsing
		i64, err := strconv.ParseInt(data.(string), 0, 64)
		return int64(i64), wrapStrconvNumError(err)
	}
}

// StringToUint64HookFunc returns a DecodeHookFunc that converts
// strings to uint64.
func StringToUint64HookFunc() DecodeHookFunc {
	return func(f reflect.Type, t reflect.Type, data any) (any, error) {
		if f.Kind() != reflect.String || t.Kind() != reflect.Uint64 {
			return data, nil
		}

		// Convert it by parsing
		u64, err := strconv.ParseUint(data.(string), 0, 64)
		return uint64(u64), wrapStrconvNumError(err)
	}
}

// StringToIntHookFunc returns a DecodeHookFunc that converts
// strings to int.
func StringToIntHookFunc() DecodeHookFunc {
	return func(f reflect.Type, t reflect.Type, data any) (any, error) {
		if f.Kind() != reflect.String || t.Kind() != reflect.Int {
			return data, nil
		}

		// Convert it by parsing
		i64, err := strconv.ParseInt(data.(string), 0, 0)
		return int(i64), wrapStrconvNumError(err)
	}
}

// StringToUintHookFunc returns a DecodeHookFunc that converts
// strings to uint.
func StringToUintHookFunc() DecodeHookFunc {
	return func(f reflect.Type, t reflect.Type, data any) (any, error) {
		if f.Kind() != reflect.String || t.Kind() != reflect.Uint {
			return data, nil
		}

		// Convert it by parsing
		u64, err := strconv.ParseUint(data.(string), 0, 0)
		return uint(u64), wrapStrconvNumError(err)
	}
}

// StringToFloat32HookFunc returns a DecodeHookFunc that converts
// strings to float32.
func StringToFloat32HookFunc() DecodeHookFunc {
	return func(f reflect.Type, t reflect.Type, data any) (any, error) {
		if f.Kind() != reflect.String || t.Kind() != reflect.Float32 {
			return data, nil
		}

		// Convert it by parsing
		f64, err := strconv.ParseFloat(data.(string), 32)
		return float32(f64), wrapStrconvNumError(err)
	}
}

// StringToFloat64HookFunc returns a DecodeHookFunc that converts
// strings to float64.
func StringToFloat64HookFunc() DecodeHookFunc {
	return func(f reflect.Type, t reflect.Type, data any) (any, error) {
		if f.Kind() != reflect.String || t.Kind() != reflect.Float64 {
			return data, nil
		}

		// Convert it by parsing
		f64, err := strconv.ParseFloat(data.(string), 64)
		return f64, wrapStrconvNumError(err)
	}
}

// StringToBoolHookFunc returns a DecodeHookFunc that converts
// strings to bool.
func StringToBoolHookFunc() DecodeHookFunc {
	return func(f reflect.Type, t reflect.Type, data any) (any, error) {
		if f.Kind() != reflect.String || t.Kind() != reflect.Bool {
			return data, nil
		}

		// Convert it by parsing
		b, err := strconv.ParseBool(data.(string))
		return b, wrapStrconvNumError(err)
	}
}

// StringToByteHookFunc returns a DecodeHookFunc that converts
// strings to byte.
func StringToByteHookFunc() DecodeHookFunc {
	return StringToUint8HookFunc()
}

// StringToRuneHookFunc returns a DecodeHookFunc that converts
// strings to rune.
func StringToRuneHookFunc() DecodeHookFunc {
	return StringToInt32HookFunc()
}

// StringToComplex64HookFunc returns a DecodeHookFunc that converts
// strings to complex64.
func StringToComplex64HookFunc() DecodeHookFunc {
	return func(f reflect.Type, t reflect.Type, data any) (any, error) {
		if f.Kind() != reflect.String || t.Kind() != reflect.Complex64 {
			return data, nil
		}

		// Convert it by parsing
		c128, err := strconv.ParseComplex(data.(string), 64)
		return complex64(c128), wrapStrconvNumError(err)
	}
}

// StringToComplex128HookFunc returns a DecodeHookFunc that converts
// strings to complex128.
func StringToComplex128HookFunc() DecodeHookFunc {
	return func(f reflect.Type, t reflect.Type, data any) (any, error) {
		if f.Kind() != reflect.String || t.Kind() != reflect.Complex128 {
			return data, nil
		}

		// Convert it by parsing
		c128, err := strconv.ParseComplex(data.(string), 128)
		return c128, wrapStrconvNumError(err)
	}
}
