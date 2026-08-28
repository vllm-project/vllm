package danger

import (
	"reflect"
	"unsafe"
)

// typeID is used as key in encoder and decoder caches to enable using
// the optimize runtime.mapaccess2_fast64 function instead of the more
// expensive lookup if we were to use reflect.Type as map key.
//
// typeID holds the pointer to the reflect.Type value, which is unique
// in the program.
//
// https://github.com/segmentio/encoding/blob/master/json/codec.go#L59-L61
type TypeID unsafe.Pointer

func MakeTypeID(t reflect.Type) TypeID {
	// reflect.Type has the fields:
	// typ unsafe.Pointer
	// ptr unsafe.Pointer
	return TypeID((*[2]unsafe.Pointer)(unsafe.Pointer(&t))[1])
}
