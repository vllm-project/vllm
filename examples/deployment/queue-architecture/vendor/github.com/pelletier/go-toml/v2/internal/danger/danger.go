package danger

import (
	"fmt"
	"reflect"
	"unsafe"
)

const maxInt = uintptr(int(^uint(0) >> 1))

func SubsliceOffset(data []byte, subslice []byte) int {
	datap := (*reflect.SliceHeader)(unsafe.Pointer(&data))
	hlp := (*reflect.SliceHeader)(unsafe.Pointer(&subslice))

	if hlp.Data < datap.Data {
		panic(fmt.Errorf("subslice address (%d) is before data address (%d)", hlp.Data, datap.Data))
	}
	offset := hlp.Data - datap.Data

	if offset > maxInt {
		panic(fmt.Errorf("slice offset larger than int (%d)", offset))
	}

	intoffset := int(offset)

	if intoffset > datap.Len {
		panic(fmt.Errorf("slice offset (%d) is farther than data length (%d)", intoffset, datap.Len))
	}

	if intoffset+hlp.Len > datap.Len {
		panic(fmt.Errorf("slice ends (%d+%d) is farther than data length (%d)", intoffset, hlp.Len, datap.Len))
	}

	return intoffset
}

func BytesRange(start []byte, end []byte) []byte {
	if start == nil || end == nil {
		panic("cannot call BytesRange with nil")
	}
	startp := (*reflect.SliceHeader)(unsafe.Pointer(&start))
	endp := (*reflect.SliceHeader)(unsafe.Pointer(&end))

	if startp.Data > endp.Data {
		panic(fmt.Errorf("start pointer address (%d) is after end pointer address (%d)", startp.Data, endp.Data))
	}

	l := startp.Len
	endLen := int(endp.Data-startp.Data) + endp.Len
	if endLen > l {
		l = endLen
	}

	if l > startp.Cap {
		panic(fmt.Errorf("range length is larger than capacity"))
	}

	return start[:l]
}

func Stride(ptr unsafe.Pointer, size uintptr, offset int) unsafe.Pointer {
	// TODO: replace with unsafe.Add when Go 1.17 is released
	//   https://github.com/golang/go/issues/40481
	return unsafe.Pointer(uintptr(ptr) + uintptr(int(size)*offset))
}
