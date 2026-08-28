// We enable 64 bit LE platforms:

//go:build (amd64 || arm64 || ppc64le || riscv64) && !nounsafe && !purego && !appengine

package le

import (
	"unsafe"
)

// Load8 will load from b at index i.
func Load8[I Indexer](b []byte, i I) byte {
	//return binary.LittleEndian.Uint16(b[i:])
	//return *(*uint16)(unsafe.Pointer(&b[i]))
	return *(*byte)(unsafe.Add(unsafe.Pointer(unsafe.SliceData(b)), i))
}

// Load16 will load from b at index i.
func Load16[I Indexer](b []byte, i I) uint16 {
	//return binary.LittleEndian.Uint16(b[i:])
	//return *(*uint16)(unsafe.Pointer(&b[i]))
	return *(*uint16)(unsafe.Add(unsafe.Pointer(unsafe.SliceData(b)), i))
}

// Load32 will load from b at index i.
func Load32[I Indexer](b []byte, i I) uint32 {
	//return binary.LittleEndian.Uint32(b[i:])
	//return *(*uint32)(unsafe.Pointer(&b[i]))
	return *(*uint32)(unsafe.Add(unsafe.Pointer(unsafe.SliceData(b)), i))
}

// Load64 will load from b at index i.
func Load64[I Indexer](b []byte, i I) uint64 {
	//return binary.LittleEndian.Uint64(b[i:])
	//return *(*uint64)(unsafe.Pointer(&b[i]))
	return *(*uint64)(unsafe.Add(unsafe.Pointer(unsafe.SliceData(b)), i))
}

// Store16 will store v at b.
func Store16(b []byte, v uint16) {
	*(*uint16)(unsafe.Pointer(unsafe.SliceData(b))) = v
}

// Store32 will store v at b.
func Store32(b []byte, v uint32) {
	*(*uint32)(unsafe.Pointer(unsafe.SliceData(b))) = v
}

// Store64 will store v at b[i:].
func Store64[I Indexer](b []byte, i I, v uint64) {
	*(*uint64)(unsafe.Add(unsafe.Pointer(unsafe.SliceData(b)), i)) = v
}
