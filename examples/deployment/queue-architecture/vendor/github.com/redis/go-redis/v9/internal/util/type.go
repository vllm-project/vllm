package util

func ToPtr[T any](v T) *T {
	return &v
}
