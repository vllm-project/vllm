package toml

import (
	"fmt"
	"strings"
	"time"

	"github.com/pelletier/go-toml/v2/unstable"
)

// LocalDate represents a calendar day in no specific timezone.
type LocalDate struct {
	Year  int
	Month int
	Day   int
}

// AsTime converts d into a specific time instance at midnight in zone.
func (d LocalDate) AsTime(zone *time.Location) time.Time {
	return time.Date(d.Year, time.Month(d.Month), d.Day, 0, 0, 0, 0, zone)
}

// String returns RFC 3339 representation of d.
func (d LocalDate) String() string {
	return fmt.Sprintf("%04d-%02d-%02d", d.Year, d.Month, d.Day)
}

// MarshalText returns RFC 3339 representation of d.
func (d LocalDate) MarshalText() ([]byte, error) {
	return []byte(d.String()), nil
}

// UnmarshalText parses b using RFC 3339 to fill d.
func (d *LocalDate) UnmarshalText(b []byte) error {
	res, err := parseLocalDate(b)
	if err != nil {
		return err
	}
	*d = res
	return nil
}

// LocalTime represents a time of day of no specific day in no specific
// timezone.
type LocalTime struct {
	Hour       int // Hour of the day: [0; 24[
	Minute     int // Minute of the hour: [0; 60[
	Second     int // Second of the minute: [0; 60[
	Nanosecond int // Nanoseconds within the second:  [0, 1000000000[
	Precision  int // Number of digits to display for Nanosecond.
}

// String returns RFC 3339 representation of d.
// If d.Nanosecond and d.Precision are zero, the time won't have a nanosecond
// component. If d.Nanosecond > 0 but d.Precision = 0, then the minimum number
// of digits for nanoseconds is provided.
func (d LocalTime) String() string {
	s := fmt.Sprintf("%02d:%02d:%02d", d.Hour, d.Minute, d.Second)

	if d.Precision > 0 {
		s += fmt.Sprintf(".%09d", d.Nanosecond)[:d.Precision+1]
	} else if d.Nanosecond > 0 {
		// Nanoseconds are specified, but precision is not provided. Use the
		// minimum.
		s += strings.Trim(fmt.Sprintf(".%09d", d.Nanosecond), "0")
	}

	return s
}

// MarshalText returns RFC 3339 representation of d.
func (d LocalTime) MarshalText() ([]byte, error) {
	return []byte(d.String()), nil
}

// UnmarshalText parses b using RFC 3339 to fill d.
func (d *LocalTime) UnmarshalText(b []byte) error {
	res, left, err := parseLocalTime(b)
	if err == nil && len(left) != 0 {
		err = unstable.NewParserError(left, "extra characters")
	}
	if err != nil {
		return err
	}
	*d = res
	return nil
}

// LocalDateTime represents a time of a specific day in no specific timezone.
type LocalDateTime struct {
	LocalDate
	LocalTime
}

// AsTime converts d into a specific time instance in zone.
func (d LocalDateTime) AsTime(zone *time.Location) time.Time {
	return time.Date(d.Year, time.Month(d.Month), d.Day, d.Hour, d.Minute, d.Second, d.Nanosecond, zone)
}

// String returns RFC 3339 representation of d.
func (d LocalDateTime) String() string {
	return d.LocalDate.String() + "T" + d.LocalTime.String()
}

// MarshalText returns RFC 3339 representation of d.
func (d LocalDateTime) MarshalText() ([]byte, error) {
	return []byte(d.String()), nil
}

// UnmarshalText parses b using RFC 3339 to fill d.
func (d *LocalDateTime) UnmarshalText(data []byte) error {
	res, left, err := parseLocalDateTime(data)
	if err == nil && len(left) != 0 {
		err = unstable.NewParserError(left, "extra characters")
	}
	if err != nil {
		return err
	}

	*d = res
	return nil
}
