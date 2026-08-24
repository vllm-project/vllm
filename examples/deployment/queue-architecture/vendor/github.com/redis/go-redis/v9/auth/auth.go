// Package auth package provides authentication-related interfaces and types.
// It also includes a basic implementation of credentials using username and password.
package auth

// StreamingCredentialsProvider is an interface that defines the methods for a streaming credentials provider.
// It is used to provide credentials for authentication.
// The CredentialsListener is used to receive updates when the credentials change.
type StreamingCredentialsProvider interface {
	// Subscribe subscribes to the credentials provider for updates.
	// It returns the current credentials, a cancel function to unsubscribe from the provider,
	// and an error if any.
	//
	// Implementations MUST be idempotent with respect to listener identity:
	// subscribing the same listener value more than once must not produce
	// duplicate notifications and must not create multiple independent
	// subscriptions that each need to be cancelled separately. Every
	// UnsubscribeFunc returned for a given listener must cancel that
	// listener's subscription; calling any one of them must be sufficient to
	// stop updates to that listener, and calling subsequent ones must be a
	// safe no-op. Callers (including go-redis internals) may retain only
	// the most recently returned UnsubscribeFunc and rely on it to fully
	// unsubscribe the listener.
	//
	// TODO(ndyakov): Should we add context to the Subscribe method?
	Subscribe(listener CredentialsListener) (Credentials, UnsubscribeFunc, error)
}

// UnsubscribeFunc is a function that is used to cancel the subscription to the credentials provider.
// It is used to unsubscribe from the provider when the credentials are no longer needed.
//
// Per the StreamingCredentialsProvider.Subscribe contract, if the same
// listener is subscribed multiple times, every UnsubscribeFunc returned for
// that listener must fully unsubscribe it on first invocation, and
// subsequent invocations (from any of the equivalent UnsubscribeFuncs) must
// be a safe no-op.
type UnsubscribeFunc func() error

// CredentialsListener is an interface that defines the methods for a credentials listener.
// It is used to receive updates when the credentials change.
// The OnNext method is called when the credentials change.
// The OnError method is called when an error occurs while requesting the credentials.
type CredentialsListener interface {
	OnNext(credentials Credentials)
	OnError(err error)
}

// Credentials is an interface that defines the methods for credentials.
// It is used to provide the credentials for authentication.
type Credentials interface {
	// BasicAuth returns the username and password for basic authentication.
	BasicAuth() (username string, password string)
	// RawCredentials returns the raw credentials as a string.
	// This can be used to extract the username and password from the raw credentials or
	// additional information if present in the token.
	RawCredentials() string
}

type basicAuth struct {
	username string
	password string
}

// RawCredentials returns the raw credentials as a string.
func (b *basicAuth) RawCredentials() string {
	return b.username + ":" + b.password
}

// BasicAuth returns the username and password for basic authentication.
func (b *basicAuth) BasicAuth() (username string, password string) {
	return b.username, b.password
}

// NewBasicCredentials creates a new Credentials object from the given username and password.
func NewBasicCredentials(username, password string) Credentials {
	return &basicAuth{
		username: username,
		password: password,
	}
}
