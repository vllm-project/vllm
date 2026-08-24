package auth

// ReAuthCredentialsListener is a struct that implements the CredentialsListener interface.
// It is used to re-authenticate the credentials when they are updated.
// It contains:
// - reAuth: a function that takes the new credentials and returns an error if any.
// - onErr: a function that takes an error and handles it.
type ReAuthCredentialsListener struct {
	reAuth func(credentials Credentials) error
	onErr  func(err error)
}

// OnNext is called when the credentials are updated.
// It calls the reAuth function with the new credentials.
// If the reAuth function returns an error, it calls the onErr function with the error.
func (c *ReAuthCredentialsListener) OnNext(credentials Credentials) {
	if c.reAuth == nil {
		return
	}

	err := c.reAuth(credentials)
	if err != nil {
		c.OnError(err)
	}
}

// OnError is called when an error occurs.
// It can be called from both the credentials provider and the reAuth function.
func (c *ReAuthCredentialsListener) OnError(err error) {
	if c.onErr == nil {
		return
	}

	c.onErr(err)
}

// NewReAuthCredentialsListener creates a new ReAuthCredentialsListener.
// Implements the auth.CredentialsListener interface.
func NewReAuthCredentialsListener(reAuth func(credentials Credentials) error, onErr func(err error)) *ReAuthCredentialsListener {
	return &ReAuthCredentialsListener{
		reAuth: reAuth,
		onErr:  onErr,
	}
}

// Ensure ReAuthCredentialsListener implements the CredentialsListener interface.
var _ CredentialsListener = (*ReAuthCredentialsListener)(nil)
