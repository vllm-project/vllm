package viper

// ExperimentalBindStruct tells Viper to use the new bind struct feature.
func ExperimentalBindStruct() Option {
	return optionFunc(func(v *Viper) {
		v.experimentalBindStruct = true
	})
}
