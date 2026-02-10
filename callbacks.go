package smartturn

// Callbacks are invoked synchronously by the engine from the same goroutine
// that calls PushPCM. The SDK does not spawn goroutines. All fields are
// optional (nil is allowed).
type Callbacks struct {
	OnListeningStarted func()
	OnListeningStopped func()

	OnSpeechStart func()
	OnSpeechEnd   func()

	OnChunk        func(chunk []float32)
	OnSegmentReady func(segment []float32)

	OnError func(err error)
}
