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
	// OnSegmentReady receives segment audio; the engine may reuse the slice after the callback returns—copy if retaining.
	OnSegmentReady func(segment []float32)

	// OnTurnPrediction receives Smart-Turn's decision when a segment ends by VAD
	// silence (not by max-duration cap). `complete` is true when the model
	// thinks the turn is finished; `probability` is the underlying score.
	OnTurnPrediction func(complete bool, probability float32)

	OnError func(err error)
}
