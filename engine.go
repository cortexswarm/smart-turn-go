package smartturn

import (
	"errors"
	"os"

	ort "github.com/yalue/onnxruntime_go"
)

// EnvONNXRuntimeLib is the environment variable read before initializing ONNX.
// If set, it is used as the ONNX Runtime shared library path (e.g. on macOS
// after brew install onnxruntime, set to the path to libonnxruntime.dylib).
const EnvONNXRuntimeLib = "ONNXRUNTIME_SHARED_LIBRARY_PATH"

var (
	ErrChunkSize = errors.New("chunk must be exactly 512 samples")
)

// Engine is the main SDK entry. It is single-threaded and not goroutine-safe;
// the caller must serialize PushPCM and lifecycle methods.
type Engine struct {
	cfg       Config
	cb        Callbacks
	vad       *sileroVAD
	segmenter *segmenter
	smartTurn *smartTurn

	listening bool
	closed    bool

	segmentEmitSamples  int // target samples per OnSegmentReady slice
	segmentEmittedSoFar int // how many samples of the current segment have been emitted
}

// New creates an engine from config and callbacks. It validates config, loads ONNX
// models, and creates sessions. The ONNX Runtime shared library path is set explicitly
// (as recommended by onnxruntime_go): first from bundled lib under BundledLibDir
// (e.g. lib/darwin_arm64/libonnxruntime.dylib), then overridden by EnvONNXRuntimeLib
// if set. Call ort.SetSharedLibraryPath before New for any other custom path.
func New(cfg Config, cb Callbacks) (*Engine, error) {
	if err := validateConfig(cfg); err != nil {
		return nil, err
	}
	// Set library path explicitly; default (onnxruntime.so on non-Windows) fails on macOS.
	if path := os.Getenv(EnvONNXRuntimeLib); path != "" {
		ort.SetSharedLibraryPath(path)
	} else if bundled := resolveBundledLib(candidateBaseDirs()); bundled != "" {
		ort.SetSharedLibraryPath(bundled)
	}
	if err := ort.InitializeEnvironment(); err != nil {
		return nil, err
	}
	e := &Engine{cfg: cfg, cb: cb}
	vad, err := newSileroVAD(cfg.SileroVADModelPath)
	if err != nil {
		return nil, err
	}
	st, err := newSmartTurn(cfg.SmartTurnModelPath)
	if err != nil {
		_ = vad.destroy()
		return nil, err
	}
	seg := newSegmenter(cfg.SampleRate, cfg.ChunkSize, cfg.PreSpeechMs, cfg.StopMs, cfg.MaxDurationSeconds)
	e.vad = vad
	e.segmenter = seg
	e.smartTurn = st
	// Derive how many samples correspond to one emit interval.
	if cfg.SegmentEmitMs > 0 {
		e.segmentEmitSamples = int(float64(cfg.SegmentEmitMs) * float64(cfg.SampleRate) / 1000.0)
		if e.segmentEmitSamples <= 0 {
			e.segmentEmitSamples = cfg.ChunkSize
		}
	} else {
		e.segmentEmitSamples = cfg.ChunkSize
	}
	return e, nil
}

// Start starts listening. Invokes OnListeningStarted callback.
func (e *Engine) Start() {
	if e.closed {
		return
	}
	e.listening = true
	if e.cb.OnListeningStarted != nil {
		e.cb.OnListeningStarted()
	}
}

// Stop stops listening. Invokes OnListeningStopped callback.
func (e *Engine) Stop() {
	if e.closed {
		return
	}
	e.listening = false
	if e.cb.OnListeningStopped != nil {
		e.cb.OnListeningStopped()
	}
}

// PushPCM processes one chunk of 512 float32 samples (mono, 16 kHz).
// Returns ErrChunkSize if len(chunk) != 512. Callbacks are invoked synchronously.
func (e *Engine) PushPCM(chunk []float32) error {
	if e.closed {
		return errors.New("engine is closed")
	}
	if len(chunk) != RequiredChunkSize {
		return ErrChunkSize
	}
	if !e.listening {
		return nil
	}

	prob, err := e.vad.speechProb(chunk)
	if err != nil {
		if e.cb.OnError != nil {
			e.cb.OnError(err)
		}
		return err
	}
	isSpeech := prob > e.cfg.VadThreshold

	res := e.segmenter.processChunk(isSpeech, chunk)
	// Reset emitted counter on a new segment.
	if res.Started {
		e.segmentEmittedSoFar = 0
	}
	if res.Started && e.cb.OnSpeechStart != nil {
		e.cb.OnSpeechStart()
	}
	if e.cb.OnChunk != nil {
		e.cb.OnChunk(chunk)
	}

	// While speech is active, res.Segment holds the full accumulated segment so far.
	if len(res.Segment) > 0 && e.segmentEmitSamples > 0 && e.cb.OnSegmentReady != nil {
		total := len(res.Segment)
		// Emit fixed-size slices as we cross each interval boundary.
		for total-e.segmentEmittedSoFar >= e.segmentEmitSamples {
			start := e.segmentEmittedSoFar
			end := start + e.segmentEmitSamples
			slice := make([]float32, end-start)
			copy(slice, res.Segment[start:end])
			e.cb.OnSegmentReady(slice)
			e.segmentEmittedSoFar = end
		}
	}

	if res.Ended {
		// Emit any remaining tail for this segment before speech end callback.
		if len(res.Segment) > e.segmentEmittedSoFar && e.cb.OnSegmentReady != nil {
			start := e.segmentEmittedSoFar
			end := len(res.Segment)
			slice := make([]float32, end-start)
			copy(slice, res.Segment[start:end])
			e.cb.OnSegmentReady(slice)
		}
		if e.cb.OnSpeechEnd != nil {
			e.cb.OnSpeechEnd()
		}
		e.segmentEmittedSoFar = 0
	}
	return nil
}

// Reset clears VAD state and segment state. Sessions are not closed.
func (e *Engine) Reset() {
	if e.closed {
		return
	}
	e.vad.resetState()
	e.segmenter.reset()
}

// Close releases ONNX sessions and resources. The engine must not be used after Close.
func (e *Engine) Close() {
	if e.closed {
		return
	}
	e.closed = true
	e.listening = false
	if err := e.vad.destroy(); err != nil && e.cb.OnError != nil {
		e.cb.OnError(err)
	}
	if err := e.smartTurn.destroy(); err != nil && e.cb.OnError != nil {
		e.cb.OnError(err)
	}
}
