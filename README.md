# smart-turn

Production-grade streaming SDK in Go for detecting speech turns from continuous PCM audio using Silero VAD and the Smart-Turn ONNX model.

## Overview

- **Language:** Go  
- **Purpose:** Detect speech turns from continuous PCM (mono, 16 kHz, float32) using fixed 512-sample chunks.  
- **Models:** Silero VAD and Smart-Turn v3.2 CPU (ONNX). No microphone capture or resampling; the host app supplies audio.

## Requirements

- Go 1.21+ with **CGO enabled** (required by [github.com/yalue/onnxruntime_go](https://github.com/yalue/onnxruntime_go)). Run `go mod tidy` or `go get github.com/yalue/onnxruntime_go@v1.25.0` to fetch the dependency and populate `go.sum`.
- ONNX Runtime shared library. Set `ort.SetSharedLibraryPath("path/to/onnxruntime.so")` before calling `smartturn.New` if the library is not on the default search path.
- Model files (provided by you; not downloaded by the SDK):
  - `silero_vad.onnx`
  - `smart-turn-v3.2-cpu.onnx`

## Project layout

```
smart-turn/
├── go.mod
├── config.go
├── callbacks.go
├── engine.go
├── silero_vad.go
├── segment.go
├── smart_turn.go
├── whisper_mel.go
├── data/
│   ├── silero_vad.onnx
│   └── smart-turn-v3.2-cpu.onnx
├── examples/
│   └── wav_test/
│       └── main.go
└── README.md
```

All code is in package `smartturn`. Only the SDK surface is exported; internal types (VAD, segmenter, Smart-Turn) are unexported.

## Configuration

```go
cfg := smartturn.Config{
    SampleRate:         16000,   // must be 16000
    ChunkSize:          512,     // must be 512
    VadThreshold:       0.5,
    PreSpeechMs:        200,
    StopMs:             1000,
    MaxDurationSeconds: 8,
    SileroVADModelPath: "data/silero_vad.onnx",
    SmartTurnModelPath: "data/smart-turn-v3.2-cpu.onnx",
}
```

All fields are validated in `New()`; invalid config or missing model files return an error.

## Callbacks

Callbacks are optional (nil allowed) and invoked **synchronously** from the same goroutine that calls `PushPCM`. The SDK does not spawn goroutines.

- `OnListeningStarted` / `OnListeningStopped`
- `OnSpeechStart` / `OnSpeechEnd`
- `OnChunk(chunk []float32)`
- `OnSegmentReady(segment []float32)`
- `OnError(err error)`

## Engine API

- `New(cfg Config, cb Callbacks) (*Engine, error)` — validates config, loads ONNX sessions.
- `Start()` / `Stop()` — toggles listening and invokes the corresponding callbacks.
- `PushPCM(chunk []float32) error` — processes one chunk of **exactly 512** samples. Returns `ErrChunkSize` if length is wrong.
- `Reset()` — resets VAD and segment state; sessions stay loaded.
- `Close()` — releases ONNX resources; engine must not be used after.

The engine is **single-threaded and not goroutine-safe**. Serialize all calls from the caller side.

## Example

From the repo root (with models in `data/`):

```bash
go run ./examples/wav_test data/test.wav
```

Or with a custom model directory:

```bash
go run ./examples/wav_test /path/to/audio.wav /path/to/models
```

The example loads a WAV file, converts to mono float32, slices into 512-sample chunks, and feeds them to the engine while printing callback events.

## Non-goals

- No microphone capture, resampling, or concurrency inside the SDK.
- No logging in hot paths.
- No public exposure of ONNX or VAD internals.

## License

See LICENSE.
