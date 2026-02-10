# smart-turn

Production-grade streaming SDK in Go for detecting speech turns from continuous PCM audio using Silero VAD and the Smart-Turn ONNX model.

**Credits.** This project is a Go implementation built on the [**Smart Turn v3.2**](https://github.com/pipecat-ai/smart-turn) model and design from [pipecat-ai/smart-turn](https://github.com/pipecat-ai/smart-turn). Smart Turn is an open source, community-driven, native audio turn detection model (BSD 2-clause); we thank the Pipecat team and contributors for the original work. VAD is powered by [**Silero VAD**](https://github.com/snakers4/silero-vad) (MIT)—pre-trained enterprise-grade voice activity detection from the Silero team. ONNX inference uses [**yalue/onnxruntime_go**](https://github.com/yalue/onnxruntime_go) (MIT), a cross-platform Go wrapper for Microsoft ONNX Runtime.

## Overview

- **Language:** Go  
- **Purpose:** Detect speech turns from continuous PCM (mono, 16 kHz, float32) using fixed 512-sample chunks.  
- **Models:** Silero VAD and Smart-Turn v3.2 CPU (ONNX). No microphone capture or resampling; the host app supplies audio.

## Requirements

- Go 1.21+ with **CGO enabled** (required by [github.com/yalue/onnxruntime_go](https://github.com/yalue/onnxruntime_go)). Run `go mod tidy` or `go get github.com/yalue/onnxruntime_go@v1.25.0` to fetch the dependency and populate `go.sum`.
- **ONNX Runtime shared library.** You do **not** need `ONNXRUNTIME_SHARED_LIBRARY_PATH` if the runtime is bundled under **`data/`** (e.g. `data/onnxruntime_arm64.dylib`) or `lib/<GOOS>_<GOARCH>/`. The SDK resolves the path in that order, then falls back to the env var for overrides. [onnxruntime_go](https://github.com/yalue/onnxruntime_go) targets **ONNX Runtime 1.23.2**.
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
├── onnxruntime_lib.go
├── silero_vad.go
├── segment.go
├── smart_turn.go
├── whisper_mel.go
├── data/
│   ├── silero_vad.onnx
│   ├── smart-turn-v3.2-cpu.onnx
│   ├── onnxruntime_arm64.dylib   # optional: bundled ONNX Runtime (darwin arm64)
│   ├── onnxruntime_amd64.dylib   # optional: darwin amd64
│   ├── onnxruntime_arm64.so      # optional: linux arm64
│   ├── onnxruntime_amd64.so      # optional: linux amd64
│   └── onnxruntime.dll           # optional: windows
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

The example uses [github.com/youpy/go-wav](https://github.com/youpy/go-wav) to load a WAV file, converts to mono float32 (averaging channels for stereo), slices into 512-sample chunks, and feeds them to the engine while printing callback events.

## Troubleshooting

- **"Protobuf parsing failed"** when loading a model: the ONNX file may be corrupted or from an incompatible export. Use the official assets: [Silero VAD](https://github.com/snakers4/silero-vad) (e.g. `silero_vad.onnx` from the repo or releases), [Smart-Turn v3.2](https://github.com/pipecat-ai/smart-turn) (e.g. `smart-turn-v3.2-cpu.onnx`). The example resolves model paths to absolute paths to avoid CWD-related load issues.

## Non-goals

- No microphone capture, resampling, or concurrency inside the SDK.
- No logging in hot paths.
- No public exposure of ONNX or VAD internals.

## License

See LICENSE.
