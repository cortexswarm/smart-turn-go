# smart-turn

A production-grade, streaming Go SDK for detecting speech turns from continuous PCM audio using Silero VAD and the Smart-Turn ONNX model.

---

## Credits

- **Smart Turn v3.2 ([pipecat-ai/smart-turn](https://github.com/pipecat-ai/smart-turn), BSD 2-Clause)**: Open source, native audio turn detection model. Special thanks to the Pipecat team and contributors.
- **Silero VAD ([snakers4/silero-vad](https://github.com/snakers4/silero-vad), MIT)**: Enterprise-grade voice activity detection models by Silero.
- **ONNX Runtime Go Wrapper ([yalue/onnxruntime_go](https://github.com/yalue/onnxruntime_go), MIT)**: Cross-platform Go bindings for Microsoft ONNX Runtime.

---

## Overview

- **Language:** Go
- **Goal:** Detect speech turns from continuous mono PCM audio (16 kHz, `float32`), processed in fixed 512-sample frames.
- **Models Used:** Silero VAD and Smart-Turn v3.2 (CPU/ONNX).
- **Input:** Audio provided by the host application. No microphone capture or resampling in SDK.

---

## Requirements

- **Go 1.21+** (with CGO enabled):  
  Fetch dependencies with `go mod tidy` or `go get github.com/yalue/onnxruntime_go@v1.25.0`.
  If you see _missing go.sum entry_ for `github.com/gen2brain/malgo` (e.g. when running `./examples/mic`), run:  
  `go mod download github.com/gen2brain/malgo` or `go mod tidy`.
- **ONNX Runtime Shared Library:**  
  - Typically bundled under `data/` (e.g. `data/onnxruntime_arm64.dylib`) or `lib/<GOOS>_<GOARCH>/`.  
  - If not found, set the `ONNXRUNTIME_SHARED_LIBRARY_PATH` environment variable.
  - [onnxruntime_go](https://github.com/yalue/onnxruntime_go) targets ONNX Runtime 1.23.2.
- **Model Files:** _(must be provided by user)_
  - `silero_vad.onnx`
  - `smart-turn-v3.2-cpu.onnx`

---

## Configuration

Example setup:

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

- All configuration fields are validated in `New()`.  
- Invalid configs or missing model files produce an error.

---

## Callbacks

All callbacks are **optional** (can be left `nil`) and are invoked **synchronously** from the same goroutine that calls `PushPCM`. The SDK does **NOT** spawn goroutines.

Available callbacks:

- `OnListeningStarted` / `OnListeningStopped`
- `OnSpeechStart` / `OnSpeechEnd`
- `OnChunk(chunk []float32)`
- `OnSegmentReady(segment []float32)`
- `OnError(err error)`

---

## Engine API

- `New(cfg Config, cb Callbacks) (*Engine, error)`  
  Validates config; loads ONNX sessions.
- `Start()` / `Stop()`  
  Toggles listening, invokes relevant callbacks.
- `PushPCM(chunk []float32) error`  
  Processes a chunk (must be **exactly 512 samples**). Returns `ErrChunkSize` when length is incorrect.
- `Reset()`  
  Resets VAD and segment state but keeps model sessions loaded.
- `Close()`  
  Releases ONNX resources. Must not use the engine after closing.

> **Note:** The engine is **single-threaded and not goroutine-safe**. All API calls should be serialized by the caller.

---

## Example Usage

From the project root (models in `data/`):

**WAV file** (defaults: `data/test.wav`, output to `output/`):

```bash
go run ./examples/file
go run ./examples/file /path/to/audio.wav my_output
```

**Live microphone** (captures from default mic, prints callbacks; requires [malgo](https://github.com/gen2brain/malgo)):

```bash
go get -u github.com/gen2brain/malgo
go run ./examples/mic
```

- The WAV example (`examples/file/main.go`) uses [github.com/youpy/go-wav](https://github.com/youpy/go-wav) to load WAVs, converts to mono `float32`, and processes 512-sample chunks. The mic example (`examples/mic/main.go`) captures at 16 kHz mono via malgo and feeds the engine in real time.

---
