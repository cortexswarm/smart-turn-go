// Live mic example: captures from the default microphone, runs Smart-Turn (VAD + turn detection),
// and prints callbacks. Run from repo root: go run ./examples/mic
//
// Requires: go get -u github.com/gen2brain/malgo
package main

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sync"

	"github.com/cortexswarm/smart-turn-go"
	"github.com/gen2brain/malgo"
)

const (
	chunkSize    = 512
	sampleRate   = 16000
	defaultModel = "data"
)

func main() {
	ctx, err := malgo.InitContext(nil, malgo.ContextConfig{}, nil)
	if err != nil {
		fmt.Fprintf(os.Stderr, "malgo init: %v\n", err)
		os.Exit(1)
	}
	defer func() {
		_ = ctx.Uninit()
		ctx.Free()
	}()

	sileroPath := filepath.Join(defaultModel, "silero_vad.onnx")
	smartTurnPath := filepath.Join(defaultModel, "smart-turn-v3.2-cpu.onnx")
	if a, err := filepath.Abs(sileroPath); err == nil {
		sileroPath = a
	}
	if a, err := filepath.Abs(smartTurnPath); err == nil {
		smartTurnPath = a
	}

	cfg := smartturn.Config{
		SampleRate:         sampleRate,
		ChunkSize:          chunkSize,
		VadThreshold:       0.75,
		PreSpeechMs:        200,
		StopMs:             500,
		MaxDurationSeconds: 600,
		SegmentEmitMs:      1000,
		SileroVADModelPath: sileroPath,
		SmartTurnModelPath: smartTurnPath,
	}
	cb := smartturn.Callbacks{
		OnListeningStarted: func() { fmt.Println("[callback] listening started") },
		OnListeningStopped: func() { fmt.Println("[callback] listening stopped") },
		OnSpeechStart:      func() { fmt.Println("[callback] speech start") },
		OnSpeechEnd:        func() { fmt.Println("[callback] speech end") },
		OnSegmentReady:     func(seg []float32) { fmt.Printf("[callback] segment ready (%d samples)\n", len(seg)) },
		OnError:            func(err error) { fmt.Printf("[callback] error: %v\n", err) },
	}

	engine, err := smartturn.New(cfg, cb)
	if err != nil {
		fmt.Fprintf(os.Stderr, "engine: %v\n", err)
		os.Exit(1)
	}
	defer engine.Close()

	// Chunks of 512 float32 sent from capture callback to engine goroutine
	chunkCh := make(chan []float32, 64)
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		engine.Start()
		defer engine.Stop()
		for ch := range chunkCh {
			_ = engine.PushPCM(ch)
		}
	}()

	deviceConfig := malgo.DefaultDeviceConfig(malgo.Capture)
	deviceConfig.Capture.Format = malgo.FormatF32
	deviceConfig.Capture.Channels = 1
	deviceConfig.SampleRate = sampleRate
	deviceConfig.Alsa.NoMMap = 1

	var buf []float32
	onRecvFrames := func(_, pSample []byte, framecount uint32) {
		if framecount == 0 {
			return
		}
		n := int(framecount) * int(deviceConfig.Capture.Channels)
		for i := 0; i < n; i++ {
			buf = append(buf, float32FromBytes(pSample[i*4:]))
		}
		for len(buf) >= chunkSize {
			chunk := make([]float32, chunkSize)
			copy(chunk, buf[:chunkSize])
			buf = append(buf[:0], buf[chunkSize:]...)
			select {
			case chunkCh <- chunk:
			default:
				// drop if consumer is slow
			}
		}
	}

	device, err := malgo.InitDevice(ctx.Context, deviceConfig, malgo.DeviceCallbacks{Data: onRecvFrames})
	if err != nil {
		fmt.Fprintf(os.Stderr, "init device: %v\n", err)
		os.Exit(1)
	}
	defer device.Uninit()

	if err := device.Start(); err != nil {
		fmt.Fprintf(os.Stderr, "device start: %v\n", err)
		os.Exit(1)
	}

	fmt.Println("Capturing from default microphone. Callbacks will print below. Press Enter to stop...")
	fmt.Scanln()

	close(chunkCh)
	wg.Wait()
}

func float32FromBytes(b []byte) float32 {
	return math.Float32frombits(binary.LittleEndian.Uint32(b))
}
