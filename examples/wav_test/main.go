// WAV test example: load a WAV file, convert to 16 kHz mono float32,
// feed 512-sample chunks into the engine, and print callback events.
package main

import (
	"encoding/binary"
	"fmt"
	"io"
	"os"
	"path/filepath"

	"github.com/bgaurav7/smart-turn-go"
)

const chunkSize = 512

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintf(os.Stderr, "Usage: %s <wav_file> [model_dir]\n", os.Args[0])
		os.Exit(1)
	}
	wavPath := os.Args[1]
	modelDir := "data"
	if len(os.Args) >= 3 {
		modelDir = os.Args[2]
	}
	sileroPath := filepath.Join(modelDir, "silero_vad.onnx")
	smartTurnPath := filepath.Join(modelDir, "smart-turn-v3.2-cpu.onnx")

	cfg := smartturn.Config{
		SampleRate:         16000,
		ChunkSize:          512,
		VadThreshold:       0.5,
		PreSpeechMs:        200,
		StopMs:             1000,
		MaxDurationSeconds: 8,
		SileroVADModelPath: sileroPath,
		SmartTurnModelPath: smartTurnPath,
	}
	cb := smartturn.Callbacks{
		OnListeningStarted: func() { fmt.Println("[event] listening started") },
		OnListeningStopped: func() { fmt.Println("[event] listening stopped") },
		OnSpeechStart:      func() { fmt.Println("[event] speech start") },
		OnSpeechEnd:        func() { fmt.Println("[event] speech end") },
		OnSegmentReady:    func(seg []float32) { fmt.Printf("[event] segment ready (%d samples)\n", len(seg)) },
		OnError:            func(err error) { fmt.Printf("[error] %v\n", err) },
	}

	engine, err := smartturn.New(cfg, cb)
	if err != nil {
		fmt.Fprintf(os.Stderr, "New: %v\n", err)
		os.Exit(1)
	}
	defer engine.Close()

	samples, sampleRate, err := loadWAV(wavPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "load WAV: %v\n", err)
		os.Exit(1)
	}
	if sampleRate != 16000 {
		fmt.Fprintf(os.Stderr, "warning: WAV is %d Hz; resampling not implemented, using as-is\n", sampleRate)
	}

	engine.Start()
	defer engine.Stop()

	for i := 0; i+chunkSize <= len(samples); i += chunkSize {
		chunk := samples[i : i+chunkSize]
		if err := engine.PushPCM(chunk); err != nil {
			fmt.Fprintf(os.Stderr, "PushPCM: %v\n", err)
			os.Exit(1)
		}
	}
	fmt.Println("done")
}

func loadWAV(path string) (samples []float32, sampleRate int, err error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, 0, err
	}
	defer f.Close()

	var riff [4]byte
	if _, err := io.ReadFull(f, riff[:]); err != nil {
		return nil, 0, err
	}
	if string(riff[:]) != "RIFF" {
		return nil, 0, fmt.Errorf("not a RIFF file")
	}
	var fileSize uint32
	if err := binary.Read(f, binary.LittleEndian, &fileSize); err != nil {
		return nil, 0, err
	}
	var wave [4]byte
	if _, err := io.ReadFull(f, wave[:]); err != nil {
		return nil, 0, err
	}
	if string(wave[:]) != "WAVE" {
		return nil, 0, fmt.Errorf("not WAVE format")
	}

	var numChannels uint16
	var bitsPerSample uint16
	var sampleRateU32 uint32
	for {
		var chunkID [4]byte
		if _, err := io.ReadFull(f, chunkID[:]); err != nil {
			return nil, 0, err
		}
		var chunkLen uint32
		if err := binary.Read(f, binary.LittleEndian, &chunkLen); err != nil {
			return nil, 0, err
		}
		switch string(chunkID[:]) {
		case "fmt ":
			var fmtCode uint16
			if err := binary.Read(f, binary.LittleEndian, &fmtCode); err != nil {
				return nil, 0, err
			}
			if fmtCode != 1 {
				return nil, 0, fmt.Errorf("unsupported format (not PCM)")
			}
			if err := binary.Read(f, binary.LittleEndian, &numChannels); err != nil {
				return nil, 0, err
			}
			if err := binary.Read(f, binary.LittleEndian, &sampleRateU32); err != nil {
				return nil, 0, err
			}
			sampleRate = int(sampleRateU32)
			var byteRate, blockAlign uint16
			if err := binary.Read(f, binary.LittleEndian, &byteRate); err != nil {
				return nil, 0, err
			}
			if err := binary.Read(f, binary.LittleEndian, &blockAlign); err != nil {
				return nil, 0, err
			}
			if err := binary.Read(f, binary.LittleEndian, &bitsPerSample); err != nil {
				return nil, 0, err
			}
			remain := int64(chunkLen) - 16
			if remain > 0 {
				_, _ = f.Seek(remain, io.SeekCurrent)
			}
		case "data":
			if numChannels == 0 {
				return nil, 0, fmt.Errorf("fmt chunk before data")
			}
			raw := make([]byte, chunkLen)
			if _, err := io.ReadFull(f, raw); err != nil {
				return nil, 0, err
			}
			if bitsPerSample != 16 {
				return nil, 0, fmt.Errorf("only 16-bit PCM supported")
			}
			n := len(raw) / 2
			samples = make([]float32, n/int(numChannels))
			for i := 0; i < len(samples); i++ {
				idx := i * int(numChannels) * 2
				s := int16(binary.LittleEndian.Uint16(raw[idx : idx+2]))
				samples[i] = float32(s) / 32768.0
			}
			return samples, sampleRate, nil
		default:
			_, _ = f.Seek(int64(chunkLen), io.SeekCurrent)
		}
	}
}
