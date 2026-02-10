// WAV test example: load a WAV file, convert to mono float32,
// feed 512-sample chunks into the engine, and print callback events.
package main

import (
	"fmt"
	"io"
	"os"
	"path/filepath"

	"github.com/bgaurav7/smart-turn-go"
	"github.com/youpy/go-wav"
)

const (
	chunkSize   = 512
	segmentRate = 16000
)

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintf(os.Stderr, "Usage: %s <wav_file> [model_dir] [output_dir]\n", os.Args[0])
		os.Exit(1)
	}
	wavPath := os.Args[1]
	modelDir := "data"
	outDir := "output"
	if len(os.Args) >= 3 {
		modelDir = os.Args[2]
	}
	if len(os.Args) >= 4 {
		outDir = os.Args[3]
	}
	if err := os.MkdirAll(outDir, 0755); err != nil {
		fmt.Fprintf(os.Stderr, "output dir: %v\n", err)
		os.Exit(1)
	}
	var segmentNum int
	sileroPath := filepath.Join(modelDir, "silero_vad.onnx")
	smartTurnPath := filepath.Join(modelDir, "smart-turn-v3.2-cpu.onnx")
	if a, err := filepath.Abs(sileroPath); err == nil {
		sileroPath = a
	}
	if a, err := filepath.Abs(smartTurnPath); err == nil {
		smartTurnPath = a
	}

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
		OnSegmentReady: func(seg []float32) {
			segmentNum++
			name := filepath.Join(outDir, fmt.Sprintf("segment_%03d.wav", segmentNum))
			if err := saveSegmentWAV(name, seg, segmentRate); err != nil {
				fmt.Fprintf(os.Stderr, "save %s: %v\n", name, err)
				return
			}
			fmt.Printf("[event] segment ready (%d samples) -> %s\n", len(seg), name)
		},
		OnError: func(err error) { fmt.Printf("[error] %v\n", err) },
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

	wavReader := wav.NewReader(f)
	format, err := wavReader.Format()
	if err != nil {
		return nil, 0, fmt.Errorf("WAV format: %w", err)
	}
	sampleRate = int(format.SampleRate)
	numChannels := int(format.NumChannels)
	if numChannels < 1 || numChannels > 2 {
		return nil, 0, fmt.Errorf("WAV: only mono or stereo supported, got %d channels", numChannels)
	}

	var out []float32
	for {
		readSamples, err := wavReader.ReadSamples()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, 0, fmt.Errorf("reading WAV samples: %w", err)
		}
		for _, s := range readSamples {
			var v float64
			if numChannels == 1 {
				v = wavReader.FloatValue(s, 0)
			} else {
				v = (wavReader.FloatValue(s, 0) + wavReader.FloatValue(s, 1)) / 2
			}
			out = append(out, float32(v))
		}
	}
	return out, sampleRate, nil
}

func saveSegmentWAV(path string, samples []float32, sampleRate int) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	// go-wav Sample.Values[0] is the PCM value (e.g. 16-bit: -32768..32767)
	wavSamples := make([]wav.Sample, len(samples))
	for i, v := range samples {
		if v < -1 {
			v = -1
		}
		if v > 1 {
			v = 1
		}
		wavSamples[i] = wav.Sample{Values: [2]int{int(v * 32767), 0}}
	}
	writer := wav.NewWriter(f, uint32(len(wavSamples)), 1, uint32(sampleRate), 16)
	return writer.WriteSamples(wavSamples)
}
