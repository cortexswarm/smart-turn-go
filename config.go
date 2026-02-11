package smartturn

import (
	"errors"
	"os"
)

const (
	RequiredSampleRate = 16000
	RequiredChunkSize  = 512
)

// Config holds SDK configuration. All fields must be set; no silent defaults.
type Config struct {
	SampleRate         int     // must be 16000
	ChunkSize          int     // must be 512
	VadThreshold       float32 // speech probability threshold (e.g. 0.5)
	PreSpeechMs        int     // ms of audio to keep before speech trigger (e.g. 200)
	StopMs             int     // ms of trailing silence to end segment (e.g. 500)
	MaxDurationSeconds float32 // hard cap per segment in seconds (e.g. 600 for 10 minutes)

	// SegmentEmitMs controls how often OnSegmentReady is called while speech is active.
	// For example, 1000 emits 1-second slices; any remaining tail is emitted before OnSpeechEnd.
	SegmentEmitMs int

	SileroVADModelPath string // path to silero_vad.onnx
	SmartTurnModelPath string // path to smart-turn-v3.2-cpu.onnx
}

// validate checks Config and returns an error on invalid or missing values.
func validateConfig(cfg Config) error {
	if cfg.SampleRate != RequiredSampleRate {
		return errors.New("config: SampleRate must be 16000")
	}
	if cfg.ChunkSize != RequiredChunkSize {
		return errors.New("config: ChunkSize must be 512")
	}
	if cfg.VadThreshold < 0 || cfg.VadThreshold > 1 {
		return errors.New("config: VadThreshold must be in [0, 1]")
	}
	if cfg.PreSpeechMs < 0 {
		return errors.New("config: PreSpeechMs must be >= 0")
	}
	if cfg.StopMs <= 0 {
		return errors.New("config: StopMs must be > 0")
	}
	if cfg.MaxDurationSeconds <= 0 {
		return errors.New("config: MaxDurationSeconds must be > 0")
	}
	if cfg.SegmentEmitMs <= 0 {
		return errors.New("config: SegmentEmitMs must be > 0")
	}
	if cfg.SileroVADModelPath == "" {
		return errors.New("config: SileroVADModelPath is required")
	}
	if cfg.SmartTurnModelPath == "" {
		return errors.New("config: SmartTurnModelPath is required")
	}
	if _, err := os.Stat(cfg.SileroVADModelPath); err != nil {
		if os.IsNotExist(err) {
			return errors.New("config: Silero VAD model file not found: " + cfg.SileroVADModelPath)
		}
		return err
	}
	if _, err := os.Stat(cfg.SmartTurnModelPath); err != nil {
		if os.IsNotExist(err) {
			return errors.New("config: Smart-Turn model file not found: " + cfg.SmartTurnModelPath)
		}
		return err
	}
	return nil
}
