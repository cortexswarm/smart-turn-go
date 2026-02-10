package smartturn

import (
	"errors"
	"time"

	ort "github.com/yalue/onnxruntime_go"
)

var errChunkSize = errors.New("chunk must be exactly 512 samples")

const (
	sileroContextSamples = 64
	sileroInputSamples   = sileroContextSamples + RequiredChunkSize // 576
	sileroStateSize      = 2 * 1 * 128
	sileroResetInterval  = 5 * time.Second
)

// sileroVAD is a stateful ONNX wrapper for Silero VAD. Not safe for concurrent use.
type sileroVAD struct {
	session  *ort.AdvancedSession
	input    *ort.Tensor[float32]   // (1, 576)
	state    *ort.Tensor[float32]   // (2, 1, 128)
	sr       *ort.Tensor[int64]     // (1,) = 16000
	output   *ort.Tensor[float32]   // (1, 1) speech prob
	stateOut *ort.Tensor[float32]   // (2, 1, 128) new state

	context [sileroContextSamples]float32
	stateBuf [sileroStateSize]float32
	lastReset time.Time
}

func newSileroVAD(modelPath string) (*sileroVAD, error) {
	inputShape := ort.NewShape(1, sileroInputSamples)
	inputData := make([]float32, sileroInputSamples)
	inputTensor, err := ort.NewTensor(inputShape, inputData)
	if err != nil {
		return nil, err
	}

	stateShape := ort.NewShape(2, 1, 128)
	stateData := make([]float32, sileroStateSize)
	stateTensor, err := ort.NewTensor(stateShape, stateData)
	if err != nil {
		_ = inputTensor.Destroy()
		return nil, err
	}

	srShape := ort.NewShape(1)
	srData := []int64{16000}
	srTensor, err := ort.NewTensor(srShape, srData)
	if err != nil {
		_ = inputTensor.Destroy()
		_ = stateTensor.Destroy()
		return nil, err
	}

	outputShape := ort.NewShape(1, 1)
	outputTensor, err := ort.NewEmptyTensor[float32](outputShape)
	if err != nil {
		_ = inputTensor.Destroy()
		_ = stateTensor.Destroy()
		_ = srTensor.Destroy()
		return nil, err
	}

	stateOutShape := ort.NewShape(2, 1, 128)
	stateOutTensor, err := ort.NewEmptyTensor[float32](stateOutShape)
	if err != nil {
		_ = inputTensor.Destroy()
		_ = stateTensor.Destroy()
		_ = srTensor.Destroy()
		_ = outputTensor.Destroy()
		return nil, err
	}

	sess, err := ort.NewAdvancedSession(modelPath,
		[]string{"input", "state", "sr"},
		[]string{"output", "stateN"},
		[]ort.Value{inputTensor, stateTensor, srTensor},
		[]ort.Value{outputTensor, stateOutTensor},
		nil)
	if err != nil {
		_ = inputTensor.Destroy()
		_ = stateTensor.Destroy()
		_ = srTensor.Destroy()
		_ = outputTensor.Destroy()
		_ = stateOutTensor.Destroy()
		return nil, err
	}

	v := &sileroVAD{
		session:   sess,
		input:     inputTensor,
		state:     stateTensor,
		sr:        srTensor,
		output:    outputTensor,
		stateOut:  stateOutTensor,
		lastReset: time.Now(),
	}
	return v, nil
}

func (v *sileroVAD) resetState() {
	for i := range v.context {
		v.context[i] = 0
	}
	for i := range v.stateBuf {
		v.stateBuf[i] = 0
	}
	v.state.ZeroContents()
	v.lastReset = time.Now()
}

func (v *sileroVAD) maybeReset() {
	if time.Since(v.lastReset) >= sileroResetInterval {
		v.resetState()
	}
}

// speechProb returns the speech probability for the given 512-sample chunk.
// Caller must not modify chunk. No allocations in hot path (reuses session tensors).
func (v *sileroVAD) speechProb(chunk []float32) (float32, error) {
	if len(chunk) != RequiredChunkSize {
		return 0, errChunkSize
	}

	v.maybeReset()

	// Build input: context (64) + chunk (512) into input tensor
	inputData := v.input.GetData()
	copy(inputData[:sileroContextSamples], v.context[:])
	copy(inputData[sileroContextSamples:], chunk)

	// Update context to last 64 samples of effective input (chunk's last 64 or context+chunk boundary)
	for i := 0; i < sileroContextSamples; i++ {
		v.context[i] = inputData[sileroInputSamples-sileroContextSamples+i]
	}

	if err := v.session.Run(); err != nil {
		return 0, err
	}

	prob := v.output.GetData()[0]

	// Copy stateN back to state for next run
	copy(v.state.GetData(), v.stateOut.GetData())

	return prob, nil
}

func (v *sileroVAD) destroy() error {
	return v.session.Destroy()
}
