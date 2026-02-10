package smartturn

import (
	"errors"

	ort "github.com/yalue/onnxruntime_go"
)

var errInvalidSegment = errors.New("invalid segment for Smart-Turn")

// smartTurnResult is the structured result from Smart-Turn inference (not exposed to SDK users).
type smartTurnResult struct {
	Complete    bool
	Probability float32
}

// smartTurn runs inference on a finalized speech segment. Unexported; used by engine only.
type smartTurn struct {
	session *ort.AdvancedSession
	input   *ort.Tensor[float32]
	output  *ort.Tensor[float32]
}

func newSmartTurn(modelPath string) (*smartTurn, error) {
	// Smart-Turn v3.2 CPU expects input_features shape (1, 80, 800) - Whisper mel for 8s.
	inputShape := ort.NewShape(1, whisperNMels, whisper8sFrames)
	inputData := make([]float32, 1*whisperNMels*whisper8sFrames)
	inputTensor, err := ort.NewTensor(inputShape, inputData)
	if err != nil {
		return nil, err
	}
	outputShape := ort.NewShape(1)
	outputTensor, err := ort.NewEmptyTensor[float32](outputShape)
	if err != nil {
		inputTensor.Destroy()
		return nil, err
	}
	sess, err := ort.NewAdvancedSession(modelPath,
		[]string{"input_features"},
		[]string{"output"},
		[]ort.Value{inputTensor},
		[]ort.Value{outputTensor},
		nil)
	if err != nil {
		inputTensor.Destroy()
		outputTensor.Destroy()
		return nil, err
	}
	return &smartTurn{session: sess, input: inputTensor, output: outputTensor}, nil
}

// run runs Smart-Turn on the segment audio. Segment is truncated to last 8s or left-padded to 8s.
func (st *smartTurn) run(segment []float32) (smartTurnResult, error) {
	mel := computeWhisperMel(segment)
	if mel == nil {
		return smartTurnResult{}, errInvalidSegment
	}
	inputData := st.input.GetData()
	copy(inputData, mel)
	if err := st.session.Run(); err != nil {
		return smartTurnResult{}, err
	}
	prob := st.output.GetData()[0]
	return smartTurnResult{
		Complete:    prob > 0.5,
		Probability: prob,
	}, nil
}

func (st *smartTurn) destroy() error {
	return st.session.Destroy()
}
