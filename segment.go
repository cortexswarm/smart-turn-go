package smartturn

// segmenter holds state for the segmentation state machine. Pure logic; no ONNX, no callbacks.
type segmenter struct {
	cfg configSegment

	preBuffer    [][]float32
	preBufIdx    int
	preBufCount  int
	segment      []float32
	speechActive bool
	trailingChunks int
	sinceTrigger  int
}

type configSegment struct {
	preChunks   int
	stopChunks  int
	maxChunks   int
	chunkSize   int
}

func newSegmenter(sampleRate, chunkSize, preSpeechMs, stopMs int, maxDurationSec float32) *segmenter {
	chunkMs := float64(chunkSize) / float64(sampleRate) * 1000
	preChunks := ceilDiv(preSpeechMs, max(1, int(chunkMs)))
	if preChunks <= 0 {
		preChunks = 1
	}
	preChunks = min(preChunks, 256)
	stopChunks := ceilDiv(stopMs, max(1, int(chunkMs)))
	if stopChunks <= 0 {
		stopChunks = 1
	}
	maxChunks := int(maxDurationSec * float32(sampleRate) / float32(chunkSize))
	if maxChunks <= 0 {
		maxChunks = 1
	}
	return &segmenter{
		cfg: configSegment{
			preChunks: preChunks,
			stopChunks: stopChunks,
			maxChunks:  maxChunks,
			chunkSize:  chunkSize,
		},
		preBuffer: make([][]float32, preChunks),
	}
}

func ceilDiv(a, b int) int {
	if b <= 0 {
		return 0
	}
	return (a + b - 1) / b
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// segmentResult is returned by processChunk on every chunk.
type segmentResult struct {
	Started        bool
	Ended          bool
	EndedBySilence bool   // true when segment ended due to trailing silence (VAD); false when capped at max duration
	Segment        []float32 // current accumulated segment (including pre-speech) while speech is active
}

// processChunk updates segment state with one VAD result and chunk.
// chunk must have length cfg.chunkSize (512). Returns Started=true on transition
// to speech, Ended=true when segment is finalized with Segment set.
func (s *segmenter) processChunk(isSpeech bool, chunk []float32) segmentResult {
	out := segmentResult{}
	if len(chunk) != s.cfg.chunkSize {
		return out
	}

	chunkCopy := make([]float32, len(chunk))
	copy(chunkCopy, chunk)

	if !s.speechActive {
		s.preBuffer[s.preBufIdx] = chunkCopy
		s.preBufIdx = (s.preBufIdx + 1) % s.cfg.preChunks
		if s.preBufCount < s.cfg.preChunks {
			s.preBufCount++
		}
		if isSpeech {
			s.speechActive = true
			out.Started = true
			s.trailingChunks = 0
			s.sinceTrigger = 1
			s.segment = s.buildSegmentWithChunk(chunkCopy)
			out.Segment = s.segment
		}
		return out
	}

	s.segment = append(s.segment, chunkCopy...)
	out.Segment = s.segment
	s.sinceTrigger++
	if isSpeech {
		s.trailingChunks = 0
	} else {
		s.trailingChunks++
	}

	if s.trailingChunks >= s.cfg.stopChunks {
		out.Ended = true
		out.EndedBySilence = true
		out.Segment = s.segment
		s.reset()
	} else if s.sinceTrigger >= s.cfg.maxChunks {
		out.Ended = true
		out.EndedBySilence = false
		out.Segment = s.segment
		s.reset()
	}
	return out
}

func (s *segmenter) buildSegmentWithChunk(firstChunk []float32) []float32 {
	n := s.preBufCount*s.cfg.chunkSize + len(firstChunk)
	seg := make([]float32, 0, n)
	startIdx := (s.preBufIdx - s.preBufCount + s.cfg.preChunks) % s.cfg.preChunks
	for i := 0; i < s.preBufCount; i++ {
		idx := (startIdx + i) % s.cfg.preChunks
		if s.preBuffer[idx] != nil {
			seg = append(seg, s.preBuffer[idx]...)
		}
	}
	seg = append(seg, firstChunk...)
	return seg
}

func (s *segmenter) reset() {
	s.segment = nil
	s.speechActive = false
	s.trailingChunks = 0
	s.sinceTrigger = 0
	s.preBufIdx = 0
	s.preBufCount = 0
	for i := range s.preBuffer {
		s.preBuffer[i] = nil
	}
}
