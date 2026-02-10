package smartturn

import "math"

// Whisper mel params (16kHz): n_fft=400, hop=160, n_mels=80.
const (
	whisperNFFT    = 400
	whisperHop     = 160
	whisperNMels   = 80
	whisper8sSamples = 128000
	whisper8sFrames  = 800
)

// computeWhisperMel converts 8s of mono float32 audio (128000 samples) to
// log-mel features shape (80, 800). Audio is truncated to last 8s or left-padded.
func computeWhisperMel(audio []float32) []float32 {
	padded := make([]float32, whisper8sSamples)
	if n := len(audio); n >= whisper8sSamples {
		copy(padded, audio[n-whisper8sSamples:])
	} else {
		copy(padded[whisper8sSamples-n:], audio)
	}
	return computeWhisperMelFromPadded(padded)
}

func computeWhisperMelFromPadded(padded []float32) []float32 {
	if len(padded) != whisper8sSamples {
		return nil
	}
	// STFT: 400 window, 160 hop -> 798 frames from 128000; we pad to 800
	nFrames := (whisper8sSamples - whisperNFFT) / whisperHop
	if nFrames > whisper8sFrames {
		nFrames = whisper8sFrames
	}
	// Power spectrum: 400-point real FFT -> 201 bins
	nBins := whisperNFFT/2 + 1
	mel := make([]float32, whisperNMels*whisper8sFrames)
	window := make([]float32, whisperNFFT)
	for i := 0; i < whisperNFFT; i++ {
		window[i] = float32(0.5 * (1 - math.Cos(2*math.Pi*float64(i)/float64(whisperNFFT))))
	}
	filters := getMelFilterbank(whisperNMels, nBins)
	fftBuf := make([]float32, whisperNFFT*2)
	for t := 0; t < whisper8sFrames; t++ {
		offset := t * whisperHop
		if offset+whisperNFFT > len(padded) {
			break
		}
		for i := 0; i < whisperNFFT; i++ {
			fftBuf[i*2] = padded[offset+i] * window[i]
			fftBuf[i*2+1] = 0
		}
		power := realFFTPower(fftBuf, whisperNFFT)
		for m := 0; m < whisperNMels; m++ {
			var v float32
			for k := 0; k < nBins; k++ {
				v += filters[m*nBins+k] * power[k]
			}
			if v < 1e-10 {
				v = 1e-10
			}
			mel[m*whisper8sFrames+t] = float32(math.Log(float64(v)))
		}
	}
	return mel
}

func realFFTPower(buf []float32, n int) []float32 {
	// Simple DFT for power spectrum (n/2+1 bins). Not optimized but correct.
	nOut := n/2 + 1
	power := make([]float32, nOut)
	for k := 0; k < nOut; k++ {
		var re, im float64
		for i := 0; i < n; i++ {
			angle := -2 * math.Pi * float64(k) * float64(i) / float64(n)
			re += float64(buf[i*2]) * math.Cos(angle)
			im += float64(buf[i*2]) * math.Sin(angle)
		}
		power[k] = float32((re*re + im*im) / float64(n*n))
	}
	return power
}

var cachedMelFilters []float32

func getMelFilterbank(nMels, nBins int) []float32 {
	if cachedMelFilters != nil && len(cachedMelFilters) == nMels*nBins {
		return cachedMelFilters
	}
	// Mel scale: 20 Hz to 8000 Hz (Nyquist at 16kHz is 8kHz)
	sampleRate := 16000.0
	lowFreq := 20.0
	highFreq := 8000.0
	lowMel := hzToMel(lowFreq)
	highMel := hzToMel(highFreq)
	melPoints := make([]float64, nMels+2)
	for i := 0; i < nMels+2; i++ {
		melPoints[i] = lowMel + (highMel-lowMel)*float64(i)/float64(nMels+1)
	}
	hzPoints := make([]float64, nMels+2)
	for i := range melPoints {
		hzPoints[i] = melToHz(melPoints[i])
	}
	binFreq := make([]float64, nBins)
	for k := 0; k < nBins; k++ {
		binFreq[k] = float64(k) * sampleRate / float64(2*(nBins-1))
	}
	filters := make([]float32, nMels*nBins)
	for m := 0; m < nMels; m++ {
		left := hzPoints[m]
		center := hzPoints[m+1]
		right := hzPoints[m+2]
		for k := 0; k < nBins; k++ {
			f := binFreq[k]
			var v float64
			if f >= left && f <= center {
				v = (f - left) / (center - left)
			} else if f > center && f <= right {
				v = (right - f) / (right - center)
			}
			filters[m*nBins+k] = float32(v)
		}
	}
	cachedMelFilters = filters
	return filters
}

func hzToMel(hz float64) float64 {
	return 2595 * math.Log10(1+hz/700)
}

func melToHz(mel float64) float64 {
	return 700 * (math.Pow(10, mel/2595) - 1)
}
