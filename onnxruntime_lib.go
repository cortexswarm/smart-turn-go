package smartturn

import (
	"os"
	"path/filepath"
	"runtime"
)

func pathExists(p string) bool {
	_, err := os.Stat(p)
	return err == nil
}

// BundledLibDir is the directory name under which platform-specific ONNX Runtime
// libraries are stored (e.g. lib/darwin_arm64/libonnxruntime.dylib).
const BundledLibDir = "lib"

// DataDir is the directory (e.g. data/) where ONNX models and optionally the
// runtime are stored. Runtime files may be named e.g. onnxruntime_arm64.dylib.
const DataDir = "data"

// bundledLibNames returns candidate filenames for the ONNX Runtime shared library
// on the current OS. On Linux, official releases use versioned .so (e.g.
// libonnxruntime.so.1.23.2); the first existing file in the list is used.
func bundledLibNames() []string {
	switch runtime.GOOS {
	case "darwin":
		return []string{"libonnxruntime.dylib"}
	case "windows":
		return []string{"onnxruntime.dll"}
	default:
		return []string{"libonnxruntime.so.1.23.2", "libonnxruntime.so"}
	}
}

// dataDirLibName returns the runtime library filename when stored in data/
// (e.g. onnxruntime_arm64.dylib, onnxruntime_amd64.so, onnxruntime.dll).
func dataDirLibName() string {
	switch runtime.GOOS {
	case "darwin":
		return "onnxruntime_" + runtime.GOARCH + ".dylib"
	case "windows":
		return "onnxruntime.dll"
	default:
		return "onnxruntime_" + runtime.GOARCH + ".so"
	}
}

// bundledLibPlatform returns the platform subdir (GOOS_GOARCH) for bundled libs.
func bundledLibPlatform() string {
	return runtime.GOOS + "_" + runtime.GOARCH
}

// candidateBaseDirs returns base directories to search for bundled ONNX lib:
// working directory first, then the directory of the running executable.
func candidateBaseDirs() []string {
	cwd, _ := os.Getwd()
	exe, err := os.Executable()
	if err != nil {
		return []string{cwd}
	}
	exeDir := filepath.Dir(exe)
	if exeDir == cwd {
		return []string{cwd}
	}
	return []string{cwd, exeDir}
}

// resolveBundledLib returns the first path that exists. It checks (1) data/
// with platform-specific names (e.g. data/onnxruntime_arm64.dylib), then
// (2) lib/<platform>/ with standard names (e.g. lib/darwin_arm64/libonnxruntime.dylib).
func resolveBundledLib(candidateBaseDirs []string) string {
	platform := bundledLibPlatform()
	dataName := dataDirLibName()
	for _, base := range candidateBaseDirs {
		if base == "" {
			continue
		}
		p := filepath.Join(base, DataDir, dataName)
		if pathExists(p) {
			return p
		}
	}
	for _, base := range candidateBaseDirs {
		if base == "" {
			continue
		}
		for _, name := range bundledLibNames() {
			p := filepath.Join(base, BundledLibDir, platform, name)
			if pathExists(p) {
				return p
			}
		}
	}
	return ""
}
