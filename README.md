# qwen3-tts.cpp

![PyTorch vs qwen3-tts.cpp benchmark](./docs/benchmark_pytorch_vs_cpp.png)

**Benchmark Snapshot (PyTorch vs qwen3-tts.cpp):** Basic 3.19x faster, Clone 4.07x faster. Peak RSS delta: Basic +19.0%, Clone +7.7%.

C++ inference for [Qwen3-TTS](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base) using the [GGML](https://github.com/ggml-org/ggml) tensor library.

Runs the full TTS pipeline in pure C++17, including text tokenization, speaker encoding, transformer code generation, and vocoder decoding, without Python or PyTorch at inference time.

## Features

- Full text-to-speech pipeline in C++17 with GGML backend
- **0.6B and 1.7B model support** (Base, CustomVoice, VoiceDesign)
- Voice cloning from reference audio (ECAPA-TDNN x-vector extraction)
- **Speaker embedding save/load** for fast repeated cloning (skip encoder)
- **Voice steering instructions** for controlling speech style and emotion
- **Unicode/multilingual tokenizer** (Chinese, Japanese, Korean, German, etc.)
- Greedy and sampled decoding (temperature, top-k, repetition penalty, seed)
- GGUF model format with **built-in C++ quantizer** (F16, Q8_0, Q4_K, etc.)
- Runtime backend selection: Metal (macOS), Vulkan (Linux/Windows), CUDA (NVIDIA)
- Cross-platform: macOS, Linux, Windows (MSVC)
- C API for FFI integration (Python, Rust, Nim, etc.)
- Deterministic reference tests comparing C++ output against Python

## Prerequisites

- **Compiler:** GCC 9+ or Clang 10+ (Linux/macOS), or MSVC 2019+ (Windows)
- **CMake:** 3.14+
- **GGML:** Built from source (vendored as git submodule)
- **GPU backends (optional):**
  - Metal — macOS only, built into Xcode Command Line Tools
  - [Vulkan SDK](https://vulkan.lunarg.com/) — Linux and Windows (AMD, NVIDIA, Intel)
  - [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) — NVIDIA GPUs
- **Python 3.10+** with [uv](https://github.com/astral-sh/uv) — model conversion only; not needed if using [pre-built GGUF files](#pre-built-models-skip-python)

## Quickstart (macOS)

Run these commands from a fresh clone. For **Linux** or **Windows**, see the [Build](#build) section below.

```bash
git clone https://github.com/predict-woo/qwen3-tts.cpp.git
cd qwen3-tts.cpp
git submodule update --init --recursive

# 1) Build GGML with Metal
cmake -S ggml -B ggml/build -DGGML_METAL=ON
cmake --build ggml/build -j4

# 2) Build qwen3-tts.cpp
cmake -S . -B build
cmake --build build -j4

# 3) Create a uv Python environment for setup/conversion tools
uv venv .venv
source .venv/bin/activate

# 4) Install Python dependencies
uv pip install --upgrade pip
uv pip install huggingface_hub gguf torch safetensors numpy tqdm coremltools

# Optional if model access requires auth:
# huggingface-cli login

# 5) Download and generate all runtime model artifacts
python scripts/setup_pipeline_models.py

# 6) Basic synthesis example
./build/qwen3-tts-cli \
  -m models \
  -t "Hello from qwen3-tts.cpp running on macOS with CoreML by default." \
  -o examples/readme_example_basic.wav

# 7) Voice-clone example using sample audio in this repo
./build/qwen3-tts-cli \
  -m models \
  -r examples/readme_clone_input.wav \
  -t "This is a voice cloning example generated from the sample audio file in this directory." \
  -o examples/readme_example_clone.wav
```

Expected model artifacts after step 5:

- `models/qwen3-tts-0.6b-f16.gguf`
- `models/qwen3-tts-tokenizer-f16.gguf`
- `models/coreml/code_predictor.mlpackage` (on macOS)

Expected audio outputs after steps 6-7:

- `examples/readme_example_basic.wav`
- `examples/readme_example_clone.wav`

Included voice-clone input/output pair (so users can compare directly):

- Input reference audio: `examples/readme_clone_input.wav`
- Generated output audio: `examples/readme_example_clone.wav`

Audio preview (inline):

<audio controls src="./examples/readme_clone_input.wav"></audio>
<br/>
<audio controls src="./examples/readme_example_clone.wav"></audio>

If your Markdown renderer does not show inline controls, use direct links:

- [Play input reference WAV](./examples/readme_clone_input.wav)
- [Play generated output WAV](./examples/readme_example_clone.wav)

## Build

### Clone and initialize

```bash
git clone https://github.com/predict-woo/qwen3-tts.cpp.git
cd qwen3-tts.cpp
git submodule update --init --recursive
```

> **Note:** The top-level CMake expects GGML in `./ggml` with libraries under `./ggml/build/src`.

### macOS (Metal)

```bash
# Build GGML with Metal GPU acceleration
cmake -S ggml -B ggml/build -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release
cmake --build ggml/build -j$(sysctl -n hw.ncpu)

# Build qwen3-tts.cpp
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(sysctl -n hw.ncpu)
```

CoreML acceleration for the code predictor stage is enabled by default when the model exists. Generate it with `python scripts/setup_pipeline_models.py --coreml on`.

### Linux (Vulkan)

Install the [Vulkan SDK](https://vulkan.lunarg.com/) first, then:

```bash
# Build GGML with Vulkan
cmake -S ggml -B ggml/build -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
cmake --build ggml/build -j$(nproc)

# Build qwen3-tts.cpp
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

### Linux (CUDA)

Requires the [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit), then:

```bash
# Build GGML with CUDA
cmake -S ggml -B ggml/build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build ggml/build -j$(nproc)

# Build qwen3-tts.cpp
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

At runtime, set `QWEN3_TTS_BACKEND=cuda` to force the CUDA backend (see [Backend Selection](#backend-selection)).

### Windows (MSVC)

Open a **Developer Command Prompt for VS 2022** (or 2019), then:

```cmd
:: Build GGML (CPU-only; add -DGGML_VULKAN=ON or -DGGML_CUDA=ON for GPU)
cmake -S ggml -B ggml\build -DCMAKE_BUILD_TYPE=Release
cmake --build ggml\build --config Release

:: Build qwen3-tts.cpp
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

**DLL placement:** Copy `ggml.dll`, `ggml-base.dll`, `ggml-cpu.dll`, and the backend DLL (e.g. `ggml-vulkan.dll` or `ggml-cuda.dll`) into the same directory as `qwen3-tts-cli.exe`.

For Vulkan on Windows, install the [Vulkan SDK](https://vulkan.lunarg.com/) and add `-DGGML_VULKAN=ON` to the GGML cmake line. For CUDA, add `-DGGML_CUDA=ON`.

## Model Setup (Recommended)

Use the one-shot setup script:

```bash
source .venv/bin/activate
python scripts/setup_pipeline_models.py
```

Useful flags:

- `--force` re-downloads and re-generates all artifacts.
- `--coreml auto|on|off` controls CoreML export behavior.
- `--skip-download` skips HF download and uses existing local model dirs.

### Pre-built Models (Skip Python)

Community-hosted GGUF files are available on HuggingFace. Download directly — no Python required:

```bash
mkdir -p models
curl -L -o models/qwen3-tts-0.6b-f16.gguf \
  "https://huggingface.co/koboldcpp/tts/resolve/main/qwen3-tts-0.6b-f16.gguf"
curl -L -o models/qwen3-tts-tokenizer-f16.gguf \
  "https://huggingface.co/koboldcpp/tts/resolve/main/qwen3-tts-tokenizer-f16.gguf"
```

These are F16 (full-precision) models. Q8\_0 quantized variants are also available and will be auto-detected if named `qwen3-tts-0.6b-q8_0.gguf`.

## Manual Model Conversion (Advanced)

Convert HuggingFace models to GGUF format:

```bash
# Download the model
huggingface-cli download Qwen/Qwen3-TTS-12Hz-0.6B-Base \
    --local-dir models/Qwen3-TTS-12Hz-0.6B-Base

# Convert TTS model (transformer + speaker encoder + tokenizer)
python scripts/convert_tts_to_gguf.py \
    models/Qwen3-TTS-12Hz-0.6B-Base \
    models/qwen3-tts-0.6b-f16.gguf

# Convert vocoder (audio decoder)
python scripts/convert_tokenizer_to_gguf.py \
    models/Qwen3-TTS-12Hz-0.6B-Base \
    models/qwen3-tts-tokenizer-f16.gguf
```

Place both `.gguf` files in a `models/` directory.

## Usage

```bash
# Basic synthesis
./build/qwen3-tts-cli -m models -t "Hello, world!" -o hello.wav

# Voice cloning from reference audio
./build/qwen3-tts-cli -m models -t "Hello! How are you?" -r reference.wav -o cloned.wav

# Greedy decoding with max length
./build/qwen3-tts-cli -m models -t "Hello!" -r ref.wav -o out.wav \
    --temperature 0 --max-tokens 2048
```

### CLI Options

| Flag | Description | Default |
|------|-------------|---------|
| `-m, --model <dir>` | Model directory containing GGUF files | (required) |
| `--tts-model <file>` | TTS model filename (overrides auto-detection) | (auto) |
| `--tokenizer-model <file>` | Tokenizer/vocoder filename | (auto) |
| `-t, --text <text>` | Text to synthesize | (required) |
| `-o, --output <file>` | Output WAV file path | `output.wav` |
| `-r, --reference <file>` | Reference audio for voice cloning | (none) |
| `--speaker-embedding <file>` | Use precomputed speaker embedding (.json/.bin) | (none) |
| `--dump-speaker-embedding <file>` | Save extracted embedding from `--reference` | (none) |
| `--instruction <text>` | Voice steering instruction (e.g. "Speak happily") | (none) |
| `--temperature <val>` | Sampling temperature (0 = greedy) | 0.9 |
| `--top-k <n>` | Top-k sampling (0 = disabled) | 50 |
| `--max-tokens <n>` | Maximum audio frames to generate | 4096 |
| `--repetition-penalty <val>` | Repetition penalty on codebook-0 token generation | 1.05 |
| `--seed <n>` | RNG seed for reproducible output | (random) |
| `--no-f32-acc` | Disable f32 matmul accumulation (faster, less precise) | (off) |
| `-l, --language <lang>` | Language: en, ru, zh, ja, ko, de, fr, es | en |
| `-j, --threads <n>` | Number of compute threads | 4 |

On macOS, CoreML code predictor is enabled by default when `models/coreml/code_predictor.mlpackage` exists.
Set `QWEN3_TTS_USE_COREML=0` to disable it. Low-memory mode is opt-in via `QWEN3_TTS_LOW_MEM=1`.

### Speaker Embedding Workflow

Precompute a speaker embedding once (saves ~20s per synthesis):

```bash
# Extract and save embedding from reference audio
./build/qwen3-tts-cli -m models -r reference.wav --dump-speaker-embedding speaker.json \
  -t "Initial extraction." -o /dev/null

# Reuse embedding for fast voice-cloned synthesis (no encoder needed)
./build/qwen3-tts-cli -m models --speaker-embedding speaker.json \
  -t "Fast voice cloning with cached embedding." -o output.wav
```

### Voice Steering Instructions

Control voice characteristics with the `--instruction` flag (works best with 1.7B VoiceDesign models):

```bash
./build/qwen3-tts-cli -m models --instruction "Speak in a cheerful tone" \
  -t "Good morning, everyone!" -o cheerful.wav
```

### Quantization

Quantize models from F16 to smaller formats without Python:

```bash
./build/qwen3-tts-quantize --input models/qwen3-tts-0.6b-f16.gguf \
  --output models/qwen3-tts-0.6b-q8_0.gguf --type q8_0
```

Supported types: `q8_0`, `q4_0`, `q4_1`, `q5_0`, `q5_1`, `q4_k`, `q5_k`, `q6_k`. The CLI auto-detects Q8_0 models when present.

### Backend Selection

At runtime, each component logs its selected backend (for example, `TTSTransformer backend: MTL0` or `BLAS`).

- Preferred order: `IGPU` -> `GPU` -> `ACCEL` -> `CPU`
- Encoder and transformer can run on Metal/other accelerators with CPU fallback in the scheduler
- Decoder now follows the same backend preference and will use Metal when available
- `QWEN3_TTS_BACKEND` overrides runtime selection: `auto` (default), `cuda`, or `cpu`
- `QWEN3_TTS_DEVICE` selects CUDA device index when `QWEN3_TTS_BACKEND=cuda` (default device is index 0)
- `QWEN3_TTS_DECODER_GPU_MAX_FRAMES` controls max frames per CUDA vocoder chunk (default: `34`)
- `QWEN3_TTS_DECODER_GPU_CONTEXT_FRAMES` controls left-context frames per CUDA vocoder chunk (default: `12`)

## Architecture

```
Text ──► [Tokenizer] ──► token IDs
                              │
Reference Audio ──► [Speaker Encoder] ──► speaker embedding
                              │
token IDs + speaker embedding ──► [TTS Transformer] ──► speech codes (N frames x 16 codebooks)
                              │
speech codes ──► [Vocoder] ──► audio waveform (24kHz)
```

### Source Files

| Module | Files | Description |
|--------|-------|-------------|
| `src/common/` | `gguf_loader`, `coreml_code_predictor`, `speaker_embedding_io` | Shared infrastructure |
| `src/tokenizer/` | `text_tokenizer`, `tokenizer_unicode` | BPE tokenizer with Unicode/GPT-2 regex |
| `src/transformer/` | `tts_transformer` | Qwen2 talker + code predictor (0.6B and 1.7B) |
| `src/encoder/` | `audio_tokenizer_encoder` | ECAPA-TDNN x-vector speaker encoder |
| `src/decoder/` | `audio_tokenizer_decoder` | WavTokenizer vocoder (codes to 24kHz audio) |
| `src/pipeline/` | `qwen3_tts`, `qwen3tts_c_api` | Pipeline orchestration + C API |
| `src/` | `main.cpp`, `qwen3_tts_quantize.cpp` | CLI entry point + quantization tool |

### TTS Transformer Details

The transformer generates speech codes in two stages per frame:

1. **Talker** (28 layers, 16 heads, 1024 hidden) produces a hidden state and codebook-0 logits.
2. **Code Predictor** (5 layers) autoregressively generates codebooks 1-15 from that hidden state.

The prefill embedding mirrors the Python pipeline exactly:
- Positions 0-2: text-projected role tokens (`<|im_start|>`, `assistant`, `\n`)
- Positions 3-6: TTS pad + codec embeddings (think tokens, language ID)
- Position 7: TTS pad + speaker embedding
- Position 8: TTS BOS + codec pad embedding
- Position 9+: text-projected text tokens + codec BOS/embeddings

## Testing

```bash
# Run full test suite
bash scripts/run_all_tests.sh

# Individual component tests
./build/test_tokenizer --model models/qwen3-tts-0.6b-f16.gguf
./build/test_encoder --tokenizer models/qwen3-tts-0.6b-f16.gguf \
    --audio clone.wav --reference reference/ref_audio_embedding.bin
./build/test_transformer --model models/qwen3-tts-0.6b-f16.gguf \
    --ref-dir reference/
./build/test_decoder --tokenizer models/qwen3-tts-tokenizer-f16.gguf \
    --codes reference/speech_codes.bin --reference reference/decoded_audio.bin

# End-to-end Python vs C++ comparison
uv run python scripts/compare_e2e.py

# Generate deterministic reference data from Python
uv run python scripts/generate_deterministic_reference.py
```

### Test Results (F16 model)

- Prefill logits: cosine similarity = 0.99999994 with Python reference
- Codebook 0 match rate: 81% (frame-level exact match)
- Codebooks 1-4: ~84% match rate
- Audio output is perceptually equivalent; low waveform correlation is expected due to autoregressive divergence from F16 precision

## Profiling

Build with compile-time timing instrumentation (zero overhead when disabled):

```bash
cmake .. -DQWEN3_TTS_TIMING=ON
make -j4
```

Example output (92 frames, 7.3s audio):

```
=== Detailed Generation Timing (92 frames) ===

  Prefill:
      Compute:           175.9 ms

  Talker forward_step:
      Graph build:        21.8 ms   (0.2 ms/frame)
      Graph alloc:        34.1 ms   (0.4 ms/frame)
      Compute:          7717.4 ms   (83.9 ms/frame)

  Code predictor:
      Init/KV/embed:       7.7 ms   (0.1 ms/frame)
      Prefill (2tok):   1393.2 ms   (15.1 ms/frame)
      Steps (14):      19531.7 ms   (212.3 ms/frame)
      Compute:         20702.6 ms   (225.0 ms/frame)

  Total generate:      28915.0 ms   (3.2 frames/s)
```

The code predictor (15 sequential forward passes per frame) accounts for ~71% of generation time.

## Troubleshooting

### Linux: shared library link error (`recompile with -fPIC`)

If you see `relocation R_X86_64_32S ... recompile with -fPIC` when building the shared library, your CMakeLists.txt is missing `set(CMAKE_POSITION_INDEPENDENT_CODE ON)`. This is fixed in current builds.

### Vulkan: segfault when loading vocoder

Older versions had a bug where `normalize_codebooks()` wrote directly to GPU-mapped memory, causing a segfault on Vulkan backends. This is fixed in current builds. If you hit this on a fork, see [issue #20](https://github.com/predict-woo/qwen3-tts.cpp/issues/20) for the fix.

### Windows: missing DLL errors at runtime

Copy all required DLLs next to `qwen3-tts-cli.exe`:
- `ggml.dll`, `ggml-base.dll`, `ggml-cpu.dll`
- Backend DLL: `ggml-vulkan.dll` or `ggml-cuda.dll` (if using GPU)

These are built under `ggml\build\bin\Release\` (or `ggml\build\src\` depending on generator).

### CUDA: slower than expected

Ensure you are using the CUDA backend at runtime:

```bash
QWEN3_TTS_BACKEND=cuda QWEN3_TTS_DEVICE=0 ./build/qwen3-tts-cli -m models -t "test" -o out.wav
```

Tune chunked vocoder decode with `QWEN3_TTS_DECODER_GPU_MAX_FRAMES` (default 34) and `QWEN3_TTS_DECODER_GPU_CONTEXT_FRAMES` (default 12) if you see OOM or slow decode on your GPU.

### Vocoder runs on CPU despite GPU backend shown

If logs show `AudioTokenizerDecoder backend: Vulkan0` but vocoder decode is still slow (~17s), this is the backend/buffer mismatch bug. Fixed in current builds — update to the latest version.

## Acknowledgments

- [Qwen3-TTS](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base) by Alibaba Qwen team
- [GGML](https://github.com/ggml-org/ggml) tensor library by Georgi Gerganov
- [WavTokenizer](https://github.com/jishengpeng/WavTokenizer) vocoder architecture
- [predict-woo/qwen3-tts.cpp](https://github.com/predict-woo/qwen3-tts.cpp) — original upstream project
- [gonwan/qwen3-tts.cpp](https://github.com/gonwan/qwen3-tts.cpp) — 1.7B model support, speaker embedding save/load, explicit model selection, seed parameter, Windows `_fseeki64` fix, `--no-f32-acc` flag
- [Danmoreng/qwen3-tts.cpp](https://github.com/Danmoreng/qwen3-tts.cpp) — Unicode/UTF-8 GPT-2 regex tokenizer (derived from [llama.cpp](https://github.com/ggml-org/llama.cpp)), voice steering instructions, C++ quantizer tool, decoder graph caching, `ggml_pad_reflect_1d` optimization
- [SiaoZeng](https://github.com/SiaoZeng) — Vulkan vocoder backend-safe `normalize_codebooks` fix ([issue #20](https://github.com/predict-woo/qwen3-tts.cpp/issues/20))
- [kevinzhow/clawd20130](https://github.com/kevinzhow) — CUDA runtime backend override and chunked GPU vocoder decode ([PR #11](https://github.com/predict-woo/qwen3-tts.cpp/pull/11))
- [DeryabinIvan](https://github.com/DeryabinIvan) — Windows build fixes ([PR #2](https://github.com/predict-woo/qwen3-tts.cpp/pull/2))
- [TheTastefulToastie](https://github.com/TheTastefulToastie) — cleaner CMake approach for MSVC `_USE_MATH_DEFINES` / `NOMINMAX`
