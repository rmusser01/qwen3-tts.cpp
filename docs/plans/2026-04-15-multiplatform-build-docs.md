# Multi-Platform Build Docs Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add Windows and Linux build instructions, pre-built model downloads, and troubleshooting to README.md.

**Architecture:** Sequential edits to README.md — expand Prerequisites, add Linux/Windows build subsections under Build, add pre-built model download option, add Troubleshooting section before Acknowledgments.

**Tech Stack:** Markdown only. No code changes.

---

### Task 1: Expand Prerequisites section

**Files:**
- Modify: `README.md:21-26`

**Step 1: Replace the Prerequisites section**

Replace lines 21-26 (from `## Prerequisites` through the Python line) with:

```markdown
## Prerequisites

- **Compiler:** GCC 9+ or Clang 10+ (Linux/macOS), or MSVC 2019+ (Windows)
- **CMake:** 3.14+
- **GGML:** Built from source (vendored as git submodule)
- **GPU backends (optional):**
  - Metal — macOS only, built into Xcode CLT
  - [Vulkan SDK](https://vulkan.lunarg.com/) — Linux and Windows (AMD, NVIDIA, Intel)
  - [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) — NVIDIA GPUs
- **Python 3.10+** with [uv](https://github.com/astral-sh/uv) — model conversion only; not needed if using pre-built GGUF files
```

**Step 2: Commit**

```bash
git add README.md
git commit -m "docs: expand prerequisites for multi-platform support"
```

---

### Task 2: Add Linux/Windows note to Quickstart

**Files:**
- Modify: `README.md:28-29`

**Step 1: Update the Quickstart heading**

Replace:
```markdown
## Quickstart (macOS, copy/paste)

Run these commands from a fresh clone:
```

With:
```markdown
## Quickstart (macOS)

Run these commands from a fresh clone. For **Linux** or **Windows**, see the [Build](#build) section below.
```

**Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add cross-platform note to quickstart"
```

---

### Task 3: Restructure Build section with platform subsections

**Files:**
- Modify: `README.md:100-117`

**Step 1: Replace the Build section**

Replace the entire `## Build` section (from `## Build` through the CUDA note) with:

````markdown
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

At runtime, set `QWEN3_TTS_BACKEND=cuda` to force CUDA (see [Backend Selection](#backend-selection)).

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
````

**Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add Linux and Windows build instructions"
```

---

### Task 4: Add pre-built model download option

**Files:**
- Modify: `README.md` — insert after "Model Setup (Recommended)" section

**Step 1: Add Pre-built Models subsection**

Insert after the `## Model Setup (Recommended)` section (after the `--skip-download` bullet), before `## Manual Model Conversion`:

```markdown
### Pre-built Models (Skip Python)

Community-hosted GGUF files are available on HuggingFace. Download directly — no Python required:

```bash
mkdir -p models
curl -L -o models/qwen3-tts-0.6b-f16.gguf \
  "https://huggingface.co/koboldcpp/tts/resolve/main/qwen3-tts-0.6b-f16.gguf"
curl -L -o models/qwen3-tts-tokenizer-f16.gguf \
  "https://huggingface.co/koboldcpp/tts/resolve/main/qwen3-tts-tokenizer-f16.gguf"
```

These are F16 (full-precision) models. Q8_0 quantized variants are also available and will be auto-detected if named `qwen3-tts-0.6b-q8_0.gguf`.
```

**Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add pre-built GGUF model download option"
```

---

### Task 5: Add Troubleshooting section

**Files:**
- Modify: `README.md` — insert before `## Acknowledgments`

**Step 1: Add Troubleshooting section**

Insert before the `## Acknowledgments` section:

```markdown
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
```

**Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add troubleshooting section for common build/runtime issues"
```

---

### Task 6: Final review and squash commit

**Step 1: Review the full README**

Read through the complete README to verify:
- Section ordering makes sense
- No broken markdown links
- Code blocks have correct language tags
- No duplicate content between Quickstart and Build sections

**Step 2: Verify all internal links resolve**

Check that `[Build](#build)`, `[Backend Selection](#backend-selection)` anchors match actual headings.

**Step 3: Push to fork**

```bash
git push fork integrate-upstream-fixes:main
```
