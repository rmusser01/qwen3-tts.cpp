# Design: Multi-Platform Build Documentation

## Goal

Update README.md to provide clear build instructions for macOS, Linux, and Windows alongside the existing macOS-only quickstart. Add a pre-built model download option and a troubleshooting section.

## Changes

### 1. Prerequisites (expand)

Add MSVC 2019+ for Windows. List GPU backend options per platform: Metal (macOS), Vulkan SDK (Linux/Windows), CUDA Toolkit (NVIDIA GPUs).

### 2. Quickstart (minor)

Keep macOS-only. Add one line pointing Linux/Windows users to the Build section.

### 3. Build section (restructure)

Replace the single macOS block with:

- **Common** clone + submodule init (shared across all platforms)
- **macOS (Metal)** — `GGML_METAL=ON`, note CoreML is optional
- **Linux (Vulkan)** — `GGML_VULKAN=ON`, note Vulkan SDK prerequisite
- **Linux (CUDA)** — `GGML_CUDA=ON`, note CUDA toolkit prerequisite
- **Windows (MSVC)** — Developer Command Prompt, cmake generator, DLL placement note, Vulkan/CUDA variants

### 4. Pre-built Models (new section)

Add a "Pre-built Models" subsection under Model Setup pointing to `koboldcpp/tts` on HuggingFace for users who want to skip the Python conversion pipeline entirely.

### 5. Troubleshooting (new section)

Cover:
- Linux shared library fPIC error (fixed in our codebase, but document for forks)
- Vulkan segfault on vocoder load (fixed, but note for older versions)
- Windows DLL placement (`ggml.dll`, `ggml-base.dll`, backend DLLs next to exe)
- CUDA slower than expected (point to `QWEN3_TTS_BACKEND=cuda` env var and chunked decode tuning)

## Files modified

- `README.md` — all changes above

## Out of scope

- No code changes
- No new build system changes (those were done in prior commits)
- No CI/CD configuration
