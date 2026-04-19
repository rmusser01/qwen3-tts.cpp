# User-Facing Smoketest

A single command that builds the project, pulls models, and exercises every
user-visible feature end-to-end. Intended for anyone who is **not** the
maintainer and wants to confirm a fresh checkout works on their machine.

## What it covers

| Area | Coverage |
|---|---|
| Build | `cmake` configure + build of `ggml` (Metal) and the project |
| Models | Downloads `qwen3-tts-0.6b-f16.gguf`, `qwen3-tts-0.6b-q8_0.gguf`, tokenizer |
| CLI synthesis | Basic text, voice cloning, instruction/steering, clone+instruction |
| Backends | Both `QWEN3_TTS_BACKEND=auto` (Metal on macOS) and `QWEN3_TTS_BACKEND=cpu` |
| Quantization | F16 and Q8_0 |
| Model sizes | 0.6B always; 1.7B if you pre-drop a GGUF in `models/` |
| Speaker embeddings | `--dump-speaker-embedding` → `--speaker-embedding` round trip |
| Python server | Starts `server/main.py`, hits `/health`, `/v1/audio/voices`, and `/v1/audio/speech` for default + cloned voice |
| WAV integrity | Every output is parsed: RIFF+WAVE header, `fmt`/`data` chunks, duration > 0.5s |

It does **not** grade perceptual quality, only structural validity. For
bit-accurate regression testing against the Python reference implementation,
use `scripts/run_all_tests.sh` and `scripts/compare_e2e.py`.

## Prerequisites

- macOS (Apple Silicon recommended) with Xcode Command Line Tools
  (`xcode-select --install`)
- `cmake`, `curl`, `python3` on `$PATH` (install via Homebrew if missing)
- ~5 GB free disk for first run (model downloads + build)
- Network access for HuggingFace downloads on first run

On Linux or Windows the script exits with code 2 — the CLI still works there,
but the smoketest currently only drives the macOS/Metal path. See the main
README for Linux/Vulkan, Linux/CUDA, and Windows/MSVC build recipes.

## Run it

```bash
bash scripts/run_user_smoketest.sh
```

Flags:

| Flag | Purpose |
|---|---|
| `--rebuild` | Force `cmake --build` even if binaries exist |
| `--skip-download` | Assume models are already in `models/` (for air-gapped or flaky networks) |
| `--skip-server` | Don't run the Python server section (skips venv + deps) |
| `--server-port N` | Bind the server to port `N` instead of `8765` |

Expected runtime:

- Cold (first run, full download + build): **15–30 min**, dominated by the
  model download.
- Warm (binaries and models cached): **~5 min**.

## Results

The script prints a colorized pass/fail table at the end and writes the
machine-readable version to `build/smoketest/results.tsv`. All generated
artifacts land in `build/smoketest/`:

```
build/smoketest/
├── 0.6b_f16_auto_basic.wav
├── 0.6b_f16_auto_clone.wav
├── 0.6b_f16_auto_instr.wav
├── 0.6b_f16_auto_clone_instr.wav
├── 0.6b_f16_cpu_basic.wav
├── …
├── embedding_reuse.wav
├── server_default.wav
├── server_cloned.wav
├── speaker.json
├── server.log
├── results.tsv
├── venv/
└── voices/smoketest.json
```

Exit codes:

- `0` — all required rows PASS (SKIP and WARN are allowed)
- `1` — at least one FAIL row
- `2` — non-macOS host
- `3` — missing required tools
- `4`/`5` — not a valid checkout / missing reference audio
- `10`–`12` — build step failed
- `20`/`21` — 0.6B model or tokenizer missing after download

**WARN** rows mean the synthesis produced a valid wav but `auto` backend
silently fell back to CPU (Metal did not initialize). The wav is fine, but
your Metal build may be broken — check `build/smoketest/*_auto_*.log` for the
backend-selection line.

## Listen to an output

```bash
afplay build/smoketest/0.6b_f16_auto_basic.wav
afplay build/smoketest/0.6b_f16_auto_clone.wav
afplay build/smoketest/server_cloned.wav
```

## Running with 1.7B models

There is no public prebuilt 1.7B GGUF on the default HuggingFace mirror at
time of writing. To include 1.7B in the smoketest matrix, drop one or both of
these files into `models/` before running:

```
models/qwen3-tts-1.7b-f16.gguf
models/qwen3-tts-1.7b-q8_0.gguf
```

**Important — convert from the `Base` repo, not `CustomVoice` or `VoiceDesign`.**
Only `Qwen/Qwen3-TTS-12Hz-1.7B-Base` ships the ECAPA-TDNN `speaker_encoder.*`
weights (76 tensors); the `CustomVoice` and `VoiceDesign` repos (and community
mirrors of them like `forkjoin-ai/qwen3-tts-12hz-1.7b-customvoice`) contain
only the talker. Converting those will produce a GGUF that fails at runtime
with `Error: Failed to load speaker encoder: No speaker encoder tensors found
in model` whenever you try to voice-clone.

Recommended conversion flow:

```bash
huggingface-cli login        # Qwen repos are gated
huggingface-cli download Qwen/Qwen3-TTS-12Hz-1.7B-Base \
    --local-dir models/hf/qwen3-tts-1.7b-base
python scripts/convert_tts_to_gguf.py \
    --input models/hf/qwen3-tts-1.7b-base \
    --output models/qwen3-tts-1.7b-f16.gguf --type f16
./build/qwen3-tts-quantize --input models/qwen3-tts-1.7b-f16.gguf \
    --output models/qwen3-tts-1.7b-q8_0.gguf --type q8_0
```

Alternatives:

1. **Project setup script** — edit the `REPOS` constants in
   `scripts/setup_pipeline_models.py` to point at
   `Qwen/Qwen3-TTS-12Hz-1.7B-Base` and re-run.
2. **Community prebuilt** — e.g. the one in `gonwan/qwen3-tts.cpp` releases.
   Verify it was produced from `Base` (not CustomVoice) before trusting the
   cloning rows. Rename to the exact names above.

The script passes `--tts-model <filename>` explicitly for every row so results
are unambiguous. Note that `--tts-model` expects a **filename relative to the
`-m` directory**, not a full path — the filenames above must match exactly so
the CLI can resolve them under `models/`.

### Why the 1.7B `basic` row is capped at `--max-tokens 250`

Per the [HF README for Qwen3-TTS-12Hz-1.7B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base):

> The Base variant is voice-cloning-only and does not support text-only
> inference without a reference audio sample.

If you invoke 1.7B Base with just `-t <text>` (no `-r`, no `--instruction`),
the model has no conditioning signal to tell it when to stop and will
generate speech up to the `--max-tokens` budget — roughly 5.5 minutes of
audio for an 8-word prompt. This is deterministic in greedy mode and
identical across all backends and quantizations, so it is a model-behavior
property, not a bug in this codebase. The `clone`, `instr`, and
`clone_instr` rows all produce bounded output because they supply the
required conditioning.

The smoketest caps the 1.7B basic row at 250 codec tokens (~21s of audio)
so the invocation plumbing is still exercised without making the run take
forever. 0.6B Base bounds its own output in text-only mode so no cap is
needed there.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ggml submodule missing` on first run | Script will `git submodule update --init --recursive` automatically. If that fails, you're offline or lack git credentials. |
| Metal init fails at runtime | Rebuild without CoreML: `rm -rf build && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DQWEN3_TTS_COREML=OFF && cmake --build build -j`. Then `--rebuild`. |
| `WARN: ran but fell back to CPU on 'auto'` | Metal didn't initialize. Check `ggml/build` was built with `-DGGML_METAL=ON` and that macOS has Metal support (Apple Silicon or AMD discrete GPU). |
| Port 8765 already in use | `--server-port 9000` (or any free port) |
| `pip install` fails | Network, or corporate proxy. Activate the venv manually: `source build/smoketest/venv/bin/activate && pip install -r server/requirements.txt` and read the error. |
| First voice-clone synthesis takes ~20s | Expected. The speaker encoder lazy-loads on first `-r` call per process. Subsequent calls reuse it. |
| First CLI run on Metal is slow | CoreML may be compiling the code-predictor mlpackage. Subsequent runs reuse the compiled cache. |
| Downloaded a wav but it won't play | Run `validate_wav` portion manually: `python3 -c "import wave; w=wave.open('path.wav'); print(w.getnframes(), w.getframerate())"`. If that fails, check `build/smoketest/*.log` for CLI errors. |

## Cleanup

```bash
rm -rf build/smoketest
```

Leaves compiled binaries (`build/qwen3-tts-cli` etc.) and downloaded models
(`models/*.gguf`) intact. To reset everything:

```bash
rm -rf build models/qwen3-tts-*.gguf
```

## Extending the smoketest

- Add a new synthesis case: append to the `run_synthesis` loop in Section 3
  of `scripts/run_user_smoketest.sh`.
- Add a new wav assertion: edit the `validate_wav` Python heredoc.
- Add Linux/Windows support: replace the `uname -s` gate with a per-platform
  `cmake` recipe and drop the `.dylib` assumption (server env var
  `QWEN3TTS_LIB_PATH` takes `.so` or `.dll`).
