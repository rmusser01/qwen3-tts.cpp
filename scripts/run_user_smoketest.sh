#!/usr/bin/env bash
# scripts/run_user_smoketest.sh
# Automated user-facing smoketest: build -> download -> synthesize (0.6B + 1.7B if
# present, F16 + Q8_0, Metal-auto + CPU-forced) -> speaker embedding -> Python
# server end-to-end. Intended for non-maintainer users validating a release.
#
# Usage:
#   bash scripts/run_user_smoketest.sh                # run everything
#   bash scripts/run_user_smoketest.sh --rebuild      # force rebuild
#   bash scripts/run_user_smoketest.sh --skip-download  # assume models already in models/
#   bash scripts/run_user_smoketest.sh --skip-server  # skip Python server test
#   bash scripts/run_user_smoketest.sh --server-port N
#   bash scripts/run_user_smoketest.sh --help

set -u
set -o pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

SMOKE_DIR="$PROJECT_ROOT/build/smoketest"
VOICES_DIR="$SMOKE_DIR/voices"
RESULTS_TSV="$SMOKE_DIR/results.tsv"
SERVER_LOG="$SMOKE_DIR/server.log"
CLONE_REF="$PROJECT_ROOT/examples/readme_clone_input.wav"

HF_BASE="https://huggingface.co/koboldcpp/tts/resolve/main"
MODEL_06_F16="models/qwen3-tts-0.6b-f16.gguf"
MODEL_06_Q8="models/qwen3-tts-0.6b-q8_0.gguf"
MODEL_TOKENIZER="models/qwen3-tts-tokenizer-f16.gguf"
MODEL_17_F16="models/qwen3-tts-1.7b-f16.gguf"
MODEL_17_Q8="models/qwen3-tts-1.7b-q8_0.gguf"

REBUILD=0
SKIP_DOWNLOAD=0
SKIP_SERVER=0
SERVER_PORT="8765"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --rebuild) REBUILD=1; shift ;;
        --skip-download) SKIP_DOWNLOAD=1; shift ;;
        --skip-server) SKIP_SERVER=1; shift ;;
        --server-port) SERVER_PORT="$2"; shift 2 ;;
        --help|-h)
            grep '^#' "$0" | head -20
            exit 0
            ;;
        *) echo "Unknown argument: $1" >&2; exit 64 ;;
    esac
done

# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

if [[ -t 1 ]]; then
    RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
else
    RED=''; GREEN=''; YELLOW=''; BLUE=''; NC=''
fi

PASS_COUNT=0
FAIL_COUNT=0
SKIP_COUNT=0
WARN_COUNT=0

banner() {
    echo ""
    echo -e "${BLUE}============================================${NC}"
    echo -e "${BLUE} $1${NC}"
    echo -e "${BLUE}============================================${NC}"
}

pass() { echo -e "${GREEN}[PASS]${NC} $1"; PASS_COUNT=$((PASS_COUNT+1)); }
fail() { echo -e "${RED}[FAIL]${NC} $1"; FAIL_COUNT=$((FAIL_COUNT+1)); }
skip() { echo -e "${YELLOW}[SKIP]${NC} $1"; SKIP_COUNT=$((SKIP_COUNT+1)); }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; WARN_COUNT=$((WARN_COUNT+1)); }
info() { echo -e "      $1"; }

tsv() {
    # tsv <model> <quant> <backend> <flow> <status> <seconds> <artifact>
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$@" >> "$RESULTS_TSV"
}

file_size_bytes() {
    if stat -f%z "$1" >/dev/null 2>&1; then
        stat -f%z "$1"
    else
        stat -c%s "$1" 2>/dev/null || echo 0
    fi
}

# Validate WAV header + fmt/data chunks + duration. Prints "OK <sec>" on success,
# or "FAIL <reason>" on failure.
validate_wav() {
    local wav="$1"
    python3 - "$wav" <<'PY'
import struct, sys, os
p = sys.argv[1]
try:
    size = os.path.getsize(p)
    if size < 10_000:
        print(f"FAIL size={size} (<10KB)"); sys.exit(0)
    with open(p, "rb") as f:
        hdr = f.read(12)
        if hdr[0:4] != b"RIFF" or hdr[8:12] != b"WAVE":
            print(f"FAIL not-RIFF-WAVE"); sys.exit(0)
        fmt_rate = None; data_bytes = None; bps = None; channels = None
        while True:
            ch = f.read(8)
            if len(ch) < 8: break
            tag, csize = ch[0:4], struct.unpack("<I", ch[4:8])[0]
            if tag == b"fmt ":
                fmt = f.read(csize)
                _fmt_tag, channels, fmt_rate, _byterate, _blk, bps = struct.unpack("<HHIIHH", fmt[:16])
            elif tag == b"data":
                data_bytes = csize
                break
            else:
                f.seek(csize, 1)
        if fmt_rate is None:
            print("FAIL no-fmt-chunk"); sys.exit(0)
        if data_bytes is None:
            print("FAIL no-data-chunk"); sys.exit(0)
        if not (8000 <= fmt_rate <= 96000):
            print(f"FAIL bad-rate={fmt_rate}"); sys.exit(0)
        if channels not in (1, 2):
            print(f"FAIL bad-channels={channels}"); sys.exit(0)
        bytes_per_sample = max(1, (bps // 8) * channels)
        samples = data_bytes // bytes_per_sample
        duration = samples / fmt_rate
        if duration < 0.5:
            print(f"FAIL short-duration={duration:.2f}s"); sys.exit(0)
        print(f"OK {duration:.2f}s rate={fmt_rate} ch={channels}")
except Exception as e:
    print(f"FAIL exc={e}")
PY
}

# ---------------------------------------------------------------------------
# Section 0: Preflight
# ---------------------------------------------------------------------------

banner "Section 0 — Preflight"

if [[ "$(uname -s)" != "Darwin" ]]; then
    fail "This script targets macOS (Metal). See scripts/USER_SMOKETEST.md for Linux/Windows notes."
    exit 2
fi
pass "macOS host detected ($(uname -sr))"

MISSING=""
for t in cmake curl python3; do
    if ! command -v "$t" >/dev/null 2>&1; then MISSING="$MISSING $t"; fi
done
if [[ -n "$MISSING" ]]; then
    fail "Missing required tools:$MISSING"
    info "Install with: xcode-select --install; brew install cmake python3"
    exit 3
fi
pass "Required tools present: cmake curl python3"

if [[ ! -f "$PROJECT_ROOT/CMakeLists.txt" || ! -d "$PROJECT_ROOT/ggml" ]]; then
    fail "Not a qwen3-tts.cpp checkout (missing CMakeLists.txt or ggml/)"
    exit 4
fi

if [[ ! -f "$PROJECT_ROOT/ggml/CMakeLists.txt" ]]; then
    info "ggml submodule missing; initializing…"
    git submodule update --init --recursive
fi
pass "ggml submodule present"

if [[ ! -f "$CLONE_REF" ]]; then
    fail "Reference clone audio missing at $CLONE_REF"
    exit 5
fi

# Disk-space warning (best-effort, macOS df reports in 512-byte blocks).
FREE_GB=$(df -g "$PROJECT_ROOT" 2>/dev/null | awk 'NR==2 {print $4}')
if [[ -n "${FREE_GB:-}" ]] && (( FREE_GB < 5 )); then
    warn "Only ${FREE_GB} GB free on this volume; first run needs ~5 GB."
fi

CPU_COUNT=$(sysctl -n hw.ncpu 2>/dev/null || echo 4)
info "Parallel build: ${CPU_COUNT} cores"

rm -rf "$SMOKE_DIR"
mkdir -p "$SMOKE_DIR" "$VOICES_DIR"
: > "$RESULTS_TSV"
info "Fresh artifacts directory: $SMOKE_DIR"

# ---------------------------------------------------------------------------
# Section 1: Build
# ---------------------------------------------------------------------------

banner "Section 1 — Build"

need_build=0
for bin in build/qwen3-tts-cli build/libqwen3tts.dylib build/qwen3-tts-quantize; do
    if [[ ! -e "$bin" ]]; then
        need_build=1
        break
    fi
done
if (( REBUILD )); then need_build=1; fi

if (( need_build )); then
    info "Building ggml with Metal…"
    if ! cmake -S ggml -B ggml/build -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release >"$SMOKE_DIR/cmake_ggml.log" 2>&1; then
        fail "ggml cmake configure failed (see $SMOKE_DIR/cmake_ggml.log)"; exit 10
    fi
    if ! cmake --build ggml/build -j"$CPU_COUNT" >>"$SMOKE_DIR/cmake_ggml.log" 2>&1; then
        fail "ggml build failed (see $SMOKE_DIR/cmake_ggml.log)"; exit 10
    fi
    pass "ggml built with Metal"

    info "Building qwen3-tts.cpp…"
    if ! cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DQWEN3_TTS_COREML=ON >"$SMOKE_DIR/cmake_project.log" 2>&1; then
        fail "project cmake configure failed (see $SMOKE_DIR/cmake_project.log)"; exit 11
    fi
    if ! cmake --build build -j"$CPU_COUNT" >>"$SMOKE_DIR/cmake_project.log" 2>&1; then
        fail "project build failed (see $SMOKE_DIR/cmake_project.log)"; exit 11
    fi
    pass "project built"
else
    pass "Build cached (pass --rebuild to force)"
fi

for bin in build/qwen3-tts-cli build/libqwen3tts.dylib build/qwen3-tts-quantize; do
    if [[ ! -e "$bin" ]]; then
        fail "Expected artifact missing after build: $bin"; exit 12
    fi
done

# ---------------------------------------------------------------------------
# Section 2: Model acquisition
# ---------------------------------------------------------------------------

banner "Section 2 — Model acquisition"

mkdir -p models

download_if_missing() {
    local rel="$1"
    local filename
    filename="$(basename "$rel")"
    local url="$HF_BASE/$filename"
    if [[ -f "$rel" && $(file_size_bytes "$rel") -gt 1000000 ]]; then
        pass "Already present: $rel ($(file_size_bytes "$rel") bytes)"
        return 0
    fi
    if (( SKIP_DOWNLOAD )); then
        skip "Download disabled; $rel missing"
        return 1
    fi
    info "Downloading $url"
    if curl -fL --retry 3 --retry-delay 2 -o "$rel" "$url" 2>&1 | tail -n 2 | sed 's/^/        /'; then
        pass "Downloaded $rel ($(file_size_bytes "$rel") bytes)"
    else
        fail "Failed to download $filename"
        rm -f "$rel"
        return 1
    fi
}

download_if_missing "$MODEL_06_F16" || true
download_if_missing "$MODEL_06_Q8"  || true
download_if_missing "$MODEL_TOKENIZER" || true

if [[ ! -f "$MODEL_06_F16" && ! -f "$MODEL_06_Q8" ]]; then
    fail "No 0.6B model available; cannot continue."
    exit 20
fi
if [[ ! -f "$MODEL_TOKENIZER" ]]; then
    fail "Tokenizer model missing; synthesis will fail."
    exit 21
fi

# 1.7B is optional — no public prebuilt GGUF at the time of writing.
if [[ -f "$MODEL_17_F16" ]]; then
    pass "Found local 1.7B F16 at $MODEL_17_F16"
fi
if [[ -f "$MODEL_17_Q8" ]]; then
    pass "Found local 1.7B Q8_0 at $MODEL_17_Q8"
fi
if [[ ! -f "$MODEL_17_F16" && ! -f "$MODEL_17_Q8" ]]; then
    skip "1.7B models not present (see scripts/USER_SMOKETEST.md to get them)"
fi

# ---------------------------------------------------------------------------
# Section 3: Synthesis matrix
# ---------------------------------------------------------------------------

banner "Section 3 — Synthesis matrix"

# run_synthesis <model_tag> <quant_tag> <backend> <flow> <tts_model_flag> <extra_args...>
# model_tag and quant_tag are string tags for filenames/log lines. backend is
# "auto" (Metal preferred) or "cpu". Flow is a string like basic/clone/instr.
run_synthesis() {
    local model_tag="$1" quant_tag="$2" backend="$3" flow="$4"
    shift 4
    local out_tag="${model_tag}_${quant_tag}_${backend}_${flow}"
    local out_wav="$SMOKE_DIR/${out_tag}.wav"
    local log_file="$SMOKE_DIR/${out_tag}.log"

    local start=$SECONDS
    local env_prefix=()
    if [[ "$backend" == "cpu" ]]; then
        env_prefix=("QWEN3_TTS_BACKEND=cpu")
    else
        env_prefix=("QWEN3_TTS_BACKEND=auto")
    fi

    local rc=0
    env "${env_prefix[@]}" ./build/qwen3-tts-cli -m models "$@" -o "$out_wav" \
        >"$log_file" 2>&1 || rc=$?
    local elapsed=$((SECONDS - start))

    if (( rc != 0 )); then
        fail "$out_tag (exit=$rc, ${elapsed}s) — see $log_file"
        tsv "$model_tag" "$quant_tag" "$backend" "$flow" "FAIL" "$elapsed" "$out_wav"
        return 0
    fi

    local v
    v="$(validate_wav "$out_wav")"
    if [[ "$v" != OK* ]]; then
        fail "$out_tag wav invalid — $v"
        tsv "$model_tag" "$quant_tag" "$backend" "$flow" "FAIL" "$elapsed" "$out_wav"
        return 0
    fi

    # For the Metal (auto) case, try to confirm Metal was actually picked.
    local status="PASS"
    if [[ "$backend" == "auto" ]]; then
        if grep -Eiq "backend.*metal|metal backend|ggml_metal" "$log_file"; then
            :
        elif grep -Eiq "using CPU backend|backend.*cpu" "$log_file"; then
            warn "$out_tag ran but fell back to CPU on 'auto' — Metal did not initialize"
            status="WARN"
        fi
    fi

    if [[ "$status" == "PASS" ]]; then
        pass "$out_tag (${elapsed}s, $v)"
    fi
    tsv "$model_tag" "$quant_tag" "$backend" "$flow" "$status" "$elapsed" "$out_wav"
}

TEXT_BASIC='Hello, this is a qwen3 TTS smoke test.'
INSTRUCTION='Speak in a calm, slow voice.'

# Build the matrix as a list of "model_tag quant_tag tts-model-flag-or-empty" rows.
# `--tts-model` takes a filename relative to the model dir (-m), not a full path —
# see src/pipeline/qwen3_tts.cpp:131-132 which prepends model_dir + "/".
# `--tts-model` takes a filename relative to the model dir (-m), not a full path —
# see src/pipeline/qwen3_tts.cpp:131-132 which prepends model_dir + "/".
# Always pass it explicitly so rows are unambiguous: auto-detect would otherwise
# prefer Q8_0 over F16 when both exist (src/pipeline/qwen3_tts.cpp:134-143).
MATRIX=()
[[ -f "$MODEL_06_F16" ]] && MATRIX+=("0.6b f16 --tts-model|$(basename "$MODEL_06_F16")")
[[ -f "$MODEL_06_Q8"  ]] && MATRIX+=("0.6b q8_0 --tts-model|$(basename "$MODEL_06_Q8")")
[[ -f "$MODEL_17_F16" ]] && MATRIX+=("1.7b f16 --tts-model|$(basename "$MODEL_17_F16")")
[[ -f "$MODEL_17_Q8"  ]] && MATRIX+=("1.7b q8_0 --tts-model|$(basename "$MODEL_17_Q8")")

for row in "${MATRIX[@]}"; do
    # shellcheck disable=SC2206
    parts=($row)
    mtag="${parts[0]}"; qtag="${parts[1]}"; mflag_raw="${parts[2]:-}"
    mflag_args=()
    if [[ -n "$mflag_raw" ]]; then
        IFS='|' read -r a b <<<"$mflag_raw"
        mflag_args=("$a" "$b")
    fi

    # 1.7B Base is voice-cloning-only per Qwen's HF README — text-only
    # inference is explicitly unsupported and causes the model to ramble to
    # the max-tokens ceiling (~327s of speech for an 8-word prompt). Cap the
    # basic row on 1.7B so the smoketest confirms the text-only invocation
    # path plumbs through without waiting 15+ minutes per combo.
    basic_extra_args=()
    if [[ "$mtag" == "1.7b" ]]; then
        basic_extra_args=(--max-tokens 250)
    fi

    for backend in auto cpu; do
        run_synthesis "$mtag" "$qtag" "$backend" "basic" \
            "${mflag_args[@]+"${mflag_args[@]}"}" "${basic_extra_args[@]+"${basic_extra_args[@]}"}" -t "$TEXT_BASIC"
        run_synthesis "$mtag" "$qtag" "$backend" "clone" \
            "${mflag_args[@]+"${mflag_args[@]}"}" -t "$TEXT_BASIC" -r "$CLONE_REF"
        run_synthesis "$mtag" "$qtag" "$backend" "instr" \
            "${mflag_args[@]+"${mflag_args[@]}"}" -t "$TEXT_BASIC" --instruction "$INSTRUCTION"
        run_synthesis "$mtag" "$qtag" "$backend" "clone_instr" \
            "${mflag_args[@]+"${mflag_args[@]}"}" -t "$TEXT_BASIC" -r "$CLONE_REF" --instruction "$INSTRUCTION"
    done
done

# ---------------------------------------------------------------------------
# Section 4: Speaker embedding workflow
# ---------------------------------------------------------------------------

banner "Section 4 — Speaker embedding workflow"

EMB_JSON="$SMOKE_DIR/speaker.json"
EMB_TMP_WAV="$SMOKE_DIR/_embedding_dump_tmp.wav"
EMB_REUSE_WAV="$SMOKE_DIR/embedding_reuse.wav"

start=$SECONDS
if QWEN3_TTS_BACKEND=auto ./build/qwen3-tts-cli -m models \
        -r "$CLONE_REF" \
        --dump-speaker-embedding "$EMB_JSON" \
        -t "embedding extraction" -o "$EMB_TMP_WAV" \
        >"$SMOKE_DIR/embedding_dump.log" 2>&1; then
    elapsed=$((SECONDS - start))
    if [[ -s "$EMB_JSON" ]]; then
        if python3 - "$EMB_JSON" <<'PY'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
    n = d["embedding_size"]; arr = d["data"]
    assert isinstance(arr, list) and len(arr) == n, f"len mismatch {len(arr)} vs {n}"
    assert all(isinstance(x, (int,float)) for x in arr[:4]), "non-numeric"
    sys.exit(0)
except Exception as e:
    print(e); sys.exit(1)
PY
        then
            pass "Speaker embedding dumped (${elapsed}s) → $EMB_JSON"
            tsv "embed" "-" "auto" "dump" "PASS" "$elapsed" "$EMB_JSON"
        else
            fail "speaker.json schema check failed"
            tsv "embed" "-" "auto" "dump" "FAIL" "$elapsed" "$EMB_JSON"
        fi
    else
        fail "Embedding file missing or empty"
        tsv "embed" "-" "auto" "dump" "FAIL" "$elapsed" "$EMB_JSON"
    fi
else
    fail "dump-speaker-embedding CLI call failed"
    tsv "embed" "-" "auto" "dump" "FAIL" "0" "$EMB_JSON"
fi

if [[ -s "$EMB_JSON" ]]; then
    start=$SECONDS
    if QWEN3_TTS_BACKEND=auto ./build/qwen3-tts-cli -m models \
            --speaker-embedding "$EMB_JSON" \
            -t "Reusing cached speaker embedding from disk." \
            -o "$EMB_REUSE_WAV" \
            >"$SMOKE_DIR/embedding_reuse.log" 2>&1; then
        elapsed=$((SECONDS - start))
        v="$(validate_wav "$EMB_REUSE_WAV")"
        if [[ "$v" == OK* ]]; then
            pass "Speaker embedding reuse (${elapsed}s, $v)"
            tsv "embed" "-" "auto" "reuse" "PASS" "$elapsed" "$EMB_REUSE_WAV"
        else
            fail "embedding_reuse.wav invalid — $v"
            tsv "embed" "-" "auto" "reuse" "FAIL" "$elapsed" "$EMB_REUSE_WAV"
        fi
    else
        fail "speaker-embedding CLI call failed"
        tsv "embed" "-" "auto" "reuse" "FAIL" "0" "$EMB_REUSE_WAV"
    fi
fi

# ---------------------------------------------------------------------------
# Section 5: Python server
# ---------------------------------------------------------------------------

SERVER_PID=""
cleanup_server() {
    if [[ -n "$SERVER_PID" ]]; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup_server EXIT

if (( SKIP_SERVER )); then
    banner "Section 5 — Python server (skipped by flag)"
    skip "Server test skipped (--skip-server)"
else
    banner "Section 5 — Python server"

    VENV="$SMOKE_DIR/venv"
    info "Creating venv at $VENV"
    if ! python3 -m venv "$VENV" >"$SMOKE_DIR/venv.log" 2>&1; then
        fail "Failed to create venv"
        tsv "server" "-" "auto" "setup" "FAIL" "0" "$VENV"
    else
        # shellcheck disable=SC1091
        source "$VENV/bin/activate"
        if ! pip install --quiet --upgrade pip >>"$SMOKE_DIR/venv.log" 2>&1; then
            warn "pip upgrade failed (non-fatal)"
        fi
        if ! pip install --quiet -r server/requirements.txt >>"$SMOKE_DIR/venv.log" 2>&1; then
            fail "pip install of server requirements failed (see $SMOKE_DIR/venv.log)"
            tsv "server" "-" "auto" "setup" "FAIL" "0" "$VENV"
        else
            pass "Python server deps installed"

            # Seed a voice from the dumped embedding so the server has >1 voice.
            if [[ -s "$EMB_JSON" ]]; then
                cp "$EMB_JSON" "$VOICES_DIR/smoketest.json"
                info "Seeded voice 'smoketest' in $VOICES_DIR"
            fi

            info "Starting server on 127.0.0.1:$SERVER_PORT"
            QWEN3TTS_MODEL_DIR="$PROJECT_ROOT/models" \
            QWEN3TTS_VOICES_DIR="$VOICES_DIR" \
            QWEN3TTS_LIB_PATH="$PROJECT_ROOT/build/libqwen3tts.dylib" \
            QWEN3TTS_HOST="127.0.0.1" \
            QWEN3TTS_PORT="$SERVER_PORT" \
                python server/main.py >"$SERVER_LOG" 2>&1 &
            SERVER_PID=$!

            # Readiness probe (/health is defined in server/main.py).
            ready=0
            for _ in $(seq 1 60); do
                if curl -fsS "http://127.0.0.1:$SERVER_PORT/health" -o /dev/null 2>/dev/null; then
                    ready=1
                    break
                fi
                if ! kill -0 "$SERVER_PID" 2>/dev/null; then
                    break
                fi
                sleep 1
            done

            if (( !ready )); then
                fail "Server did not become ready within 60s (log: $SERVER_LOG)"
                tsv "server" "-" "auto" "startup" "FAIL" "60" "$SERVER_LOG"
            else
                pass "Server responded to /health"
                tsv "server" "-" "auto" "startup" "PASS" "0" "$SERVER_LOG"

                # Voice listing endpoint
                if curl -fsS "http://127.0.0.1:$SERVER_PORT/v1/audio/voices" \
                        -o "$SMOKE_DIR/server_voices.json" 2>/dev/null; then
                    if python3 -c 'import json,sys;d=json.load(open(sys.argv[1]));assert "voices" in d and "default" in d["voices"]' \
                            "$SMOKE_DIR/server_voices.json" 2>/dev/null; then
                        pass "/v1/audio/voices returned valid list"
                    else
                        warn "/v1/audio/voices returned malformed JSON"
                    fi
                else
                    warn "/v1/audio/voices request failed"
                fi

                # Default voice
                start=$SECONDS
                if curl -fsS -X POST "http://127.0.0.1:$SERVER_PORT/v1/audio/speech" \
                        -H 'Content-Type: application/json' \
                        -d '{"input":"Server default voice end to end test.","model":"qwen3-tts"}' \
                        -o "$SMOKE_DIR/server_default.wav"; then
                    elapsed=$((SECONDS - start))
                    v="$(validate_wav "$SMOKE_DIR/server_default.wav")"
                    if [[ "$v" == OK* ]]; then
                        pass "Server /v1/audio/speech default voice (${elapsed}s, $v)"
                        tsv "server" "-" "auto" "default_voice" "PASS" "$elapsed" "$SMOKE_DIR/server_default.wav"
                    else
                        fail "server_default.wav invalid — $v"
                        tsv "server" "-" "auto" "default_voice" "FAIL" "$elapsed" "$SMOKE_DIR/server_default.wav"
                    fi
                else
                    fail "/v1/audio/speech (default) returned non-2xx"
                    tsv "server" "-" "auto" "default_voice" "FAIL" "0" "$SMOKE_DIR/server_default.wav"
                fi

                # Cloned voice (only if we seeded it)
                if [[ -s "$VOICES_DIR/smoketest.json" ]]; then
                    start=$SECONDS
                    if curl -fsS -X POST "http://127.0.0.1:$SERVER_PORT/v1/audio/speech" \
                            -H 'Content-Type: application/json' \
                            -d '{"input":"Server cloned voice end to end test.","model":"qwen3-tts","voice":"smoketest"}' \
                            -o "$SMOKE_DIR/server_cloned.wav"; then
                        elapsed=$((SECONDS - start))
                        v="$(validate_wav "$SMOKE_DIR/server_cloned.wav")"
                        if [[ "$v" == OK* ]]; then
                            pass "Server /v1/audio/speech cloned voice (${elapsed}s, $v)"
                            tsv "server" "-" "auto" "cloned_voice" "PASS" "$elapsed" "$SMOKE_DIR/server_cloned.wav"
                        else
                            fail "server_cloned.wav invalid — $v"
                            tsv "server" "-" "auto" "cloned_voice" "FAIL" "$elapsed" "$SMOKE_DIR/server_cloned.wav"
                        fi
                    else
                        fail "/v1/audio/speech (cloned) returned non-2xx"
                        tsv "server" "-" "auto" "cloned_voice" "FAIL" "0" "$SMOKE_DIR/server_cloned.wav"
                    fi
                fi
            fi

            cleanup_server
            SERVER_PID=""
        fi
        deactivate 2>/dev/null || true
    fi
fi

# ---------------------------------------------------------------------------
# Section 6: Summary
# ---------------------------------------------------------------------------

banner "Section 6 — Summary"

if [[ -s "$RESULTS_TSV" ]]; then
    echo ""
    printf '%-8s %-6s %-6s %-14s %-6s %-7s  %s\n' \
        "MODEL" "QUANT" "BACKEND" "FLOW" "STATUS" "TIME_s" "ARTIFACT"
    printf '%-8s %-6s %-6s %-14s %-6s %-7s  %s\n' \
        "--------" "------" "------" "--------------" "------" "-------" "--------"
    while IFS=$'\t' read -r m q b f s t a; do
        # Colorize status.
        case "$s" in
            PASS) sc="${GREEN}${s}${NC}" ;;
            FAIL) sc="${RED}${s}${NC}" ;;
            WARN) sc="${YELLOW}${s}${NC}" ;;
            SKIP) sc="${YELLOW}${s}${NC}" ;;
            *)    sc="$s" ;;
        esac
        printf '%-8s %-6s %-6s %-14s %-16b %-7s  %s\n' \
            "$m" "$q" "$b" "$f" "$sc" "$t" "$a"
    done < "$RESULTS_TSV"
    echo ""
fi

echo "Passes:   $PASS_COUNT"
echo "Warnings: $WARN_COUNT"
echo "Skips:    $SKIP_COUNT"
echo "Failures: $FAIL_COUNT"
echo ""
echo "Raw results: $RESULTS_TSV"
echo "Listen to an output:  afplay $SMOKE_DIR/0.6b_f16_auto_basic.wav"
echo ""

if (( FAIL_COUNT > 0 )); then
    echo -e "${RED}Smoketest FAILED.${NC}"
    exit 1
fi

echo -e "${GREEN}Smoketest PASSED.${NC}"
if (( WARN_COUNT > 0 )); then
    echo -e "${YELLOW}(${WARN_COUNT} warning(s) — review for Metal fallback etc.)${NC}"
fi
exit 0
