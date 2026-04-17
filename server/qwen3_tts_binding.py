"""Python ctypes binding for libqwen3tts shared library."""

import ctypes
import ctypes.util
import os
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# C struct mirrors (must match qwen3tts_c_api.h exactly)
# ---------------------------------------------------------------------------

class Qwen3TtsParams(ctypes.Structure):
    _fields_ = [
        ("max_audio_tokens", ctypes.c_int32),
        ("temperature", ctypes.c_float),
        ("top_p", ctypes.c_float),
        ("top_k", ctypes.c_int32),
        ("n_threads", ctypes.c_int32),
        ("repetition_penalty", ctypes.c_float),
        ("language_id", ctypes.c_int32),
    ]


class Qwen3TtsAudio(ctypes.Structure):
    _fields_ = [
        ("samples", ctypes.POINTER(ctypes.c_float)),
        ("n_samples", ctypes.c_int32),
        ("sample_rate", ctypes.c_int32),
    ]


# ---------------------------------------------------------------------------
# Library discovery
# ---------------------------------------------------------------------------

def _find_library() -> str:
    """Find libqwen3tts shared library."""
    # 1. Explicit env var
    env_path = os.environ.get("QWEN3TTS_LIB_PATH")
    if env_path and os.path.isfile(env_path):
        return env_path

    # 2. Relative to this script (../build/)
    script_dir = Path(__file__).resolve().parent
    for suffix in ("dylib", "so", "dll"):
        candidate = script_dir.parent / "build" / f"libqwen3tts.{suffix}"
        if candidate.is_file():
            return str(candidate)

    # 3. System library search
    found = ctypes.util.find_library("qwen3tts")
    if found:
        return found

    raise RuntimeError(
        "Cannot find libqwen3tts. Set QWEN3TTS_LIB_PATH or build with cmake first."
    )


def _load_library() -> ctypes.CDLL:
    """Load and configure the shared library."""
    lib_path = _find_library()
    lib = ctypes.CDLL(lib_path)

    # -- qwen3_tts_default_params --
    lib.qwen3_tts_default_params.argtypes = [ctypes.POINTER(Qwen3TtsParams)]
    lib.qwen3_tts_default_params.restype = None

    # -- qwen3_tts_create --
    lib.qwen3_tts_create.argtypes = [ctypes.c_char_p, ctypes.c_int32]
    lib.qwen3_tts_create.restype = ctypes.c_void_p

    # -- qwen3_tts_is_loaded --
    lib.qwen3_tts_is_loaded.argtypes = [ctypes.c_void_p]
    lib.qwen3_tts_is_loaded.restype = ctypes.c_int

    # -- qwen3_tts_synthesize --
    lib.qwen3_tts_synthesize.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(Qwen3TtsParams),
    ]
    lib.qwen3_tts_synthesize.restype = ctypes.POINTER(Qwen3TtsAudio)

    # -- qwen3_tts_synthesize_with_voice_file --
    lib.qwen3_tts_synthesize_with_voice_file.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p,
        ctypes.POINTER(Qwen3TtsParams),
    ]
    lib.qwen3_tts_synthesize_with_voice_file.restype = ctypes.POINTER(Qwen3TtsAudio)

    # -- qwen3_tts_synthesize_with_voice_samples --
    lib.qwen3_tts_synthesize_with_voice_samples.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_float), ctypes.c_int32,
        ctypes.POINTER(Qwen3TtsParams),
    ]
    lib.qwen3_tts_synthesize_with_voice_samples.restype = ctypes.POINTER(Qwen3TtsAudio)

    # -- qwen3_tts_extract_embedding_file --
    lib.qwen3_tts_extract_embedding_file.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_float), ctypes.c_int32,
    ]
    lib.qwen3_tts_extract_embedding_file.restype = ctypes.c_int32

    # -- qwen3_tts_synthesize_with_embedding --
    lib.qwen3_tts_synthesize_with_embedding.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_float), ctypes.c_int32,
        ctypes.POINTER(Qwen3TtsParams),
    ]
    lib.qwen3_tts_synthesize_with_embedding.restype = ctypes.POINTER(Qwen3TtsAudio)

    # -- qwen3_tts_sample_rate --
    lib.qwen3_tts_sample_rate.argtypes = [ctypes.c_void_p]
    lib.qwen3_tts_sample_rate.restype = ctypes.c_int32

    # -- qwen3_tts_free_audio --
    lib.qwen3_tts_free_audio.argtypes = [ctypes.POINTER(Qwen3TtsAudio)]
    lib.qwen3_tts_free_audio.restype = None

    # -- qwen3_tts_destroy --
    lib.qwen3_tts_destroy.argtypes = [ctypes.c_void_p]
    lib.qwen3_tts_destroy.restype = None

    # -- qwen3_tts_get_error --
    lib.qwen3_tts_get_error.argtypes = [ctypes.c_void_p]
    lib.qwen3_tts_get_error.restype = ctypes.c_char_p

    return lib


# ---------------------------------------------------------------------------
# High-level wrapper
# ---------------------------------------------------------------------------

class QwenTTS:
    """High-level Python wrapper for the qwen3-tts C API."""

    def __init__(self, model_dir: str, n_threads: int = 4):
        self._lib = _load_library()
        self._handle = self._lib.qwen3_tts_create(
            model_dir.encode("utf-8"), n_threads
        )
        if not self._handle:
            raise RuntimeError(f"Failed to load models from {model_dir}")

    def synthesize(
        self,
        text: str,
        temperature: float = 0.9,
        top_k: int = 50,
        language_id: int = 2050,
        max_audio_tokens: int = 4096,
        repetition_penalty: float = 1.05,
    ) -> tuple[list[float], int]:
        """Synthesize text to audio. Returns (samples, sample_rate)."""
        params = self._make_params(
            temperature=temperature, top_k=top_k, language_id=language_id,
            max_audio_tokens=max_audio_tokens, repetition_penalty=repetition_penalty,
        )
        audio_ptr = self._lib.qwen3_tts_synthesize(
            self._handle, text.encode("utf-8"), ctypes.byref(params)
        )
        return self._extract_audio(audio_ptr)

    def synthesize_with_embedding(
        self,
        text: str,
        embedding: list[float],
        temperature: float = 0.9,
        top_k: int = 50,
        language_id: int = 2050,
        max_audio_tokens: int = 4096,
        repetition_penalty: float = 1.05,
    ) -> tuple[list[float], int]:
        """Synthesize with a pre-computed speaker embedding."""
        params = self._make_params(
            temperature=temperature, top_k=top_k, language_id=language_id,
            max_audio_tokens=max_audio_tokens, repetition_penalty=repetition_penalty,
        )
        emb_arr = (ctypes.c_float * len(embedding))(*embedding)
        audio_ptr = self._lib.qwen3_tts_synthesize_with_embedding(
            self._handle, text.encode("utf-8"),
            emb_arr, len(embedding), ctypes.byref(params),
        )
        return self._extract_audio(audio_ptr)

    def extract_embedding(self, wav_path: str) -> list[float]:
        """Extract speaker embedding from a WAV file."""
        buf_size = 2048
        buf = (ctypes.c_float * buf_size)()
        result = self._lib.qwen3_tts_extract_embedding_file(
            self._handle, wav_path.encode("utf-8"), buf, buf_size
        )
        if result < 0:
            err = self._get_error()
            raise RuntimeError(f"Failed to extract embedding: {err}")
        return list(buf[:result])

    def close(self):
        """Destroy the engine and release resources."""
        if self._handle:
            self._lib.qwen3_tts_destroy(self._handle)
            self._handle = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        self.close()

    # -- Private helpers --

    def _make_params(self, **kwargs) -> Qwen3TtsParams:
        params = Qwen3TtsParams()
        self._lib.qwen3_tts_default_params(ctypes.byref(params))
        for key, value in kwargs.items():
            setattr(params, key, value)
        return params

    def _extract_audio(self, audio_ptr) -> tuple[list[float], int]:
        if not audio_ptr:
            err = self._get_error()
            raise RuntimeError(f"Synthesis failed: {err}")
        audio = audio_ptr.contents
        samples = [audio.samples[i] for i in range(audio.n_samples)]
        sample_rate = audio.sample_rate
        self._lib.qwen3_tts_free_audio(audio_ptr)
        return samples, sample_rate

    def _get_error(self) -> str:
        err = self._lib.qwen3_tts_get_error(self._handle)
        return err.decode("utf-8") if err else "unknown error"
