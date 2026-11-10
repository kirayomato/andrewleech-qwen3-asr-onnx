#!/usr/bin/env python3
"""
Thin inference wrapper around the exported Qwen3-ASR ONNX package.

Library API:
    init(...)                      # configure / preload, optional
    transcribe(wav: np.ndarray) -> str
    batch_transcribe(audio_batch: List[np.ndarray]) -> List[str]

CLI:
    python transcribe.py --wav a.wav [--wav b.wav ...]

Expects the self-contained export layout (encoder.onnx, embed_tokens.bin,
config.json, tokenizer.json, plus either decoder_init/decoder_step or
decoder_merged) produced by export.py; quantized variants resolve via the
.int4/.int8 filename suffix.
"""

import argparse
import json
import os
import re
import sys
from typing import List, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.inference import greedy_decode_merged, greedy_decode_onnx  # noqa: E402
from src.mel import log_mel_spectrogram  # noqa: E402
from src.prompt import EOS_TOKEN_IDS, build_prompt_ids  # noqa: E402

# Model output may start with "language {name}<asr_text>" (or the degraded
# "<asr text>" variant); strip it so transcribe() returns plain text.
_LANG_PREFIX_RE = re.compile(
    r"^language\s+\w+(?:<asr_text>|[\s\n]+asr\s+text\s*)", re.IGNORECASE
)


def _to_mono_f32(wav: np.ndarray) -> np.ndarray:
    wav = np.asarray(wav)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if wav.dtype == np.int16:
        wav = wav.astype(np.float32) / 32768.0
    elif wav.dtype == np.int32:
        wav = wav.astype(np.float32) / 2147483648.0
    else:
        wav = wav.astype(np.float32, copy=False)
    return np.clip(wav, -1.0, 1.0)


def _find_model_path(model_dir: str, name: str, quant: Optional[str]) -> Optional[str]:
    if quant:
        path = os.path.join(model_dir, f"{name}.{quant}.onnx")
        if os.path.exists(path):
            return path
    path = os.path.join(model_dir, f"{name}.onnx")
    return path if os.path.exists(path) else None


def _resolve_model_path(model_dir: str, name: str, quant: Optional[str]) -> str:
    path = _find_model_path(model_dir, name, quant)
    if path is None:
        raise FileNotFoundError(f"No ONNX file for {name} (quant={quant}) in {model_dir}")
    return path


class Transcriber:
    """Loads the ONNX package once and transcribes 16 kHz mono waveforms.

    Both decoder layouts are supported: split (decoder_init + decoder_step) and
    merged (decoder_merged), selected by the files present in onnx_dir.
    """

    def __init__(
        self,
        onnx_dir: str = "output/qwen3-asr-0.6b",
        quant: Optional[str] = "int4",
        max_new_tokens: int = 256,
        context: Optional[str] = None,
        language: Optional[str] = None,
    ) -> None:
        import onnxruntime as ort
        from tokenizers import Tokenizer

        opts = ort.SessionOptions()
        opts.log_severity_level = 3

        def session(path: str):
            return ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])

        self.sessions = {"encoder": session(_resolve_model_path(onnx_dir, "encoder", quant))}
        merged = _find_model_path(onnx_dir, "decoder_merged", quant)
        if merged is not None:
            self.sessions["decoder_merged"] = session(merged)
        else:
            for name in ("decoder_init", "decoder_step"):
                self.sessions[name] = session(_resolve_model_path(onnx_dir, name, quant))

        self.embed = self._load_embed(onnx_dir)
        self.tokenizer = Tokenizer.from_file(os.path.join(onnx_dir, "tokenizer.json"))
        self.max_new_tokens = int(max_new_tokens)
        self.context = context
        self.language = language

    @staticmethod
    def _load_embed(model_dir: str) -> np.ndarray:
        with open(os.path.join(model_dir, "config.json")) as f:
            cfg = json.load(f)
        if "embed_tokens_shape" in cfg:
            shape = cfg["embed_tokens_shape"]
        else:
            shape = [cfg["decoder"]["vocab_size"], cfg["decoder"]["hidden_size"]]
        dtype = np.float16 if cfg.get("embed_tokens_dtype", "float32") == "float16" else np.float32
        embed = np.fromfile(os.path.join(model_dir, "embed_tokens.bin"), dtype=dtype).reshape(shape)
        return embed.astype(np.float32)

    def transcribe(self, wav: np.ndarray) -> str:
        wav = _to_mono_f32(wav)
        if wav.size == 0:
            raise ValueError("empty audio")

        mel = log_mel_spectrogram(wav)
        mel_np = mel.cpu().numpy() if hasattr(mel, "cpu") else np.asarray(mel)
        (audio_features,) = self.sessions["encoder"].run(["audio_features"], {"mel": mel_np})
        audio_features = np.asarray(audio_features, dtype=np.float32)

        prompt_ids = build_prompt_ids(
            int(audio_features.shape[1]),
            language=self.language,
            context=self.context,
            tokenizer=self.tokenizer,
        )
        if "decoder_merged" in self.sessions:
            tokens = greedy_decode_merged(
                self.sessions["decoder_merged"],
                self.embed,
                audio_features,
                prompt_ids,
                max_tokens=self.max_new_tokens,
            )
        else:
            tokens = greedy_decode_onnx(
                self.sessions, self.embed, audio_features, prompt_ids, max_tokens=self.max_new_tokens
            )
        while tokens and tokens[-1] in EOS_TOKEN_IDS:
            tokens.pop()
        text = self.tokenizer.decode(tokens, skip_special_tokens=True)
        return _LANG_PREFIX_RE.sub("", text).strip()

    def batch_transcribe(self, audio_batch: List[np.ndarray]) -> List[str]:
        return [self.transcribe(w) for w in audio_batch]


# ---------------------------------------------------------------------------
# Module-level convenience API (lazy default instance)
# ---------------------------------------------------------------------------

_default: Optional[Transcriber] = None


def init(**kwargs) -> Transcriber:
    """Configure the module-level transcriber (see Transcriber for options)."""
    global _default
    _default = Transcriber(**kwargs)
    return _default


def _get() -> Transcriber:
    global _default
    if _default is None:
        _default = Transcriber()
    return _default


def transcribe(wav: np.ndarray) -> str:
    return _get().transcribe(wav)


def batch_transcribe(audio_batch: List[np.ndarray]) -> List[str]:
    return _get().batch_transcribe(audio_batch)


def _load_audio_any(path: str) -> np.ndarray:
    import scipy.io.wavfile
    import scipy.signal

    rate, data = scipy.io.wavfile.read(path)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if data.dtype == np.int16:
        audio = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        audio = data.astype(np.float32) / 2147483648.0
    else:
        audio = data.astype(np.float32)
    if rate != 16000:
        audio = scipy.signal.resample_poly(audio, 16000, rate).astype(np.float32)
    return np.clip(audio, -1.0, 1.0)


def main() -> None:
    p = argparse.ArgumentParser(description="Qwen3-ASR ONNX transcription")
    p.add_argument("--wav", nargs="+", required=True, help="WAV files (any rate, mono/mixed)")
    p.add_argument("--onnx-dir", default="output/qwen3-asr-0.6b", help="Exported model dir")
    p.add_argument("--quant", default="int4", help="Quant suffix (int4/int8/...), empty for FP32")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--context", default=None, help="System-message context (hotwords)")
    p.add_argument("--language", default=None, help="Forced language, e.g. English")
    args = p.parse_args()

    tr = Transcriber(
        onnx_dir=args.onnx_dir,
        quant=args.quant or None,
        max_new_tokens=args.max_new_tokens,
        context=args.context,
        language=args.language,
    )
    for path in args.wav:
        print(tr.transcribe(_load_audio_any(path)))


if __name__ == "__main__":
    main()
