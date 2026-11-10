#!/usr/bin/env python3
"""
Common ASR model interface: abstract base + implementation for the exported
Qwen3-ASR ONNX package.

The concrete class builds directly on the repo's inference pieces
(src/mel.py, src/prompt.py, src/inference.py); no wrapper layer in between.
"""

import json
import os
import re
import sys
from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.inference import greedy_decode_onnx
from src.mel import log_mel_spectrogram
from src.prompt import EOS_TOKEN_IDS, build_prompt_ids

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


class BaseASRModel(ABC):
    """ASR模型基类"""

    def __init__(self, model_id: str):
        self.model_id = model_id
        self.model = None

    @abstractmethod
    def load(self):
        """加载模型"""
        pass

    @abstractmethod
    def transcribe(self, audio_batch: List[np.ndarray]) -> List[str]:
        """转录一批音频"""
        pass


class Qwen3ASRModel(BaseASRModel):
    """Qwen3-ASR ONNX implementation.

    model_id is the exported package directory (encoder/decoder_init/decoder_step,
    embed_tokens.bin, config.json, tokenizer.json). quant selects the quantized
    variant, with fallback to the unsuffixed FP32 file per component.
    """

    def __init__(
        self,
        model_id: str,
        quant: Optional[str] = "int4",
        max_new_tokens: int = 256,
        context: Optional[str] = None,
        language: Optional[str] = None,
    ) -> None:
        super().__init__(model_id)
        self.quant = quant
        self.max_new_tokens = int(max_new_tokens)
        self.context = context
        self.language = language
        self._embed_tokens: Optional[np.ndarray] = None
        self._tokenizer = None

    def _resolve(self, name: str) -> str:
        if self.quant:
            path = os.path.join(self.model_id, f"{name}.{self.quant}.onnx")
            if os.path.exists(path):
                return path
        path = os.path.join(self.model_id, f"{name}.onnx")
        if os.path.exists(path):
            return path
        raise FileNotFoundError(f"No ONNX file for {name} (quant={self.quant}) in {self.model_id}")

    def _load_embed(self) -> np.ndarray:
        with open(os.path.join(self.model_id, "config.json")) as f:
            cfg = json.load(f)
        if "embed_tokens_shape" in cfg:
            shape = cfg["embed_tokens_shape"]
        else:
            shape = [cfg["decoder"]["vocab_size"], cfg["decoder"]["hidden_size"]]
        dtype = np.float16 if cfg.get("embed_tokens_dtype", "float32") == "float16" else np.float32
        embed = np.fromfile(os.path.join(self.model_id, "embed_tokens.bin"), dtype=dtype).reshape(shape)
        return embed.astype(np.float32)

    def load(self):
        import onnxruntime as ort
        from transformers import AutoTokenizer

        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        self.model = {
            name: ort.InferenceSession(
                self._resolve(name),
                opts,
                providers=["CPUExecutionProvider"],
            )
            for name in ("encoder", "decoder_init", "decoder_step")
        }
        self._embed_tokens = self._load_embed()
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        return self

    def _transcribe_one(self, wav: np.ndarray) -> str:
        mel = log_mel_spectrogram(wav)
        mel_np = mel.cpu().numpy() if hasattr(mel, "cpu") else np.asarray(mel)
        (audio_features,) = self.model["encoder"].run(["audio_features"], {"mel": mel_np})
        audio_features = np.asarray(audio_features, dtype=np.float32)

        prompt_ids = build_prompt_ids(
            int(audio_features.shape[1]),
            language=self.language,
            context=self.context,
            tokenizer=self._tokenizer,
        )
        tokens = greedy_decode_onnx(
            self.model,
            self._embed_tokens,
            audio_features,
            prompt_ids,
            max_tokens=self.max_new_tokens,
        )
        while tokens and tokens[-1] in EOS_TOKEN_IDS:
            tokens.pop()
        text = self._tokenizer.decode(tokens, skip_special_tokens=True)
        return _LANG_PREFIX_RE.sub("", text).strip()

    def transcribe(self, audio_batch: List[np.ndarray]) -> List[str]:
        if isinstance(audio_batch, np.ndarray):
            raise TypeError("audio_batch must be a list of 1-D arrays; wrap a single array as [wav]")
        if self.model is None:
            raise RuntimeError("model not loaded; call load() first")

        results: List[str] = []
        for wav in audio_batch:
            wav = _to_mono_f32(wav)
            if wav.size == 0:
                raise ValueError("empty audio")
            results.append(self._transcribe_one(wav))
        return results
