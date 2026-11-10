"""
Prompt template construction and token ID constants for Qwen3-ASR.

The prompt structure for ASR transcription:
    <|im_start|>system\n{context}<|im_end|>\n
    <|im_start|>user\n<|audio_start|><|audio_pad|>...<|audio_pad|><|audio_end|><|im_end|>\n
    <|im_start|>assistant\n

Where <|audio_pad|> is repeated N times (N = encoder output sequence length)
and {context} is the optional system-message context (empty by default).
"""

from typing import Any

# Special token IDs — shared across all Qwen3-ASR model sizes; validated by
# verify_special_tokens() in export.py at export time against the actual tokenizer.
ENDOFTEXT_TOKEN_ID = 151643  # <|endoftext|> - pad token, also EOS
IM_START_TOKEN_ID = 151644  # <|im_start|>
IM_END_TOKEN_ID = 151645  # <|im_end|> - also EOS
AUDIO_START_TOKEN_ID = 151669  # <|audio_start|>
AUDIO_END_TOKEN_ID = 151670  # <|audio_end|>
AUDIO_PAD_TOKEN_ID = 151676  # <|audio_pad|> - replaced by encoder output
ASR_TEXT_TOKEN_ID = 151704  # <asr_text>

# Both are EOS tokens — generation stops on either
EOS_TOKEN_IDS = [ENDOFTEXT_TOKEN_ID, IM_END_TOKEN_ID]


def get_feat_extract_output_lengths(input_lengths: int) -> int:
    """Compute the number of audio tokens the encoder produces from a given mel frame count.

    Delegates to encoder_wrapper._get_feat_extract_output_lengths which uses
    ONNX-safe integer division.
    """
    from src.encoder_wrapper import _get_feat_extract_output_lengths

    return _get_feat_extract_output_lengths(input_lengths)  # type: ignore[no-any-return]


def _prompt_text(audio_token_count: int, context: str, language: str | None) -> str:
    """Render the full prompt as text, matching the official chat template."""
    pads = "<|audio_pad|>" * audio_token_count
    text = (
        f"<|im_start|>system\n{context}<|im_end|>\n"
        f"<|im_start|>user\n<|audio_start|>{pads}<|audio_end|><|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    if language is not None:
        text += f"language {language}<asr_text>"
    return text


def _encode_text(tokenizer: Any, text: str) -> list[int]:
    """Encode text with either a HuggingFace tokenizer or a tokenizers.Tokenizer."""
    ids = tokenizer.encode(text, add_special_tokens=False)
    if hasattr(ids, "ids"):
        ids = ids.ids
    return [int(t) for t in ids]


def build_prompt_ids(
    audio_token_count: int,
    language: str | None = None,
    *,
    context: str | None = None,
    tokenizer: Any,
) -> list[int]:
    """
    Build the full prompt token ID sequence for ASR transcription.

    The template is rendered as text and encoded in one pass, so token
    boundaries match the official processor's chat-template tokenization.

    Args:
        audio_token_count: Number of encoder output tokens. Use
            get_feat_extract_output_lengths(mel_frames) to compute this
            from mel frame count.
            This determines how many <|audio_pad|> tokens are inserted.
        language: Optional language name to force (e.g. "English"). If provided,
            the assistant prefix becomes "language {name}<asr_text>".
        context: Optional system-message content (hotwords / domain hints).
        tokenizer: Tokenizer used to encode the template (HuggingFace tokenizer
            or tokenizers.Tokenizer).

    Returns:
        List of token IDs forming the complete prompt.
    """
    ids = _encode_text(tokenizer, _prompt_text(audio_token_count, context or "", language))
    pad_count = sum(1 for t in ids if t == AUDIO_PAD_TOKEN_ID)
    if pad_count != audio_token_count:
        raise ValueError(
            f"prompt contains {pad_count} <|audio_pad|> tokens, expected {audio_token_count}"
        )
    return ids


def get_audio_pad_range(prompt_ids: list[int]) -> tuple[int, int]:
    """
    Find the start and end indices of <|audio_pad|> tokens in the prompt.

    Returns:
        (start_idx, end_idx) — the range [start_idx, end_idx) where audio
        features should replace token embeddings.
    """
    start = None
    end = None
    for i, tid in enumerate(prompt_ids):
        if tid == AUDIO_PAD_TOKEN_ID:
            if start is None:
                start = i
            end = i + 1
    if start is None or end is None:
        raise ValueError("No <|audio_pad|> tokens found in prompt")
    return start, end
