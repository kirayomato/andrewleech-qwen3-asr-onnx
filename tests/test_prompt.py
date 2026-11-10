"""
Unit tests for prompt construction (build_prompt_ids).

The template is always encoded by a real tokenizer, so these tests are
skipped when the Qwen3-ASR tokenizer is not accessible.
"""

import pytest

from src.prompt import (
    ASR_TEXT_TOKEN_ID,
    AUDIO_END_TOKEN_ID,
    AUDIO_PAD_TOKEN_ID,
    AUDIO_START_TOKEN_ID,
    build_prompt_ids,
)


@pytest.fixture(scope="module")
def tokenizer():
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained("Qwen/Qwen3-ASR-0.6B", trust_remote_code=True)
    except Exception:
        pytest.skip("Qwen3-ASR-0.6B tokenizer not accessible")


def test_tokenizer_required():
    with pytest.raises(TypeError):
        build_prompt_ids(3)


class TestPrompt:
    def test_audio_pad_count_and_boundaries(self, tokenizer):
        n = 5
        ids = build_prompt_ids(n, tokenizer=tokenizer)
        assert ids.count(AUDIO_PAD_TOKEN_ID) == n
        start = ids.index(AUDIO_PAD_TOKEN_ID)
        assert ids[start - 1] == AUDIO_START_TOKEN_ID
        assert ids[start + n] == AUDIO_END_TOKEN_ID

    def test_context_lands_in_system_turn(self, tokenizer):
        n = 4
        ids = build_prompt_ids(n, context="特斯拉", tokenizer=tokenizer)
        assert ids.count(AUDIO_PAD_TOKEN_ID) == n
        text = tokenizer.decode(ids, skip_special_tokens=True)
        assert "特斯拉" in text
        assert text.index("特斯拉") < text.index("user")

    def test_context_changes_prompt(self, tokenizer):
        base = build_prompt_ids(3, tokenizer=tokenizer)
        with_ctx = build_prompt_ids(3, context="特斯拉", tokenizer=tokenizer)
        assert with_ctx != base

    def test_language_forcing_suffix(self, tokenizer):
        ids = build_prompt_ids(3, language="English", tokenizer=tokenizer)
        assert ids.count(AUDIO_PAD_TOKEN_ID) == 3
        assert ids[-1] == ASR_TEXT_TOKEN_ID

    def test_stray_audio_pad_in_context_raises(self, tokenizer):
        with pytest.raises(ValueError, match="audio_pad"):
            build_prompt_ids(3, context="<|audio_pad|>", tokenizer=tokenizer)
