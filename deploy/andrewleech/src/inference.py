"""Shared ONNX greedy-decode loop for Qwen3-ASR."""

import numpy as np

from src.prompt import EOS_TOKEN_IDS, get_audio_pad_range

# Additive attention mask for the merged decoder: 0 attends, this blocks.
_MASK_BLOCKED = np.finfo(np.float32).min


def _causal_mask(q_len: int, past_len: int) -> np.ndarray:
    """[1, 1, q_len, past_len + q_len] additive mask over a growing KV cache.

    Derived here rather than inside the graph because shape arithmetic done in
    the exporter gets frozen into constants.
    """
    new = np.triu(np.full((q_len, q_len), _MASK_BLOCKED, dtype=np.float32), k=1)
    if past_len == 0:
        return new[np.newaxis, np.newaxis]
    past = np.zeros((q_len, past_len), dtype=np.float32)
    return np.concatenate([past, new], axis=-1)[np.newaxis, np.newaxis]


def _empty_kv(merged) -> np.ndarray:
    """[num_layers, 1, kv_heads, 0, head_dim] — present_keys layout, empty seq."""
    shape = list(merged.get_outputs()[1].shape)
    if any(shape[i] is None for i in (0, 2, 4)):
        raise ValueError(
            "merged decoder exposes a dynamic KV shape; the empty cache cannot be sized"
        )
    return np.zeros((shape[0], 1, shape[2], 0, shape[4]), dtype=np.float32)


def merged_prefill(
    merged,
    embed_tokens: np.ndarray,
    token_ids: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Run a prompt span through the merged decoder and return the KV it produces.

    Used for the system turn, whose KV depends only on the context text.
    """
    empty = _empty_kv(merged)
    _, keys, values = merged.run(
        ["logits", "present_keys", "present_values"],
        {
            "input_embeds": embed_tokens[token_ids][np.newaxis, :, :],
            "position_ids": np.arange(len(token_ids), dtype=np.int64)[np.newaxis, :],
            "attention_mask": _causal_mask(len(token_ids), 0),
            "past_keys": empty,
            "past_values": empty,
        },
    )
    return keys, values


def greedy_decode_merged(
    merged,
    embed_tokens: np.ndarray,
    audio_features: np.ndarray,
    prompt_ids: list[int],
    max_tokens: int = 256,
    past_keys: np.ndarray | None = None,
    past_values: np.ndarray | None = None,
    n_past: int = 0,
) -> list[int]:
    """
    Greedy decode on the merged decoder.

    The merged graph takes pre-looked-up embeddings plus a KV cache, so a caller
    holding the KV of a decoded prefix (see merged_prefill) passes it as
    past_keys/past_values with n_past set, and only the tail is prefill here.

    Args:
        merged: ORT session for decoder_merged
        embed_tokens: [vocab, hidden] embedding matrix
        audio_features: [1, audio_len, hidden] encoder output
        prompt_ids: list of token IDs including <|audio_pad|> placeholders
        max_tokens: maximum number of tokens to generate
        past_keys / past_values: KV of prompt_ids[:n_past]; empty when omitted
        n_past: number of prompt tokens already covered by the cache

    Returns:
        List of generated token IDs (including EOS if hit).
    """
    audio_start, audio_end = get_audio_pad_range(prompt_ids)
    audio_len = audio_end - audio_start
    if audio_features.shape[1] != audio_len:
        raise ValueError(f"Audio feature length {audio_features.shape[1]} != audio_pad count {audio_len}")

    if past_keys is None or past_values is None:
        if n_past:
            raise ValueError("n_past is set but no prefix cache was passed")
        past_keys = past_values = _empty_kv(merged)
    elif int(past_keys.shape[3]) != n_past:
        raise ValueError(f"Prefix cache holds {past_keys.shape[3]} tokens, n_past is {n_past}")

    tail_ids = prompt_ids[n_past:]
    if not tail_ids:
        raise ValueError("n_past covers the whole prompt, nothing left to prefill")

    audio_at = audio_start - n_past
    if audio_at < 0:
        raise ValueError("audio placeholders start inside the cached prefix")

    input_embeds = embed_tokens[tail_ids][np.newaxis, :, :]
    input_embeds[0, audio_at : audio_at + audio_len] = audio_features[0]

    logits, present_keys, present_values = merged.run(
        ["logits", "present_keys", "present_values"],
        {
            "input_embeds": input_embeds,
            "position_ids": np.arange(n_past, len(prompt_ids), dtype=np.int64)[np.newaxis, :],
            "attention_mask": _causal_mask(len(tail_ids), n_past),
            "past_keys": past_keys,
            "past_values": past_values,
        },
    )

    next_token = int(np.argmax(logits[0, -1, :]))
    output_tokens = [next_token]

    if next_token in EOS_TOKEN_IDS:
        return output_tokens

    pos = len(prompt_ids)
    for _ in range(max_tokens - 1):
        token_embed = embed_tokens[next_token][np.newaxis, np.newaxis, :]

        logits, present_keys, present_values = merged.run(
            ["logits", "present_keys", "present_values"],
            {
                "input_embeds": token_embed,
                "position_ids": np.array([[pos]], dtype=np.int64),
                "attention_mask": _causal_mask(1, int(present_keys.shape[3])),
                "past_keys": present_keys,
                "past_values": present_values,
            },
        )

        next_token = int(np.argmax(logits[0, -1, :]))
        output_tokens.append(next_token)
        pos += 1

        if next_token in EOS_TOKEN_IDS:
            break

    return output_tokens


def greedy_decode_onnx(
    sessions: dict,
    embed_tokens: np.ndarray,
    audio_features: np.ndarray,
    prompt_ids: list[int],
    max_tokens: int = 256,
) -> list[int]:
    """
    Greedy decode using ONNX Runtime sessions.

    Supports both v1 (input_embeds for both init and step) and v3 (input_ids
    for init, input_embeds for step) decoder formats. The format is detected
    by checking decoder_init's input names.

    Args:
        sessions: dict with keys "decoder_init" and "decoder_step" (ORT sessions)
        embed_tokens: [vocab, hidden] embedding matrix
        audio_features: [1, audio_len, hidden] encoder output
        prompt_ids: list of token IDs including <|audio_pad|> placeholders
        max_tokens: maximum number of tokens to generate

    Returns:
        List of generated token IDs (including EOS if hit).
    """
    position_ids = np.arange(len(prompt_ids), dtype=np.int64)[np.newaxis, :]

    # Detect decoder format from input names
    init_input_names = {inp.name for inp in sessions["decoder_init"].get_inputs()}

    if "input_ids" in init_input_names:
        # v3 format: decoder_init accepts input_ids + audio_features + audio_offset
        audio_start, _ = get_audio_pad_range(prompt_ids)
        input_ids = np.array(prompt_ids, dtype=np.int64)[np.newaxis, :]
        audio_offset = np.array([audio_start], dtype=np.int64)

        logits, present_keys, present_values = sessions["decoder_init"].run(
            ["logits", "present_keys", "present_values"],
            {
                "input_ids": input_ids,
                "position_ids": position_ids,
                "audio_features": audio_features,
                "audio_offset": audio_offset,
            },
        )
    else:
        # v1 format: decoder_init accepts input_embeds
        input_embeds = embed_tokens[prompt_ids].copy()

        audio_start, audio_end = get_audio_pad_range(prompt_ids)
        audio_len = audio_end - audio_start
        if audio_features.shape[1] != audio_len:
            raise ValueError(f"Audio feature length {audio_features.shape[1]} != audio_pad count {audio_len}")
        input_embeds[audio_start:audio_end] = audio_features[0]
        input_embeds = input_embeds[np.newaxis, :, :]

        logits, present_keys, present_values = sessions["decoder_init"].run(
            ["logits", "present_keys", "present_values"],
            {"input_embeds": input_embeds, "position_ids": position_ids},
        )

    next_token = int(np.argmax(logits[0, -1, :]))
    output_tokens = [next_token]

    if next_token in EOS_TOKEN_IDS:
        return output_tokens

    # Autoregressive loop — decoder_step always accepts input_embeds
    pos = len(prompt_ids)
    for _ in range(max_tokens - 1):
        token_embed = embed_tokens[next_token][np.newaxis, np.newaxis, :]
        step_pos = np.array([[pos]], dtype=np.int64)

        logits, present_keys, present_values = sessions["decoder_step"].run(
            ["logits", "present_keys", "present_values"],
            {
                "input_embeds": token_embed,
                "position_ids": step_pos,
                "past_keys": present_keys,
                "past_values": present_values,
            },
        )

        next_token = int(np.argmax(logits[0, -1, :]))
        output_tokens.append(next_token)
        pos += 1

        if next_token in EOS_TOKEN_IDS:
            break

    return output_tokens
