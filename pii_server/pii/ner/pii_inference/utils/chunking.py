"""Split tokenized source files into StarPII model windows."""

# ref: https://github.com/bigcode-project/bigcode-dataset/blob/bebec929edd826f19b5fa3538f22d18d5b50da4b/pii/ner/pii_inference/utils/chunking.py#L40

from transformers import PreTrainedTokenizerBase


# (modified): Omit upstream's training-label path because this service only runs inference.
def chunk_inputs(
    input_ids: list[int],
    attention_mask: list[int],
    id: str,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    overlap: int = 0,
    **kwargs: object,
) -> list[dict[str, object]]:
    """Split one tokenized source file into overlapping model windows.

    Args:
        input_ids (list[int]): Token IDs for the complete source file.
        attention_mask (list[int]): Attention mask for the complete source file.
        id (str): Source-file identifier.
        tokenizer (PreTrainedTokenizerBase): StarPII tokenizer.
        max_length (int): Tokens in each window before special tokens.
        overlap (int): Tokens shared by adjacent windows.
        **kwargs (object): Unused dataset columns.

    Returns:
        list[dict[str, object]]: Model-ready windows for the source file.
    """
    # (modified): Accept an exact overlap so the service's CLI value is preserved.
    step = _get_chunking_step(max_length, overlap)

    def _chunked_seq(sequence: list[int]):
        for index in range(len(sequence) // step + 1):
            if index * step < len(sequence):
                yield sequence[index * step : index * step + max_length]

    chunks = zip(*(_chunked_seq(sequence) for sequence in (input_ids, attention_mask)))
    prepared_chunks = (
        _prepare_for_model(*chunk, tokenizer=tokenizer) for chunk in chunks
    )

    # (modified): Omit chunk IDs because grouping uses source IDs and offsets.
    return [
        {
            "input_ids": chunk_input_ids,
            "attention_mask": chunk_attention_mask,
            "offset": index * step,
            "id": id,
        }
        for index, (chunk_input_ids, chunk_attention_mask) in enumerate(prepared_chunks)
    ]


def _prepare_for_model(
    input_ids: list[int],
    attention_mask: list[int],
    *,
    tokenizer: PreTrainedTokenizerBase,
) -> tuple[list[int], list[int]]:
    """Add model special tokens to one window.

    Args:
        input_ids (list[int]): Window token IDs.
        attention_mask (list[int]): Window attention mask.
        tokenizer (PreTrainedTokenizerBase): StarPII tokenizer.

    Returns:
        tuple[list[int], list[int]]: Prepared IDs and attention mask.
    """
    start_token_id = (
        tokenizer.cls_token_id
        if tokenizer.cls_token_id is not None
        else tokenizer.bos_token_id
    )
    end_token_id = (
        tokenizer.sep_token_id
        if tokenizer.sep_token_id is not None
        else tokenizer.eos_token_id
    )
    input_ids = [start_token_id, *input_ids, end_token_id]
    attention_mask = [1, *attention_mask, 1]
    return input_ids, attention_mask


# (modified): Interpret the service option as an exact token overlap.
def _get_chunking_step(length: int, overlap: int) -> int:
    """Compute the distance between adjacent model windows.

    Args:
        length (int): Maximum tokens in a window.
        overlap (int): Tokens shared by adjacent windows.

    Returns:
        int: Distance between window starts.
    """
    return length - overlap
