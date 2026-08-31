"""Post-process entities emitted by the StarPII inference pipeline."""

# ref: https://github.com/bigcode-project/bigcode-dataset/blob/bebec929edd826f19b5fa3538f22d18d5b50da4b/pii/ner/pii_inference/utils/postprocessing.py#L7

import re


def postprocess(entity: dict[str, object]) -> dict[str, object]:
    """Trim punctuation from an entity using BigCode's inference recipe.

    Args:
        entity (dict[str, object]): Entity emitted by StarPII.

    Returns:
        dict[str, object]: Entity with adjusted value and offsets.
    """
    start = int(entity["start"])
    old_value = str(entity["value"])
    new_value = old_value.lstrip("!\"'()*+,-./:;<=>?[\\]^_`{|}~")
    if entity["tag"] != "KEY":
        new_value = re.sub(r"[\s\W]+$", "", new_value)
    new_value = new_value.strip()
    entity["start"] = start + old_value.find(new_value)
    entity["end"] = int(entity["start"]) + len(new_value)
    entity["value"] = new_value
    return entity


# (modified): Omit upstream's training-only retokenize_with_logits helper.
