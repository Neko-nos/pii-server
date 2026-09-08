"""Apply BigCode's redaction-time filters without rewriting source text."""

# ref: https://github.com/bigcode-project/bigcode-dataset/blob/bebec929edd826f19b5fa3538f22d18d5b50da4b/pii/ner/pii_redaction/utils.py

# (modified): Share IP exclusions with the regex detection pipeline.
from pii_server.pii.pii_redaction import is_invalid_private_or_popular_dns_ip


def is_key(matched_str: str, gibberish) -> bool:
    """Return whether a candidate is long and gibberish.

    Args:
        matched_str (str): Candidate key.
        gibberish: Loaded gibberish detector.

    Returns:
        bool: Whether the candidate should be retained.
    """
    # (modified): Reuse the service's loaded model instead of reading it per finding.
    return gibberish.is_gibberish(matched_str.lower()) and len(matched_str) > 8


def is_secret(matched_str: str) -> bool:
    """Return whether a PII span is long enough to retain.

    Args:
        matched_str (str): Candidate PII span.

    Returns:
        bool: Whether the span is longer than three characters.
    """
    return len(matched_str) > 3


def is_full_name(matched_str: str) -> bool:
    """Return whether a name contains more than one word.

    Args:
        matched_str (str): Candidate name.

    Returns:
        bool: Whether the candidate is a full name.
    """
    return len(matched_str.split()) > 1


# (modified): Filter during detection so detect and mask use identical spans.
def keep_entity(entity: dict[str, object], gibberish) -> bool:
    """Return whether a StarPII entity survives upstream redaction filters.

    Args:
        entity (dict[str, object]): StarPII entity.
        gibberish: Loaded gibberish detector.

    Returns:
        bool: Whether the service should report the entity.
    """
    tag = str(entity["tag"])
    value = str(entity["value"])
    # (modified): StarPII confuses usernames with decorators and path values;
    # BigCode dropped them because of frequent false positives and negatives.
    # ref: https://arxiv.org/html/2305.06161v2#S4.SS3
    if tag in {"AMBIGUOUS", "USERNAME"} or not is_secret(value):
        return False
    if tag == "IP_ADDRESS":
        return not is_invalid_private_or_popular_dns_ip(value)
    if tag == "KEY":
        return is_key(value, gibberish)
    if tag == "NAME":
        return is_full_name(value)
    return True


# (modified): Omit upstream redaction functions here because the shared redactor handles findings from both detectors.
