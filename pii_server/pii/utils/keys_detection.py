"""Detect secret keys with detect-secrets."""

# ref: https://github.com/bigcode-project/bigcode-dataset/blob/bebec929edd826f19b5fa3538f22d18d5b50da4b/pii/utils/keys_detection.py

import tempfile

from detect_secrets import SecretsCollection
from detect_secrets.settings import transient_settings
from gibberish_detector.detector import Detector as GibberishDetector

# ref: https://github.com/bigcode-project/bigcode-dataset/blob/bebec929edd826f19b5fa3538f22d18d5b50da4b/pii/utils/keys_detection.py#L8
filters = [
    # some filters from [original list](https://github.com/Yelp/detect-secrets/blob/master/docs/filters.md#built-in-filters)
    # were removed based on their targets
    {"path": "detect_secrets.filters.heuristic.is_potential_uuid"},
    {"path": "detect_secrets.filters.heuristic.is_likely_id_string"},
    {"path": "detect_secrets.filters.heuristic.is_templated_secret"},
    {"path": "detect_secrets.filters.heuristic.is_sequential_string"},
]
plugins = [
    {"name": "ArtifactoryDetector"},
    {"name": "AWSKeyDetector"},
    # the entropy detectors esp Base64 need the gibberish detector on top
    {"name": "Base64HighEntropyString"},
    {"name": "HexHighEntropyString"},
    {"name": "AzureStorageKeyDetector"},
    {"name": "CloudantDetector"},
    {"name": "DiscordBotTokenDetector"},
    {"name": "GitHubTokenDetector"},
    {"name": "IbmCloudIamDetector"},
    {"name": "IbmCosHmacDetector"},
    {"name": "JwtTokenDetector"},
    {"name": "MailchimpDetector"},
    {"name": "NpmDetector"},
    {"name": "SendGridDetector"},
    {"name": "SlackDetector"},
    {"name": "SoftlayerDetector"},
    {"name": "StripeDetector"},
    {"name": "TwilioKeyDetector"},
]


# ref: https://github.com/bigcode-project/bigcode-dataset/blob/bebec929edd826f19b5fa3538f22d18d5b50da4b/pii/utils/keys_detection.py#L55
def is_hash(content: str, value: str) -> bool:
    """Return whether a candidate is a hash after gibberish detection.

    Args:
        content (str): Complete file content.
        value (str): Candidate secret.

    Returns:
        bool: Whether the candidate appears on a hash-related line.
    """
    value_index = content.index(value)
    # (modified): Slice the containing line directly so offset-zero candidates have empty context.
    line_start = content.rfind("\n", 0, value_index) + 1
    target_line = content[line_start:value_index]
    if len(value) in [32, 40, 64]:
        keywords = ["sha", "md5", "hash", "byte"]
        if any(x in target_line.lower() for x in keywords):
            return True
    return False


# ref: https://github.com/bigcode-project/bigcode-dataset/blob/bebec929edd826f19b5fa3538f22d18d5b50da4b/pii/utils/keys_detection.py#L73
def file_has_hashes(content: str, coeff: float = 0.02) -> bool:
    """Return whether hash-related literals exceed a file-density threshold.

    Args:
        content (str): Complete file content.
        coeff (float): Maximum allowed proportion of hash-related lines.

    Returns:
        bool: Whether ``hash`` or ``sha`` occurrences exceed the threshold.
    """
    count_sha = 0
    count_hash = 0
    nlines = content.count("\n")
    threshold = int(coeff * nlines)
    for line in content.splitlines():
        count_sha += line.lower().count("sha")
        count_hash += line.lower().count("hash")
        if count_sha > threshold or count_hash > threshold:
            return True
    return False


# ref: https://github.com/bigcode-project/bigcode-dataset/blob/bebec929edd826f19b5fa3538f22d18d5b50da4b/pii/utils/keys_detection.py#L87
def get_indexes(text: str, value: str) -> list[tuple[int, int]]:
    """Return every non-overlapping span of a value in text.

    Args:
        text (str): Text to search.
        value (str): Value whose spans should be returned.

    Returns:
        list[tuple[int, int]]: Start and end indices for every occurrence.
    """
    string = text
    indexes = []
    new_start = 0
    while True:
        try:
            start = string.index(value)
            indexes.append(new_start + start)
            new_start = new_start + start + len(value)
            string = text[new_start:]
        except ValueError:
            break
    return [(x, x + len(value)) for x in indexes]


# ref: https://github.com/bigcode-project/bigcode-dataset/blob/bebec929edd826f19b5fa3538f22d18d5b50da4b/pii/utils/keys_detection.py#L103
# (modified): Require callers to preserve the source suffix instead of defaulting to .txt.
def detect_keys(
    content: str,
    suffix: str,
    # (modified): Reuse the server's detector instead of loading its model for every content group.
    gibberish: GibberishDetector,
) -> list[dict[str, str | int]]:
    """Detect secret keys in content with detect-secrets.

    Args:
        content (str): Text to analyze.
        suffix (str): File suffix supplied to detect-secrets plugins.
        gibberish (GibberishDetector): Loaded gibberish detector.

    Returns:
        list[dict[str, str | int]]: Matches containing the tag, value, and start
            and end indices.
    """

    secrets = SecretsCollection()
    # (modified): Keep temporary-file cleanup active while detect-secrets scans it.
    with tempfile.NamedTemporaryFile(suffix=suffix, mode="w") as fp:
        fp.write(content)
        fp.flush()
        # (modified): Omit upstream's unused transient-settings binding.
        with transient_settings({"plugins_used": plugins, "filters_used": filters}):
            secrets.scan_file(fp.name)
    matches = []
    # (modified): Ignore transformed values that cannot produce a span in the source text.
    for secrets_set in secrets.data.values():
        content_has_hashes = file_has_hashes(content)
        for secret in secrets_set:
            if not gibberish.is_gibberish(secret.secret_value.lower()):
                continue
            indexes = get_indexes(content, secret.secret_value)
            if not indexes:
                continue
            if is_hash(content, secret.secret_value) or content_has_hashes:
                continue
            for start, end in indexes:
                matches.append(
                    {
                        "tag": "KEY",
                        "value": secret.secret_value,
                        "start": start,
                        "end": end,
                    }
                )
    return matches
