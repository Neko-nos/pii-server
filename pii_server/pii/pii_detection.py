from gibberish_detector.detector import Detector as GibberishDetector

# (modified): Use package imports because the service imports the upstream tree as a package.
from pii_server.pii.utils.emails_ip_addresses_detection import detect_email_addresses
from pii_server.pii.utils.keys_detection import detect_keys
from pii_server.pii.utils.usernames_detection import detect_home_name

# ref: https://github.com/bigcode-project/bigcode-dataset/blob/bebec929edd826f19b5fa3538f22d18d5b50da4b/pii/pii_detection.py#L7
# (modified): Omit postprocess_secrets because the server consumes findings directly.


# ref: https://github.com/bigcode-project/bigcode-dataset/blob/bebec929edd826f19b5fa3538f22d18d5b50da4b/pii/pii_detection.py#L18
def scan_pii_batch(
    examples: dict[str, list[str]],
    # (modified): Reuse the server's loaded gibberish model across all staged contents.
    gibberish: GibberishDetector,
    # (modified): Name upstream's vague "other" option after the detect-secrets implementation it invokes.
    key_detector: str = "detect-secrets",
) -> list[list[dict[str, str | int]]]:
    """Scan source contents for PII.

    Args:
        examples (dict[str, list[str]]): Parallel content and suffix lists.
        gibberish (GibberishDetector): Loaded gibberish detector.
        key_detector (str): ``detect-secrets`` or ``regex`` credential detector.

    Returns:
        list[list[dict[str, str | int]]]: Findings grouped by input text.

    Raises:
        ValueError: If ``key_detector`` is not ``detect-secrets`` or ``regex``.
    """
    # (modified): Validate once instead of silently routing every unknown name to detect-secrets.
    if key_detector not in {"detect-secrets", "regex"}:
        raise ValueError(f"Unsupported key detector: {key_detector}")

    findings = []
    # (modified): Preserve each staged file's suffix for detect-secrets plugins.
    for text, suffix in zip(examples["content"], examples["suffix"]):
        if key_detector == "regex":
            # use a regex to detect keys + emails + ips
            secrets = detect_email_addresses(
                text,
                tag_types={"KEY", "EMAIL", "IP_ADDRESS"},
                gibberish=gibberish,
            )
        else:
            # detect emails and ip addresses with regexes
            secrets = detect_email_addresses(
                text,
                tag_types={"EMAIL", "IP_ADDRESS"},
                gibberish=gibberish,
            )
            # for keys use detect-secrets tool
            secrets.extend(detect_keys(text, suffix, gibberish))
        # (modified): Retain known local account names that StarPII's filters discard.
        secrets.extend(detect_home_name(text))
        findings.append(secrets)
    # (modified): Return findings directly instead of upstream's dataset columns.
    return findings
