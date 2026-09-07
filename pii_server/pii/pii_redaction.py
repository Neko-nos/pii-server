"""Apply upstream redaction exclusions and mask detected PII."""

import ipaddress

# ref: https://github.com/bigcode-project/bigcode-dataset/blob/bebec929edd826f19b5fa3538f22d18d5b50da4b/pii/pii_redaction.py#L19
POPULAR_DNS_SERVERS = [
    "8.8.8.8",
    "8.8.4.4",
    "1.1.1.1",
    "1.0.0.1",
    "76.76.19.19",
    "76.223.122.150",
    "9.9.9.9",
    "149.112.112.112",
    "208.67.222.222",
    "208.67.220.220",
    "8.26.56.26",
    "8.20.247.20",
    "94.140.14.14",
    "94.140.15.15",
]

# (modified): Omit random replacements and dataset helpers because masking uses type tokens.


# (modified): Share the same IP exclusions between regex and StarPII findings.
def is_invalid_private_or_popular_dns_ip(value: str | int) -> bool:
    """Return whether an IP address should not be reported.

    Args:
        value (str | int): Candidate IP address.

    Returns:
        bool: Whether the value is invalid, private, or a popular DNS server.
    """
    try:
        address = ipaddress.ip_address(value)
    except ValueError:
        return True
    return address.is_private or value in POPULAR_DNS_SERVERS


# ref: https://github.com/bigcode-project/bigcode-dataset/blob/bebec929edd826f19b5fa3538f22d18d5b50da4b/pii/pii_redaction.py#L110
# (modified): This service returns findings directly, so apply upstream's redaction filters during detection.
def remove_redaction_exclusions(
    secrets: list[dict[str, str | int]],
) -> list[dict[str, str | int]]:
    """Remove IP addresses that upstream excludes during redaction.

    Args:
        secrets (list[dict[str, str | int]]): Detected PII spans.

    Returns:
        list[dict[str, str | int]]: Spans eligible for service output.
    """
    return [
        secret
        for secret in secrets
        if secret["tag"] != "IP_ADDRESS"
        or not is_invalid_private_or_popular_dns_ip(secret["value"])
    ]


# ref: https://github.com/bigcode-project/bigcode-dataset/blob/bebec929edd826f19b5fa3538f22d18d5b50da4b/pii/ner/pii_redaction/utils.py#L102
# (modified): Consume the shared scan response, whose spans already passed redaction filters.
def redact_pii_text(text: str, findings: list[dict[str, object]]) -> str:
    """Mask detected spans using BigCode's replacement order and value cache.

    Args:
        text (str): Original text to mask.
        findings (list[dict[str, object]]): Filtered spans with start, end, and type.

    Returns:
        str: Text with detected spans replaced by ``<TYPE_MASK>`` tokens.
    """
    replaced_secrets = {}
    subparts = []
    step = 0
    for finding in sorted(findings, key=lambda item: int(item["start"])):
        start = int(finding["start"])
        end = int(finding["end"])
        # (modified): Skip contained findings that would move back into already masked text.
        if end < step:
            continue
        subtext = text[step:start]
        subparts.append(subtext if subtext else " ")
        value = text[start:end]
        if value in replaced_secrets:
            replacement = replaced_secrets[value]
        else:
            # (modified): Use the requested type tokens for all PII, including IPs.
            replacement = f"<{finding['type']}_MASK>"
            replaced_secrets[value] = replacement
        subparts.append(replacement)
        step = end
    subparts.append(text[step:])
    return "".join(subparts)
