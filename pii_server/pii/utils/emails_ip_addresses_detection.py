"""Detect email addresses, IP addresses, and regex-shaped keys."""

# ref: https://github.com/bigcode-project/bigcode-dataset/blob/bebec929edd826f19b5fa3538f22d18d5b50da4b/pii/utils/emails_ip_addresses_detection.py

import ipaddress

import regex
from gibberish_detector.detector import Detector as GibberishDetector

# Regexes for PII detection
# (modified): Compile reusable regexes once and uppercase their constant names.

# ref: https://github.com/bigcode-project/bigcode-dataset/blob/bebec929edd826f19b5fa3538f22d18d5b50da4b/pii/utils/emails_ip_addresses_detection.py#L26
YEAR_PATTERNS: list[regex.Pattern] = [
    regex.compile(
        r"(?:^|[\b\s@?,!;:\'\")(.\p{Han}])([1-2][0-9]{3}[\p{Pd}/][1-2][0-9]{3})(?:$|[\s@,?!;:\'\"(.\p{Han}])"
    ),  # yyyy-yyyy or yyyy/yyyy
    regex.compile(
        r"(?:^|[\b\s@?,!;:\'\")(.\p{Han}])([1-2][0-9]{3}[\p{Pd}/.][0-3][0-9][\p{Pd}/.][0-3][0-9])(?:$|[\s@,?!;:\'\"(.\p{Han}])"
    ),  # yyyy-mm-dd or yyyy-dd-mm or yyyy/mm/dd or yyyy/dd/mm or yyyy.mm.dd or yyyy.dd.mm
    regex.compile(
        r"(?:^|[\b\s@?,!;:\'\")(.\p{Han}])([0-3][0-9][\p{Pd}/.][0-3][0-9][\p{Pd}/.](?:[0-9]{2}|[1-2][0-9]{3}))(?:$|[\s@,?!;:\'\"(.\p{Han}])"
    ),  # mm-dd-yyyy or dd-mm-yyyy or mm/dd/yyyy or dd/mm/yyyy or mm.dd.yyyy or dd.mm.yyyy or the same but with yy instead of yyyy
    regex.compile(
        r"(?:^|[\b\s@?,!;:\'\")(.\p{Han}])([0-3][0-9][\p{Pd}/](?:[0-9]{2}|[1-2][0-9]{3}))(?:$|[\s@,?!;:\'\"(.\p{Han}])"
    ),  # mm-yyyy or mm/yyyy or the same but with yy
    regex.compile(
        r"(?:^|[\b\s@?,!;:\'\")(.\p{Han}])([1-2][0-9]{3}-[0-3][0-9])(?:$|[\s@,?!;:\'\"(.\p{Han}])"
    ),  # yyyy-mm or yyyy/mm
]

# (modified): Omit upstream's unused original key pattern.
KEY_PATTERN: regex.Pattern = regex.compile(
    r"(?!(?:\/[^ .]+){2,})((?:(?:[A-Za-z]+[\p{Nd}\p{Pd}\/\+\=:_]+|[\p{Nd}\p{Pd}\/\+\=:]+[A-Za-z]+)){4,})(?:$|[\b\s\p{Han}@?,!;:\'\")(.])",
    flags=regex.MULTILINE,
)
IPV4_PATTERN: regex.Pattern = regex.compile(
    r"(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)(?:\.(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)){3}"
)
IPV6_PATTERN: regex.Pattern = regex.compile(
    r"(?:[0-9a-fA-F]{1,4}:){7,7}[0-9a-fA-F]{1,4}|(?:[0-9a-fA-F]{1,4}:){1,7}:|(?:[0-9a-fA-F]{1,4}:){1,6}:[0-9a-fA-F]{1,4}|(?:[0-9a-fA-F]{1,4}:){1,5}(?::[0-9a-fA-F]{1,4}){1,2}|(?:[0-9a-fA-F]{1,4}:){1,4}(?::[0-9a-fA-F]{1,4}){1,3}|(?:[0-9a-fA-F]{1,4}:){1,3}(?::[0-9a-fA-F]{1,4}){1,4}|(?:[0-9a-fA-F]{1,4}:){1,2}(?::[0-9a-fA-F]{1,4}){1,5}|[0-9a-fA-F]{1,4}:(?:(?::[0-9a-fA-F]{1,4}){1,6})|:(?:(?::[0-9a-fA-F]{1,4}){1,7}|:)|fe80:(?::[0-9a-fA-F]{0,4}){0,4}%[0-9a-zA-Z]{1,}|::(?:ffff(?::0{1,4}){0,1}:){0,1}(?:(?:25[0-5]|(?:2[0-4]|1{0,1}[0-9]){0,1}[0-9])\.){3,3}(?:25[0-5]|(?:2[0-4]|1{0,1}[0-9]){0,1}[0-9])|(?:[0-9a-fA-F]{1,4}:){1,4}:(?:(?:25[0-5]|(?:2[0-4]|1{0,1}[0-9]){0,1}[0-9])\.){3,3}(25[0-5]|(?:2[0-4]|1{0,1}[0-9]){0,1}[0-9])"
)
IP_PATTERN: regex.Pattern = regex.compile(
    r"(?:^|[\b\s@?,!;:\'\")(.\p{Han}])("
    + f"{IPV4_PATTERN.pattern}|{IPV6_PATTERN.pattern}"
    + ")(?:$|[\\s@,?!;:'\"(.\\p{Han}])",
    flags=regex.MULTILINE,
)

# Note: to reduce false positives, a number of technically-valid-but-rarely-used
# email address patterns (e.g. with parenthesis or slashes) will not match
EMAIL_PATTERN: regex.Pattern = regex.compile(
    r"""
    (?<= ^ | [[({<\b\s@,?!;'"\p{Han}¿¡:.] | \\['"] )  # left delimiter
    (
      (?:                                             # local part
        [^][(){}<>\b\s@,?!;'":#/\\=.\-]               # arbitrary character
        |
        (?: [=.\-] (?! [.@]) )                        # ".=-" not before ".@"
      )+
      @
      (?:
        (?:
             \w                                       # single-letter subdomain
           |
             [^.\b\s@?!;,/()>\-:]                     # subdomain (>=2 letter)
             [^.\b\s@?!;,/()>]{0,62}
             [^.\b\s@?!;,/()>\-:'"]
        )
        \.
      ){1,10}
      (?: [\p{L}\p{M}]{2,63} | xn-- \w+ )             # TLD, including IDN
    )
    (?= $ | [])}>\b\s@,?!;'"\p{Han}] | \\['"] | : (?! \d) | \. (?! \S))   # right delim
""",
    flags=regex.MULTILINE | regex.VERBOSE,
)


# ref: https://github.com/bigcode-project/bigcode-dataset/blob/bebec929edd826f19b5fa3538f22d18d5b50da4b/pii/utils/emails_ip_addresses_detection.py#L84
def get_regexes(
    high_risk_tags: set[str],
) -> dict[str, regex.Pattern]:
    """Return regexes for the requested PII tags.

    Args:
        high_risk_tags (set[str]): Tags to include.

    Returns:
        dict[str, regex.Pattern]: Compiled regexes keyed by PII tag.
    """

    # (modified): Keep only the three tags used by the service.
    patterns = {
        "EMAIL": EMAIL_PATTERN,
        "IP_ADDRESS": IP_PATTERN,
        "KEY": KEY_PATTERN,
    }
    return {tag: patterns[tag] for tag in high_risk_tags}


# ref: https://github.com/bigcode-project/bigcode-dataset/blob/bebec929edd826f19b5fa3538f22d18d5b50da4b/pii/utils/emails_ip_addresses_detection.py#L111
def ip_has_digit(matched_str: str) -> bool:
    """Return whether an IP-like PII span contains a digit.

    Args:
        matched_str (str): Candidate IP address.

    Returns:
        bool: Whether the candidate contains at least one digit.
    """
    return any(map(str.isdigit, matched_str))


# ref: https://github.com/bigcode-project/bigcode-dataset/blob/bebec929edd826f19b5fa3538f22d18d5b50da4b/pii/utils/emails_ip_addresses_detection.py#L117
def matches_date_pattern(matched_str: str) -> bool:
    """Return whether an IP-like span is a date false positive.

    Args:
        matched_str (str): Candidate IP address.

    Returns:
        bool: Whether the candidate matches a supported date pattern.
    """
    for year_regex in YEAR_PATTERNS:
        if year_regex.match(matched_str):
            return True
    return False


# ref: https://github.com/bigcode-project/bigcode-dataset/blob/bebec929edd826f19b5fa3538f22d18d5b50da4b/pii/utils/emails_ip_addresses_detection.py#L125
def filter_versions(matched_str: str, context: str) -> bool:
    """Return whether a short dotted value is probably a version.

    Args:
        matched_str (str): Candidate IP address.
        context (str): Neighboring text used to recognize DNS/server addresses.

    Returns:
        bool: Whether the candidate is probably a version number.
    """
    dot_count = matched_str.count(".")
    exclude = dot_count == 3 and len(matched_str) == 7
    if exclude and ("dns" in context.lower() or "server" in context.lower()):
        return False
    return exclude


# ref: https://github.com/bigcode-project/bigcode-dataset/blob/bebec929edd826f19b5fa3538f22d18d5b50da4b/pii/utils/emails_ip_addresses_detection.py#L137
def not_ip_address(matched_str: str) -> bool:
    """Return whether a candidate does not have a valid IP address format.

    For example, ``33.01.33.33`` is invalid because ``01`` has a leading zero.

    Args:
        matched_str (str): Candidate IP address.

    Returns:
        bool: Whether the candidate is invalid.
    """
    try:
        ipaddress.ip_address(matched_str)
        return False
    except ValueError:
        return True


# ref: https://github.com/bigcode-project/bigcode-dataset/blob/bebec929edd826f19b5fa3538f22d18d5b50da4b/pii/utils/emails_ip_addresses_detection.py#L157
def detect_email_addresses(
    content: str,
    tag_types: set[str],
    # (modified): Reuse the server's detector instead of loading its model for every content group.
    gibberish: GibberishDetector,
) -> list[dict[str, str | int]]:
    """Detect email addresses in a string using regex matching.

    Args:
        content (str): Text to analyze.
        tag_types (set[str]): Tags to detect.
        gibberish (GibberishDetector): Loaded gibberish detector.

    Returns:
        list[dict[str, str | int]]: Matches containing the tag, value, and start
            and end indices.
    """
    mst_regexes = get_regexes(tag_types)
    matches = []
    for tag in tag_types:
        label_pattern = mst_regexes[tag]
        # (modified): finditer yields only matches, and every selected regex requires a nonempty group 1.
        # (modified): Therefore, the upstream match and group guards are safe to omit.
        for match in label_pattern.finditer(content):
            value = match.group(1)
            start, end = match.span(1)
            if tag == "IP_ADDRESS":
                # Filter out false positive IPs
                if not ip_has_digit(value) or matches_date_pattern(value):
                    continue
                if filter_versions(
                    value, content[start - 100 : end + 100]
                ) or not_ip_address(value):
                    continue
            # (modified): Reuse the caller's loaded model instead of loading it for every key candidate.
            if tag == "KEY" and not gibberish.is_gibberish(value.lower()):
                continue
            matches.append(
                {
                    "tag": tag,
                    "value": value,
                    "start": start,
                    "end": end,
                }
            )
    return matches
