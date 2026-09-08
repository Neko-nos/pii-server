"""Detect personal account names in the local home-directory path."""

import os
import re


def detect_home_name(
    content: str, home_directory: str | None = None
) -> list[dict[str, str | int]]:
    """Detect the final directory name within occurrences of the expanded home path.

    Args:
        content (str): Text to scan.
        home_directory (str | None): Expanded home path. When None, use the
            server's HOME environment variable.

    Returns:
        list[dict[str, str | int]]: NAME findings containing the final directory
            name of the expanded home path.
    """
    if home_directory is None:
        home_directory = os.environ["HOME"]
    prefix, separator, name = home_directory.rstrip("/").rpartition("/")
    # Shared OS, cloud-image, and automation accounts do not identify an individual.
    whitelist = {
        "admin",
        "administrator",
        "almalinux",
        "centos",
        "cloud-user",
        "debian",
        "ec2-user",
        "fedora",
        "guest",
        "jenkins",
        "opc",
        "rocky",
        "root",
        "runner",
        "ubuntu",
        "user",
        "vagrant",
    }
    if not name or name.casefold() in whitelist:
        return []

    # Capture source spans because escaped slashes shift name offsets.
    pattern = re.compile(
        re.escape(prefix + separator).replace("/", r"\\?/")
        + f"(?P<name>{re.escape(name)})"
    )
    matches = []
    for match in pattern.finditer(content):
        start, end = match.span("name")
        matches.append(
            {
                "tag": "NAME",
                "value": name,
                "start": start,
                "end": end,
            }
        )
    return matches
