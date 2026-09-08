from pathlib import Path

import pytest
from gibberish_detector import detector

from pii_server.pii.pii_detection import scan_pii_batch
from pii_server.pii.pii_redaction import redact_pii_text
from pii_server.pii.utils.usernames_detection import detect_home_name


def test_regex_detection_detects_dummy_email() -> None:
    dummy_email = "placeholder@example.com"
    content = f'const supportEmail = "{dummy_email}";'
    gibberish = detector.create_from_model(
        Path(__file__).parents[1]
        / "pii_server"
        / "pii"
        / "gibberish_data"
        / "big.model"
    )

    (findings,) = scan_pii_batch(
        {"content": [content], "suffix": [".js"]},
        gibberish,
        key_detector="regex",
    )

    assert findings == [
        {
            "tag": "EMAIL",
            "value": dummy_email,
            "start": content.index(dummy_email),
            "end": content.index(dummy_email) + len(dummy_email),
        }
    ]


@pytest.mark.parametrize("separator", ["/", r"\/"])
@pytest.mark.parametrize(
    "home_directory,name,prefix",
    [
        ("/Users/dummyuser", "dummyuser", "/Users/"),
        ("/home/dummyuser", "dummyuser", "/home/"),
        ("/Users/dummy.user+test", "dummy.user+test", "/Users/"),
        ("/home/xy", "xy", "/home/"),
        ("/home/dummyuser/", "dummyuser", "/home/"),
        ("/srv/dummyuser", "dummyuser", "/srv/"),
        ("/home/staff/dummyuser", "dummyuser", "/home/staff/"),
        ("/usr/home/dummyuser", "dummyuser", "/usr/home/"),
        ("/home/staff/dummyuser/", "dummyuser", "/home/staff/"),
        ("/dummyuser", "dummyuser", "/"),
    ],
)
def test_home_name_detection_masks_home_path_occurrences(
    home_directory: str, name: str, prefix: str, separator: str
) -> None:
    directory = home_directory.rstrip("/").replace("/", separator)
    prefix = prefix.replace("/", separator)
    content = (
        f'架空: "{directory}{separator}project"\r\n'
        f"{directory} {directory}:42\n"
        f"{directory}2 {directory}-other {directory}.txt {directory}_other\n"
        f"/backup{directory}/project file://{directory}/project $HOME/project {name}"
    )

    findings = detect_home_name(content, home_directory)

    assert len(findings) == 9
    for finding in findings:
        assert finding["tag"] == "NAME"
        assert finding["value"] == name
        assert content[finding["start"] : finding["end"]] == name
    assert redact_pii_text(
        content, [{"type": finding["tag"], **finding} for finding in findings]
    ) == (
        f'架空: "{prefix}<NAME_MASK>{separator}project"\r\n'
        f"{prefix}<NAME_MASK> {prefix}<NAME_MASK>:42\n"
        f"{prefix}<NAME_MASK>2 {prefix}<NAME_MASK>-other "
        f"{prefix}<NAME_MASK>.txt {prefix}<NAME_MASK>_other\n"
        f"/backup{prefix}<NAME_MASK>/project file://{prefix}<NAME_MASK>/project "
        f"$HOME/project {name}"
    )


def test_home_name_detection_masks_paths_with_mixed_slash_escaping() -> None:
    content = r'"/Users\/dummyuser\/project" "\/Users/dummyuser/project"'

    findings = detect_home_name(content, "/Users/dummyuser")

    assert (
        redact_pii_text(
            content, [{"type": finding["tag"], **finding} for finding in findings]
        )
        == r'"/Users\/<NAME_MASK>\/project" "\/Users/<NAME_MASK>/project"'
    )


@pytest.mark.parametrize("prefix", ["/", "/home/", "/Users/", "/usr/home/"])
@pytest.mark.parametrize(
    "username", ["ubuntu", "Ubuntu", "debian", "ec2-user", "root", "runner"]
)
def test_home_name_detection_excludes_whitelisted_accounts(
    prefix: str, username: str
) -> None:
    home_directory = f"{prefix}{username}"

    assert detect_home_name(f"{home_directory}/project", home_directory) == []


def test_home_name_detection_ignores_the_filesystem_root() -> None:
    assert detect_home_name("/project", "/") == []
