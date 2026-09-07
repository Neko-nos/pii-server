import pytest

from pii_server.pii.pii_redaction import redact_pii_text


def test_redaction_preserves_unicode_and_line_endings() -> None:
    text = "架空: DUMMY_NAME\r\nplaceholder@example.com\tDUMMY_PASSWORD"
    findings = [
        {
            "start": text.index(value),
            "end": text.index(value) + len(value),
            "type": tag,
        }
        for value, tag in [
            ("DUMMY_PASSWORD", "PASSWORD"),
            ("DUMMY_NAME", "NAME"),
            ("placeholder@example.com", "EMAIL"),
        ]
    ]

    assert redact_pii_text(text, findings) == (
        "架空: <NAME_MASK>\r\n<EMAIL_MASK>\t<PASSWORD_MASK>"
    )


@pytest.mark.parametrize("tag", ["NAME", "EMAIL", "KEY", "PASSWORD", "IP_ADDRESS"])
def test_redaction_masks_repeated_adjacent_spans(tag: str) -> None:
    findings = [
        {"start": 0, "end": 7, "type": tag},
        {"start": 7, "end": 14, "type": tag},
    ]

    assert redact_pii_text("<dummy><dummy>", findings) == f" <{tag}_MASK> <{tag}_MASK>"


def test_redaction_skips_nested_findings() -> None:
    findings = [
        {"start": 7, "end": 12, "type": "NAME"},
        {"start": 9, "end": 10, "type": "KEY"},
        {"start": 10, "end": 17, "type": "PASSWORD"},
        {"start": 7, "end": 12, "type": "NAME"},
    ]

    assert redact_pii_text("prefix abcdefghij suffix", findings) == (
        "prefix <NAME_MASK> <NAME_MASK> <PASSWORD_MASK> suffix"
    )


@pytest.mark.parametrize("tags", [("NAME", "PASSWORD"), ("PASSWORD", "NAME")])
def test_redaction_reuses_first_replacement_for_conflicting_types(
    tags: tuple[str, str],
) -> None:
    text = "Value: Avery Placeholder"
    findings = [
        {"start": text.index("Avery"), "end": len(text), "type": tag} for tag in tags
    ]

    assert redact_pii_text(text, findings) == (
        f"Value: <{tags[0]}_MASK> <{tags[0]}_MASK>"
    )


@pytest.mark.parametrize("text", ["", "\r\n\t  ", "架空のサンプル\r\nplain text"])
def test_redaction_preserves_text_without_findings(text: str) -> None:
    assert redact_pii_text(text, []) == text
