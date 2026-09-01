from pathlib import Path

from gibberish_detector import detector

from pii_server.pii.pii_detection import scan_pii_batch


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
