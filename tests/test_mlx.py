import sys

import pytest

from pii_server.starpii import StarPIIDetector


@pytest.mark.skipif(sys.platform != "darwin", reason="MLX requires macOS")
def test_mlx_model_inference_detects_dummy_email() -> None:
    dummy_email = "placeholder@example.com"
    content = f'const supportEmail = "{dummy_email}";'
    detector = StarPIIDetector(device="mps")

    ((entity,),) = detector.detect([content], batch_size=1)
    padded_results = detector.detect(
        [content, f"// Longer dummy input.\n{content}"], batch_size=2
    )

    assert {
        "tag": entity["tag"],
        "start": entity["start"],
        "end": entity["end"],
        "value": entity["value"],
    } == {
        "tag": "EMAIL",
        "start": content.index(dummy_email),
        "end": content.index(dummy_email) + len(dummy_email),
        "value": dummy_email,
    }
    assert entity["score"] > 0.99
    assert padded_results[0][0]["value"] == dummy_email
