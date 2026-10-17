import sys

import pytest
import torch

from pii_server.starpii import StarPIIDetector


@pytest.mark.skipif(sys.platform == "darwin", reason="macOS uses the Apple GPU.")
@pytest.mark.parametrize("dtype", ["auto", "float32", "bfloat16"])
def test_cpu_detector_detects_emails_across_overlapping_batches(dtype: str) -> None:
    detector = StarPIIDetector("cpu", dtype=dtype)
    assert {
        parameter.device.type for parameter in detector.pipeline.model.parameters()
    } == {"cpu"}
    assert detector.pipeline.model.dtype == (
        torch.bfloat16 if dtype == "bfloat16" else torch.float32
    )

    dummy_email = "placeholder@example.com"
    line = f"Email: {dummy_email}\r\n"
    contents = [line, line * 30]
    results = detector.detect(contents, window_size=64, window_overlap=16, batch_size=2)

    assert len(results) == len(contents)
    for content, entities in zip(contents, results):
        assert [
            {
                "tag": entity["tag"],
                "start": entity["start"],
                "end": entity["end"],
                "value": entity["value"],
            }
            for entity in entities
        ] == [
            {
                "tag": "EMAIL",
                "start": start,
                "end": start + len(dummy_email),
                "value": dummy_email,
            }
            for start in range(line.index(dummy_email), len(content), len(line))
        ]
        assert all(entity["score"] > 0.99 for entity in entities)
