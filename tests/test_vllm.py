import sys

import pytest
from datasets import Dataset

if sys.platform != "darwin":
    from pii_server.pii.ner.pii_inference.utils.vllm.backend import (
        VllmPiiNERPipeline,
    )


@pytest.mark.skipif(sys.platform == "darwin", reason="vLLM is not used on macOS")
def test_vllm_model_inference_detects_dummy_email() -> None:
    dummy_email = "placeholder@example.com"
    content = f'const supportEmail = "{dummy_email}";'
    dataset = Dataset.from_dict(
        {
            "content": [content, f"// Longer dummy input.\n{content}"],
            "id": ["short", "long"],
        }
    )
    pipeline = VllmPiiNERPipeline("bigcode/starpii")
    pipeline.batch_size = 2
    pipeline.window_size = 512
    pipeline.window_overlap = 0

    first_result, second_result = pipeline(dataset)
    (entity,) = first_result["entities"]

    assert first_result["content"] == content
    assert first_result["id"] == "short"
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
    assert second_result["entities"][0]["value"] == dummy_email
