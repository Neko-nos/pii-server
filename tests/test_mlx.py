import mlx.core as mx
import pytest
from datasets import Dataset

from pii_server.pii.ner.pii_inference.utils.mlx.backend import MlxPiiNERPipeline
from pii_server.pii.ner.pii_inference.utils.mlx.conversion import (
    prepare_mlx_checkpoint,
)


@pytest.fixture(scope="module", autouse=True)
def mlx_gpu() -> None:
    """Require and select the Apple GPU for MLX integration tests."""
    assert mx.metal.is_available(), "MLX inference requires the macOS GPU."
    mx.set_default_device(mx.gpu)


@pytest.mark.parametrize("dtype", ["bfloat16", "float32"])
def test_mlx_model_inference_detects_dummy_email(dtype: str) -> None:
    dummy_email = "placeholder@example.com"
    content = f'const supportEmail = "{dummy_email}";'
    dataset = Dataset.from_dict(
        {
            "content": [content, f"// Longer dummy input.\n{content}"],
            "id": ["short", "long"],
        }
    )
    checkpoint_path = prepare_mlx_checkpoint("bigcode/starpii", dtype=dtype)
    pipeline = MlxPiiNERPipeline(checkpoint_path)
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


def test_mlx_inference_detects_emails_across_overlapping_batches() -> None:
    dummy_email = "placeholder@example.com"
    line = f"Email: {dummy_email}\r\n"
    content = line * 100
    dataset = Dataset.from_dict({"content": [content], "id": ["long"]})
    checkpoint_path = prepare_mlx_checkpoint("bigcode/starpii")
    pipeline = MlxPiiNERPipeline(checkpoint_path)
    pipeline.batch_size = 2
    pipeline.window_size = 512
    pipeline.window_overlap = 64

    (result,) = pipeline(dataset)

    assert [
        {
            "tag": entity["tag"],
            "start": entity["start"],
            "end": entity["end"],
            "value": entity["value"],
        }
        for entity in result["entities"]
    ] == [
        {
            "tag": "EMAIL",
            "start": start,
            "end": start + len(dummy_email),
            "value": dummy_email,
        }
        for start in range(line.index(dummy_email), len(content), len(line))
    ]
