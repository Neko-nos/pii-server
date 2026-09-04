import pytest
import torch
from datasets import Dataset

from pii_server.pii.ner.pii_inference.utils.pipeline import PiiNERPipeline


@pytest.mark.parametrize(
    "device",
    [
        pytest.param(torch.device("cpu"), id="cpu"),
        pytest.param(
            torch.device("cuda"),
            marks=pytest.mark.skipif(
                not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(),
                reason="a BF16-capable CUDA device is unavailable",
            ),
            id="cuda",
        ),
    ],
)
def test_model_inference_detects_dummy_email(device: torch.device) -> None:
    dummy_email = "placeholder@example.com"
    content = f'const supportEmail = "{dummy_email}";'
    dataset = Dataset.from_dict({"content": [content], "id": ["dummy"]})
    pipeline = PiiNERPipeline(
        "bigcode/starpii", device=device, batch_size=1, num_workers=0
    )

    (result,) = pipeline(dataset)
    (entity,) = result["entities"]

    assert result["content"] == content
    assert result["id"] == "dummy"
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
