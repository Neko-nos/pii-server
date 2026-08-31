"""Run the upstream StarPII pipeline for collections of source files."""

from pathlib import Path

import torch
from datasets import Dataset
from gibberish_detector import detector

from pii_server.pii.ner.pii_inference.utils.pipeline import PiiNERPipeline
from pii_server.pii.ner.pii_redaction.utils import keep_entity


class StarPIIDetector:
    """Run StarPII inference and upstream redaction-time filters."""

    def __init__(self, device: int | str = -1) -> None:
        """Load StarPII on the configured device.

        Args:
            device (int | str): Torch device index or ``mps`` backend name.
        """
        # This service always uses StarPII rather than accepting an arbitrary model.
        # str -> mps, -1 -> cpu, other -> gpu
        pipeline_device = torch.device(device) if isinstance(device, str) else device
        self.pipeline = PiiNERPipeline("bigcode/starpii", device=pipeline_device)
        # Loading the packaged gibberish model once avoids repeated disk reads.
        self.gibberish = detector.create_from_model(
            Path(__file__).with_name("pii") / "gibberish_data" / "big.model"
        )

    def detect(
        self,
        contents: list[str],
        window_size: int = 512,
        window_overlap: int = 256,
        batch_size: int = 1,
    ) -> list[list[dict[str, object]]]:
        """Detect PII across a collection of source files.

        Args:
            contents (list[str]): Source-file contents.
            window_size (int): Tokens in each StarPII model window.
            window_overlap (int): Tokens shared by adjacent windows.
            batch_size (int): Model windows evaluated together.

        Returns:
            list[list[dict[str, object]]]: Findings grouped by source file.
        """
        self.pipeline.window_size = window_size
        self.pipeline.window_overlap = window_overlap
        self.pipeline.batch_size = batch_size
        dataset = Dataset.from_dict(
            {
                "content": contents,
                "id": [str(index) for index in range(len(contents))],
            }
        )
        return [
            [
                entity
                for entity in result["entities"]
                if keep_entity(entity, self.gibberish)
            ]
            for result in self.pipeline(dataset)
        ]
