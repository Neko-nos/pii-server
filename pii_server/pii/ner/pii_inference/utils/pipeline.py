"""BigCode's StarPII inference pipeline adapted for the Git-hook service."""

# ref: https://github.com/bigcode-project/bigcode-dataset/blob/bebec929edd826f19b5fa3538f22d18d5b50da4b/pii/ner/pii_inference/utils/pipeline.py

from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from datasets import Dataset
from jaxtyping import Float
from seqeval.metrics.sequence_labeling import get_entities
from torch.utils.data import DataLoader, IterableDataset
from transformers import (
    DataCollatorForTokenClassification,
    PreTrainedTokenizerBase,
)

from .chunking import chunk_inputs
from .postprocessing import postprocess


class BasePiiNERPipeline(ABC):
    """Share backend-independent StarPII inference processing."""

    # (modified): Omit unused call arguments because the service passes only a dataset.
    def __call__(self, inputs: Dataset) -> Iterator[dict[str, object]]:
        """Yield detected entities alongside each dataset entry.

        Args:
            inputs (Dataset): Source files with ``content`` and ``id`` columns.

        Yields:
            dict[str, object]: Original entry with its detected entities.
        """
        dataset_iterator = inputs.to_iterable_dataset()
        predict_iterator = self._predict_iterator(inputs, batch_size=self.batch_size)
        for entry, entities in zip(dataset_iterator, predict_iterator):
            yield dict(entities=entities, **entry)

    @abstractmethod
    def forward(self, model_inputs: dict[str, object]) -> dict[str, object]:
        """Evaluate one group of model windows with the selected backend.

        Args:
            model_inputs (dict[str, object]): Collated model inputs and metadata.

        Returns:
            dict[str, object]: Model outputs and metadata as CPU arrays.
        """

    # (modified): Keep only overlap averaging, the sole aggregation used by inference.
    @staticmethod
    def combine_chunks(
        chunks: list[Float[np.ndarray, "_chunk_tokens labels"]],
        offsets: list[int],
    ) -> Float[np.ndarray, "total_tokens labels"]:
        """Combine overlapping prediction windows for one source file.

        Args:
            chunks (list[Float[np.ndarray, "_chunk_tokens labels"]]): Predictions
                for individual windows, whose token lengths may differ.
            offsets (list[int]): Window start positions in the source token sequence.

        Returns:
            Float[np.ndarray, "total_tokens labels"]: Predictions aligned to the
                complete source token sequence.
        """
        total_length = np.max(offsets) + len(chunks[np.argmax(offsets)])
        total_shape = (total_length, np.shape(chunks[0])[-1])
        combined_chunks = np.zeros(total_shape, dtype=np.array(chunks[0]).dtype)
        for chunk, offset in zip(chunks, offsets):
            combined_chunks[offset : offset + len(chunk)] += chunk
        return combined_chunks / combined_chunks.sum(axis=-1, keepdims=True)

    @staticmethod
    def _get_pipeline_dataloader(
        dataset: Dataset,
        tokenizer: PreTrainedTokenizerBase,
        batch_size: int | None,
        num_workers: int,
        window_size: int | None = None,
        # (modified): Propagate the service's exact token-overlap setting.
        window_overlap: int = 0,
    ) -> DataLoader:
        """Build upstream's window iterator and model DataLoader.

        Args:
            dataset (Dataset): Source files to tokenize.
            tokenizer (PreTrainedTokenizerBase): Tokenizer matching the model.
            batch_size (int | None): Model windows evaluated together.
            num_workers (int): DataLoader worker count.
            window_size (int | None): Maximum tokens in each model window.
            window_overlap (int): Tokens shared by adjacent windows.

        Returns:
            DataLoader: Loader that yields collated model windows.
        """
        iterator = PipelineIterator(
            dataset,
            tokenizer,
            window_size=window_size,
            window_overlap=window_overlap,
        )
        loader = DataLoader(
            iterator,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=DataCollator(tokenizer),
        )
        return loader

    def _predict_iterator(
        self, inputs: Dataset, batch_size: int | None
    ) -> Iterator[list[dict[str, object]]]:
        """Yield entities for each source file in a dataset.

        Args:
            inputs (Dataset): Source files to evaluate.
            batch_size (int | None): Model windows evaluated together.

        Yields:
            list[dict[str, object]]: Entities detected in one source file.
        """
        loader = self._get_pipeline_dataloader(
            inputs,
            self.tokenizer,
            batch_size=batch_size,
            num_workers=self.num_workers,
            window_size=self.window_size,
            window_overlap=self.window_overlap,
        )
        processing_iterator = self.process_inputs(loader)
        for processed in self.combine_chunked_inputs(processing_iterator):
            yield self.extract_entities(
                text=processed["text"],
                logits=processed["logits"],
                offset_mapping=processed["offset_mapping"],
            )

    def combine_chunked_inputs(
        self, processing_iterator: Iterable[dict[str, object]]
    ) -> Iterator[dict[str, object]]:
        """Average overlapping windows belonging to each source file.

        Args:
            processing_iterator (Iterable[dict[str, object]]): Window predictions.

        Yields:
            dict[str, object]: Combined predictions for one source file.
        """
        for group in self.group_processed_chunks(processing_iterator):
            group["logits"] = self.combine_chunks(group["logits"], group["offset"])
            yield group

    @staticmethod
    def group_processed_chunks(
        processing_iterator: Iterable[dict[str, object]],
    ) -> Iterator[dict[str, object]]:
        """Collate adjacent windows that have the same source identifier.

        Args:
            processing_iterator (Iterable[dict[str, object]]): Window predictions.

        Yields:
            dict[str, object]: Collated windows and metadata for one source file.
        """
        for group in iterator_group_by(processing_iterator, column="id"):
            text, offset_mapping, identifier = (
                group[0]["text"],
                group[0]["offset_mapping"],
                group[0]["id"],
            )
            group = collate(group)
            group.update(id=identifier, text=text, offset_mapping=offset_mapping)
            yield group

    def process_inputs(self, loader: DataLoader) -> Iterator[dict[str, object]]:
        """Evaluate and ungroup each DataLoader result.

        Args:
            loader (DataLoader): Loader that yields collated model windows.

        Yields:
            dict[str, object]: Prediction and metadata for one model window.
        """
        for batch in loader:
            outputs = self.forward(batch)
            yield from uncollate(outputs)

    def extract_entities(
        self, text: str, logits, offset_mapping: list[tuple[int, int]]
    ) -> list[dict[str, object]]:
        """Convert token predictions into source spans.

        Args:
            text (str): Complete source text.
            logits (np.ndarray): Token label probabilities.
            offset_mapping (list[tuple[int, int]]): Token spans in ``text``.

        Returns:
            list[dict[str, object]]: Detected and post-processed entities.
        """

        def construct_entity(
            tag: str, start: int, end: int, score: float
        ) -> dict[str, object]:
            """Construct and post-process one detected entity.

            Args:
                tag (str): Predicted entity label.
                start (int): Entity start offset in the source text.
                end (int): Entity end offset in the source text.
                score (float): Mean token probability for the entity.

            Returns:
                dict[str, object]: Post-processed entity metadata.
            """
            entity = {
                "tag": tag,
                "start": start,
                "end": end,
                "value": text[start:end],
                "context": text[max(start - 50, 0) : min(end + 50, len(text))],
                "score": score,
            }
            entity = postprocess(entity)
            return entity

        logits = logits[: len(offset_mapping)]
        pred_labels = np.argmax(logits, axis=-1)
        pred_labels = [self.id_to_label[label] for label in pred_labels]
        label_prob = np.max(logits, axis=-1)
        pred_entities = get_entities([pred_labels])
        entities = [
            construct_entity(
                tag=tag,
                start=offset_mapping[start_idx][0],
                end=offset_mapping[end_idx][-1],
                score=np.mean(label_prob[start_idx : end_idx + 1]),
            )
            for tag, start_idx, end_idx in pred_entities
        ]
        return entities


@dataclass
class DataCollator(DataCollatorForTokenClassification):
    """Collate tensors while retaining source metadata."""

    # (modified): Retain only metadata consumed by grouping and post-processing.
    _dont_touch: ClassVar[list[str]] = [
        "text",
        "offset",
        "id",
        "offset_mapping",
    ]

    def torch_call(self, features: list[dict[str, object]]) -> dict[str, object]:
        """Collate model inputs and preserve metadata lists.

        Args:
            features (list[dict[str, object]]): Model windows and source metadata.

        Returns:
            dict[str, object]: Collated tensors and metadata lists.
        """
        keys = [key for key in features[0] if key in self._dont_touch]
        batch = {key: [feature.pop(key) for feature in features] for key in keys}
        batch.update(super().torch_call(features))
        return batch


class PipelineIterator(IterableDataset):
    """Yield model windows from every source file in a dataset."""

    def __init__(
        self,
        dataset: Dataset,
        tokenizer: PreTrainedTokenizerBase,
        text_column: str = "content",
        window_size: int = 512,
        # (modified): Pipeline windows use the service's exact token-overlap setting.
        window_overlap: int = 0,
    ) -> None:
        """Initialize the iterable model-window dataset.

        Args:
            dataset (Dataset): Source files to tokenize.
            tokenizer (PreTrainedTokenizerBase): Tokenizer matching the model.
            text_column (str): Dataset column containing source text.
            window_size (int): Maximum tokens in each model window.
            window_overlap (int): Tokens shared by adjacent windows.
        """
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.text_column = text_column
        self.window_overlap = window_overlap
        self.window_size = window_size

    def __len__(self) -> int:
        """Return the number of source files.

        Returns:
            int: Number of source files in the dataset.
        """
        return len(self.dataset)

    def __iter__(self) -> Iterator[dict[str, object]]:
        """Tokenize source files and yield their model windows.

        Yields:
            dict[str, object]: Model window and its source metadata.
        """
        iterator = self.dataset.to_iterable_dataset()
        iterator = iterator.map(
            lambda entry: self.tokenizer.encode_plus(
                entry[self.text_column],
                return_offsets_mapping=True,
                add_special_tokens=False,
            )
        )
        for item in iterator:
            # (modified): Pass the exact overlap rather than upstream's boolean-derived frequency.
            for chunk in chunk_inputs(
                **item,
                tokenizer=self.tokenizer,
                max_length=self.window_size,
                overlap=self.window_overlap,
            ):
                yield dict(
                    **chunk,
                    text=item[self.text_column],
                    offset_mapping=item["offset_mapping"],
                )


def uncollate(inputs: dict[str, object]) -> list[dict[str, object]]:
    """Split grouped model outputs into individual window results.

    Args:
        inputs (dict[str, object]): Model outputs grouped by field.

    Returns:
        list[dict[str, object]]: Model outputs grouped by window.

    Raises:
        AssertionError: If grouped fields have different lengths.
    """
    keys = list(inputs)
    assert all(len(inputs[key]) == len(inputs[keys[0]]) for key in keys), (
        "All entries must be same length. Inputs lengths: "
        + ", ".join(f"{key}: {len(inputs[key])}" for key in keys)
    )
    return [dict(zip(keys, values)) for values in zip(*[inputs[key] for key in keys])]


def collate(inputs: list[dict[str, object]]) -> dict[str, list[object]]:
    """Group window results by key.

    Args:
        inputs (list[dict[str, object]]): Individual window results.

    Returns:
        dict[str, list[object]]: Window results grouped by field.
    """
    keys = inputs[0].keys()
    return {key: [entry[key] for entry in inputs] for key in keys}


def iterator_group_by(
    iterator: Iterable[dict[str, object]], column: str
) -> Iterator[list[dict[str, object]]]:
    """Group adjacent window results by source identifier.

    Args:
        iterator (Iterable[dict[str, object]]): Ordered window results.
        column (str): Field containing the grouping identifier.

    Yields:
        list[dict[str, object]]: Adjacent results with the same identifier.
    """
    current_identifier = None
    grouped_items = []
    for item in iterator:
        if item[column] != current_identifier and current_identifier is not None:
            yield grouped_items
            grouped_items = []
        current_identifier = item[column]
        grouped_items.append(item)
    if grouped_items:
        yield grouped_items
