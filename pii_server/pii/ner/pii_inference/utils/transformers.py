"""Transformers implementation of StarPII token classification."""

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from .pipeline import BasePiiNERPipeline


class TransformersPiiNERPipeline(BasePiiNERPipeline):
    """Run StarPII on the CPU using its original Transformers model."""

    def __init__(self, model_name_or_path: str, dtype: str = "auto") -> None:
        """Initialize the Transformers inference pipeline on the CPU.

        Args:
            model_name_or_path (str): Model identifier.
            dtype (str): Model precision; auto selects FP32.
        """
        self.dtype = "float32" if dtype == "auto" else dtype
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_name_or_path, dtype=self.dtype
        ).to("cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, add_prefix_space=True
        )
        self.num_workers = 0
        self.id_to_label = self.model.config.id2label

    @torch.inference_mode()
    def forward(self, model_inputs: dict[str, object]) -> dict[str, object]:
        """Evaluate one group of model windows on the CPU.

        Args:
            model_inputs (dict[str, object]): Collated model inputs and metadata.

        Returns:
            dict[str, object]: Token probabilities and metadata as CPU arrays.
        """
        logits = self.model(
            input_ids=model_inputs.pop("input_ids"),
            attention_mask=model_inputs.pop("attention_mask"),
        ).logits
        # Match the shared pipeline's offsets, which exclude boundary special tokens.
        logits = logits.softmax(dim=-1)[:, 1:-1]
        # NumPy cannot consume bfloat16 buffers directly.
        return {"logits": logits.float().numpy(), **model_inputs}
