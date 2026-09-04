"""vLLM implementation of StarPII token classification."""

import os

from transformers import AutoConfig, AutoTokenizer
from vllm import LLM

from ..pipeline import BasePiiNERPipeline


class VllmPiiNERPipeline(BasePiiNERPipeline):
    """Run StarPII through vLLM's token-classification pooler."""

    def __init__(self, model_name_or_path: str, device: int = -1) -> None:
        """Initialize the vLLM inference pipeline.

        Args:
            model_name_or_path (str): Model identifier.
            device (int): CUDA device number; a negative value uses vLLM's
                automatically selected platform.
        """
        if device >= 0:
            # vLLM selects a concrete accelerator through its visibility setting.
            os.environ["CUDA_VISIBLE_DEVICES"] = str(device)

        config = AutoConfig.from_pretrained(model_name_or_path)
        self.model = LLM(
            model=model_name_or_path,
            runner="pooling",
            pooler_config={"task": "token_classify", "use_activation": True},
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            # ref: https://github.com/bigcode-project/bigcode-dataset/blob/bebec929edd826f19b5fa3538f22d18d5b50da4b/pii/ner/pii_inference/utils/pipeline.py#L41
            model_name_or_path,
            add_prefix_space=True,
        )
        self.num_workers = 1
        self.id_to_label = config.id2label

    def forward(self, model_inputs: dict[str, object]) -> dict[str, object]:
        """Evaluate one group of model windows with vLLM.

        Args:
            model_inputs (dict[str, object]): Collated model inputs and metadata.

        Returns:
            dict[str, object]: Model probabilities and metadata as CPU arrays.
        """
        input_ids = model_inputs.pop("input_ids")
        attention_mask = model_inputs.pop("attention_mask")
        prompts = [
            {"prompt_token_ids": ids[mask.bool()].tolist()}
            for ids, mask in zip(input_ids, attention_mask)
        ]
        outputs = self.model.encode(
            prompts,
            pooling_task="token_classify",
            use_tqdm=False,
        )
        # Each model window has one special token at each boundary.
        logits = [output.outputs.data[1:-1].float().numpy() for output in outputs]
        return {"logits": logits, **model_inputs}
