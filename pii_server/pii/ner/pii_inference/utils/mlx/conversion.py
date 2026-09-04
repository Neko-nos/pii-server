"""Convert StarPII weights into a cached MLX checkpoint."""

import os
from pathlib import Path

import mlx.core as mx
from transformers import AutoModelForTokenClassification, AutoTokenizer, BertConfig


# ref: https://github.com/ml-explore/mlx-examples/blob/796f5b53cab69a3d48a44233ce21aae889e94a08/bert/convert.py#L7
def _convert_weight_name(name: str) -> str:
    """Map Hugging Face BERT parameter names to the official MLX layout.

    Args:
        name (str): Hugging Face parameter name.

    Returns:
        str: MLX parameter name.
    """
    for old, new in (
        (".layer.", ".layers."),
        (".self.key.", ".key_proj."),
        (".self.query.", ".query_proj."),
        (".self.value.", ".value_proj."),
        (".attention.output.dense.", ".attention.out_proj."),
        (".attention.output.LayerNorm.", ".ln1."),
        (".output.LayerNorm.", ".ln2."),
        (".intermediate.dense.", ".linear1."),
        (".output.dense.", ".linear2."),
        (".LayerNorm.", ".norm."),
    ):
        name = name.replace(old, new)
    return name


def convert(
    model_name_or_path: str,
    output_path: Path,
) -> None:
    """Convert a StarPII checkpoint to MLX safetensors.

    Args:
        model_name_or_path (str): Hugging Face model identifier or local directory.
        output_path (Path): Destination for the converted checkpoint.
    """
    model = AutoModelForTokenClassification.from_pretrained(model_name_or_path)
    weights = {
        _convert_weight_name(name): mx.array(value.numpy())
        for name, value in model.state_dict().items()
    }
    mx.save_safetensors(output_path, weights)


def prepare_mlx_checkpoint(
    model_name_or_path: str,
) -> Path:
    """Create or reuse a locally converted MLX checkpoint.

    Args:
        model_name_or_path (str): Hugging Face model identifier or local directory.

    Returns:
        Path: Cached MLX safetensors checkpoint.
    """
    cache_root = (
        Path(os.environ["XDG_CACHE_HOME"])
        if "XDG_CACHE_HOME" in os.environ
        else Path.home() / ".cache"
    )
    checkpoint_path = (
        cache_root / "mlx" / model_name_or_path.replace("/", "--") / "model.safetensors"
    )
    if checkpoint_path.exists():
        return checkpoint_path

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    config = BertConfig.from_pretrained(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    config.save_pretrained(checkpoint_path.parent)
    tokenizer.save_pretrained(checkpoint_path.parent)
    convert(model_name_or_path, checkpoint_path)
    return checkpoint_path
