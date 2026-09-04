"""MLX implementation of StarPII's BERT token-classification model."""

from pathlib import Path

import mlx.core as mx
import numpy as np
from jaxtyping import Bool, Float, Int
from mlx import nn
from transformers import AutoTokenizer, BertConfig

from ..pipeline import BasePiiNERPipeline


class BertEncoderLayer(nn.Module):
    """Original BERT post-normalized encoder layer."""

    def __init__(self, config: BertConfig) -> None:
        """Initialize a BERT encoder layer.

        Args:
            config (BertConfig): BERT model configuration.
        """
        super().__init__()
        self.attention = nn.MultiHeadAttention(
            config.hidden_size, config.num_attention_heads, bias=True
        )
        self.ln1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.ln2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.linear1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.linear2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.gelu = nn.GELU()

    def __call__(
        self,
        hidden_states: Float[mx.array, "batch tokens hidden"],
        attention_mask: Bool[mx.array, "batch 1 1 tokens"] | None,
    ) -> Float[mx.array, "batch tokens hidden"]:
        """Transform hidden states with self-attention and a feed-forward layer.

        Args:
            hidden_states (Float[mx.array, "batch tokens hidden"]): Input states.
            attention_mask (Bool[mx.array, "batch 1 1 tokens"] | None): Tokens
                available to self-attention.

        Returns:
            Float[mx.array, "batch tokens hidden"]: Encoded hidden states.
        """
        attention_output = self.attention(
            hidden_states,
            hidden_states,
            hidden_states,
            attention_mask,
        )
        hidden_states = self.ln1(hidden_states + attention_output)
        output = self.linear2(self.gelu(self.linear1(hidden_states)))
        return self.ln2(hidden_states + output)


class BertEncoder(nn.Module):
    """Stack of BERT encoder layers."""

    def __init__(self, config: BertConfig) -> None:
        """Initialize the encoder stack.

        Args:
            config (BertConfig): BERT model configuration.
        """
        super().__init__()
        self.layers = [
            BertEncoderLayer(config) for _ in range(config.num_hidden_layers)
        ]

    def __call__(
        self, hidden_states: mx.array, attention_mask: mx.array | None
    ) -> mx.array:
        """Apply each encoder layer in sequence.

        Args:
            hidden_states (mx.array): Input hidden states.
            attention_mask (mx.array | None): Tokens available to self-attention.

        Returns:
            mx.array: Encoded hidden states.
        """
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        return hidden_states


class BertEmbeddings(nn.Module):
    """BERT word, position, and token-type embeddings."""

    def __init__(self, config: BertConfig) -> None:
        """Initialize the BERT embedding tables.

        Args:
            config (BertConfig): BERT model configuration.
        """
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size
        )
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def __call__(
        self, input_ids: Int[mx.array, "batch tokens"]
    ) -> Float[mx.array, "batch tokens hidden"]:
        """Embed token IDs and add their position and token-type embeddings.

        Args:
            input_ids (Int[mx.array, "batch tokens"]): Token IDs.

        Returns:
            Float[mx.array, "batch tokens hidden"]: Normalized embeddings.
        """
        position_ids = mx.broadcast_to(mx.arange(input_ids.shape[1]), input_ids.shape)
        # BigCode's StarPII inference supplies no segments, so BERT uses segment zero.
        token_type_ids = mx.zeros_like(input_ids)
        return self.norm(
            self.word_embeddings(input_ids)
            + self.position_embeddings(position_ids)
            + self.token_type_embeddings(token_type_ids)
        )


class BertModel(nn.Module):
    """BERT encoder without the unused pooling layer."""

    def __init__(self, config: BertConfig) -> None:
        """Initialize the embedding and encoder modules.

        Args:
            config (BertConfig): BERT model configuration.
        """
        super().__init__()
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)

    def __call__(
        self, input_ids: mx.array, attention_mask: mx.array | None
    ) -> mx.array:
        """Encode token IDs with an optional attention mask.

        Args:
            input_ids (mx.array): Token IDs.
            attention_mask (mx.array | None): Tokens available to self-attention.

        Returns:
            mx.array: Encoded token states.
        """
        hidden_states = self.embeddings(input_ids)
        if attention_mask is not None:
            attention_mask = attention_mask[:, None, None, :]
        return self.encoder(hidden_states, attention_mask)


class BertForTokenClassification(nn.Module):
    """BERT encoder with StarPII's token classifier."""

    def __init__(self, config: BertConfig) -> None:
        """Initialize BERT and its token-classification head.

        Args:
            config (BertConfig): BERT model configuration.
        """
        super().__init__()
        self.bert = BertModel(config)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def __call__(
        self, input_ids: mx.array, attention_mask: mx.array | None
    ) -> mx.array:
        """Compute classification logits for each token.

        Args:
            input_ids (mx.array): Token IDs.
            attention_mask (mx.array | None): Tokens available to self-attention.

        Returns:
            mx.array: Per-token classification logits.
        """
        return self.classifier(self.bert(input_ids, attention_mask))


class MlxPiiNERPipeline(BasePiiNERPipeline):
    """Run StarPII through the official MLX BERT structure."""

    def __init__(
        self,
        checkpoint_path: Path,
    ) -> None:
        """Initialize the MLX inference pipeline.

        Args:
            checkpoint_path (Path): Converted MLX safetensors checkpoint.
        """
        model_path = checkpoint_path.parent
        config = BertConfig.from_pretrained(model_path, local_files_only=True)
        model = BertForTokenClassification(config)
        model.load_weights(str(checkpoint_path))
        # Materialize lazy weights during initialization instead of the first request.
        mx.eval(model.parameters())
        self.model = model
        self.tokenizer = AutoTokenizer.from_pretrained(
            # ref: https://github.com/bigcode-project/bigcode-dataset/blob/bebec929edd826f19b5fa3538f22d18d5b50da4b/pii/ner/pii_inference/utils/pipeline.py#L41
            model_path,
            add_prefix_space=True,
            local_files_only=True,
        )
        self.num_workers = 0
        self.id_to_label = config.id2label

    def forward(self, model_inputs: dict[str, object]) -> dict[str, object]:
        """Evaluate one group of model windows with MLX.

        Args:
            model_inputs (dict[str, object]): Collated model inputs and metadata.

        Returns:
            dict[str, object]: Model outputs and metadata as CPU arrays.
        """
        input_ids = mx.asarray(model_inputs.pop("input_ids"))
        attention_mask = mx.asarray(model_inputs.pop("attention_mask"), dtype=mx.bool_)
        logits = self.model(input_ids, attention_mask)
        logits = mx.softmax(logits, axis=-1)
        # Each model window has one special token at each boundary.
        logits = logits[:, 1:-1]
        return {"logits": np.asarray(logits), **model_inputs}
