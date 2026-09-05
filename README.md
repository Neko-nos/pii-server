# pii-server

`pii-server` is a Git-hook service that detects personally identifiable
information and credentials in staged files before they are committed.

When it finds suspicious text, it shows the findings and asks whether they
contain real private data. Confirmed findings block the commit;
confirmed false positives allow it to continue.

## Installation

install the command via [uv](https://docs.astral.sh/uv/concepts/tools/):

```console
# macOS
uv tool install -e ".[mlx]"

# Other OS
uv tool install -e ".[vllm]"
```

Start the persistent detector before running the hook:

```console
pii_server init
```

The first initialization downloads and loads StarPII, so it can take several
minutes. Without `--device`, macOS uses the CPU and vLLM on other operating
systems automatically detects its device platform. To select a specific CUDA
device, pass its zero-based device number when initializing the server:

```console
pii_server init --device 0
```

On a supported Apple silicon Mac, select the Apple GPU:

```console
pii_server init --device mps
```

The first MLX initialization converts the original StarPII checkpoint into a
safetensors file under `~/.cache/mlx`. Later initializations load that
converted checkpoint, its configuration, and its tokenizer entirely from the
local cache. `XDG_CACHE_HOME` replaces `~/.cache` when configured.

Stop it and release its resources with:

```console
pii_server unload
```

## Git hook

Configure the installed command as a local pre-commit hook:

```yaml
repos:
  - repo: local
    hooks:
      - id: pii-server
        name: Detect PII and credentials in staged files
        entry: pii_server detect
        language: system
        types: [text]
        # batch inference
        require_serial: true
```

Pre-commit passes the staged filenames to `pii_server detect`; the command scans
only their added lines stored in the Git index.

## Detection pipeline

The PII model is [StarPII](https://huggingface.co/bigcode/starpii), so this
project matches its inference and post-processing pipeline to the pinned
[BigCode upstream pipeline](https://github.com/bigcode-project/bigcode-dataset/tree/bebec929edd826f19b5fa3538f22d18d5b50da4b/pii/ner).
The relevant upstream sources are the
[inference pipeline](https://github.com/bigcode-project/bigcode-dataset/blob/bebec929edd826f19b5fa3538f22d18d5b50da4b/pii/ner/pii_inference/utils/pipeline.py)
and the
[redaction-time filters](https://github.com/bigcode-project/bigcode-dataset/blob/bebec929edd826f19b5fa3538f22d18d5b50da4b/pii/ner/pii_redaction/utils.py).
Intentional adaptations for the persistent, single-user Git-hook service are
marked in the source with `# (modified): <why>` comments.
