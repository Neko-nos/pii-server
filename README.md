# pii-server

Detect PII and credentials in staged Git changes, or mask them in text files.

## Installation

Install with [uv](https://docs.astral.sh/uv/concepts/tools/):

```console
# macOS (Apple silicon)
uv tool install -e ".[mlx]"

# Other OS
uv tool install -e ".[vllm]"
```

Start the detector before using `detect` or `mask`. The first run may take
several minutes:

```console
pii_server init
```

To select a specific CUDA device, pass its zero-based device number
when initializing the server:

```console
pii_server init --device 0
```

## Git hook

Add this to `.pre-commit-config.yaml`:

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

The hook scans added staged lines and asks about each finding. Answer `n` for
a false positive; otherwise, the commit is blocked.

## Masking a text file

Pass a UTF-8 text file with `--file` (`-f`):

```console
# Print the result and copy it to the clipboard
pii_server mask -f input.txt

# Overwrite the input file (--in-place can also be written as -i)
pii_server mask --file input.txt --in-place
```

Detected PII becomes tokens such as `<NAME_MASK>` and `<EMAIL_MASK>`.

## Stop the detector

```console
pii_server unload
```

Run `pii_server <command> --help` to see all options for a command.
