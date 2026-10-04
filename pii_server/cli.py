import io
import subprocess
import sys
import time
from pathlib import Path
from typing import Annotated, Literal

import pyperclip
import typer

from pii_server.pii.pii_redaction import redact_pii_text
from pii_server.utils import parse_device, request, runtime_dir, server_running


def initialize(
    device: Annotated[
        str,
        typer.Option(
            "-d",
            "--device",
            parser=lambda value: str(parse_device(value)),
            metavar="DEVICE",
            help="Accelerator: -1 for automatic selection, a CUDA index, or mps.",
        ),
    ] = "-1",
    dtype: Annotated[
        Literal["auto", "bfloat16", "float32"],
        typer.Option(
            "-t",
            "--dtype",
            help="Model precision; auto selects BF16 on GPU and FP32 on CPU.",
        ),
    ] = "auto",
) -> int:
    """Start the persistent server and load StarPII.

    Args:
        device (str): Accelerator selection.
        dtype (str): Model precision; auto selects BF16 on GPU and FP32 on CPU.

    Returns:
        int: Zero after the server accepts requests.

    Raises:
        RuntimeError: The server does not start successfully.
    """
    if server_running():
        print("PII detector is already initialized.")
        return 0

    directory = runtime_dir()
    # The server log and socket concern text that may contain private data.
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    directory.chmod(0o700)
    print("Starting and loading the PII detector...", file=sys.stderr)
    with (directory / "service.log").open("a") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "pii_server.server",
                "--device",
                str(device),
                "--dtype",
                dtype,
            ],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )

    # The initial model download can take several minutes.
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        if server_running():
            print("PII detector initialized.")
            return 0
        if process.poll() is not None:
            break
        # Frequent polling avoids adding noticeable delay after a cached model loads.
        time.sleep(0.1)
    raise RuntimeError(f"PII detector failed to start; see {directory / 'service.log'}")


def staged_additions(filename: str) -> list[str]:
    """Read added contents from a staged Git diff.

    Args:
        filename (str): Repository-relative filename supplied by pre-commit.

    Returns:
        list[str]: Contents of each contiguous group of added lines.

    Raises:
        RuntimeError: Git cannot read the staged diff.
    """
    result = subprocess.run(
        [
            "git",
            "diff",
            "--cached",
            "--unified=0",
            "--no-color",
            "--no-ext-diff",
            "--",
            filename,
        ],
        capture_output=True,
        check=False,
        encoding="utf-8",
        errors="replace",
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip())

    additions = []
    current_range = []
    in_hunk = False
    for line in result.stdout.splitlines():
        if line.startswith("@@"):
            if current_range:
                additions.append("\n".join(current_range))
                current_range = []
            in_hunk = True
        elif in_hunk and line.startswith("+"):
            current_range.append(line[1:])
        elif current_range:
            additions.append("\n".join(current_range))
            current_range = []
    if current_range:
        additions.append("\n".join(current_range))
    # Empty added ranges cannot contain private text and produce no model tokens.
    return [addition for addition in additions if addition.strip()]


def terminal_answer(
    terminal_input: io.TextIOBase, terminal_output: io.TextIOBase
) -> str:
    """Ask whether detected text contains real private data.

    Args:
        terminal_input (io.TextIOBase): Controlling terminal input.
        terminal_output (io.TextIOBase): Controlling terminal output.

    Returns:
        str: Normalized user response.
    """
    prompt = "Does the suspicious text contain actual PII or credentials? [Y/n]: "
    terminal_output.write(prompt)
    terminal_output.flush()
    return terminal_input.readline().strip().lower()


def scan(
    files: list[dict[str, str]],
    key_detector: str,
    window_size: int,
    window_overlap: int,
    batch_size: int,
) -> list[dict[str, object]]:
    """Scan text with the initialized server for detection or masking.

    Args:
        files (list[dict[str, str]]): Filenames and text to scan.
        key_detector (str): ``detect-secrets`` or ``regex`` credential detector.
        window_size (int): Tokens in each StarPII input window.
        window_overlap (int): Tokens shared by adjacent StarPII input windows.
        batch_size (int): StarPII windows evaluated together.

    Returns:
        list[dict[str, object]]: Filtered PII spans returned by both detectors.
    """
    return request(
        {
            "operation": "scan",
            "files": files,
            "key_detector": key_detector,
            "window_size": window_size,
            "window_overlap": window_overlap,
            "batch_size": batch_size,
        },
        600,
    )["findings"]


def detect(
    filenames: Annotated[
        list[str],
        typer.Argument(help="Repository-relative staged files to scan.", path_type=str),
    ],
    key_detector: Annotated[
        Literal["detect-secrets", "regex"],
        typer.Option("-k", "--key-detector", help="Credential detector."),
    ] = "detect-secrets",
    window_size: Annotated[
        int, typer.Option("-w", "--window-size", help="Tokens in each StarPII window.")
    ] = 512,
    window_overlap: Annotated[
        int,
        typer.Option(
            "-o", "--window-overlap", help="Tokens shared by adjacent windows."
        ),
    ] = 0,
    batch_size: Annotated[
        int,
        typer.Option("-b", "--batch-size", help="StarPII windows evaluated together."),
    ] = 1,
) -> int:
    """Scan staged files with the initialized server.

    Args:
        filenames (list[str]): Repository-relative staged filenames.
        key_detector (str): ``detect-secrets`` or ``regex`` credential detector.
        window_size (int): Tokens in each StarPII input window.
        window_overlap (int): Tokens shared by adjacent StarPII input windows.
        batch_size (int): StarPII windows evaluated together.

    Returns:
        int: One when real private data is reported, otherwise zero.
    """
    files = [
        {"filename": filename, "text": addition}
        for filename in filenames
        for addition in staged_additions(filename)
    ]
    # Deletion-only changes contain no added text to scan.
    if not files:
        return 0

    # Send all added contents together so the pipeline processes one collection.
    findings = scan(files, key_detector, window_size, window_overlap, batch_size)
    if not findings:
        return 0

    # Separate streams avoid the seekable buffering that r+ requires but terminals do not support.
    with (
        Path("/dev/tty").open("r", encoding="utf-8") as terminal_input,
        Path("/dev/tty").open("w", encoding="utf-8") as terminal_output,
    ):
        # Pre-commit leaves its progress message open while the hook is running.
        print(file=terminal_output)
        print("Suspicious staged text found", file=terminal_output)

        for finding in findings:
            text = finding["text"].replace("\n", "\\n")
            print(
                f"{finding['filename']}: {finding['type']}: {text}",
                file=terminal_output,
            )
            if terminal_answer(terminal_input, terminal_output) not in {"n", "no"}:
                print("Commit blocked.", file=terminal_output)
                return 1

        print(
            "All findings confirmed false positives; continuing commit.",
            file=terminal_output,
        )
        return 0


def mask(
    filename: Annotated[
        Path, typer.Option("-f", "--file", help="Text file to mask in full.")
    ],
    in_place: Annotated[
        bool, typer.Option("-i", "--in-place", help="Overwrite the input file.")
    ] = False,
    key_detector: Annotated[
        Literal["detect-secrets", "regex"],
        typer.Option("-k", "--key-detector", help="Credential detector."),
    ] = "detect-secrets",
    window_size: Annotated[
        int, typer.Option("-w", "--window-size", help="Tokens in each StarPII window.")
    ] = 512,
    window_overlap: Annotated[
        int,
        typer.Option(
            "-o", "--window-overlap", help="Tokens shared by adjacent windows."
        ),
    ] = 0,
    batch_size: Annotated[
        int,
        typer.Option("-b", "--batch-size", help="StarPII windows evaluated together."),
    ] = 1,
) -> int:
    """Replace detected PII in a text file with type-specific mask tokens.

    Args:
        filename (Path): UTF-8 text file to mask in full.
        in_place (bool): Overwrite the input file when true; otherwise write to
            stdout and copy to the clipboard.
        key_detector (str): ``detect-secrets`` or ``regex`` credential detector.
        window_size (int): Tokens in each StarPII input window.
        window_overlap (int): Tokens shared by adjacent StarPII input windows.
        batch_size (int): StarPII windows evaluated together.

    Returns:
        int: Zero after writing the masked text.
    """
    # Avoid newline conversion because it changes otherwise untouched text.
    with filename.open(encoding="utf-8", newline="") as source:
        text = source.read()
    findings = scan(
        [{"filename": str(filename), "text": text}],
        key_detector,
        window_size,
        window_overlap,
        batch_size,
    )
    masked_text = redact_pii_text(text, findings)
    if in_place:
        with filename.open("w", encoding="utf-8", newline="") as destination:
            destination.write(masked_text)
    else:
        sys.stdout.write(masked_text)
        pyperclip.copy(masked_text)
    return 0


def unload() -> int:
    """Stop the persistent PII server.

    Returns:
        int: Zero after the server stops or when it was not running.
    """
    if not server_running():
        print("PII detector is not loaded.")
        return 0
    # A local server should stop promptly when it is not processing another request.
    request({"operation": "unload"}, 10)
    print("PII detector unloaded.")
    return 0


def main() -> None:
    """Run the requested PII server command."""
    app = typer.Typer(
        help="Detect PII in staged Git content or mask PII in text files.",
        context_settings={"help_option_names": ["-h", "--help"]},
        # Pre-commit needs the exit status returned by detect.
        result_callback=sys.exit,
    )
    app.command("init", help="Start the server and load StarPII.")(initialize)
    app.command(help="Scan staged file contents.")(detect)
    app.command(help="Mask PII in a text file and copy the result to the clipboard.")(
        mask
    )
    app.command(help="Stop the server and release model memory.")(unload)
    app()
