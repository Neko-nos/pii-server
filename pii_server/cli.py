import argparse
import io
import os
import subprocess
import sys
import time
from pathlib import Path

from pii_server.utils import parse_device, request, runtime_dir, server_running


def initialize(device: int | str = -1) -> int:
    """Start the persistent server and load StarPII.

    Args:
        device (int | str): Torch device index or ``mps`` backend name.

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
    environment = os.environ.copy()
    if device == "mps":
        # PyTorch reads MPS backend switches while the child process initializes.
        environment["PYTORCH_MPS_PREFER_METAL"] = "1"
    with (directory / "service.log").open("a") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "pii_server.server",
                "--device",
                str(device),
            ],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
            env=environment,
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


def detect(
    filenames: list[str],
    key_detector: str,
    window_size: int,
    window_overlap: int,
    batch_size: int,
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
    findings = request(
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


def main() -> int:
    """Run the requested PII server command.

    Returns:
        int: Process exit status.
    """
    parser = argparse.ArgumentParser(
        description="Scan staged Git content for PII and credentials."
    )
    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
        title="commands",
    )

    init_parser = subparsers.add_parser(
        "init",
        help="start the server and load StarPII",
        description="Start the persistent server and load StarPII.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    init_parser.add_argument(
        "--device",
        type=parse_device,
        default=-1,
        help="Torch device: -1 selects CPU, an integer selects CUDA, and mps selects Apple GPU",
    )
    detect_parser = subparsers.add_parser(
        "detect",
        help="scan staged file contents",
        description="Scan the staged contents of files passed by pre-commit.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    detect_parser.add_argument(
        "--key-detector",
        choices=("detect-secrets", "regex"),
        default="detect-secrets",
        help=(
            "credential detector: detect-secrets uses format-specific plugins and "
            "filters; regex uses BigCode's broad key pattern and gibberish filter"
        ),
    )
    detect_parser.add_argument(
        "--window-size",
        type=int,
        default=512,
        help="tokens in each StarPII input window",
    )
    detect_parser.add_argument(
        "--window-overlap",
        type=int,
        default=0,
        help="tokens shared by adjacent StarPII input windows",
    )
    detect_parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="StarPII windows evaluated together",
    )
    detect_parser.add_argument(
        "filenames",
        nargs="+",
        help="repository-relative staged files to scan",
    )

    subparsers.add_parser(
        "unload",
        help="stop the server and release model memory",
        description="Stop the persistent server and release model memory.",
    )

    args = parser.parse_args()
    if args.command == "init":
        return initialize(args.device)
    if args.command == "detect":
        return detect(
            args.filenames,
            args.key_detector,
            args.window_size,
            args.window_overlap,
            args.batch_size,
        )
    return unload()
