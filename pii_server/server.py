import argparse
import json
import socket
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from pii_server.pii.pii_detection import scan_pii_batch
from pii_server.pii.pii_redaction import remove_redaction_exclusions
from pii_server.starpii import StarPIIDetector
from pii_server.utils import parse_device, runtime_dir


class Detector:
    """Combine BigCode's pattern and credential pipeline with StarPII."""

    def __init__(self, device: int | str = -1) -> None:
        """Initialize both detection pipelines.

        Args:
            device (int | str): Accelerator selection.
        """
        self.starpii = StarPIIDetector(device)

    def scan(
        self,
        files: list[dict[str, str]],
        key_detector: str,
        window_size: int,
        window_overlap: int,
        batch_size: int,
    ) -> list[dict[str, object]]:
        """Detect private data with both BigCode pipelines.

        Args:
            files (list[dict[str, str]]): Source filenames and staged contents.
            key_detector (str): ``detect-secrets`` or ``regex`` credential detector.
            window_size (int): Tokens in each StarPII input window.
            window_overlap (int): Tokens shared by adjacent StarPII input windows.
            batch_size (int): StarPII windows evaluated together.

        Returns:
            list[dict[str, object]]: Merged PII spans.
        """
        contents = [file["text"] for file in files]
        suffixes = [Path(file["filename"]).suffix for file in files]

        def scan_rule_based() -> list[list[dict[str, str | int]]]:
            """Run rule-based detection.

            Returns:
                list[list[dict[str, str | int]]]: Findings grouped by input text.
            """
            # Pass the full staged-file collection to the PII scanner once.
            return scan_pii_batch(
                {"content": contents, "suffix": suffixes},
                self.starpii.gibberish,
                key_detector=key_detector,
            )

        with ThreadPoolExecutor(max_workers=2) as executor:
            rule_based_future = executor.submit(scan_rule_based)
            starpii_future = executor.submit(
                self.starpii.detect,
                contents,
                window_size,
                window_overlap,
                batch_size,
            )
            rule_based_results = rule_based_future.result()
            starpii_results = starpii_future.result()

        findings = []
        for file, file_rule_based_findings, file_starpii_findings in zip(
            files, rule_based_results, starpii_results
        ):
            file_findings = remove_redaction_exclusions(file_rule_based_findings)
            file_findings.extend(file_starpii_findings)
            seen = set()
            for finding in sorted(
                file_findings,
                key=lambda item: (
                    int(item["start"]),
                    int(item["end"]),
                    str(item["tag"]),
                ),
            ):
                key = (finding["start"], finding["end"], finding["tag"])
                if key in seen:
                    continue
                seen.add(key)
                findings.append(
                    {
                        "filename": file["filename"],
                        "start": finding["start"],
                        "end": finding["end"],
                        "type": finding["tag"],
                        "text": finding["value"],
                    }
                )
        return findings


def receive(connection: socket.socket) -> dict[str, object]:
    """Read one JSON request from a client connection.

    Args:
        connection (socket.socket): Accepted local socket connection.

    Returns:
        dict[str, object]: Decoded request body.
    """
    data = bytearray()
    # Bounded reads allow requests of any size without allocating a guessed maximum.
    while chunk := connection.recv(65_536):
        data.extend(chunk)
    return json.loads(data)


def serve(device: int | str = -1) -> None:
    """Load StarPII and serve detection requests until unloaded.

    Args:
        device (int | str): Accelerator selection.
    """
    directory = runtime_dir()
    # The socket and log directory concern text that may contain private data.
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    directory.chmod(0o700)
    socket_path = directory / "service.sock"
    if socket_path.exists():
        socket_path.unlink()

    print("Loading PII detector")
    detector = Detector(device)
    print("PII detector loaded")
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        server.bind(str(socket_path))
        socket_path.chmod(0o600)
        server.listen()
        stop = False
        while not stop:
            connection, _ = server.accept()
            with connection:
                # One malformed request should not terminate future hook requests.
                try:
                    request = receive(connection)
                    if request["operation"] == "ping":
                        response = {"running": True}
                    elif request["operation"] == "unload":
                        response = {"unloaded": True}
                        stop = True
                    elif request["operation"] == "scan":
                        response = {
                            "findings": detector.scan(
                                request["files"],
                                request["key_detector"],
                                request["window_size"],
                                request["window_overlap"],
                                request["batch_size"],
                            )
                        }
                    else:
                        raise ValueError(
                            f"Unsupported operation: {request['operation']}"
                        )
                except Exception as error:  # noqa: BLE001
                    response = {"error": f"{type(error).__name__}: {error}"}
                connection.sendall(json.dumps(response, ensure_ascii=False).encode())
    finally:
        server.close()
        if socket_path.exists():
            socket_path.unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--device", type=parse_device, default=-1)
    serve(parser.parse_args().device)
