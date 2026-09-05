import json
import os
import socket
import tempfile
from pathlib import Path


def parse_device(value: str) -> int | str:
    """Parse a device argument accepted by the server.

    Args:
        value (str): Automatic selection (-1), CUDA device number, or ``mps``.

    Returns:
        int | str: Integer device number or ``mps`` backend name.
    """
    if value == "mps":
        return value
    return int(value)


def runtime_dir() -> Path:
    """Return the private directory used by the current user's server."""
    return Path(tempfile.gettempdir()) / f"pii-server-{os.getuid()}"


def request(payload: dict[str, object], timeout: float) -> dict[str, object]:
    """Send a request to the PII server.

    Args:
        payload (dict[str, object]): JSON request body.
        timeout (float): Maximum seconds to wait for the server.

    Returns:
        dict[str, object]: Decoded JSON response.

    Raises:
        RuntimeError: The server returned an error.
    """
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.settimeout(timeout)
        client.connect(str(runtime_dir() / "service.sock"))
        client.sendall(json.dumps(payload, ensure_ascii=False).encode())
        client.shutdown(socket.SHUT_WR)
        response = bytearray()
        # Bounded reads allow responses of any size without allocating a guessed maximum.
        while data := client.recv(65_536):
            response.extend(data)

    result = json.loads(response)
    if "error" in result:
        raise RuntimeError(result["error"])
    return result


def server_running() -> bool:
    """Return whether the current user's PII server accepts requests."""
    try:
        # Initialization checks should fail quickly when no server is running.
        request({"operation": "ping"}, 1)
    except OSError:
        return False
    return True
