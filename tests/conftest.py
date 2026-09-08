import sys
from pathlib import Path


def pytest_ignore_collect(collection_path: Path) -> bool | None:
    """Exclude the unsupported backend before its dependencies are imported.

    Args:
        collection_path (Path): File or directory considered for collection.

    Returns:
        bool | None: True for the unsupported backend, otherwise defer to pytest.
    """
    excluded_module = "test_vllm.py" if sys.platform == "darwin" else "test_mlx.py"
    if collection_path.name == excluded_module:
        return True
    return None
