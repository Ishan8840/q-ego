"""Offline batch replay example; no inference is performed.

Set QC_RESPONSES_DIR to a directory containing <video SHA256>.json responses.
Pass --provider examples.adapters:make_provider to either evaluate command.
A live implementation can replace evaluate() while preserving this interface.
"""

import hashlib
import os
from pathlib import Path
from typing import Any


class DirectoryProvider:
    def __init__(self, directory: Path) -> None:
        self.responses = {path.stem: path.read_text() for path in sorted(directory.glob("*.json"))}
        if not self.responses:
            raise ValueError(f"No response JSON files in {directory}")
        digest = hashlib.sha256()
        for identifier, response in self.responses.items():
            digest.update(identifier.encode())
            digest.update(response.encode())
        self.cache_id = "directory-replay-v1:" + digest.hexdigest()

    def evaluate(self, request: dict[str, Any]) -> str:
        return self.responses[request["video_id"]]


def make_provider() -> DirectoryProvider:
    return DirectoryProvider(Path(os.environ["QC_RESPONSES_DIR"]))
