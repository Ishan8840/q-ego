"""Explicit model download; evaluation never downloads assets implicitly."""

import hashlib
import os
import tempfile
import urllib.request
from pathlib import Path

HAND_MODEL_SHA256 = "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1"
HAND_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"


def download_hand_model(destination: Path, expected_sha256: str | None = None) -> str:
    if destination.exists():
        raise ValueError(f"Model already exists: {destination}; choose a new path")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    digest = hashlib.sha256()
    try:
        with (
            urllib.request.urlopen(HAND_MODEL_URL, timeout=60) as response,
            tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as output,
        ):
            temporary = Path(output.name)
            total = 0
            while chunk := response.read(1024 * 1024):
                total += len(chunk)
                if total > 30 * 1024 * 1024:
                    raise ValueError("Model download exceeded 30 MiB limit")
                digest.update(chunk)
                output.write(chunk)
        value = digest.hexdigest()
        if value != (expected_sha256 or HAND_MODEL_SHA256):
            raise ValueError("Downloaded model checksum mismatch")
        os.replace(temporary, destination)
        return value
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
