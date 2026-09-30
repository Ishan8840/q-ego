"""Content identity and external decoder operations (no shell interpolation)."""

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


class InfrastructureError(RuntimeError):
    """A missing executable or timeout is not evidence of a bad submission."""


def video_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_decoder(command: list[str], timeout: float) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=True)
    except (FileNotFoundError, subprocess.TimeoutExpired, PermissionError) as exc:
        raise InfrastructureError(f"{command[0]} unavailable or timed out: {exc}") from exc


def probe(path: Path, timeout: float = 60) -> dict[str, Any]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_streams",
        "-show_format",
        "-of",
        "json",
        str(path.resolve()),
    ]
    result = run_decoder(command, timeout)
    data = json.loads(result.stdout)
    if not data.get("streams"):
        raise ValueError("No video stream found")
    return data


def verify_decode(path: Path, timeout: float) -> None:
    """Fail on decoding errors even when OpenCV would silently conceal them."""
    run_decoder(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-xerror",
            "-i",
            str(path.resolve()),
            "-map",
            "0:v:0",
            "-an",
            "-f",
            "null",
            "-",
        ],
        timeout,
    )
