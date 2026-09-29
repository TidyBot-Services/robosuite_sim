"""Durable request receipts. An unresolved intent is never executed again."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path


def durable_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
        temporary = stream.name
    os.replace(temporary, path)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class RequestJournal:
    def __init__(self, directory: str | Path | None = None):
        self.directory = Path(directory or os.environ.get(
            "TIDYBOT_RECEIPT_DIR", str(Path.home() / ".local/state/robosuite_sim/receipts"))).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)

    def path(self, session_id: str, request_id: str) -> Path:
        digest = hashlib.sha256(json.dumps([session_id, request_id]).encode()).hexdigest()
        return self.directory / (digest + ".json")

    def read(self, session_id: str, request_id: str):
        path = self.path(session_id, request_id)
        return json.loads(path.read_text()) if path.exists() else None

    def write(self, record: dict):
        path = self.path(record["session_id"], record["request_id"])
        durable_json(path, record)
        return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
