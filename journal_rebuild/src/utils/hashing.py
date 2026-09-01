from __future__ import annotations

import hashlib
import io
import json
import pickle
from pathlib import Path
from typing import Any


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(obj: Any) -> str:
    payload = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256_bytes(payload)


def sha256_pickle(obj: Any) -> tuple[str, bytes]:
    buffer = io.BytesIO()
    pickle.dump(obj, buffer, protocol=pickle.HIGHEST_PROTOCOL)
    payload = buffer.getvalue()
    return sha256_bytes(payload), payload
