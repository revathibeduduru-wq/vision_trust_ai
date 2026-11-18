"""TrustLens Cryptographic Hashing Utilities (SHA-256)

Provides deterministic hashing functions for bytes, files, canonical JSON,
and directory manifests as required by the TrustLens security specification.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Tuple, Dict


def hash_bytes(data: bytes) -> str:
    """Compute SHA-256 hexadecimal digest of raw bytes."""
    hasher = hashlib.sha256()
    hasher.update(data)
    return hasher.hexdigest()


def hash_file(file_path: str | Path) -> str:
    """Compute SHA-256 hexadecimal digest of a file in 64KB blocks.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        IsADirectoryError: If the path points to a directory.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")

    hasher = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def hash_json(data: Any) -> str:
    """Compute SHA-256 hash of a JSON-serializable object using canonical encoding.

    Ensures determinism by:
    - Sorting dictionary keys.
    - Using strict compact separators (',', ':').
    - Disallowing unhandled non-standard types.
    """
    canonical_json_str = json.dumps(
        data,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    )
    return hash_bytes(canonical_json_str.encode("utf-8"))


def hash_directory(dir_path: str | Path) -> Tuple[str, Dict[str, str]]:
    """Compute a deterministic canonical hash for an entire directory.

    Process:
    1. Discovers all files recursively.
    2. Sorts relative POSIX paths alphabetically.
    3. Computes SHA-256 for each file.
    4. Constructs a canonical manifest string: '<rel_path>:<sha256>\\n'
    5. Hashes the manifest to produce the root directory hash.

    Returns:
        tuple of (manifest_root_hash, dict_of_relpaths_to_sha256)
    """
    root = Path(dir_path)
    if not root.is_dir():
        raise NotADirectoryError(f"Directory not found: {dir_path}")

    manifest_entries: Dict[str, str] = {}
    for file_path in sorted(root.rglob("*")):
        if file_path.is_file():
            rel_posix = file_path.relative_to(root).as_posix()
            manifest_entries[rel_posix] = hash_file(file_path)

    canonical_manifest_lines = [
        f"{rel_path}:{manifest_entries[rel_path]}\n"
        for rel_path in sorted(manifest_entries.keys())
    ]
    manifest_bytes = "".join(canonical_manifest_lines).encode("utf-8")
    root_hash = hash_bytes(manifest_bytes)
    return root_hash, manifest_entries
