"""Unit Tests for TrustLens Hashing Service"""

from pathlib import Path
from app.core.hashing import hash_bytes, hash_file, hash_json, hash_directory


def test_hash_bytes():
    data = b"TrustLens Offline Security Platform"
    digest = hash_bytes(data)
    assert isinstance(digest, str)
    assert len(digest) == 64
    # Determinism
    assert hash_bytes(data) == digest


def test_hash_file(temp_workspace: Path):
    test_file = temp_workspace / "sample.txt"
    test_file.write_text("Integrity Check Data", encoding="utf-8")
    digest = hash_file(test_file)
    assert isinstance(digest, str)
    assert len(digest) == 64
    assert digest == hash_bytes("Integrity Check Data".encode("utf-8"))


def test_hash_json_canonical_ordering():
    # Dictionary with unsorted keys
    obj1 = {"zebra": 1, "apple": 2, "mango": {"inner_b": 10, "inner_a": 20}}
    obj2 = {"apple": 2, "zebra": 1, "mango": {"inner_a": 20, "inner_b": 10}}
    assert hash_json(obj1) == hash_json(obj2)


def test_hash_directory_determinism(temp_workspace: Path):
    d1 = temp_workspace / "dir_a"
    d1.mkdir()
    (d1 / "file_b.txt").write_text("Hello B")
    (d1 / "file_a.txt").write_text("Hello A")

    d2 = temp_workspace / "dir_b"
    d2.mkdir()
    (d2 / "file_a.txt").write_text("Hello A")
    (d2 / "file_b.txt").write_text("Hello B")

    hash1, manifest1 = hash_directory(d1)
    hash2, manifest2 = hash_directory(d2)

    assert hash1 == hash2
    assert len(manifest1) == 2
