"""Integration Tests for FastAPI Dataset Endpoints"""

from pathlib import Path
from fastapi.testclient import TestClient
from main import app
from app.schemas.dataset import DatasetFormatEnum

client = TestClient(app)


def test_root_and_health_endpoints():
    r_root = client.get("/")
    assert r_root.status_code == 200
    assert r_root.json()["service"] == "TrustLens Platform"

    r_health = client.get("/health")
    assert r_health.status_code == 200
    assert r_health.json()["status"] == "healthy"


def test_analyze_dataset_endpoint(synthetic_folder_dataset: Path):
    payload = {
        "dataset_path": str(synthetic_folder_dataset),
        "format_hint": DatasetFormatEnum.FOLDER.value,
        "similarity_threshold": 0.85,
        "enable_embeddings": True,
    }
    response = client.post("/api/datasets/analyze", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert "dataset_manifest" in data
    assert "quality_overview" in data
    assert "duplicate_analysis" in data
    assert "evidence_objects" in data
    assert len(data["evidence_objects"]) >= 2


def test_analyze_single_image_endpoint(synthetic_folder_dataset: Path):
    valid_image = synthetic_folder_dataset / "img_01.jpg"
    with valid_image.open("rb") as f:
        response = client.post(
            "/api/datasets/analyze-image",
            files={"file": ("img_01.jpg", f, "image/jpeg")},
        )
    assert response.status_code == 200
    data = response.json()
    assert data["file_name"] == "img_01.jpg"
    assert data["is_corrupted"] is False
    assert len(data["sha256"]) == 64
    assert data["perceptual_hashes"]["phash"] is not None
