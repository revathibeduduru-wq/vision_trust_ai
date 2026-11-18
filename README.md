# TrustLens — CV & Dataset Integrity Engine (SIH26228)
## Member 2 Implementation Reference & Architecture Guide

---

## 1. Overview
The **CV / Dataset Integrity Engine** forms the foundational layer of the TrustLens security platform. It performs automated, offline-first forensic inspections across multi-format computer vision training and evaluation datasets (plain folders, CSV metadata, and YOLO formats) to ensure data authenticity, detect duplicate flooding, isolate distribution anomalies (OOD), and uncover label inconsistencies without making unsubstantiated adversarial accusations.

---

## 2. Directory Architecture

```
trustlens/
├── backend/
│   ├── app/
│   │   ├── core/
│   │   │   ├── __init__.py
│   │   │   └── hashing.py              # SHA-256 file, byte, and canonical manifest hashing
│   │   ├── cv/
│   │   │   ├── __init__.py
│   │   │   ├── dataset_loader.py       # Ingests Folders, CSV metadata, and YOLO datasets
│   │   │   ├── image_analyzer.py       # Computes SHA-256, pHash/dHash/aHash, blur, brightness, contrast
│   │   │   ├── duplicate_detector.py   # DSU graph clustering: exact (SHA-256) & near-duplicates (pHash)
│   │   │   └── embedding_extractor.py  # 512-dim normalized embeddings (TorchVision ResNet-18 / offline)
│   │   ├── analytics/
│   │   │   ├── __init__.py
│   │   │   ├── ood_detector.py         # Isolation Forest & centroid distance OOD scoring (0.0 - 1.0)
│   │   │   ├── label_consistency.py    # k-NN neighborhood consensus label verification
│   │   │   └── evidence_builder.py     # Canonical Section 22 Evidence Object transformer
│   │   ├── schemas/
│   │   │   ├── cv.py                   # Pydantic models for per-image metrics & duplicate clusters
│   │   │   ├── dataset.py              # Manifest, request, response, and quality overview schemas
│   │   │   └── evidence.py             # Section 22 Evidence Object schema
│   │   ├── services/
│   │   │   └── dataset_service.py      # End-to-end dataset audit orchestrator
│   │   └── api/
│   │       └── endpoints/
│   │           └── dataset_analysis.py # FastAPI route handlers for POST /api/datasets/analyze
│   ├── main.py                         # FastAPI application entrypoint with CORS & Swagger
│   ├── requirements.txt                # Pinned production & development dependencies
│   └── tests/
│       ├── conftest.py                 # Synthetic test generators (folders, CSVs, YOLO, corruptions)
│       ├── test_api_endpoints.py       # FastAPI route tests
│       ├── test_dataset_service.py     # End-to-end dataset analysis integration tests
│       ├── test_duplicate_detector.py  # Exact & near-duplicate clustering tests
│       ├── test_hashing.py             # SHA-256 and directory manifest tests
│       ├── test_image_analyzer.py      # Statistical and perceptual image metrics tests
│       ├── test_label_consistency.py   # k-NN label inconsistency tests
│       └── test_ood_detector.py        # Isolation Forest OOD scoring tests
└── data/
    └── demo/
        └── sample_dataset/             # Pre-seeded offline demo images for judging & team integration
```

---

## 3. Specifications & Phrasing Compliance

| Spec Section | Feature | Implementation Details | Strict Phrasing / Tone Mandate |
|---|---|---|---|
| **Section 8** | Per-Image Analysis | SHA-256, perceptual hashes (pHash, dHash, aHash, whash), width, height, aspect ratio, brightness, contrast, Laplacian blur score, corruption flag | Gracefully captures corrupted files with descriptive failure reasons without crashing the pipeline |
| **Section 9** | Duplicate Detection | Exact matching (SHA-256) + Near-duplicate clustering (Hamming distance on pHash $\ge 90\%$) with Union-Find graph clustering | Always uses the exact phrase: `Potential duplicate flooding indicator` (never asserts intent) |
| **Section 10** | OOD / Anomaly Detection | Isolation Forest + cosine distance from dataset centroid, mapped to $[0.0, 1.0]$ with `CRITICAL`, `HIGH`, `MEDIUM`, `LOW` severities | *"Sample features deviate significantly from the baseline distribution. This indicates atypical visual features or potential out-of-distribution domain shift, but does not definitively prove malicious poisoning or corruption."* |
| **Section 11** | Label Consistency | $k$-Nearest Neighbors cosine feature consensus (evaluated on representation similarity, NOT semantic comprehension) | Mandatory finding description: `Potential label inconsistency` (never says "confirmed poisoned sample") |
| **Section 22** | Evidence Objects | Standardized structured finding objects | Includes `id`, `finding_type`, `what_happened`, `why_flagged`, `evidence`, `severity`, `confidence`, `affected_asset`, `recommended_action`, `limitations` |

---

## 4. API Endpoints

### `POST /api/datasets/analyze`
**Request Payload**:
```json
{
  "dataset_path": "data/demo/sample_dataset",
  "format_hint": "AUTO",
  "similarity_threshold": 0.90,
  "ood_contamination": 0.08,
  "knn_k": 5,
  "enable_embeddings": true
}
```

**Response Payload**:
```json
{
  "dataset_manifest": {
    "dataset_id": "DS-SAMPLE_DATA-1a2b3c4d",
    "dataset_name": "sample_dataset",
    "detected_format": "FOLDER",
    "total_samples": 11,
    "classes": []
  },
  "quality_overview": {
    "total_samples": 11,
    "valid_images": 10,
    "corrupted_images": 1,
    "mean_brightness": 128.4,
    "mean_contrast": 42.1,
    "mean_blur_score": 115.8,
    "duplicate_clusters_count": 2,
    "duplicate_samples_count": 2,
    "ood_anomalies_count": 1,
    "label_inconsistencies_count": 0,
    "integrity_status": "FLAGGED"
  },
  "duplicate_analysis": {
    "clusters": [
      {
        "cluster_id": "DUP-CLUSTER-001",
        "representative_image": "data/demo/sample_dataset/images/sample_01.jpg",
        "duplicates": [
          {"file_path": "sample_01.jpg", "sha256": "...", "similarity": 100.0, "duplicate_type": "EXACT", "hamming_distance": 0},
          {"file_path": "sample_01_copy.jpg", "sha256": "...", "similarity": 100.0, "duplicate_type": "EXACT", "hamming_distance": 0}
        ],
        "cluster_size": 2,
        "indicator": "Potential duplicate flooding indicator"
      }
    ]
  },
  "evidence_objects": [
    {
      "id": "EVID-DS-2026-00001",
      "finding_type": "CORRUPTED_FILE_DETECTED",
      "what_happened": "Image file 'sample_corrupted.jpg' could not be decoded or failed format verification.",
      "why_flagged": "Integrity check failed: Corrupted or unsupported image file: UnidentifiedImageError",
      "severity": "HIGH",
      "confidence": 1.0,
      "affected_asset": "data/demo/sample_dataset/images/sample_corrupted.jpg",
      "recommended_action": "Quarantine or re-ingest corrupted image file before training pipeline ingestion.",
      "limitations": "File corruption may result from incomplete network transfers, disk sector faults, or invalid container headers rather than deliberate tampering."
    }
  ]
}
```

### `POST /api/datasets/analyze-image`
Uploads a single image multipart file and returns hashes, dimensions, brightness, contrast, blur score, and corruption status.

---

## 5. Offline-First Guarantee
Under SIH26228, TrustLens must execute in air-gapped field environments without internet access:
- If TorchVision ResNet-18 model weights are available locally, the engine extracts deep penultimate layer features.
- If running completely offline with uninitialized weights or without PyTorch, the engine automatically falls back to an internal multi-scale spatial pyramid and 2D-DCT frequency texture extractor producing deterministic, normalized 512-dimensional feature vectors.
- **Zero internet calls are made at runtime.**

---

## 6. How to Run Tests
Inside `backend/`:
```bash
python -m pytest tests -v
```
All 22 unit, analytical, and API integration tests execute and pass out-of-the-box.
