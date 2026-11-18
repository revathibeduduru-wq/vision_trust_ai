"""TrustLens Dataset Forensic API Endpoints

Provides route definitions for:
- POST /api/datasets/analyze: In-depth dataset integrity and security inspection
- POST /api/datasets/analyze-image: Single image forensics (hashes, dimensions, blur, brightness)
"""

from __future__ import annotations

import shutil
from pathlib import Path
import tempfile
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from app.cv.image_analyzer import analyze_image
from app.schemas.cv import ImageAnalysisResult
from app.schemas.dataset import (
    DatasetAnalysisRequest,
    DatasetAnalysisResponse,
    DatasetFormatEnum,
)
from app.services.dataset_service import get_dataset_service

router = APIRouter()


@router.post("/analyze", response_model=DatasetAnalysisResponse, summary="Analyze dataset integrity")
async def analyze_dataset_endpoint(request: DatasetAnalysisRequest) -> DatasetAnalysisResponse:
    """Run full dataset forensic pipeline across folders, CSV, or YOLO datasets.
    
    Identifies:
    - Per-image statistics (brightness, contrast, blur score)
    - File corruption and decode failures
    - Exact duplicate files and near-duplicate clusters (with 'Potential duplicate flooding indicator')
    - Out-Of-Distribution (OOD) statistical anomalies via Isolation Forest
    - Label consistency discrepancies using k-NN neighborhood consensus
    - Section 22 Evidence Objects for automated risk scoring
    """
    target = Path(request.dataset_path)
    if not target.exists():
        raise HTTPException(status_code=404, detail=f"Dataset path not found: {request.dataset_path}")

    service = get_dataset_service()
    try:
        response = service.analyze_dataset(request)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Dataset analysis failed: {str(e)}")


@router.post("/analyze-image", response_model=ImageAnalysisResult, summary="Analyze single image file")
async def analyze_single_image(file: UploadFile = File(...)) -> ImageAnalysisResult:
    """Perform forensic analysis on an uploaded image file."""
    # Save uploaded file to temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)

    try:
        result = analyze_image(tmp_path)
        # Preserve original filename in response
        if file.filename:
            result.file_name = file.filename
        return result
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass
