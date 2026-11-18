"""TrustLens Member 2 — Dataset Integrity CLI & Demonstration Runner

Executes full dataset forensics on data/demo/sample_dataset/ and outputs
structured metrics, duplicate clusters, OOD scores, and Section 22 Evidence.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Add backend directory to sys.path so app can be imported directly from root
ROOT_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = ROOT_DIR / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.schemas.dataset import DatasetAnalysisRequest, DatasetFormatEnum
from app.services.dataset_service import get_dataset_service


def run_demo(dataset_path: str = "data/demo/sample_dataset") -> None:
    print("=" * 72)
    print("      TrustLens — Computer Vision & Dataset Integrity Audit       ")
    print("                  SIH26228 Offline Verification                  ")
    print("=" * 72)

    target = ROOT_DIR / dataset_path
    if not target.exists():
        print(f"[!] Error: Dataset target path does not exist: {target}")
        sys.exit(1)

    print(f"\n[*] Auditing dataset at: {target}")
    request = DatasetAnalysisRequest(
        dataset_path=str(target),
        format_hint=DatasetFormatEnum.AUTO,
        similarity_threshold=0.88,
        ood_contamination=0.10,
        enable_embeddings=True,
    )

    service = get_dataset_service()
    print(f"[*] Embedding backend active: {service.embedding_extractor.backend_name}")
    print("[*] Running forensic pipeline (Hashing -> Per-Image -> Duplicates -> OOD -> Evidence)...")

    response = service.analyze_dataset(request)
    overview = response.quality_overview
    manifest = response.dataset_manifest

    print("\n" + "-" * 72)
    print(f"1. DATASET MANIFEST: {manifest.dataset_id}")
    print(f"   - Name: {manifest.dataset_name}")
    print(f"   - Detected Format: {manifest.detected_format}")
    print(f"   - Total Samples: {manifest.total_samples}")
    print(f"   - Manifest Hash: {manifest.manifest_hash}")

    print("\n" + "-" * 72)
    print("2. QUALITY & INTEGRITY METRICS:")
    print(f"   - Valid Images: {overview.valid_images} / {overview.total_samples}")
    print(f"   - Corrupted Files: {overview.corrupted_images}")
    print(f"   - Mean Luminance (Brightness): {overview.mean_brightness:.2f} / 255.0")
    print(f"   - Mean Contrast: {overview.mean_contrast:.2f}")
    print(f"   - Mean Blur Score (Laplacian Var): {overview.mean_blur_score:.2f}")
    print(f"   - Integrity Status: [{overview.integrity_status}]")

    print("\n" + "-" * 72)
    print(f"3. DUPLICATE FLOODING CLUSTERS (Total Duplicates: {overview.duplicate_samples_count}):")
    if not response.duplicate_analysis.clusters:
        print("   - No duplicate clusters detected.")
    else:
        for cl in response.duplicate_analysis.clusters:
            print(f"   [+] Cluster ID: {cl.cluster_id} (Size: {cl.cluster_size})")
            print(f"       Indicator: \"{cl.indicator}\"")
            print(f"       Representative: {Path(cl.representative_image).name}")
            for d in cl.duplicates:
                print(f"         * {Path(d.file_path).name} — {d.duplicate_type} (Similarity: {d.similarity}%, Hamming: {d.hamming_distance})")

    print("\n" + "-" * 72)
    print(f"4. OUT-OF-DISTRIBUTION (OOD) ANOMALIES (Detected: {overview.ood_anomalies_count}):")
    if response.ood_analysis and response.ood_analysis.samples:
        outliers = [s for s in response.ood_analysis.samples if s.is_outlier]
        if not outliers:
            print("   - No statistical outliers exceeded decision threshold.")
        for s in outliers:
            print(f"   [!] Sample: {Path(s.file_path).name} (Score: {s.ood_score:.4f}, Severity: {s.severity})")
            print(f"       Limitation Note: \"{s.limitation_note}\"")

    print("\n" + "-" * 72)
    print(f"5. SECTION 22 EVIDENCE OBJECTS GENERATED ({len(response.evidence_objects)} items):")
    for idx, ev in enumerate(response.evidence_objects, 1):
        print(f"   [{idx}] {ev.id} | Type: {ev.finding_type} | Severity: {ev.severity}")
        print(f"       What: {ev.what_happened}")
        print(f"       Why: {ev.why_flagged}")
        print(f"       Action: {ev.recommended_action}")

    # Export report to reports/ directory
    reports_dir = ROOT_DIR / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_file = reports_dir / "dataset_integrity_report.json"
    with report_file.open("w", encoding="utf-8") as rf:
        json.dump(response.model_dump(), rf, indent=2, default=str)

    print("\n" + "=" * 72)
    print(f"[SUCCESS] Complete audit report saved to: {report_file}")
    print("=" * 72)


if __name__ == "__main__":
    path_arg = sys.argv[1] if len(sys.argv) > 1 else "data/demo/sample_dataset"
    run_demo(path_arg)
