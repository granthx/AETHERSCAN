"""Automated download, preparation, and leakage audit for the Kaggle COVID CT dataset.

Dataset: 'CT Scans for COVID-19 Classification'
Kaggle Slug: azaemon/preprocessed-ct-scans-for-covid19
Target Counts: nCT = 9,979 · NiCT = 5,705 · pCT = 4,001 (Total = 19,685)
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

TARGET_COUNTS = {
    "nCT": 9979,
    "NiCT": 5705,
    "pCT": 4001,
}
DEFAULT_SLUG = "azaemon/preprocessed-ct-scans-for-covid19"


def verify_directory_counts(data_root: Path) -> dict:
    """Verify file counts per class folder against the tutorial video's numbers."""
    counts = {}
    mismatches = {}
    for cls_name, expected in TARGET_COUNTS.items():
        cls_dir = data_root / cls_name
        if not cls_dir.exists():
            counts[cls_name] = 0
            mismatches[cls_name] = f"Folder missing (expected {expected})"
        else:
            files = [f for f in cls_dir.iterdir() if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png"}]
            counts[cls_name] = len(files)
            if len(files) != expected:
                diff = len(files) - expected
                mismatches[cls_name] = f"Found {len(files)}, expected {expected} (diff: {diff:+d})"

    total_found = sum(counts.values())
    total_expected = sum(TARGET_COUNTS.values())
    return {
        "counts": counts,
        "expected": TARGET_COUNTS,
        "total_found": total_found,
        "total_expected": total_expected,
        "mismatches": mismatches,
        "is_exact_match": len(mismatches) == 0,
    }


def analyze_patient_leakage(data_root: Path) -> dict:
    """Check filename patterns to identify patient ID groupings and leakage risk.

    Many chest CT datasets contain multiple 2D axial slices from the same 3D scan/patient.
    If slices from Patient A appear in both Train and Test splits, image-level
    stratification leaks patient anatomical features, artificially boosting accuracy.
    """
    patterns = [
        re.compile(r"^(?:patient[_-]?)?([a-zA-Z0-9]+)[_-]slice[_-]?\d+", re.IGNORECASE),
        re.compile(r"^([a-zA-Z0-9]+)[_-]\d+\.(?:png|jpg|jpeg)$", re.IGNORECASE),
        re.compile(r"^([a-zA-Z]+[0-9]+)[_-]", re.IGNORECASE),
        re.compile(r"^(rec[0-9]+)[_-]", re.IGNORECASE),
    ]

    patient_stats = {}
    for cls_name in ["nCT", "pCT", "NiCT"]:
        cls_dir = data_root / cls_name
        if not cls_dir.exists():
            continue

        files = [f.name for f in cls_dir.iterdir() if f.is_file()]
        patients = set()
        matched_count = 0

        for fname in files:
            for pat in patterns:
                m = pat.match(fname)
                if m:
                    pid = m.group(1)
                    patients.add(pid)
                    matched_count += 1
                    break

        patient_stats[cls_name] = {
            "total_images": len(files),
            "matched_with_patient_id": matched_count,
            "unique_patients_inferred": len(patients),
            "avg_slices_per_patient": round(matched_count / max(len(patients), 1), 1) if patients else 0,
        }

    return patient_stats


def download_dataset(slug: str, dest_dir: Path) -> None:
    """Download Kaggle dataset via Kaggle API and unzip."""
    try:
        import kaggle
    except ImportError:
        print("Error: `kaggle` package is required. Install via `pip install kaggle`.")
        sys.exit(1)

    print(f"Downloading Kaggle dataset: {slug} -> {dest_dir}")
    dest_dir.mkdir(parents=True, exist_ok=True)
    kaggle.api.dataset_download_files(slug, path=str(dest_dir), unzip=True, quiet=False)
    print("Download and extraction completed.")


def flatten_folders(dest_dir: Path) -> None:
    """Flatten nested folder structures into top-level nCT/, pCT/, NiCT/."""
    for cls_name in ["nCT", "pCT", "NiCT"]:
        target_dir = dest_dir / cls_name
        target_dir.mkdir(parents=True, exist_ok=True)

        # Look for nested folders matching class name case-insensitively
        for nested in dest_dir.rglob("*"):
            if nested.is_dir() and nested != target_dir and nested.name.lower() == cls_name.lower():
                for f in nested.iterdir():
                    if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                        dest_file = target_dir / f.name
                        if not dest_file.exists():
                            shutil.move(str(f), str(dest_file))


def main(argv=None):
    p = argparse.ArgumentParser(description="Download and verify Kaggle COVID CT dataset")
    p.add_argument("--slug", default=DEFAULT_SLUG, help="Kaggle dataset slug")
    p.add_argument("--dest", default="data/real", help="Destination directory")
    p.add_argument("--check-counts-only", action="store_true", help="Only verify counts without downloading")
    p.add_argument("--check-leakage", action="store_true", help="Analyze patient ID patterns and slice leakage")
    p.add_argument("--dry-run", action="store_true", help="Dry run without downloading")
    a = p.parse_args(argv)

    dest_dir = Path(a.dest)

    if a.check_counts_only:
        res = verify_directory_counts(dest_dir)
        print("\nDataset Count Verification:")
        for cls, count in res["counts"].items():
            exp = res["expected"][cls]
            print(f"  {cls:5s}: {count:,} images (expected {exp:,})")
        print(f"Total: {res['total_found']:,} / {res['total_expected']:,}")
        if res["mismatches"]:
            print("\nMismatches found:")
            for cls, msg in res["mismatches"].items():
                print(f"  - {cls}: {msg}")
        else:
            print("\nExact match with video dataset!")
        return

    if a.check_leakage:
        stats = analyze_patient_leakage(dest_dir)
        print("\nPatient-Level Leakage Assessment:")
        print(stats)
        return

    if a.dry_run:
        print(f"[Dry Run] Would download {a.slug} to {dest_dir} and verify counts:")
        for cls, count in TARGET_COUNTS.items():
            print(f"  - {cls}: {count:,}")
        return

    download_dataset(a.slug, dest_dir)
    flatten_folders(dest_dir)
    res = verify_directory_counts(dest_dir)
    print("\nVerification Results:", res)


if __name__ == "__main__":
    main()
