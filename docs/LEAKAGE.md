# Medical Data Leakage Assessment (LEAKAGE.md)

> **Critical Methodological Note**: In medical image analysis, data leakage between training and evaluation splits can severely inflate apparent model accuracy, leading to deceptive real-world clinical performance.

---

## 1. The Slice-Level Data Leakage Problem

In Computed Tomography (CT), an axial chest scan consists of a 3D volumetric acquisition comprising **50 to 300+ contiguous 2D axial slices** per patient. 

Adjacent slices within the same patient share:
- Identical anatomical geometry (lung volume, rib cage morphology, heart position).
- Identical scanner physics (scanner model, reconstruction kernel, slice thickness, kVp/mA settings).
- Uniform tissue density and contrast levels.
- Identical spatial distribution of COVID-19 pulmonary infiltrates (ground-glass opacities spanning multiple consecutive slices).

### The Failure Mode of Image-Level Random/Stratified Splitting

When an image-wise split (such as the standard 70/15/15 stratified split used in the tutorial video) is performed across individual 2D slices:
1. Slices 45, 46, and 48 of Patient A are placed in the **Train Set**.
2. Slice 47 of Patient A is placed in the **Test Set**.

In this scenario, the convolutional network does not merely learn generalizable pathophysiological features of COVID-19 pneumonia; it memorizes the unique patient-specific anatomy and scanner artifacts of Patient A. When evaluated on Slice 47, the model performs near-perfect nearest-neighbor retrieval on a patient it has already seen during training.

```
Volumetric Scan (Patient A):
[Slice 44]  ──► Train
[Slice 45]  ──► Train
[Slice 46]  ──► Train
[Slice 47]  ──► TEST  <─── LEAKAGE! Model already memorized Patient A
[Slice 48]  ──► Train
```

---

## 2. Dataset Examination: Kaggle "CT Scans for COVID-19 Classification"

The dataset utilized in the source video (`azaemon/preprocessed-ct-scans-for-covid19`, derived from the iCTCF / HUST-19 repository) consists of pre-extracted 2D slices organized into three folders:
- `nCT/` (9,979 images)
- `pCT/` (4,001 images)
- `NiCT/` (5,705 images)

### Audit of Filename Conventions & Patient Identifiers

Running our automated audit script:
```bash
python scripts/download_real_dataset.py --check-leakage --dest data/real
```

1. **Filename Standardization**:
   In this preprocessed Kaggle distribution, filenames are flattened and re-indexed sequentially (e.g., `pCT_0001.jpg` through `pCT_4001.jpg`, or generic hash identifiers), stripping original DICOM metadata, Study Instance UIDs (`0020,000D`), and Patient IDs (`0010,0020`).
2. **Missing Group Metadata**:
   Because slice-to-patient mapping manifests without explicit patient ID tables in this public mirror, strict patient-grouped splitting (`GroupKFold` or `StratifiedGroupKFold`) cannot be guaranteed without access to the raw DICOM headers.

---

## 3. Methodological Conclusion & Honesty Warning

* **Why 99.51% is an Optimistic Figure**:
  Because the splits in the video were generated using slice-wise stratified splitting (`random.Random(42)` on image index arrays), slices from the same patient and scanner are distributed across both the training and test sets.
* **Realistic Generalization Gap**:
  Published clinical studies on the original HUST-19 / iCTCF dataset that enforce strict patient-level separation typically report test accuracies between **88% and 94%**, rather than **99.5%**.
* **Clinical Significance**:
  A model deployed in a hospital encounters completely unseen patients scanned on different machines. Reporting 99.51% without disclosing slice-level leakage creates false clinical confidence.

---

## 4. Recommendations for Research & Production

1. **Always Require DICOM Metadata**: When sourcing medical imaging, preserve `PatientID` and `StudyInstanceUID` to enforce patient-grouped cross-validation.
2. **Patient-Wise Stratified Splitting**:
   ```python
   from sklearn.model_selection import StratifiedGroupKFold
   sgkf = StratifiedGroupKFold(n_splits=5)
   for train_idx, test_idx in sgkf.split(X, y, groups=patient_ids):
       # Zero patient overlap between train and test
       pass
   ```
3. **Multi-Center Validation**: Always evaluate models on external datasets acquired from different hospitals, scanner manufacturers (GE, Siemens, Philips), and geographic populations.
