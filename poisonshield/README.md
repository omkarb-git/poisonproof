# PoisonShield

**An Intelligent Framework for Detecting and Mitigating Data Poisoning Attacks in Machine Learning**

PoisonShield is a local, free, end-to-end data-auditing application for tabular and image
datasets. It combines Cleanlab Datalab detectors with a relationship engine, a unified risk
engine, traceable explanations, human review, sanitisation and reporting. It does not modify
uploaded files; every cleaned output is a new copy.

## Highlights

- Tabular and image modalities, labelled and unlabelled.
- Cleanlab Datalab detectors: label issues, outliers, near-duplicates, class imbalance and non-IID.
- Per-detector execution status and errors; a failed detector is never presented as a clean dataset.
- Relationship engine: k-nearest neighbours, similarity, neighbourhood label consistency and a 2-D projection.
- Unified risk engine with configurable weights, normalisation and explicit coverage.
- Traceable explanations for every flagged sample.
- Human review with durable history and sanitisation that never overwrites originals.
- Model/quality evaluation against a real, separate holdout dataset.
- Self-contained offline HTML report in a ZIP, plus CSV/JSON exports.

## Project layout

```
poisonshield/
  app.py                Streamlit entry point
  run.py                Convenience launcher
  requirements.txt
  backend/              Auditing, features, ingestion, intelligence, evaluation, reporting
  ui/                   Streamlit shell, theme, components and 11 sections
  tests/                Test suite
  README.md
```

## Requirements

- Python 3.10-3.13
- The dependencies in `requirements.txt`
- For image embeddings: DINOv2 weights are downloaded on first use. If no network or no
  pretrained weights are available, PoisonShield falls back to ResNet-50 and then to an
  offline handcrafted descriptor. The extractor actually used is recorded in the metadata.

## Installation

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

If you want GPU acceleration, install the PyTorch build matching your CUDA version before
installing the rest:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

or

```bash
python run.py
```

Then open `http://localhost:8501`.

## Workflow

The application has eleven sections:

1. **Overview** - dataset health, detector coverage, risk priorities and next actions.
2. **Upload** - add a tabular table or an image archive; configure labels and ground truth.
3. **Inspection** - profiling, missingness, duplicates and image integrity.
4. **Tabular Analysis** - Cleanlab results and tabular risk evidence.
5. **Image Analysis** - extractor metadata, image risk and integrity.
6. **Relationship Explorer** - neighbourhood structure and projection.
7. **Suspicious Samples** - filter and triage every flag.
8. **Label Review** - human decisions with durable history.
9. **Sanitisation** - generate a cleaned copy without touching originals.
10. **Model / Quality Evaluation** - metrics against real ground truth and a separate holdout.
11. **Reports / Export** - HTML report archive and data exports.

## Data formats

### Tabular

- CSV, XLSX or XLS.
- A label column is optional. Without one, feature-only detectors run.
- Identifier-like, constant and entirely missing columns are excluded from features.
- A separate binary column (for example `is_poisoned`) can be declared as the evaluation
  ground truth. It is excluded from the feature representation to avoid leakage.

### Images

- One ZIP archive (recommended, preserves folders) or a batch of image files.
- A CSV/Excel mapping is optional. It must contain a path column and, optionally, a label
  column. Recognised column names include `path`, `file`, `filename`, `image_path` for paths
  and `label`, `class`, `target`, `category` for labels.
- Folder names can be used as labels when no mapping row matches.
- Extra mapping columns are available for evaluation but are never used as image features.
- Unreadable files and format mismatches are retained as integrity findings.

## Detection and risk

Cleanlab Datalab detectors produce quality scores (higher is better). PoisonShield converts
them to risk evidence as `1 - quality`. The unified risk score is a weighted mean of the
available evidence groups:

- label and neighbourhood evidence: 0.50
- outlier evidence: 0.20
- duplicate evidence: 0.10
- integrity evidence: 0.20

Label and neighbourhood signals are fused non-additively because they are correlated. Groups
that are unavailable for a sample are excluded and the coverage is reported. Class imbalance
and non-IID are dataset-level findings and are excluded from per-sample risk.

Risk categories:

- Low: score <= 0.30
- Moderate: score <= 0.55
- High: score <= 0.75
- Critical: score > 0.75

Thresholds and weights are configurable in the UI and can be re-applied without re-running
detectors.

## Sanitisation

- No sample is removed or relabelled because of a detector score alone.
- Review decisions are recorded with timestamps, notes and the reviewer name.
- The cleaned copy is written to a new file or ZIP; the uploaded original is never modified.
- The original file fingerprint is checked on the Sanitisation and Reports pages.

## Evaluation

- Detection metrics (precision, recall, F1, accuracy, and ROC AUC / average precision when
  applicable) are computed against a real ground-truth column.
- Model comparison trains the same model on original versus reviewed labels and evaluates it
  on a separate, untouched dataset. The evaluation data is never used for fitting.
- Sanitisation can reduce model performance. The measured delta is reported as-is, with
  explicit caveats.

## GPU support

For NVIDIA GPU acceleration, install the CUDA build of PyTorch instead of the CPU build:

```bash
pip install -r requirements-gpu.txt
```

The application automatically uses CUDA when `torch.cuda.is_available()` is true. CPU execution remains fully supported.

```bash
pytest -q
```

The suite covers tabular and image ingestion, labelled and unlabelled datasets, detector
failure surfacing, untouched originals, sanitisation and evaluation metrics.

## Limitations

- Risk scores are review priorities, not proof of poisoning.
- Neighbourhood analysis can be misled by correlated errors in the neighbourhood.
- Image embeddings depend on the chosen extractor and domain; general-purpose models can
  under-represent specialised imagery.
- Detection metrics depend on the quality and independence of the ground-truth column.
- Duplicate files are not automatically poisoning.
- Model comparison is one model family on one representation and one split.
- Sanitisation does not guarantee improved model accuracy.

## Troubleshooting

- **Windows multiprocessing errors**: PoisonShield runs Cleanlab with `n_jobs=1`. If you see
  spawn errors, keep that setting and avoid running inside a multiprocessing worker.
- **DINOv2 download is slow or blocked**: the first run downloads weights. If it fails,
  PoisonShield falls back to ResNet-50 and then to the offline descriptor; the extractor used
  is recorded in the metadata.
- **Large uploads**: the default Streamlit upload limit is 2 GB (`maxUploadSize = 2048`).
- **Unreadable images**: files that cannot be decoded are kept as integrity findings and
  excluded from embedding-based detectors.
- **Memory**: embedding and Cleanlab work is bounded by the configured row/image limits.
