# RadiantGuard AI

**Medical Imaging AI Research Sandbox**

RadiantGuard AI is an educational, open-source prototype exploring how medical-imaging AI can be made more transparent, uncertainty-aware, explainable, and accountable to human review.

> **Safety boundary:** This project is not clinically validated, is not a medical device, and must not be used to diagnose or treat any person. All information must be reviewed and double-checked with a qualified physician or radiologist. Use only de-identified public, synthetic, or educational data.

## Foundation build: v0.1

The first build intentionally does **not** generate medical findings. It establishes the responsible application foundation:

- Polished Streamlit interface
- PNG/JPEG and DICOM pixel viewing
- Privacy-conscious metadata display
- Basic non-clinical pixel quality preview
- Rules-based AI report language auditor
- Explicit prototype and human-review boundaries
- Tests and technical design documentation

## Run locally

```bash
python -m venv .venv
```

Activate the environment:

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS/Linux
source .venv/bin/activate
```

Install and run:

```bash
pip install -r requirements.txt
python -m streamlit run app.py
```

Run tests:

```bash
pip install -r requirements-dev.txt
python -m pytest
```

## Responsible roadmap

### Phase 1 — Foundation
- [x] Image upload and viewer
- [x] DICOM pixel support
- [x] Technical display QA preview
- [x] AI report wording auditor
- [x] Safety and design documentation

### Phase 2 — Chest X-ray research module
- [ ] Select a documented public model and dataset
- [ ] Record model license, intended use, and known limitations
- [ ] Reproduce the model's exact preprocessing
- [ ] Add versioned model loading and deterministic inference
- [ ] Add calibrated, clearly labeled research outputs
- [ ] Publish a model card and evaluation notebook

### Phase 3 — Explainability and trust evaluation
- [ ] Grad-CAM or equivalent visual attribution
- [ ] Heatmap sanity checks
- [ ] Model disagreement view
- [ ] Confidence calibration display
- [ ] Report-to-model consistency checks
- [ ] Human-readable limitations panel

### Phase 4 — Research evaluation
- [ ] De-identified benchmark cases
- [ ] Sensitivity, specificity, AUROC, and calibration metrics
- [ ] Subgroup and distribution-shift analysis where dataset metadata permits
- [ ] False-positive and false-negative case review
- [ ] Reproducible experiment reports

## Privacy

DICOM files can contain identifying metadata. RadiantGuard displays only a small allowlist of non-patient fields, but the prototype is not a certified de-identification system. Never upload real patient files unless they have been properly de-identified through an approved workflow.

## Repository structure

```text
radiantguard-ai/
├── app.py
├── src/
│   ├── imaging.py
│   ├── safety.py
│   └── ui.py
├── tests/
├── .streamlit/config.toml
├── DESIGN.md
├── SAFETY.md
├── requirements.txt
├── requirements-dev.txt
└── requirements-model.txt
```

## Author

**Lori DeGandi**  
Healthcare AI · AI Evaluation · Responsible AI Development
