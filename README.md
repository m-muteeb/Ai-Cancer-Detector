# AI-Cancer-Detector

**Affordable AI-Based Multi-Cancer Screening & Diagnostic System for Low-Resource Settings**

---

## 1 — Problem statement (ready-to-submit)

**Title:** Affordable AI-Based Multi-Cancer Screening & Diagnostic System for Low-Resource Settings

**Problem:** Access to high-quality cancer diagnostics in many Asia-Pacific regions is limited by few pathologists, long turnaround times, and costly digital pathology equipment. Histopathology diagnosis is the gold standard but requires slide scanning, specialized training, and centralized labs — causing delays and inequitable access.

**Solution:** An end-to-end system that:

1. enables low-cost image acquisition (smartphone-based microscopy or low-cost slide scanners),
2. runs a compact, explainable two-stage AI pipeline (cancer-type classification → benign/malignant classifier per type), and
3. ships as a field-friendly web app + FastAPI backend that can run inference on an edge device or modest cloud instance.

Targets: **breast, lung, colon, and prostate** cancer. Prioritizes affordability, field validation, and alignment with WHO diagnostic priorities.

**Impact:** Shorter diagnosis time, triage at PHC level, reduced patient loss-to-follow-up, and scalable deployment with demonstrable pilot evidence to reach TRL ≥ 7.

---

## 2 — High-level objectives & acceptance criteria

**Functional:** Model that classifies (breast, lung, colon, prostate) and then benign vs malignant for each, with explainability heatmaps per image.

**Affordability:** Inference feasible on smartphone-class hardware or a single 8–16GB GPU cloud instance; model quantized to **<100MB combined** for on-device use.

**Evidence:** Validation on public datasets + at least one retrospective pilot with clinical slides (report sensitivity and specificity).

**Deployment:** FastAPI backend, React frontend (mobile-responsive), TFLite/ONNX model for edge.

**Regulatory readiness:** Data and validation strategy to support ethical review and regulatory dialogue.

**Label scheme — two options:**

* **Single-stage:** 8-class — `breast_benign, breast_malignant, lung_benign, lung_malignant, colon_benign, colon_malignant, prostate_benign, prostate_malignant`.
* **Recommended (two-stage):** Stage A — 4-class cancer-type. Stage B — one binary classifier per cancer type (benign/malignant).

**Data splits & strategy:**

* Balanced per-class slices if possible; for imbalance use oversampling/undersampling + class-balanced loss.
* **Standard split:** train / val / test with test containing hospitals not used in training (domain generalization).

**Data acquisition for LMIC pilot:**

* Partner with 1–2 regional hospitals to obtain de-identified slides (smartphone microscope photos + scanned slides).
* Collect metadata: patient age, sex, specimen type, stain protocol.

---

## 4 — Data preprocessing pipeline (detailed)

**Image normalization & sizing**

* Resize to 224×224 (or 256×256 if using ViT-style models).
* Scale pixel values to [0,1].
* Optional stain normalization (Macenko or Reinhard).

**Augmentations (train-time)**

* Rotation ±180°, flips, brightness/contrast, hue jitter.
* Scale/crop, elastic deformation (light), Cutout/RandomErasing.
* Domain augment: color jitter with stain-like transforms; simulate smartphone artefacts (blur, compression noise).

**Patch extraction (WSI)**

* Extract tiles at 10x/20x, discard low-tissue tiles via Otsu or grayscale threshold.
* Tile-level labels inherited from slide label or use pathologist annotations.

**Data loader**

* PyTorch DataLoader or TF Dataset with on-the-fly augmentation and prefetching.
* Cache preprocessed tiles to fast storage (NVMe).

---

## 5 — Modeling strategy (recommended): Two-stage pipeline

**Why:** modular, easier to fine-tune per-organ, interpretable, reduces class imbalance issues.

**Stage A — Cancer type classifier (4-way)**

* Recommended: EfficientNetV2-B0 (compact, strong). Lightweight alternative: MobileNetV3-Large.
* If more compute: ConvNeXt-Tiny or Swin-Tiny.

**TensorFlow Keras template:**

```python
base = tf.keras.applications.EfficientNetV2B0(include_top=False, weights='imagenet', input_shape=(224,224,3))
x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
x = tf.keras.layers.Dropout(0.3)(x)
out = tf.keras.layers.Dense(4, activation='softmax')(x)
model = tf.keras.Model(inputs=base.input, outputs=out)
```

**Stage B — Per-type malignancy classifiers (binary)**

* Per-organ fine-tuned EfficientNetV2-B0 or ConvNeXt-Tiny.
* Outputs probability of malignancy + Grad-CAM map.

**Model compression for edge:**

* Convert to TFLite + int8 quantization or export to ONNX + quantization.
* Use knowledge distillation (teacher → student) to get small MobileNetV3 students.

---

## 6 — Training recipes & hyperparameters (exact)

**General settings**

* Mixed precision if GPU supports it.
* Optimizer: **AdamW** (weight decay = 1e-4).
* LR scheduler: Cosine annealing or ReduceLROnPlateau.
* Batch size: 16–64 (use gradient accumulation if needed).
* Epochs: until convergence + early stopping (monitor val F1 / AUC).

**EfficientNetV2-B0 example**

* Initial lr: 1e-4 (warmup 5 epochs at 1e-5).
* Loss: categorical_crossentropy (stage A) / binary_crossentropy (stage B).
* Regularization: label smoothing 0.05, dropout 0.3.
* Class imbalance: focal loss or class-weighting.

**Validation metrics every epoch**

* Per-class accuracy, macro F1, ROC-AUC per class, confusion matrix.
* Precision, Recall, Specificity for malignant class.

---

## 7 — Explainability / XAI

* Grad-CAM / Grad-CAM++ for heatmaps.
* Integrated Gradients or LIME as secondary evidence.
* Store heatmap overlays as PNGs and return with API responses.
* Provide per-prediction confidence and top-3 class probabilities.

---

## 8 — Backend: FastAPI design (full)

**Stack:** FastAPI + Uvicorn + Gunicorn. Serve inference in-process or as separate microservice.

**Project layout**

```
app/
├─ main.py
├─ models/
├─ schemas.py
├─ routes/
├─ services/
├─ storage/
└─ auth/
```

**Key endpoints**

* `GET /health`
* `POST /predict/type` (image → cancer-type probs)
* `POST /predict/malignancy` (image + optional type → malignant prob + heatmap)
* `POST /predict/full` (combined)
* `POST /upload/batch`
* `GET /model/version`
* `POST /feedback`

**Example skeleton:**

```python
# app/main.py
from fastapi import FastAPI, File, UploadFile
from app.services.inference import predict_full
app = FastAPI(title="PathAI-FASTAPI")

@app.get("/health")
async def health():
    return {"status": "ok", "models_loaded": True}

@app.post("/predict/full")
async def predict_full_endpoint(file: UploadFile = File(...)):
    img = await file.read()
    result = predict_full(img)
    return result
```

**Production notes:** Gunicorn + Uvicorn workers, Redis/RabbitMQ for job queue, ONNX Runtime for CPU inference.

---

## 9 — Frontend: React + Vite + Tailwind (mobile-responsive)

**Pages/components**

* Login/role-based access
* Upload / Capture (file or smartphone capture)
* Prediction result page: probs, malignancy score, heatmap overlay, confidence, recommendation
* Batch uploads, Audit & logs, Settings

**UX details**

* Explicit disclaimer: AI-assist only.
* Large touch targets, simple flows for low-resource clinics.

**Stack**

* React + Vite, TailwindCSS, Axios
* PWA for offline capture and delayed upload

---

## 11 — Validation, clinical pilot & TRL pathway

**Milestones:**

* M1: Technical prototype (public datasets + demo UI).
* M2: Domain adaptation (smartphone pipeline + quantized model within 5% baseline).
* M3: Retrospective clinical validation (report sensitivity/specificity).
* M4: Prospective pilot in 1–2 PHCs.
* M5: TRL evidence pack for TRL ≥ 7.

**Validation deliverables:** confusion matrices, per-organ ROC-AUC, sensitivity at clinically relevant thresholds, explainability artifacts, usability reports.

**Regulatory & ethics:** IRB approvals; de-identify images; encrypt data; prepare local regulatory docs (Malaysia/Philippines/Pakistan considerations).

---

## 12 — Features to add (priority & optional)

**MUST-HAVE**

* Image capture & upload
* Two-stage predictions with confidence
* Grad-CAM overlay
* Audit log & feedback
* Model versioning
* Offline capture + delayed upload

**NICE-TO-HAVE**

* Active learning loop
* LIS/HL7/FHIR integration
* Auto report generation (PDF)
* Multi-modal input (clinical metadata)
* Tissue segmentation (U-Net) for tile selection

---

## 13 — Compute, storage & resource estimates

* Working dataset: hundreds of GB for tiles (your earlier 191GB → 440GB realistic).
* Storage: 1 TB NVMe recommended; S3 for cold storage.
* RAM: 32GB comfortable; 16GB workable.
* GPU: single GPU with 12–16GB VRAM (RTX 3080/4070) for EfficientNetV2-B0.
* Edge: Raspberry Pi + Coral TPU or mid-range Android device.

---

## 14 — Evaluation & success metrics

* Clinical: Sensitivity & specificity per organ (prioritize sensitivity).
* AI: Macro F1, per-class AUC, calibration (Brier score).
* Operational: inference latency, model size, memory usage.
* User: time-to-decision, clinician satisfaction, reduction in TAT.
* Equity: performance disparity across sites/stains/devices.

---

## 15 — Security, privacy & ethics

* Encrypt images at rest (AES-256) and in transit (TLS 1.2+).
* Role-based access control; audit logs.
* Clear informed consent & de-identification pipeline.
* Present model outputs as assistive, not definitive.

---

## 16 — Cost & funding levers

* Use open-source stack (FastAPI, React, TensorFlow/PyTorch).
* Apply for cloud/academic credits (ADB/WHO/Path grants).
* Local partnerships to reduce costs.
* Trial cheap smartphone microscope attachments before scanners.

---

## 17 — Example repo layout & next-code deliverables

```
/path-ai-project
├─ app/                      # FastAPI app
├─ frontend/                 # React Vite + Tailwind app
├─ models/                   # Training scripts, model definitions
├─ data/                     # ingestion & preprocessing
├─ notebooks/                # EDA & baseline experiments
├─ infra/                    # Dockerfiles, k8s manifests
├─ tests/                    # unit & integration tests
└─ README.md
```

**Pick one code deliverable I can produce now:**

* Full FastAPI app with `/predict/full` endpoint and TF model loader.
* React frontend for capture/upload + result view.
* Training notebook + PyTorch/TensorFlow scripts for two-stage pipeline.
* TFLite conversion + example Android wrapper.

Tell me which and I will generate a complete runnable package (code + Dockerfile + brief run instructions).

---

## 18 — Hard truths & risks

* > 90% accuracy per cancer type is optimistic — expect performance drops in real-world heterogeneous data.
* WSI dependence is a deployment risk; low-cost imaging is critical for scaling.
* Clinical validation is mandatory for adoption and TRL advancement.
* Plan for data drift and fairness monitoring.

---

*If you want, I can convert this into a ready-to-commit `README.md` file in the repo structure, and then generate the code deliverable you choose next.*
