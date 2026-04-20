# 🌿 PlantSnap — Offline Herb Identification for iOS

A production ML system that identifies **71 medicinal herbs** in real-time using a **hybrid ResNet18 + CLIP architecture** — ResNet18 via CoreML for fast on-device inference (62ms, fully offline), with **CLIP ViT-B/32 zero-shot fallback** for uncertain predictions and unknown herbs. Built with a complete **automated retraining pipeline** powered by user feedback.

**Live API:** [computer-vision-yin8.onrender.com/docs](https://computer-vision-yin8.onrender.com/docs)
**Medium Series:** [medium.com/@rachana.gupta_7569](https://medium.com/@rachana.gupta_7569)

---

## How It Works

```
Camera → ResNet18 (on-device, 62ms) → confidence check
   │
   ├── confidence ≥ 50% → show result (offline, fast)
   │
   └── confidence < 50% → CLIP ViT-B/32 fallback (server)
         → zero-shot comparison across 71 herb embeddings
         → returns best match + similarity score
   
User corrects wrong prediction (offline)
   ↓
Corrections queue locally (Store and Forward)
   ↓
Signal returns → batch sync to FastAPI
   ↓
3-layer quality gate (blur + brightness + CLIP)
   ↓
Corrections saved to S3
   ↓
Retraining pipeline triggers (MLflow)
   ↓
New CoreML model → OTA update to iOS
   ↓
Better predictions → fewer corrections needed
```

---

## Architecture

```
plantsnap/
├── ios/              # SwiftUI iOS app
│   ├── ContentView   # Camera + 3-tier feedback UI
│   ├── ImagePicker   # Camera/album with CoreML inference
│   └── FeedbackBanner # Correction submission (offline-capable)
│
├── api/              # FastAPI backend (deployed on Render)
│   ├── main.py       # Endpoints: /feedback, /clip, /version, /metrics
│   ├── database.py   # SQLite + S3 backup/restore
│   ├── models.py     # SQLAlchemy models
│   ├── schemas.py    # Pydantic validation
│   └── clip_fallback.py  # CLIP zero-shot inference
│
├── ml/               # Model training + MLOps
│   ├── train_herbs_mlflow.py  # PyTorch training with MLflow tracking
│   ├── clip_fallback.py       # CLIP ViT-B/32 fallback classifier
│   └── data/raw/              # 71 herb image classes
│
└── .github/
    └── workflows/    # CI/CD: 17/17 tests → auto-deploy
```

---

## Key Features

### On-Device ML (iOS)
- **ResNet18 fine-tuned** on 5,400+ herb images (71 classes) using PyTorch + Apple MPS
- **CoreML export** for Neural Engine inference at **62ms** — fully offline
- **3-tier feedback UI**: known herbs dropdown → searchable extended list → free text input

### FastAPI Backend
- **3-layer image quality gate**: blur detection (OpenCV), brightness check, CLIP zero-shot plant verification
- **CLIP fallback endpoint** (`POST /clip`): when ResNet confidence < 50%, CLIP ViT-B/32 zero-shot classifies across all 71 herb names
- **Store and Forward**: iOS queues corrections offline, syncs when signal returns
- **Idempotent submissions**: UUID-based deduplication prevents duplicate training data
- **Confidence thresholding**: only saves corrections where model confidence < 70% (highest value for retraining)

### MLOps Pipeline
- **MLflow experiment tracking** across 5 training runs (45% → 62.7% val accuracy)
- **Automated retraining**: S3 corrections → train/val split → augmentation → new CoreML model
- **OTA model updates**: iOS checks `/version` on launch, downloads new model silently
- **S3 database backup**: SQLite backed up to S3 on every write, restored on Render restart

---

## Training Runs

| Run | Key Change | Val Accuracy | Insight |
|-----|-----------|-------------|---------|
| 1 | Baseline (no val split) | ~45% | Flying blind without validation |
| 2 | Augmentation + val split | 56.0% | Proper measurement enabled progress |
| 3 | +60% more data (iNaturalist) | 47.3% | **More data made it worse** — noisy images |
| 4 | Unfreeze layer4 | **62.7%** | **One line of code gained 16%** |
| 5 | Dropout + stronger augmentation | 61.3% | Reduced overfitting but slightly lower accuracy |

**Run 4 is deployed to production.** Full story: [I Ran My Herb Classifier 4 Times. The Third Run Made It Worse.](https://medium.com/@rachana.gupta_7569)

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **iOS** | SwiftUI · AVFoundation · CoreML · Vision |
| **ML** | PyTorch · ResNet18 (fine-tuned) · CLIP ViT-B/32 (zero-shot fallback) · MLflow |
| **API** | FastAPI · Pydantic · SQLAlchemy · SQLite |
| **Cloud** | AWS S3 · Render.com · GitHub Actions |
| **Testing** | 17 integration tests · CI/CD auto-deploy |

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/feedback` | POST | Submit herb correction + image |
| `/clip` | POST | CLIP zero-shot fallback (base64 image) |
| `/version` | GET | Current model version for OTA updates |
| `/feedback/all` | GET | All corrections (newest first) |
| `/feedback/stats` | GET | Correction count + retraining readiness |
| `/metrics` | POST | Log inference confidence per herb |

**Interactive docs:** [computer-vision-yin8.onrender.com/docs](https://computer-vision-yin8.onrender.com/docs)

---

## Quick Start

### API (local)
```bash
cd api
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload
# → http://localhost:8000/docs
```

### Training
```bash
cd ml
source venv/bin/activate
python train_herbs_mlflow.py
# → MLflow UI: http://localhost:5000
```

### CLIP Fallback (test)
```bash
cd ml
source venv/bin/activate
python clip_fallback.py data/raw/lavender/Image_1.jpg
# → 🌿 CLIP result: lavender (0.3000)
```

### iOS
Open `ios/TrailBotanist.xcodeproj` in Xcode → Run on device

---

## Medium Articles

1. [I Trained ResNet18 to Identify 70 Medicinal Herbs](https://medium.com/@rachana.gupta_7569/i-trained-resnet18-to-identify-70-medicinal-herbs-heres-what-i-learned-373bbc089055)
2. [I Built an ML App for Foragers in Forests. Server-Side Inference Would Have Killed It.](https://medium.com/p/8841c882ec2b)
3. [I Ran My Herb Classifier 4 Times. The Third Run Made It Worse. Here's Why.](https://medium.com/@rachana.gupta_7569)
4. [I Built a Production ML Feedback Loop with FastAPI and Render. Here's What Actually Broke.](https://medium.com/@rachana.gupta_7569/i-built-a-production-ml-feedback-loop-with-fastapi-and-render-heres-what-actually-broke-487bedc6bcae)

---

## What's Next

- [ ] CLIP fallback deployed to Render
- [ ] Run CLIP accuracy benchmark across all 71 herbs
- [ ] Article 5: ResNet vs CLIP vs Gemma — edge ML tradeoffs
- [ ] GitHub Actions: automated retraining on S3 correction threshold
- [ ] EfficientNet comparison for on-device accuracy improvement

---

## License

MIT

---

*Built by [Rachana Gupta](https://linkedin.com/in/rachana1101) — Apple Lead Software Engineer
