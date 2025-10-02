# BlazePose Exercise Classification & Repetition Counting

Production‑style realtime exercise classification + rep counting with:

* Landmark extraction → feature engineering → temporal windows → model training
* Multiple models (RF baseline, RF enhanced, XGBoost) with optional filtering
* Strict rep logic (amplitude, velocity, dwell) + adaptive thresholds
* Jitter suppression & angle smoothing to avoid false reps from small noise
* Per‑exercise JSON configs for one‑command launches
* Research paper sources under `paper/` (`clean_paper.tex` primary)

---
## 1. Environment (Windows PowerShell)
```powershell
py -3.11 -m venv .venv
./.venv/Scripts/Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

> Already have the `.venv` from earlier work? Just run the last two lines.

---
## 2. Quick Real‑Time Inference (Models Present)
Simplest run (auto exercise detection, all models):
```powershell
python src/realtime_inference.py --models models --show-angle --adaptive-reps --smooth 5
```
Interactive keys: `q` quit · `m` cycle model · `r` reset reps · `s` snapshot.

### New / Important Flags
| Flag | Purpose |
|------|---------|
| `--config <file>` | Load JSON defaults (command line overrides) |
| `--model-include rf_enhanced,xgb_enhanced` | Only load specific models |
| `--force-exercise bicep_curls` | Skip classifier for rep counting of chosen exercise |
| `--rep-mode strict --strict-reps` | Enable advanced repetition gating |
| `--adaptive-reps` | Learn personalized down/up angle thresholds |
| `--smooth 5` | Majority vote smoothing over last 5 predictions |
| `--angle-jitter-thresh 2.0` | Ignore tiny angle changes below this delta (deg) |
| `--min-still-frames 4` | Frames of low motion before accepting small drift |
| `--elbow-down-thresh / --elbow-up-thresh` | Override elbow thresholds (pushups/curls) |
| `--skip-n 1` | Process every other frame for speed |
| `--auto-recover` | Attempt backend switch if feed stalls |

### Minimal + Fast (XGBoost only)
```powershell
python src/realtime_inference.py --models models --model-include xgb_enhanced --smooth 5 --adaptive-reps
```

### Raw Preview (debug camera)
```powershell
python src/realtime_inference.py --raw-preview --device 0 --width 640 --height 360
```

---
## 3. Per‑Exercise One‑Command Configs
Use JSON configs instead of long CLI strings. Example for bicep curls:
```powershell
python src/realtime_inference.py --config configs/bicep_curls.json
```
Each config sets: forced exercise, strict mode, amplitude threshold, smoothing, jitter suppression, model filtering, resolution, adaptive reps, angle overlay.

Available config files:
```
configs/bicep_curls.json
configs/pushups.json
configs/squats.json
configs/lunges.json
configs/jumping_jacks.json
configs/situps.json
configs/dumbbell_rows.json
configs/dumbbell_shoulder_press.json
configs/lateral_shoulder_raises.json
configs/tricep_extensions.json
```

Override anything ad‑hoc:
```powershell
python src/realtime_inference.py --config configs/pushups.json --angle-jitter-thresh 3.0 --min-rep-amplitude 28
```

---
## 4. Full Data → Model Pipeline
Only needed if you are re‑training.

### 4.1 Extract Landmarks from Videos
```powershell
./.venv/Scripts/python.exe src/extract_landmarks.py --input data/mmfit_videos data/kaggle_pushup --output landmarks --name mmfit_videos_landmarks.csv --sample-rate 2
```

### 4.2 Frame Feature Engineering (angles / ratios)
```powershell
./.venv/Scripts/python.exe src/feature_engineering.py --landmarks landmarks/mmfit_videos_landmarks.csv --out features/mmfit_features_labeled.pkl
```

### 4.3 Sliding Window Aggregation
```powershell
./.venv/Scripts/python.exe src/windowing.py --features features/mmfit_features_labeled.pkl --out features/mmfit_windows.pkl --window-size 60 --stride 8
```

### 4.4 Train Random Forest Baseline + Enhanced
```powershell
./.venv/Scripts/python.exe src/train_models.py --windows features/mmfit_windows.pkl --results results --models models --seed 42
```

### 4.5 Train XGBoost (Optional)
```powershell
./.venv/Scripts/python.exe src/train_xgboost.py --windows features/mmfit_windows.pkl --models models --results results --variant enhanced --seed 42
```

---
## 5. Repetition Counting & Noise Control
| Mode | Description | When to Use |
|------|-------------|-------------|
| Simple | Fast threshold crossing (single angle) | Quick demos, low noise |
| Strict | FSM + amplitude + velocity + hold + separation | Quality / form focus |

Enhancers:
* `--adaptive-reps`: Personalizes down/up after enough motion span is observed.
* Jitter suppression: Freezes angle if delta < `--angle-jitter-thresh` until `--min-still-frames` passes.
* Angle EMA smoothing: `--angle-smooth-alpha` (default 0.3).

---
## 6. Project Layout (Simplified)
```
src/                # Source pipeline & realtime inference
models/             # Trained model artifacts (.pkl + metadata json)
features/           # Generated feature/window pickles
landmarks/          # Raw landmark CSV(s)
results/            # Reports, confusion matrices, session logs, snapshots
paper/              # Research paper (use clean_paper.tex for Overleaf)
README.md           # (this file)
requirements.txt    # Pinned dependencies
```
Legacy / exploratory scripts retained (e.g. `blazepose_classifier.py`, `deploy.py`, `dataset_generator.py`)—they are not required for the main pipeline but kept for reference. Remove or archive if you no longer need them.

> A helper script `scripts/push_to_github.ps1` can automate initial git init + push.

---
## 7. Housekeeping / Cleanup Suggestions
Already performed: merged duplicate documentation; removed obsolete `README_new.md`.

Optional (manual) if you want an even leaner tree:
* Move legacy one-off scripts (`dataset_generator.py`, `model_evaluator.py`, `deploy.py`) into a `legacy/` folder.
* Keep only `clean_paper.tex` inside `paper/` for the manuscript; archive others.

---
## 8. Troubleshooting Cheatsheet
| Issue | Hint |
|-------|------|
| No models found | Re-run training (section 4) |
| First prediction slow / freeze | Large RF models spinning threads; try `--model-include xgb_enhanced` |
| False tiny reps | Increase `--angle-jitter-thresh` or `--min-rep-amplitude` |
| Reps not counted | Lower `--min-rep-amplitude` or disable strict mode |
| Misclassification early | Wait for full window (`Window: 60/60`) |
| Black / frozen feed | Use `--auto-recover` or lower resolution |
| High CPU | Add `--skip-n 1` or restrict models |
| Elbow thresholds off | Tune `--elbow-down-thresh / --elbow-up-thresh` |

---
## 9. Paper
If you need a local compile (even though you use Overleaf):
```powershell
pdflatex clean_paper.tex
bibtex clean_paper
pdflatex clean_paper.tex
pdflatex clean_paper.tex
```

---
## 10. License & Acknowledgements
See paper acknowledgements and dependencies; MediaPipe BlazePose, scikit-learn, XGBoost, MM-Fit dataset.

---
**Enjoy building with an interpretable, real‑time pose exercise system!**
