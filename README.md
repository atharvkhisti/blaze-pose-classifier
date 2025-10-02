# BlazePose Exercise Classification & Repetition Counting

Unified, cleaned README for the project. This repository implements:

* Dataset → landmarks → features → temporal windows → model training
* Real‑time multi‑model exercise classification (RF baseline / RF enhanced / XGBoost)
* Simple and strict (FSM + amplitude/velocity gated) repetition counting
* Adaptive rep thresholds & probability smoothing
* Research paper (IEEE format) under `paper/` (primary source: `clean_paper.tex` used for Overleaf)

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
## 2. Quick Real‑Time Inference (If Models Already Trained)
```powershell
./.venv/Scripts/python.exe src/realtime_inference.py --models models --show-angle
```
Interactive keys: `q` quit · `m` cycle model · `r` reset reps · `s` snapshot.

### Optional performance / UX flags
* Lower resolution: `--width 640 --height 360`
* Smooth predictions: `--smooth 5`
* Adaptive rep thresholds: `--adaptive-reps`
* Strict FSM reps: `--rep-mode strict --strict-reps`
* List webcams first: `--list-devices`

---
## 3. Per‑Exercise Run Commands (Force Counting)
The classifier normally auto‑detects the exercise; you can force rep counting mode for focused sessions / calibration. Each command below assumes models already exist in `models/` and uses angle overlay + adaptive thresholds.

| Exercise | Command (PowerShell) |
|----------|----------------------|
| Squats | `./.venv/Scripts/python.exe src/realtime_inference.py --models models --force-exercise squats --show-angle --adaptive-reps` |
| Lunges | `./.venv/Scripts/python.exe src/realtime_inference.py --models models --force-exercise lunges --show-angle --adaptive-reps` |
| Situps | `./.venv/Scripts/python.exe src/realtime_inference.py --models models --force-exercise situps --show-angle --adaptive-reps` |
| Pushups | `./.venv/Scripts/python.exe src/realtime_inference.py --models models --force-exercise pushups --show-angle --adaptive-reps --elbow-down-thresh 55 --elbow-up-thresh 160` |
| Jumping Jacks | `./.venv/Scripts/python.exe src/realtime_inference.py --models models --force-exercise jumping_jacks --show-angle --adaptive-reps` |
| Bicep Curls | `./.venv/Scripts/python.exe src/realtime_inference.py --models models --force-exercise bicep_curls --show-angle --adaptive-reps` |
| Tricep Extensions | `./.venv/Scripts/python.exe src/realtime_inference.py --models models --force-exercise tricep_extensions --show-angle --adaptive-reps` |
| Dumbbell Rows | `./.venv/Scripts/python.exe src/realtime_inference.py --models models --force-exercise dumbbell_rows --show-angle --adaptive-reps` |
| Dumbbell Shoulder Press | `./.venv/Scripts/python.exe src/realtime_inference.py --models models --force-exercise dumbbell_shoulder_press --show-angle --adaptive-reps` |
| Lateral Shoulder Raises | `./.venv/Scripts/python.exe src/realtime_inference.py --models models --force-exercise lateral_shoulder_raises --show-angle --adaptive-reps` |

Add strict counting (quality gating) by appending: `--rep-mode strict --strict-reps --min-rep-amplitude 25`.

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
## 5. Repetition Counting Modes
* **Simple** (default): lightweight threshold crossing (`RepCounter`).
* **Strict**: amplitude, velocity, dwell, separation checks (`AdvancedRepCounter`). Use `--rep-mode strict --strict-reps`.
* Adaptive thresholds automatically refine down/up angles with `--adaptive-reps` once sufficient range is observed.

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
| No models found | Re-run training step 4.4 |
| Low FPS | Lower resolution or add `--skip-n 1` |
| Few reps counted | Use `--adaptive-reps` or strict mode for form checking |
| Misclassifications early | Wait until window fills (`Window: 60/60`) |
| Black camera feed | Add `--auto-recover` or test with `--list-devices` |

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
