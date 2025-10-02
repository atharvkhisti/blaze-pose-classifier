# BlazePose Exercise Classifier

This project provides a comprehensive system for real-time exercise classification and repetition counting using Google's BlazePose model and machine learning techniques.

## Features

- **Real-time exercise classification** (squats, pushups, situps, jumping jacks, lunges)
- **Accurate repetition counting** with finite state machine approach
- **Feature engineering** to extract biomechanically relevant features
- **Model training and evaluation** pipeline with Random Forest and XGBoost
- **Visualization tools** for model performance and exercise analysis
- **Complete documentation** including an IEEE-format research paper

## Project Structure

```
blazepose_classifier/
│
├── blazepose_classifier.py     # Main classifier implementation
├── dataset_generator.py        # Tool to generate training datasets
├── model_evaluator.py          # Model training and evaluation
├── visualization.py            # Visualization utilities
├── deploy.py                   # Deployment script
├── setup.py                    # Setup utility
├── paper.tex                   # LaTeX paper documenting the system
│
├── figures/                    # Generated plots and visualizations
│   └── system_architecture.dot # System architecture diagram
│
├── results/                    # Evaluation results and LaTeX tables
└── models/                     # Trained model files
```

## Installation

### Requirements

- Python 3.7+
- OpenCV
- MediaPipe
- NumPy
- Pandas
- scikit-learn
- Matplotlib
- Seaborn
- XGBoost
- NetworkX

### Setup

1. Clone this repository
2. Run the setup script:

```bash
python setup.py
```

This will:
- Check if your Python version is compatible
- Install required packages
- Verify camera access
- Test MediaPipe functionality
- Create necessary directories
- Optionally create sample data and model files

## Usage

### 1. Generate Training Data

To create a dataset from your webcam:

```bash
python dataset_generator.py
```

This will guide you through recording samples for each exercise type.

### 2. Train and Evaluate Models

Once you've collected your dataset, train and evaluate classification models:

```bash
python model_evaluator.py
```

This will:
- Train a Random Forest classifier
- Train a Gradient Boosting classifier
- Perform hyperparameter tuning
- Generate evaluation metrics and plots
- Export results to LaTeX format
- Save the best performing model

### 3. Run the Exercise Classifier

To use the classifier with your webcam:

```bash
python deploy.py --display
```

Additional options:
```bash
python deploy.py --help
```

### 4. Visualization

Generate additional visualizations:

```bash
python visualization.py
```

## System Architecture

The system consists of the following components:

1. **Pose Estimation**: Uses BlazePose to extract 33 landmarks from each video frame
2. **Feature Extraction**: Converts raw landmarks into biomechanically meaningful features
3. **Classification**: Machine learning models that identify the exercise being performed
4. **Repetition Counting**: Finite state machine that tracks exercise states and counts reps
5. **Visualization**: Real-time display with exercise type, rep count, and pose overlay

## Feature Engineering

The system extracts several categories of features from the BlazePose landmarks:

1. **Static Joint Angles**: Key angles between body segments
2. **Dynamic Features**: Velocity and acceleration of joint movements
3. **Posture Features**: Body orientation and alignment
4. **Anthropometric Features**: Body proportion measurements
5. **Temporal Features**: Patterns across time windows
6. **Relative Position Features**: Spatial relationships between joints

## Repetition Counting

The system uses a finite state machine (FSM) approach to count exercise repetitions:

- STARTING: Initial position detection
- TOP: Upper position of the exercise
- BOTTOM: Lower position of the exercise

The FSM automatically adapts thresholds for different exercise types.

## Original Pipeline Implementation

This repository also includes the original pipeline implementation for extracting landmarks, engineering features, and training models:

### Quickstart (Windows PowerShell)

1) Create a virtual environment and install dependencies

```
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2) Prepare data folders and labels
- Put your videos under `data/mmfit_videos/` and/or `data/kaggle_pushup/`
- Create `data/video_labels.csv` with columns:
```
video,exercise,form
subject1_squat1.mp4,squat,correct
subject2_biceps1.mp4,biceps,incorrect
```

3) Run the pipeline step-by-step

- Extract landmarks:
```
py src\extract_landmarks.py --input data\mmfit_videos data\kaggle_pushup --output landmarks
```

- Feature engineering (per-frame):
```
py src\feature_engineering.py --landmarks landmarks\mmfit_videos_landmarks.csv --out features\mmfit_features.pkl
```

- Windowing + labels:
```
py src\windowing.py --features features\mmfit_features.pkl --labels data\video_labels.csv --out features\mmfit_windows.pkl
```

- Train models and produce results:
```
py src\train_models.py --windows features\mmfit_windows.pkl --results results --models models
```

- Rep counting (example):
```
py src\rep_counting.py --features features\mmfit_features.pkl --video <video_name.mp4> --exercise biceps
```

### Using MM-Fit pose_2d.npy instead of videos

If your `data\mmfit_videos` contains MM-Fit pose arrays (no `.mp4` videos), convert them first:

```
py src\adapters\pose2d_to_landmarks.py --input "data\mmfit_videos\mm-fit\mm-fit" --output landmarks --name mmfit_pose2d_landmarks.csv --schema auto --stride 2
```

## Documentation

A complete paper documenting the system is included in `paper.tex`. To compile it:

```bash
pdflatex paper.tex
bibtex paper
pdflatex paper.tex
pdflatex paper.tex
```

## Contributing

Contributions are welcome! Some potential areas for improvement:

- Add support for more exercise types
- Implement more sophisticated repetition counting
- Create a graphical user interface (GUI)
- Add exercise form feedback and corrections
- Support for custom exercise definitions

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- Google's MediaPipe for the BlazePose model
- scikit-learn team for the machine learning framework
- The MM-Fit dataset for exercise videos