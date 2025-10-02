# BlazePose Exercise Classification and Repetition Counting System

This repository contains the implementation and research paper for a BlazePose-based exercise classification and repetition counting system.

## Research Paper

The research paper is located in the `paper/` directory and is written in LaTeX using the IEEE conference format. The paper documents the methodology, implementation, and results of the BlazePose-based exercise classification system.

### Paper Structure

- `paper/main.tex`: The main LaTeX document
- `paper/references.bib`: Bibliography file with references
- `paper/figures/`: Directory containing figure files for the paper

### Compiling the Paper

#### Windows
```
compile_paper.bat
```

#### Cross-platform (Python)
```
python compile_paper.py
```

The compiled PDF will be saved as `paper/BlazePoseClassifier_Paper.pdf`.

## System Implementation

The implementation of the BlazePose classifier system includes:

- `blazepose_classifier.py`: Main classifier implementation
- `dataset_generator.py`: Tools for generating training data
- `model_evaluator.py`: Evaluation scripts for the classifier
- `visualization.py`: Visualization utilities

## Requirements

To run the classifier:
```
pip install -r requirements.txt
```

To compile the LaTeX paper:
- LaTeX distribution (e.g., TeX Live, MiKTeX)
- pdflatex
- bibtex

## Citation

If you use this work, please cite:
```
@article{blazepose_classifier,
  title={BlazePose-Based Exercise Classification and Repetition Counting System},
  author={Khisti, Atharv},
  journal={},
  year={2023}
}
```