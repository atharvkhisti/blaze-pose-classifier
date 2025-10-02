#!/usr/bin/env python
"""
LaTeX paper compilation script for the BlazePose Classifier project.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

def compile_paper():
    """Compile the LaTeX paper using pdflatex and bibtex."""
    print("Compiling LaTeX paper...")
    
    # Change to paper directory
    os.chdir("paper")
    
    # Create a build directory if it doesn't exist
    build_dir = Path("build")
    build_dir.mkdir(exist_ok=True)
    
    # Define commands to run
    commands = [
        ["pdflatex", "-output-directory=build", "main.tex"],
        ["bibtex", "build/main"],
        ["pdflatex", "-output-directory=build", "main.tex"],
        ["pdflatex", "-output-directory=build", "main.tex"]
    ]
    
    # Run commands
    for cmd in commands:
        print(f"Running: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error during compilation: {e}")
            return False
    
    # Copy the final PDF to the paper directory
    output_file = build_dir / "main.pdf"
    final_file = Path("BlazePoseClassifier_Paper.pdf")
    
    if output_file.exists():
        shutil.copy(output_file, final_file)
        print(f"Paper compilation completed successfully.")
        print(f"Output saved as paper/{final_file.name}")
        return True
    else:
        print("Compilation did not produce a PDF file.")
        return False

if __name__ == "__main__":
    # Change to the script's directory
    os.chdir(Path(__file__).parent)
    
    if not compile_paper():
        sys.exit(1)