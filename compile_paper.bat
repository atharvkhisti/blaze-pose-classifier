@echo off
echo Compiling LaTeX paper...
cd paper

:: Create a build directory for output files
if not exist build mkdir build

:: Run pdflatex twice to resolve references
pdflatex -output-directory=build main.tex
bibtex build/main
pdflatex -output-directory=build main.tex
pdflatex -output-directory=build main.tex

:: Copy the final PDF to the main paper directory
copy build\main.pdf BlazePoseClassifier_Paper.pdf

echo Paper compilation completed.
echo Output saved as paper\BlazePoseClassifier_Paper.pdf

pause