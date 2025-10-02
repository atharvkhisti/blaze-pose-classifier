# Requires: PowerShell
# Purpose: Create a Python 3.11 virtual environment and install requirements

Write-Host "[setup] Checking for Python 3.11 via py launcher..."
$py311 = & py -3.11 -c "import sys; print(sys.version)" 2>$null
if ($LASTEXITCODE -eq 0) {
  Write-Host "[setup] Found Python 3.11 via 'py -3.11'"
  & py -3.11 -m venv .venv
} else {
  Write-Host "[setup] 'py -3.11' not found. Trying common install path..."
  $common = "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe"
  if (Test-Path $common) {
    Write-Host "[setup] Using $common"
    & $common -m venv .venv
  } else {
    Write-Host "[setup][ERROR] Python 3.11 not found. Please install from https://www.python.org/downloads/release/python-3119/ (check 'Add to PATH')" -ForegroundColor Red
    exit 1
  }
}

Write-Host "[setup] Activating venv and upgrading pip/setuptools/wheel"
. .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel

Write-Host "[setup] Installing project requirements"
pip install -r requirements.txt

Write-Host "[setup] Done. To activate later, run: .\\.venv\\Scripts\\Activate.ps1"