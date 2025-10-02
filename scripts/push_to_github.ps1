<#
Requires: Git installed and available in PATH.

Usage examples:
  powershell -ExecutionPolicy Bypass -File .\scripts\push_to_github.ps1 -RemoteUrl "https://github.com/USER/REPO.git"
  powershell -File .\scripts\push_to_github.ps1 -RemoteUrl "git@github.com:USER/REPO.git" -Branch dev -CommitMessage "feat: add pipeline" -Force

Parameters:
  -RemoteUrl      (required) GitHub repository URL (HTTPS or SSH)
  -Branch         Branch name to use/create (default: main)
  -CommitMessage  Commit message for initial or latest staged changes
  -Force          Skip confirmation prompts
  -SkipModels     If set, will NOT stage models/ directory even if present

Script behavior:
  1. Initializes repo if .git not present
  2. Adds a sensible default .gitignore if missing
  3. Optionally excludes models/ when -SkipModels is used
  4. Adds remote 'origin' if not already defined
  5. Pushes to remote with upstream tracking

Note: You will be prompted for GitHub credentials / PAT on first HTTPS push.
#>

param(
    [Parameter(Mandatory=$true)] [string]$RemoteUrl,
    [string]$Branch = "main",
    [string]$CommitMessage = "Initial commit",
    [switch]$Force,
    [switch]$SkipModels
)

function Write-Step($msg) { Write-Host "[>] $msg" -ForegroundColor Cyan }
function Write-Warn($msg) { Write-Host "[!] $msg" -ForegroundColor Yellow }
function Write-Ok($msg)   { Write-Host "[OK] $msg" -ForegroundColor Green }

$ErrorActionPreference = 'Stop'

# 1. Repo init
if (-not (Test-Path .git)) {
    Write-Step "Initializing new git repository"
    git init | Out-Null
} else {
    Write-Step ".git already exists; skipping init"
}

# 2. Ensure branch
$currentBranch = git branch --show-current 2>$null
if (-not $currentBranch) {
    Write-Step "Creating and switching to branch $Branch"
    git checkout -b $Branch | Out-Null
} elseif ($currentBranch -ne $Branch) {
    Write-Step "Switching to branch $Branch"
    git checkout $Branch 2>$null | Out-Null || git checkout -b $Branch | Out-Null
}

# 3. Create .gitignore if missing
if (-not (Test-Path .gitignore)) {
    Write-Step "Creating .gitignore"
@'
__pycache__/
.venv/
*.py[cod]
*.log
dist/
build/
.mypy_cache/
.pytest_cache/
*.npy
*.npz
*.csv
*.tsv
*.parquet
data/
landmarks/
features/
results/
.ipynb_checkpoints/
*.aux
*.out
*.bbl
*.blg
*.toc
*.synctex.gz
*.fdb_latexmk
*.fls
Thumbs.db
.vscode/
.idea/
__pycache__/
'@ | Set-Content .gitignore -Encoding UTF8
    Write-Ok ".gitignore created"
}

# 4. Stage files
Write-Step "Staging files"
git add .

if ($SkipModels) {
    if (Test-Path models) {
        Write-Step "Unstaging models/ due to -SkipModels"
        git reset models | Out-Null
    }
}

# 5. Commit (only if changes present)
$status = git status --porcelain
if (-not $status) {
    Write-Warn "No changes to commit"
} else {
    Write-Step "Creating commit"
    git commit -m $CommitMessage | Out-Null
    Write-Ok "Committed: $CommitMessage"
}

# 6. Remote handling
$existingRemote = git remote 2>$null | Select-String -SimpleMatch origin
if (-not $existingRemote) {
    Write-Step "Adding remote origin -> $RemoteUrl"
    git remote add origin $RemoteUrl
} else {
    Write-Step "Remote 'origin' already set"
}

# 7. Confirm push
if (-not $Force) {
    $resp = Read-Host "Push branch '$Branch' to origin? (y/n)"
    if ($resp -notin 'y','Y') { Write-Warn "Aborted by user"; exit 1 }
}

Write-Step "Pushing to origin/$Branch"
try {
    git push -u origin $Branch
    Write-Ok "Push complete"
} catch {
    Write-Warn "Push failed: $($_.Exception.Message)"
    Write-Warn "If first HTTPS push, ensure you use a Personal Access Token as password."
    exit 2
}

Write-Ok "Repository is now on GitHub."
