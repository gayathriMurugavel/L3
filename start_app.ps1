$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonPath = Join-Path $projectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $pythonPath)) {
    Write-Error "Project virtual environment not found at .venv\Scripts\python.exe"
    exit 1
}

Set-Location $projectRoot
& $pythonPath -m streamlit run app.py