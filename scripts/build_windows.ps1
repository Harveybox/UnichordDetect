# PowerShell script intended to be run on Windows (GitHub Actions or local dev)
# Usage: Open PowerShell (Admin if necessary) and run: .\build_windows.ps1

Write-Host "Starting Windows build for UnichordDetect"
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install pyinstaller

# Build single-file, no console window
pyinstaller --noconsole --onefile main.py

if (Test-Path -Path dist\main.exe) {
    Write-Host "Build succeeded: dist\main.exe"
} else {
    Write-Error "Build did not produce dist\\main.exe. Check PyInstaller output."
    exit 1
}
