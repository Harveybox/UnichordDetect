# PowerShell script intended to be run on Windows (GitHub Actions or local dev)
# Usage: Open PowerShell (Admin if necessary) and run: .\build_windows.ps1

Write-Host "Starting Windows build for UnichordDetect"
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install pyinstaller

# Build GUI single-file (no console window)
pyinstaller --noconsole --onefile --name unichord_gui main.py

# Build console-enabled diagnostic binary
pyinstaller --onefile --name unichord_console main.py

if (Test-Path -Path dist\unichord_console.exe) {
    Write-Host "Build succeeded: dist\unichord_console.exe (console) and dist\unichord_gui.exe (GUI)"
} else {
    Write-Error "Build did not produce dist\\unichord_console.exe. Check PyInstaller output."
    exit 1
}
