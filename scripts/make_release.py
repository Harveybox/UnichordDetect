"""
Create a distributable zip for UnichordDetect.
The archive contains source code and docs for offline sharing.
"""
from __future__ import annotations

import argparse
import pathlib
import zipfile
from typing import Iterable

ROOT = pathlib.Path(__file__).resolve().parent.parent
DIST = ROOT / "dist"


def iter_files() -> Iterable[pathlib.Path]:
    includes = [
        "README.md",
        "RELEASE.md",
        "requirements.txt",
        "VERSION",
        "main.py",
    ]
    include_dirs = ["audio_capture", "chord_estimator", "timeline", "ui"]
    for rel in includes:
        path = ROOT / rel
        if path.exists():
            yield path
    for folder in include_dirs:
        for path in (ROOT / folder).rglob("*.py"):
            yield path


def create_archive(version: str) -> pathlib.Path:
    DIST.mkdir(exist_ok=True)
    zip_path = DIST / f"UnichordDetect-{version}.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in iter_files():
            arcname = path.relative_to(ROOT)
            zf.write(path, arcname)
    return zip_path


def read_version() -> str:
    version_file = ROOT / "VERSION"
    return version_file.read_text().strip()


def main(version: str | None = None) -> None:
    resolved_version = version or read_version()
    archive = create_archive(resolved_version)
    print(f"Created release archive: {archive}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create UnichordDetect release archive")
    parser.add_argument("--version", help="Override version (default: read VERSION)")
    args = parser.parse_args()
    main(args.version)
