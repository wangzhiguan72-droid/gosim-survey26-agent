#!/usr/bin/env python3
"""Zip this project into a ZIP the platform can prepare and run as a complete project.

    python3 pack_agent.py [--out ../python-agent.zip]

Mirrors the platform's project rules (see observer.project.json and the hackathon's
docs): observer.project.json stays at the zip root with "protocol":
"jsonl-v4"; .env is never packed (the platform rejects ZIPs that contain one, and a
permanent key must never be uploaded -- set it on the Participate page instead);
__pycache__, run_output, .git and editor/OS clutter are skipped.

Standard library only.
"""
from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MANIFEST_NAME = "observer.project.json"
EXCLUDED_DIRS = {"__pycache__", ".git", ".venv", "venv", "run_output", ".pytest_cache", ".idea", ".vscode"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".zip"}
EXCLUDED_NAMES = {".DS_Store", "Thumbs.db"}
ENV_TEMPLATES = {".env.example", ".env.sample", ".env.template"}


def collect(root: Path) -> list[Path]:
    files = []
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        if any(part in EXCLUDED_DIRS for part in rel.parts):
            continue
        if not path.is_file() or path.name in EXCLUDED_NAMES or path.suffix in EXCLUDED_SUFFIXES:
            continue
        if path.name == ".env" or (path.name.startswith(".env.") and path.name not in ENV_TEMPLATES):
            continue
        files.append(path)
    return files


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=ROOT.parent / "python-agent.zip", help="output zip path")
    args = parser.parse_args(argv)

    manifest_path = ROOT / MANIFEST_NAME
    if not manifest_path.is_file():
        raise SystemExit(f"missing {MANIFEST_NAME} at {ROOT}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol") != "jsonl-v4":
        raise SystemExit(f"{MANIFEST_NAME} must declare \"protocol\": \"jsonl-v4\"")

    out = args.out.resolve()
    if ROOT in out.parents or out.parent == ROOT:
        raise SystemExit("write the zip outside this project folder, or it would package itself")
    out.parent.mkdir(parents=True, exist_ok=True)

    files = collect(ROOT)
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, arcname=path.relative_to(ROOT).as_posix())

    print(f"packed {len(files)} files from {ROOT} -> {out} ({out.stat().st_size} bytes)")
    print(f"  manifest: image {manifest['image']}, run {' '.join(manifest['run'])}")
    if (ROOT / ".env").is_file():
        print("  note: .env exists but was left out of the ZIP (never upload it; set your key on the Participate page)")
    print("  next: upload this ZIP as a complete project, or push this folder to a GitHub repository")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
