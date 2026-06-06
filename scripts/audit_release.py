from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

TEXT_SUFFIXES = {
    ".bib",
    ".cff",
    ".csv",
    ".json",
    ".md",
    ".py",
    ".tex",
    ".txt",
    ".yml",
    ".yaml",
}

FORBIDDEN_PATTERNS = [
    re.compile(r"/Users/catmushroomliu"),
    re.compile(r"\bDesktop\b"),
    re.compile(r"\bDownloads\b"),
    re.compile(r"<org-or-user>"),
    re.compile(r"will be deposited", re.IGNORECASE),
]

ALLOWLIST = {
    "FIGURE_MANIFEST.md",
    "scripts/audit_release.py",
}

EXCLUDED_DIRS = {".git", "__pycache__", ".ipynb_checkpoints"}
EXCLUDED_SUFFIXES = {".pyc"}
MANIFEST_PATH = Path("metadata/release_file_manifest.tsv")


def iter_files() -> list[Path]:
    files: list[Path] = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(ROOT)
        if rel == MANIFEST_PATH:
            continue
        if any(part in EXCLUDED_DIRS for part in rel.parts):
            continue
        if path.suffix.lower() in EXCLUDED_SUFFIXES:
            continue
        files.append(path)
    return sorted(files)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def scan_text(path: Path) -> list[str]:
    if path.suffix.lower() not in TEXT_SUFFIXES:
        return []
    rel = str(path.relative_to(ROOT))
    if path.name in ALLOWLIST or rel in ALLOWLIST:
        return []
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return [f"{path.relative_to(ROOT)}: cannot decode as UTF-8"]
    hits: list[str] = []
    for i, line in enumerate(text.splitlines(), start=1):
        for patt in FORBIDDEN_PATTERNS:
            if patt.search(line):
                hits.append(f"{path.relative_to(ROOT)}:{i}: {line.strip()[:220]}")
                break
    return hits


def main() -> int:
    files = iter_files()
    large = [p for p in files if p.stat().st_size > 100 * 1024 * 1024]
    text_hits: list[str] = []
    for path in files:
        text_hits.extend(scan_text(path))

    manifest = ROOT / MANIFEST_PATH
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", encoding="utf-8") as handle:
        handle.write("relative_path\tsize_bytes\tsha256\n")
        for path in files:
            handle.write(f"{path.relative_to(ROOT)}\t{path.stat().st_size}\t{sha256(path)}\n")

    print(f"Release root: {ROOT}")
    print(f"Files scanned: {len(files)}")
    print(f"Manifest written: {manifest.relative_to(ROOT)}")

    if large:
        print("\nFiles larger than 100 MB:")
        for path in large:
            print(f"  {path.relative_to(ROOT)} ({path.stat().st_size} bytes)")

    if text_hits:
        print("\nForbidden local/private patterns:")
        for hit in text_hits[:200]:
            print(f"  {hit}")
        if len(text_hits) > 200:
            print(f"  ... {len(text_hits) - 200} more")

    if large or text_hits:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
