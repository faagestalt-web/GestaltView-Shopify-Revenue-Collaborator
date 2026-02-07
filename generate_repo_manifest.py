#!/usr/bin/env python3
"""Generate repo_manifest.json for GestaltView Sidekick Studio.

Usage:
  python scripts/generate_repo_manifest.py --root .. --out ../repo_manifest.json
  python scripts/generate_repo_manifest.py --no-hash   (faster)
"""
from __future__ import annotations
import argparse, os, json, hashlib, mimetypes
from pathlib import Path
from datetime import datetime
import re

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()

def build_manifest(root: Path, include_hash: bool = True) -> dict:
    # discover files (exclude venv / pycache)
    records = []
    for p in root.rglob('*'):
        if not p.is_file():
            continue
        rel = p.relative_to(root).as_posix()
        if rel.startswith('.venv/') or '/.venv/' in rel or '__pycache__' in rel:
            continue
        st = p.stat()
        rec = {
            "path": rel,
            "size_bytes": st.st_size,
            "modified_utc": datetime.utcfromtimestamp(st.st_mtime).isoformat() + "Z",
            "ext": p.suffix.lower(),
            "mime": mimetypes.guess_type(rel)[0] or "application/octet-stream",
        }
        if include_hash:
            rec["sha256"] = sha256_file(p)
        records.append(rec)

    # entrypoints (best effort)
    entrypoints = []
    for candidate in [
        "gestaltview-sidekick-starter/backend/app/main.py",
        "gestaltview-sidekick-starter/frontend/src/main.tsx",
        "gestaltview-sidekick-starter/docker-compose.yml",
        "gestaltview-sidekick-starter/docker-compose.dev.yml",
    ]:
        if (root / candidate).exists():
            entrypoints.append(candidate)

    # lightweight keyword index from paths
    kw = {}
    for r in records:
        toks = re.split(r'[/\._\-]+', r["path"].lower())
        for t in toks:
            if not t or len(t) < 3:
                continue
            if t in {"src","app","json","md","tsx","ts","py","yml","yaml","lock","test","spec","main"}:
                continue
            kw.setdefault(t, set()).add(r["path"])
    kw = {k: sorted(list(v)) for k,v in kw.items() if len(v) <= 40}

    return {
        "$schema": "https://gestaltview.ai/schemas/repo_manifest.v1.json",
        "manifest_version": "1.0.0",
        "generated_utc": datetime.utcnow().isoformat() + "Z",
        "repo": {
            "name": "GestaltView Sidekick Studio (Starter) - Enhanced",
            "root": root.name,
            "entrypoints": entrypoints,
        },
        "file_index": {"total_files": len(records), "files": records},
        "keyword_index": {"note": "Path-derived keywords. For semantic search, run the Manifest Index Layer.", "keywords": kw},
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='..', help='Repo root (default: .. from scripts/)') 
    ap.add_argument('--out', default=None, help='Output path (default: <root>/repo_manifest.json)')
    ap.add_argument('--no-hash', action='store_true', help='Skip SHA256 for speed')
    args = ap.parse_args()

    root = Path(args.root).resolve()
    out = Path(args.out).resolve() if args.out else (root / "repo_manifest.json")
    manifest = build_manifest(root, include_hash=not args.no_hash)
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f"Wrote {out} (files: {manifest['file_index']['total_files']})")

if __name__ == '__main__':
    main()
