"""Local artifact identity and reproducibility metadata. Never records secrets."""
from __future__ import annotations

import importlib.metadata
import platform
import subprocess
import sys
from pathlib import Path

from .data import sha256_file


MODEL_FILES = ["*.json", "*.safetensors", "*.txt", "tokenizer.model"]


def model_path(name: str) -> Path:
    path = Path(name)
    if path.is_dir():
        return path.resolve()
    from huggingface_hub import snapshot_download
    # Request only model/tokenizer assets, not missing README/.gitattributes files.
    return Path(snapshot_download(name, local_files_only=True, allow_patterns=MODEL_FILES))


def model_identity(name: str) -> dict:
    path = model_path(name)
    return {
        "name": name, "resolved_path": str(path),
        "snapshot_revision": path.name if path.parent.name == "snapshots" else None,
        "files": {str(p.relative_to(path)): sha256_file(p) for p in sorted(path.rglob("*"))
                  if p.is_file() and p.suffix in (".json", ".safetensors", ".txt", ".model")},
    }


def environment_identity() -> dict:
    root = Path(__file__).resolve().parents[2]
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True)
    return {
        "python": sys.version, "platform": platform.platform(),
        "packages": {name: importlib.metadata.version(name)
                     for name in ("mlx", "mlx-lm", "transformers", "huggingface-hub", "numpy")},
        "git_head_at_capture": git.stdout.strip() if git.returncode == 0 else None,
        # These hashes identify actual imported code, even in a frozen export or
        # when another commit is made during a long experiment.
        "source_sha256": {str(p.relative_to(root)): sha256_file(p)
                          for directory in ("src", "scripts")
                          for p in sorted((root / directory).rglob("*.py"))},
    }
