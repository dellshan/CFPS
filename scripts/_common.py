from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def index_by_stem(paths: List[Path]) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for p in paths:
        out[p.stem] = p
    return out


def match_pairs(a_paths: List[Path], b_paths: List[Path]) -> List[Tuple[Path, Path]]:
    a = index_by_stem(a_paths)
    b = index_by_stem(b_paths)
    keys = sorted(set(a.keys()) & set(b.keys()), key=natural_key)
    return [(a[k], b[k]) for k in keys]


def list_images(folder: Path) -> List[Path]:
    if not folder.exists():
        return []
    items = [p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS and p.is_file()]
    return sorted(items, key=lambda p: natural_key(p.name))


def list_heatmaps(folder: Path) -> List[Path]:
    if not folder.exists():
        return []
    items = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in {".npy", ".png", ".jpg", ".jpeg"}]
    return sorted(items, key=lambda p: natural_key(p.name))


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
