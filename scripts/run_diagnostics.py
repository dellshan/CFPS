from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

from _common import ensure_dir, list_images, list_heatmaps, match_pairs


@dataclass
class Config:
    top_percent: float = 10.0
    overlay_stride: int = 10
    overlay_alpha: float = 0.45
    eps: float = 1e-12


def read_rgb_uint8(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.asarray(img).astype(np.uint8)


def load_heatmap(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        hm = np.load(path)
        hm = hm.astype(np.float32)
    else:
        hm = np.asarray(Image.open(path).convert("L")).astype(np.float32) / 255.0
    # sanitize
    hm = np.nan_to_num(hm, nan=0.0, posinf=0.0, neginf=0.0)
    return hm


def heatmap_energy(hm: np.ndarray, top_percent: float) -> float:
    # Use mean of top p% pixels (robust and interpretable)
    p = np.clip(top_percent, 0.1, 100.0)
    thr = np.percentile(hm, 100.0 - p)
    mask = hm >= thr
    if not np.any(mask):
        return float(np.mean(hm))
    return float(np.mean(hm[mask]))


def instability_absdiff(curr: np.ndarray, prev: np.ndarray) -> float:
    # curr/prev uint8 RGB
    diff = np.abs(curr.astype(np.float32) - prev.astype(np.float32))
    return float(np.mean(diff) / 255.0)


class LPIPSMetric:
    def __init__(self, net: str = "alex"):
        import torch
        import lpips
        self.torch = torch
        self.loss_fn = lpips.LPIPS(net=net)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loss_fn = self.loss_fn.to(self.device)
        self.loss_fn.eval()

    def __call__(self, curr: np.ndarray, prev: np.ndarray) -> float:
        # uint8 RGB -> [-1,1] tensor
        torch = self.torch
        def to_t(x):
            t = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0
            return t.to(self.device)
        with torch.no_grad():
            v = self.loss_fn(to_t(curr), to_t(prev)).item()
        return float(v)


def overlay_image(img_rgb: np.ndarray, hm: np.ndarray, alpha: float) -> Image.Image:
    # Normalize heatmap
    h = hm.astype(np.float32)
    h = h - h.min()
    h = h / (h.max() + 1e-12)

    # Use matplotlib colormap (jet-like without requiring cv2)
    cmap = plt.get_cmap("turbo") if "turbo" in plt.colormaps() else plt.get_cmap("jet")
    colored = (cmap(h)[..., :3] * 255.0).astype(np.uint8)  # RGB

    base = Image.fromarray(img_rgb).convert("RGBA")
    heat = Image.fromarray(colored).convert("RGBA")

    # alpha mask proportional to heat
    a = (h * (255.0 * np.clip(alpha, 0.0, 1.0))).astype(np.uint8)
    heat.putalpha(Image.fromarray(a))

    return Image.alpha_composite(base, heat)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--renders_dir", type=str, required=True)
    ap.add_argument("--heatmaps_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="results")
    ap.add_argument("--instability", type=str, default="absdiff", choices=["absdiff", "lpips"])
    ap.add_argument("--top_percent", type=float, default=10.0)
    ap.add_argument("--overlay_stride", type=int, default=10)
    ap.add_argument("--overlay_alpha", type=float, default=0.45)
    args = ap.parse_args()

    cfg = Config(top_percent=args.top_percent, overlay_stride=args.overlay_stride, overlay_alpha=args.overlay_alpha)

    renders_dir = Path(args.renders_dir)
    heatmaps_dir = Path(args.heatmaps_dir)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    overlay_dir = out_dir / "overlays"
    ensure_dir(overlay_dir)

    render_paths = list_images(renders_dir)
    heat_paths = list_heatmaps(heatmaps_dir)
    pairs = match_pairs(render_paths, heat_paths)
    if len(pairs) < 2:
        raise SystemExit("Need at least 2 matched frames (render + heatmap) to compute temporal instability.")

    metric_lpips = LPIPSMetric() if args.instability == "lpips" else None

    E: List[float] = []
    M: List[float] = []

    prev_img = None
    prev_stem = None

    for idx, (img_path, hm_path) in enumerate(tqdm(pairs, desc="diagnostics")):
        img = read_rgb_uint8(img_path)
        hm = load_heatmap(hm_path)

        # Resize heatmap if needed
        if hm.shape[0] != img.shape[0] or hm.shape[1] != img.shape[1]:
            hm = np.asarray(Image.fromarray((hm * 255).astype(np.uint8)).resize((img.shape[1], img.shape[0]), resample=Image.BILINEAR)).astype(np.float32) / 255.0

        e = heatmap_energy(hm, cfg.top_percent)

        if prev_img is None:
            prev_img = img
            prev_stem = img_path.stem
            continue

        if metric_lpips is not None:
            m = metric_lpips(img, prev_img)
        else:
            m = instability_absdiff(img, prev_img)

        E.append(e)
        M.append(m)

        # overlays
        if cfg.overlay_stride > 0 and (idx % cfg.overlay_stride == 0):
            out_img = overlay_image(img, hm, cfg.overlay_alpha)
            out_img.save(overlay_dir / f"overlay_{img_path.stem}.png")

        prev_img = img
        prev_stem = img_path.stem

    E_arr = np.asarray(E, dtype=np.float64)
    M_arr = np.asarray(M, dtype=np.float64)

    np.save(out_dir / 'energies.npy', np.asarray(E_arr, dtype=np.float32))
    np.save(out_dir / 'instabilities.npy', np.asarray(M_arr, dtype=np.float32))
    pearson_r, pearson_p = pearsonr(E_arr, M_arr)
    spearman_r, spearman_p = spearmanr(E_arr, M_arr)

    # hit-rate: top 10% instability frames, how many are also top 10% heatmap energy?
    q = 0.90
    event = M_arr >= np.quantile(M_arr, q)
    pred = E_arr >= np.quantile(E_arr, q)
    hit_rate = float(np.sum(event & pred) / max(1, np.sum(event)))

    stats = {
        "n_frames": int(len(E_arr)),
        "instability": args.instability,
        "energy_top_percent": float(cfg.top_percent),
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman_r": float(spearman_r),
        "spearman_p": float(spearman_p),
        "top10_event_hit_rate": hit_rate,
    }

    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2))

    # scatter
    plt.figure(figsize=(5.2, 5.2))
    plt.scatter(E_arr, M_arr, s=10, alpha=0.6)
    plt.xlabel("Heatmap energy E_t")
    plt.ylabel("Temporal instability M_t")
    plt.title(f"Diagnostics validity (Pearson r={pearson_r:.2f})")
    plt.tight_layout()
    plt.savefig(out_dir / "scatter.pdf")
    plt.savefig(out_dir / "scatter.png", dpi=200)
    plt.close()

    # timeseries (z-score)
    def z(x):
        return (x - x.mean()) / (x.std() + cfg.eps)

    plt.figure(figsize=(10, 3.2))
    plt.plot(z(E_arr), label="Heatmap energy")
    plt.plot(z(M_arr), label="Instability")
    plt.legend()
    plt.title("Temporal alignment")
    plt.tight_layout()
    plt.savefig(out_dir / "timeseries.pdf")
    plt.savefig(out_dir / "timeseries.png", dpi=200)
    plt.close()

    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
