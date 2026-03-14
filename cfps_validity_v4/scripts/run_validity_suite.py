
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm


# -----------------------------
# Utilities
# -----------------------------
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def list_files(dir_path: Path, exts: Tuple[str, ...]) -> List[Path]:
    files: List[Path] = []
    for e in exts:
        files.extend(sorted(dir_path.glob(f"*{e}")))
    return sorted(files)


def stem_key(p: Path) -> str:
    s = p.stem
    digits = "".join([c for c in s if c.isdigit()])
    return digits if digits else s


def match_pairs(renders: List[Path], heatmaps: List[Path]) -> List[Tuple[Path, Path]]:
    hm_map = {stem_key(h): h for h in heatmaps}
    pairs: List[Tuple[Path, Path]] = []
    for r in renders:
        k = stem_key(r)
        if k in hm_map:
            pairs.append((r, hm_map[k]))
    return pairs


def read_rgb_uint8(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.asarray(img).astype(np.uint8)


def load_heatmap(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        hm = np.load(path).astype(np.float32)
    else:
        hm = (np.asarray(Image.open(path).convert("L")).astype(np.float32) / 255.0)
    hm = np.nan_to_num(hm, nan=0.0, posinf=0.0, neginf=0.0)
    return hm


def resize_heatmap_to_img(hm: np.ndarray, h: int, w: int) -> np.ndarray:
    if hm.shape[0] == h and hm.shape[1] == w:
        return hm

    # Portable resize using PIL: map to uint16 to keep gradients.
    if np.all(hm == hm.flat[0]):
        # constant map
        return np.full((h, w), float(hm.flat[0]), dtype=np.float32)

    # clip to reduce extreme outliers breaking scaling
    hi = float(np.percentile(hm, 99.9))
    if hi <= 0:
        hi = float(np.max(hm)) if float(np.max(hm)) != 0 else 1.0
    hm_clip = np.clip(hm, 0.0, hi)
    hm_u16 = (hm_clip / (np.max(hm_clip) + 1e-12) * 65535.0).astype(np.uint16)

    hm_img = Image.fromarray(hm_u16, mode="I;16").resize((w, h), resample=Image.BILINEAR)
    out = np.asarray(hm_img).astype(np.float32)
    out = out / (out.max() + 1e-12)
    return out


def instability_absdiff(curr: np.ndarray, prev: np.ndarray) -> float:
    diff = np.abs(curr.astype(np.float32) - prev.astype(np.float32))
    return float(np.mean(diff) / 255.0)


def overlay_image(img_rgb: np.ndarray, hm: np.ndarray, alpha: float) -> Image.Image:
    h = hm.astype(np.float32)
    h = h - h.min()
    h = h / (h.max() + 1e-12)

    cmap = plt.get_cmap("turbo") if "turbo" in plt.colormaps() else plt.get_cmap("jet")
    colored = (cmap(h)[..., :3] * 255.0).astype(np.uint8)

    base = Image.fromarray(img_rgb).convert("RGBA")
    heat = Image.fromarray(colored).convert("RGBA")

    a = (h * (255.0 * np.clip(alpha, 0.0, 1.0))).astype(np.uint8)
    heat.putalpha(Image.fromarray(a))
    return Image.alpha_composite(base, heat)


# -----------------------------
# Energy definitions (E_t)
# -----------------------------
def energy_mean(hm: np.ndarray) -> float:
    return float(np.mean(hm))


def energy_sum(hm: np.ndarray) -> float:
    return float(np.sum(hm))


def energy_l2(hm: np.ndarray) -> float:
    return float(np.sqrt(np.mean(hm * hm) + 1e-12))


def energy_p95(hm: np.ndarray) -> float:
    return float(np.percentile(hm, 95.0))


def energy_top_percent_mean(hm: np.ndarray, top_percent: float) -> float:
    p = np.clip(top_percent, 0.1, 100.0)
    thr = np.percentile(hm, 100.0 - p)
    mask = hm >= thr
    if not np.any(mask):
        return float(np.mean(hm))
    return float(np.mean(hm[mask]))


def energy_topk_mean(hm: np.ndarray, k: int) -> float:
    k = int(max(1, k))
    flat = hm.reshape(-1)
    if k >= flat.size:
        return float(np.mean(flat))
    idx = np.argpartition(flat, -k)[-k:]
    return float(np.mean(flat[idx]))


def energy_area_tau(hm: np.ndarray, tau: float) -> float:
    return float(np.mean(hm >= tau))


ENERGY_MODES = ("mean", "sum", "l2", "p95", "top_percent", "topk", "area_tau")


def compute_energy(hm: np.ndarray, mode: str, top_percent: float, topk: int, tau: float) -> float:
    if mode == "mean":
        return energy_mean(hm)
    if mode == "sum":
        return energy_sum(hm)
    if mode == "l2":
        return energy_l2(hm)
    if mode == "p95":
        return energy_p95(hm)
    if mode == "top_percent":
        return energy_top_percent_mean(hm, top_percent)
    if mode == "topk":
        return energy_topk_mean(hm, topk)
    if mode == "area_tau":
        return energy_area_tau(hm, tau)
    raise ValueError(f"Unknown energy_mode={mode}")


# -----------------------------
# Stats helpers
# -----------------------------
def safe_corr(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    if x.size < 3 or y.size < 3:
        return {"pearson_r": float("nan"), "pearson_p": float("nan"),
                "spearman_r": float("nan"), "spearman_p": float("nan")}
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return {"pearson_r": float("nan"), "pearson_p": float("nan"),
                "spearman_r": float("nan"), "spearman_p": float("nan")}
    pr, pp = pearsonr(x, y)
    sr, sp = spearmanr(x, y)
    return {"pearson_r": float(pr), "pearson_p": float(pp),
            "spearman_r": float(sr), "spearman_p": float(sp)}


def auc_rank(scores: np.ndarray, labels: np.ndarray) -> float:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)
    pos = scores[labels]
    neg = scores[~labels]
    if pos.size == 0 or neg.size == 0:
        return float("nan")

    order = scores.argsort()
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, scores.size + 1, dtype=np.float64)

    # tie handling: average rank for identical scores
    s = scores[order]
    i = 0
    while i < s.size:
        j = i + 1
        while j < s.size and s[j] == s[i]:
            j += 1
        if j - i > 1:
            avg = float(np.mean(ranks[order[i:j]]))
            ranks[order[i:j]] = avg
        i = j

    sum_ranks_pos = float(np.sum(ranks[labels]))
    n_pos = float(pos.size)
    n_neg = float(neg.size)
    u = sum_ranks_pos - n_pos * (n_pos + 1.0) / 2.0
    return float(u / (n_pos * n_neg))


def precision_recall_at_k(scores: np.ndarray, labels: np.ndarray, k: int) -> Dict[str, float]:
    scores = np.asarray(scores)
    labels = np.asarray(labels, dtype=bool)
    k = int(max(1, min(k, scores.size)))
    idx = np.argsort(scores)[::-1][:k]
    tp = int(np.sum(labels[idx]))
    fp = k - tp
    fn = int(np.sum(labels)) - tp
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    return {"precision": float(precision), "recall": float(recall)}


def likely_perframe_normalized(hm_stats: List[Dict[str, float]]) -> bool:
    if not hm_stats:
        return False
    mins = np.array([s["min"] for s in hm_stats], dtype=np.float64)
    maxs = np.array([s["max"] for s in hm_stats], dtype=np.float64)
    ok = (mins >= -1e-3) & (maxs <= 1.001)
    return float(np.mean(ok)) > 0.9


# -----------------------------
# Main
# -----------------------------
@dataclass
class SuiteConfig:
    top_percent: float = 10.0
    topk: int = 2000
    tau: float = 0.5
    overlay_stride: int = 10
    overlay_alpha: float = 0.45
    lag_max: int = 5
    event_quantile: float = 0.90


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--renders_dir", type=str, required=True)
    ap.add_argument("--heatmaps_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="results/validity_suite")

    ap.add_argument("--instability", type=str, default="absdiff", choices=["absdiff"])

    ap.add_argument("--energy_mode", type=str, default="top_percent", choices=list(ENERGY_MODES))
    ap.add_argument("--top_percent", type=float, default=10.0)
    ap.add_argument("--topk", type=int, default=2000)
    ap.add_argument("--tau", type=float, default=0.5)

    ap.add_argument("--overlay_stride", type=int, default=10)
    ap.add_argument("--overlay_alpha", type=float, default=0.45)

    ap.add_argument("--lag_max", type=int, default=5)
    ap.add_argument("--event_quantile", type=float, default=0.90)

    ap.add_argument("--sweep", action="store_true", help="Run sensitivity sweeps.")
    args = ap.parse_args()

    cfg = SuiteConfig(
        top_percent=args.top_percent,
        topk=args.topk,
        tau=args.tau,
        overlay_stride=args.overlay_stride,
        overlay_alpha=args.overlay_alpha,
        lag_max=args.lag_max,
        event_quantile=args.event_quantile,
    )

    renders_dir = Path(args.renders_dir)
    heatmaps_dir = Path(args.heatmaps_dir)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    overlay_dir = out_dir / "overlays"
    ensure_dir(overlay_dir)

    render_paths = list_files(renders_dir, (".png", ".jpg", ".jpeg"))
    heat_paths = list_files(heatmaps_dir, (".npy", ".png", ".jpg", ".jpeg"))
    pairs = match_pairs(render_paths, heat_paths)
    if len(pairs) < 2:
        raise SystemExit("Need at least 2 matched frames (render + heatmap).")

    # Load all (fast for small sequences) so we can do sweeps without re-reading disk.
    hm_stats: List[Dict[str, float]] = []
    imgs: List[np.ndarray] = []
    hms: List[np.ndarray] = []

    for (img_path, hm_path) in tqdm(pairs, desc="loading"):
        img = read_rgb_uint8(img_path)
        hm = load_heatmap(hm_path)
        hm = resize_heatmap_to_img(hm, img.shape[0], img.shape[1])
        imgs.append(img)
        hms.append(hm)
        hm_stats.append({"min": float(np.min(hm)), "max": float(np.max(hm)), "mean": float(np.mean(hm))})

    norm_flag = likely_perframe_normalized(hm_stats)

    # M_t computed once
    M = [instability_absdiff(imgs[i], imgs[i - 1]) for i in range(1, len(imgs))]
    M_arr = np.asarray(M, dtype=np.float64)

    # E_t aligned with M_t uses heatmap at time t (current frame)
    def compute_E(mode: str, top_percent: float, topk: int, tau: float) -> np.ndarray:
        E = [compute_energy(hms[i], mode, top_percent, topk, tau) for i in range(1, len(hms))]
        return np.asarray(E, dtype=np.float64)

    E_arr = compute_E(args.energy_mode, cfg.top_percent, cfg.topk, cfg.tau)

    # overlays (for current run only)
    if cfg.overlay_stride > 0:
        for i in range(1, len(imgs)):
            if (i % cfg.overlay_stride) == 0:
                out_img = overlay_image(imgs[i], hms[i], cfg.overlay_alpha)
                out_img.save(overlay_dir / f"overlay_{stem_key(pairs[i][0])}.png")

    corr = safe_corr(E_arr, M_arr)

    # events: top q of instability
    q = float(np.clip(cfg.event_quantile, 0.5, 0.99))
    labels = M_arr >= np.quantile(M_arr, q)
    auc = auc_rank(E_arr, labels)

    pr10 = precision_recall_at_k(E_arr, labels, k=max(5, int(0.10 * E_arr.size)))
    pr5 = precision_recall_at_k(E_arr, labels, k=max(3, int(0.05 * E_arr.size)))

    # lag correlation: E_t vs M_{t+k}
    lag_corrs: Dict[str, float] = {}
    for k in range(0, int(max(0, cfg.lag_max)) + 1):
        if k == 0:
            x, y = E_arr, M_arr
        else:
            if E_arr.size <= k:
                continue
            x, y = E_arr[:-k], M_arr[k:]
        c = safe_corr(x, y)
        lag_corrs[f"k={k}"] = float(c.get("pearson_r", float("nan")))

    stats = {
        "n_frames": int(M_arr.size),
        "instability": args.instability,
        "energy_mode": args.energy_mode,
        "energy_top_percent": float(cfg.top_percent),
        "energy_topk": int(cfg.topk),
        "energy_tau": float(cfg.tau),
        "likely_perframe_normalized_heatmap": bool(norm_flag),
        "corr": corr,
        "event_quantile": float(q),
        "auc_rank": float(auc),
        "precision_recall@10%": pr10,
        "precision_recall@5%": pr5,
        "lag_pearson_r": lag_corrs,
    }
    (out_dir / "stats_suite.json").write_text(json.dumps(stats, indent=2))

    # ---- plots ----
    # scatter
    plt.figure(figsize=(5.2, 5.2))
    plt.scatter(E_arr, M_arr, s=10, alpha=0.6)
    plt.xlabel("Heatmap score E_t")
    plt.ylabel("Temporal instability M_t")
    r0 = corr.get("pearson_r", float("nan"))
    plt.title(f"Diagnostics validity (Pearson r={r0:.2f})")
    plt.tight_layout()
    plt.savefig(out_dir / "scatter.pdf")
    plt.savefig(out_dir / "scatter.png", dpi=200)
    plt.close()

    # timeseries (z-score)
    def z(x: np.ndarray) -> np.ndarray:
        return (x - x.mean()) / (x.std() + 1e-12)

    plt.figure(figsize=(10, 3.2))
    plt.plot(z(E_arr), label="Heatmap score")
    plt.plot(z(M_arr), label="Instability")
    plt.legend()
    plt.title("Temporal alignment (z-score)")
    plt.tight_layout()
    plt.savefig(out_dir / "timeseries.pdf")
    plt.savefig(out_dir / "timeseries.png", dpi=200)
    plt.close()

    # lag plot
    ks, rs = [], []
    for k_str, r in lag_corrs.items():
        ks.append(int(k_str.split("=")[1]))
        rs.append(r)
    plt.figure(figsize=(6.2, 3.4))
    plt.plot(ks, rs, marker="o")
    plt.xlabel("Lag k (E_t vs M_{t+k})")
    plt.ylabel("Pearson r")
    plt.title("Lag correlation (early warning check)")
    plt.tight_layout()
    plt.savefig(out_dir / "lag_corr.pdf")
    plt.savefig(out_dir / "lag_corr.png", dpi=200)
    plt.close()

    # event retrieval plot
    labels_names = ["@5%", "@10%"]
    precisions = [pr5["precision"], pr10["precision"]]
    recalls = [pr5["recall"], pr10["recall"]]
    x = np.arange(len(labels_names))
    w = 0.35
    plt.figure(figsize=(6.2, 3.4))
    plt.bar(x - w / 2, precisions, width=w, label="Precision")
    plt.bar(x + w / 2, recalls, width=w, label="Recall")
    plt.xticks(x, labels_names)
    plt.ylim(0.0, 1.05)
    plt.title(f"Event retrieval (AUC≈{auc:.2f})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "event_retrieval.pdf")
    plt.savefig(out_dir / "event_retrieval.png", dpi=200)
    plt.close()

    # ---- sweeps (optional) ----
    if args.sweep:
        sweep_top = [1.0, 5.0, 10.0, 20.0]
        sweep_modes = ["mean", "sum", "l2", "p95", "top_percent", "area_tau"]
        sweep: Dict[str, Dict[str, float]] = {}

        for mode in sweep_modes:
            sweep[mode] = {}
            for tp in sweep_top:
                # only top_percent uses tp; others ignore but we still store in grid for easy plotting
                use_mode = "top_percent" if mode == "top_percent" else mode
                E = compute_E(use_mode, tp, cfg.topk, cfg.tau)
                c = safe_corr(E, M_arr)
                sweep[mode][f"tp={tp:g}"] = float(c.get("pearson_r", float("nan")))

        (out_dir / "sweep.json").write_text(json.dumps(sweep, indent=2))

        # plot top-percent sensitivity
        tps = sweep_top
        r_tp = [sweep["top_percent"][f"tp={tp:g}"] for tp in tps]
        plt.figure(figsize=(6.2, 3.4))
        plt.plot(tps, r_tp, marker="o")
        plt.xlabel("Top-percent p (mean over top p% pixels)")
        plt.ylabel("Pearson r")
        plt.title("Sensitivity to top-percent choice")
        plt.tight_layout()
        plt.savefig(out_dir / "sweep_top_percent.pdf")
        plt.savefig(out_dir / "sweep_top_percent.png", dpi=200)
        plt.close()

        # compare energy modes at default top_percent
        modes = sweep_modes
        r_modes = [sweep[m][f"tp={cfg.top_percent:g}"] for m in modes]
        plt.figure(figsize=(7.4, 3.6))
        plt.bar(np.arange(len(modes)), r_modes)
        plt.xticks(np.arange(len(modes)), modes, rotation=25, ha="right")
        plt.ylabel("Pearson r")
        plt.title("Which heatmap score works best")
        plt.tight_layout()
        plt.savefig(out_dir / "sweep_modes.pdf")
        plt.savefig(out_dir / "sweep_modes.png", dpi=200)
        plt.close()

    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
