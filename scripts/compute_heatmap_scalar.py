import argparse
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm

def scalar_heatmap_from_residual(res2d: np.ndarray, tile: int, stride: int) -> np.ndarray:
    """
    res2d: HxW scalar residual (mean abs diff)
    returns: HxW heatmap aggregated from overlapping tiles (NO per-frame normalization)
    """
    H, W = res2d.shape
    ii = np.pad(res2d, ((1, 0), (1, 0)), mode="constant").cumsum(0).cumsum(1)

    diff = np.zeros((H + 1, W + 1), dtype=np.float32)
    cnt  = np.zeros((H + 1, W + 1), dtype=np.float32)

    for y in range(0, H - tile + 1, stride):
        y2 = y + tile
        for x in range(0, W - tile + 1, stride):
            x2 = x + tile
            s = ii[y2, x2] - ii[y, x2] - ii[y2, x] + ii[y, x]

            diff[y,  x ] += s
            diff[y2, x ] -= s
            diff[y,  x2] -= s
            diff[y2, x2] += s

            cnt[y,  x ] += 1
            cnt[y2, x ] -= 1
            cnt[y,  x2] -= 1
            cnt[y2, x2] += 1

    hm = diff.cumsum(0).cumsum(1)[:H, :W]
    c  = cnt.cumsum(0).cumsum(1)[:H, :W]
    hm = hm / np.maximum(c, 1.0)
    return hm.astype(np.float32)

def load_rgb_float(path: Path) -> np.ndarray:
    im = Image.open(path).convert("RGB")
    arr = np.asarray(im).astype(np.float32) / 255.0
    return arr

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", required=True)
    ap.add_argument("--obs_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--tile", type=int, default=32)
    ap.add_argument("--stride", type=int, default=16)
    args = ap.parse_args()

    pred_dir = Path(args.pred_dir)
    obs_dir  = Path(args.obs_dir)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    obs_imgs = sorted(obs_dir.glob("*.png"))
    if len(obs_imgs) == 0:
        raise SystemExit(f"No png found in {obs_dir}")

    n_save = 0
    for op in tqdm(obs_imgs, desc="scalar-heatmaps"):
        pp = pred_dir / op.name
        if not pp.exists():
            continue

        obs = load_rgb_float(op)
        pred = load_rgb_float(pp)

        if obs.shape != pred.shape:
            raise SystemExit(f"Shape mismatch: {op} {obs.shape} vs {pp} {pred.shape}")

        res = np.mean(np.abs(obs - pred), axis=2).astype(np.float32)  # HxW
        hm = scalar_heatmap_from_residual(res, tile=args.tile, stride=args.stride)

        np.save(out_dir / (op.stem + ".npy"), hm)
        n_save += 1

    print(f"Done. out={out_dir} saved={n_save}")

if __name__ == "__main__":
    main()
