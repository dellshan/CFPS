from __future__ import annotations
import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm
from _common import ensure_dir, list_images, match_pairs

@dataclass
class Config:
    mode: str = "fft_amp"
    packet_reduce: str = "coherent"
    tile: int = 64
    stride: int = 32
    beta: float = 0.5
    eps: float = 1e-6
    q_power: float = 1.0

def read_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB")).astype(np.float32) / 255.0

def to_luminance(img: np.ndarray) -> np.ndarray:
    return 0.2989 * img[..., 0] + 0.5870 * img[..., 1] + 0.1140 * img[..., 2]

def hann2d(h: int, w: int) -> np.ndarray:
    return np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)

def radial_weight(h: int, w: int, power: float = 1.0) -> np.ndarray:
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    r = np.sqrt(fx * fx + fy * fy)
    r = r / (r.max() + 1e-12)
    return (r ** power).astype(np.float32)

def tile_score_pixel(obs_tile, pred_tile, cfg):
    return float(np.mean(np.abs(obs_tile - pred_tile)))

def tile_score_fft_amp(obs_g, pred_g, win, q, cfg):
    Fo = np.fft.fft2(obs_g * win)
    Fp = np.fft.fft2(pred_g * win)
    amp_residual = np.abs(np.log(np.abs(Fp) + cfg.eps) - np.log(np.abs(Fo) + cfg.eps))
    return float(np.sum(q * amp_residual))

def tile_score_fft_phase(obs_g, pred_g, win, q, cfg):
    Fo = np.fft.fft2(obs_g * win)
    Fp = np.fft.fft2(pred_g * win)
    denom = (np.abs(Fp) + cfg.eps) * (np.abs(Fo) + cfg.eps)
    coh = np.real(Fp * np.conj(Fo) / denom)
    phase_residual = 1.0 - coh
    return float(np.sum(q * phase_residual))

def tile_score_packet(obs_g, pred_g, win, q, cfg):
    Fo = np.fft.fft2(obs_g * win)
    Fp = np.fft.fft2(pred_g * win)
    A = np.abs(np.log(np.abs(Fp) + cfg.eps) - np.log(np.abs(Fo) + cfg.eps))
    denom = (np.abs(Fp) + cfg.eps) * (np.abs(Fo) + cfg.eps)
    C = Fp * np.conj(Fo) / denom
    Z = q * A * C
    if cfg.packet_reduce == "coherent":
        return float(np.abs(np.sum(Z)))
    else:
        return float(np.sum(np.abs(Z)))

def compute_heatmap(obs, pred, cfg):
    assert obs.shape == pred.shape
    H, W, _ = obs.shape
    tile = cfg.tile
    stride = cfg.stride
    win = hann2d(tile, tile)
    q = radial_weight(tile, tile, cfg.q_power)
    if cfg.mode != "pixel":
        obs_g = to_luminance(obs)
        pred_g = to_luminance(pred)
    score_acc = np.zeros((H + 1, W + 1), dtype=np.float64)
    count_acc = np.zeros((H + 1, W + 1), dtype=np.float64)
    for y in range(0, max(1, H - tile + 1), stride):
        for x in range(0, max(1, W - tile + 1), stride):
            y2, x2 = y + tile, x + tile
            if y2 > H or x2 > W:
                continue
            if cfg.mode == "pixel":
                s = tile_score_pixel(obs[y:y2, x:x2], pred[y:y2, x:x2], cfg)
            elif cfg.mode == "fft_amp":
                s = tile_score_fft_amp(obs_g[y:y2, x:x2], pred_g[y:y2, x:x2], win, q, cfg)
            elif cfg.mode == "fft_phase":
                s = tile_score_fft_phase(obs_g[y:y2, x:x2], pred_g[y:y2, x:x2], win, q, cfg)
            elif cfg.mode == "packet":
                s = tile_score_packet(obs_g[y:y2, x:x2], pred_g[y:y2, x:x2], win, q, cfg)
            else:
                raise ValueError(f"Unknown mode: {cfg.mode}")
            score_acc[y, x] += s;  score_acc[y2, x] -= s
            score_acc[y, x2] -= s; score_acc[y2, x2] += s
            count_acc[y, x] += 1.0;  count_acc[y2, x] -= 1.0
            count_acc[y, x2] -= 1.0; count_acc[y2, x2] += 1.0
    hm = score_acc.cumsum(0).cumsum(1)[:H, :W]
    cnt = count_acc.cumsum(0).cumsum(1)[:H, :W]
    hm = hm / np.maximum(cnt, 1.0)
    return hm.astype(np.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["pixel","fft_amp","fft_phase","packet"])
    ap.add_argument("--packet_reduce", default="coherent", choices=["coherent","scalar"])
    ap.add_argument("--obs_dir", required=True)
    ap.add_argument("--pred_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--tile", type=int, default=64)
    ap.add_argument("--stride", type=int, default=32)
    ap.add_argument("--q_power", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=0.5)
    args = ap.parse_args()
    cfg = Config(mode=args.mode, packet_reduce=args.packet_reduce,
                 tile=args.tile, stride=args.stride, q_power=args.q_power, beta=args.beta)
    obs_dir = Path(args.obs_dir); pred_dir = Path(args.pred_dir); out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    pairs = match_pairs(list_images(obs_dir), list_images(pred_dir))
    if not pairs:
        raise SystemExit("No matched obs/pred pairs found.")
    energies = []
    for obs_path, pred_path in tqdm(pairs, desc=f"heatmap [{cfg.mode}]"):
        obs = read_rgb(obs_path); pred = read_rgb(pred_path)
        hm = compute_heatmap(obs, pred, cfg)
        np.save(out_dir / f"{obs_path.stem}.npy", hm)
        energies.append(float(np.mean(hm)))
    np.save(out_dir / "energies_frame.npy", np.array(energies, dtype=np.float32))
    meta = asdict(cfg); meta["n_frames"] = len(pairs)
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Done. mode={cfg.mode} packet_reduce={cfg.packet_reduce} frames={len(pairs)} out={out_dir}")

if __name__ == "__main__":
    main()
