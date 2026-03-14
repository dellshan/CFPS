from __future__ import annotations

import numpy as np
from pathlib import Path
from PIL import Image

from _common import ensure_dir


def main():
    out = Path("data")
    ensure_dir(out / "renders")
    ensure_dir(out / "heatmaps")

    H, W = 360, 640
    rng = np.random.default_rng(0)

    # synthetic: generate a moving edge + noise bursts that correlate with heatmap
    for t in range(60):
        img = np.zeros((H, W, 3), dtype=np.uint8)
        x0 = 50 + t * 4
        img[:, :x0, :] = 40
        img[:, x0:x0+3, :] = 240
        img += rng.integers(0, 5, size=img.shape, dtype=np.uint8)

        hm = np.zeros((H, W), dtype=np.float32)
        hm[:, max(0, x0-30):min(W, x0+30)] = 1.0

        # inject random bursts (simulate instability)
        if t % 12 == 0:
            y = rng.integers(0, H-60)
            x = rng.integers(0, W-80)
            hm[y:y+60, x:x+80] += 2.0
            img[y:y+60, x:x+80, :] = 255

        hm = hm - hm.min()
        hm = hm / (hm.max() + 1e-12)

        Image.fromarray(img).save(out / "renders" / f"{t:06d}.png")
        np.save(out / "heatmaps" / f"{t:06d}.npy", hm)

    print("Demo data written to data/renders and data/heatmaps")


if __name__ == "__main__":
    main()
