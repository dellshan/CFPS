#!/usr/bin/env python3
"""
Generate 2D Failure Map: x=temporal instability, y=reconstruction error, color=M3b energy.
Usage:
    python make_failure_map.py \
        --recon results/3dgs_real/recon_lpips.npy \
        --instab results/3dgs_real/diag_pixel_lpips/instabilities.npy \
        --energy results/3dgs_real/diag_packet_coherent_lpips/energies.npy \
        --out fig/failure_map.pdf
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recon", required=True, help="recon_lpips.npy (LPIPS render vs GT)")
    parser.add_argument("--instab", required=True, help="instabilities.npy (LPIPS render_t vs render_{t-1})")
    parser.add_argument("--energy", required=True, help="energies.npy from M3b (packet coherent)")
    parser.add_argument("--out", default="failure_map.pdf")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    recon = np.load(args.recon)
    instab = np.load(args.instab)
    energy = np.load(args.energy)

    # Align lengths (instab may be 1 shorter due to pairwise diff)
    n = min(len(recon), len(instab), len(energy))
    recon = recon[:n]
    instab = instab[:n]
    energy = energy[:n]

    print(f"Frames: {n}")
    print(f"Recon  error: mean={recon.mean():.4f}, std={recon.std():.4f}")
    print(f"Instability:  mean={instab.mean():.4f}, std={instab.std():.4f}")
    print(f"M3b energy:   mean={energy.mean():.4f}, std={energy.std():.4f}")

    # Compute medians for quadrant lines
    med_x = np.median(instab)
    med_y = np.median(recon)

    fig, ax = plt.subplots(1, 1, figsize=(5.5, 4.5))

    sc = ax.scatter(
        instab, recon,
        c=energy, cmap="inferno", s=12, alpha=0.7, edgecolors="none",
        norm=Normalize(vmin=np.percentile(energy, 5),
                       vmax=np.percentile(energy, 95))
    )

    # Quadrant lines
    ax.axvline(med_x, color="gray", ls="--", lw=0.8, alpha=0.5)
    ax.axhline(med_y, color="gray", ls="--", lw=0.8, alpha=0.5)

    # Quadrant labels
    fs = 7.5
    # upper-left: low instab (x) + high error (y) = stable-but-wrong
    ax.text(0.02, 0.98, "High error\nLow flicker\n(stable-but-wrong)",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=fs, color="crimson", weight="bold", style="italic")
    # upper-right: high instab + high error = catastrophic
    ax.text(0.98, 0.98, "High error\nHigh flicker\n(catastrophic)",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=fs, color="0.4", style="italic")
    # lower-left: low instab + low error = healthy
    ax.text(0.02, 0.02, "Low error\nLow flicker\n(healthy)",
            transform=ax.transAxes, ha="left", va="bottom",
            fontsize=fs, color="0.4", style="italic")
    # lower-right: high instab + low error = flicker-only
    ax.text(0.98, 0.02, "Low error\nHigh flicker",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=fs, color="0.4", style="italic")

    ax.set_xlabel("Temporal instability  $D_t^{\\mathrm{temp}}$  (LPIPS)", fontsize=10)
    ax.set_ylabel("Reconstruction error  $D_t^{\\mathrm{recon}}$  (LPIPS)", fontsize=10)

    cb = fig.colorbar(sc, ax=ax, shrink=0.85, pad=0.02)
    cb.set_label("M3b (packet coherent) energy", fontsize=9)

    ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved → {args.out}")

    # Print quadrant stats
    q1 = (instab < med_x) & (recon >= med_y)  # low instab, high error = stable-but-wrong
    q2 = (instab >= med_x) & (recon >= med_y)  # high instab, high error = catastrophic
    q3 = (instab < med_x) & (recon < med_y)    # low instab, low error = healthy
    q4 = (instab >= med_x) & (recon < med_y)   # high instab, low error = flicker-only
    print(f"\nQuadrant distribution:")
    print(f"  Stable-but-wrong (low instab, high error): {q1.sum()} ({100*q1.mean():.1f}%)")
    print(f"  Catastrophic     (high instab, high error): {q2.sum()} ({100*q2.mean():.1f}%)")
    print(f"  Healthy          (low instab, low error):   {q3.sum()} ({100*q3.mean():.1f}%)")
    print(f"  Flicker-only     (high instab, low error):  {q4.sum()} ({100*q4.mean():.1f}%)")

    # M3b energy per quadrant
    print(f"\nMean M3b energy per quadrant:")
    print(f"  Stable-but-wrong: {energy[q1].mean():.4f}")
    print(f"  Catastrophic:     {energy[q2].mean():.4f}")
    print(f"  Healthy:          {energy[q3].mean():.4f}")
    print(f"  Flicker-only:     {energy[q4].mean():.4f}")


if __name__ == "__main__":
    main()