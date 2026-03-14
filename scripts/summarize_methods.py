import argparse, json
from pathlib import Path

METHODS = [
    ("pixel",           "M0: Pixel residual"),
    ("fft_amp",         "M1: Spectral amplitude"),
    ("fft_phase",       "M2: Phase coherence"),
    ("packet_scalar",   "M3a: Packet (scalar)"),
    ("packet_coherent", "M3b: Packet (coherent)"),
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_base", required=True)
    ap.add_argument("--instability", default="lpips", choices=["absdiff","lpips"])
    args = ap.parse_args()
    base = Path(args.results_base)
    print(f"\n{'='*70}")
    print(f"  Method Comparison — instability={args.instability}")
    print(f"{'='*70}\n")
    header = f"{'Method':<28} {'Pearson r':>10} {'Spearman p':>12} {'Hit@Top10%':>12}"
    print(header); print("-" * len(header))
    rows = []
    for key, label in METHODS:
        sp = base / f"diag_{key}_{args.instability}" / "stats.json"
        if not sp.exists():
            print(f"{label:<28} {'N/A':>10} {'N/A':>12} {'N/A':>12}"); continue
        s = json.loads(sp.read_text())
        pr, sr, hit = s["pearson_r"], s["spearman_r"], s["top10_event_hit_rate"]
        print(f"{label:<28} {pr:>10.4f} {sr:>12.4f} {hit:>12.4f}")
        rows.append((label, pr, sr, hit))
    print(f"\n--- LaTeX table ---\n")
    print(r"\begin{table}[t]")
    print(r"\caption{Residual representation comparison (" + args.instability + r" instability).}")
    print(r"\centering\small")
    print(r"\begin{tabular}{@{}lccc@{}}")
    print(r"\toprule")
    print(r"Method & Pearson $r$ & Spearman $\rho$ & Hit@Top-10\% \\")
    print(r"\midrule")
    for label, pr, sr, hit in rows:
        print(f"{label} & {pr:.3f} & {sr:.3f} & {hit:.3f} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\label{tab:method_comparison}")
    print(r"\end{table}")

if __name__ == "__main__":
    main()
