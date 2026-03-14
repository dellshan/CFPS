# CFPS Diagnostics Validity (v4)

Lightweight, iHPC-friendly validity suite for your "Explainable Frequency Heatmap" diagnostics.

## Inputs
- `data/renders/*.png`   rendered RGB frames
- `data/heatmaps/*.npy`  float heatmaps OR `*.png` grayscale heatmaps

## Outputs
- `scatter.(pdf/png)`      E_t vs M_t
- `timeseries.(pdf/png)`   z-scored alignment over time
- `lag_corr.(pdf/png)`     does E_t predict future instability?
- `event_retrieval.(pdf/png)` AUC + precision/recall bars
- `stats_suite.json`       numbers for paper
- `overlays/*.png`         qualitative overlay frames

## Install (no conda env required)
```bash
python -m pip install --user -r requirements_min.txt
```

## Run
```bash
python scripts/run_validity_suite.py \
  --renders_dir data/renders \
  --heatmaps_dir data/heatmaps \
  --out_dir results/validity_suite \
  --instability absdiff \
  --energy_mode top_percent \
  --top_percent 10 \
  --overlay_stride 10 \
  --overlay_alpha 0.45 \
  --lag_max 5 \
  --event_quantile 0.90 \
  --sweep
```

## Pro tip
If your heatmaps are per-frame min-max normalized to [0,1], `top-1% mean` can become nearly constant.
Then use `--energy_mode mean` or `--energy_mode area_tau` (and tune `--tau`).
