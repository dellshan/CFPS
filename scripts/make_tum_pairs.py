import argparse
from pathlib import Path
from PIL import Image

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tum_root", required=True, help="e.g., .../rgbd_dataset_freiburg1_360")
    ap.add_argument("--out_obs", required=True)
    ap.add_argument("--out_pred", required=True)
    ap.add_argument("--max_frames", type=int, default=800)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--resize_w", type=int, default=0)
    ap.add_argument("--resize_h", type=int, default=0)
    args = ap.parse_args()

    tum_root = Path(args.tum_root)
    rgb_dir = tum_root / "rgb"
    assert rgb_dir.exists(), f"Missing {rgb_dir}"

    out_obs = Path(args.out_obs); out_obs.mkdir(parents=True, exist_ok=True)
    out_pred = Path(args.out_pred); out_pred.mkdir(parents=True, exist_ok=True)

    imgs = sorted(rgb_dir.glob("*.png"))
    imgs = imgs[::args.stride][:args.max_frames]

    prev_img = None
    for i, p in enumerate(imgs):
        im = Image.open(p).convert("RGB")
        if args.resize_w > 0 and args.resize_h > 0:
            im = im.resize((args.resize_w, args.resize_h), Image.BILINEAR)

        name = f"{i:06d}.png"
        im.save(out_obs / name)

        if prev_img is None:
            im.save(out_pred / name)
        else:
            prev_img.save(out_pred / name)

        prev_img = im

    print(f"Done. obs={out_obs} pred={out_pred} N={len(imgs)}")

if __name__ == "__main__":
    main()
