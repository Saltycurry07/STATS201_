import argparse
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm


def predict_one(net, img_path: Path) -> float:
    img = cv2.imread(str(img_path))
    if img is None:
        return None

    img = cv2.resize(img, (224, 224))
    blob = cv2.dnn.blobFromImage(
        img,
        scalefactor=1.0,
        size=(224, 224),
        mean=(104, 117, 123),
        swapRB=False,
        crop=False,
    )
    net.setInput(blob)
    out = net.forward()
    return float(out.squeeze())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=None, help="Input CSV filename (default: auto-detect the only .csv in folder)")
    parser.add_argument("--out", default="predictions.csv", help="Output CSV filename")
    parser.add_argument("--path_col", default="local_path", help="Column name for local image path")
    parser.add_argument("--name_col", default="name", help="Column name for person name")
    args = parser.parse_args()

    ROOT = Path(__file__).resolve().parent

    prototxt = ROOT / "resnet18_deploy.prototxt"
    caffemodel = ROOT / "resnet18.caffemodel"

    if not prototxt.exists():
        raise FileNotFoundError(f"Missing model file: {prototxt}")
    if not caffemodel.exists():
        raise FileNotFoundError(f"Missing model file: {caffemodel}")

    # 读取 CSV：如果你不传 --csv，就自动找当前文件夹里唯一的 .csv
    if args.csv is None:
        csv_candidates = sorted([p for p in ROOT.glob("*.csv") if p.name != args.out])
        if len(csv_candidates) != 1:
            raise FileNotFoundError(
                f"Cannot auto-detect CSV. Found {len(csv_candidates)} csv files: {[p.name for p in csv_candidates]}\n"
                f"Please run with --csv YOUR_FILE.csv"
            )
        csv_path = csv_candidates[0]
    else:
        csv_path = ROOT / args.csv
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing CSV: {csv_path}")

    df = pd.read_csv(csv_path)

    required = {args.name_col, args.path_col}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"CSV must contain columns {sorted(required)}; got {list(df.columns)}")

    net = cv2.dnn.readNetFromCaffe(str(prototxt), str(caffemodel))

    scores = []
    missing = 0
    read_fail = 0

    for rel in tqdm(df[args.path_col].astype(str).tolist(), desc="Scoring"):
        img_path = (ROOT / rel).resolve()
        if not img_path.exists():
            scores.append(None)
            missing += 1
            continue

        s = predict_one(net, img_path)
        if s is None:
            scores.append(None)
            read_fail += 1
        else:
            scores.append(s)

    df["pred_score_raw"] = scores
    df["pred_score_1_5"] = pd.to_numeric(df["pred_score_raw"], errors="coerce").clip(1.0, 5.0)

    out_path = ROOT / args.out
    df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print("DONE.")
    print("Input CSV:", csv_path.name)
    print("Saved:", out_path.name)
    print("Missing images:", missing)
    print("Read fail:", read_fail)


if __name__ == "__main__":
    main()
