"""Build one LKA example per matched frame and audit the supplied split."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "TESLA_AP3_MODEL3_v1"
OUT = HERE / "results"
FLAGS = ["op_lanes_visible", "op_lane_left_visible", "op_lane_right_visible"]
SIGNALS = [
    "vEgo", "op_state_steer_torque", "op_curvature_desired",
    "op_curvature_actual", "op_ll_left_prob", "op_ll_right_prob", *FLAGS,
]
FEATURES = {
    "A": SIGNALS,
    "B": SIGNALS + ["torque_sd"],
    "C": SIGNALS + ["torque_sd", "confidence_gap", "curvature_gap"],
}
READ = [*SIGNALS, "Time", "wallTimeCentiseconds", "LKA_error"]


def read_can(path):
    raw = pd.read_csv(path, usecols=READ, dtype=str)
    raw[FLAGS] = raw[FLAGS].replace({"True": "1", "False": "0"})
    for column in raw:
        raw[column] = pd.to_numeric(raw[column], errors="coerce").astype(float)
    return raw.replace([np.inf, -np.inf], np.nan)


def summarize(split, record, index, paths):
    row = {
        "split": split, "record": record, "frame": index,
        **{f"has_{name}": name in paths for name in ("can", "frame", "binary", "instance")},
    }
    if not row["has_can"]:
        row["status"] = "no_can"
        return row
    raw = read_can(paths["can"])
    with paths["can"].open("rb") as stream:
        row["can_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
    row["raw_rows"] = len(raw)
    row["out_of_order"] = int(raw["Time"].diff().lt(0).sum())
    # Only collapse identical projected observations. Keep different readings at the same time.
    clean = raw.drop_duplicates().sort_values("Time", kind="stable")
    row["duplicate_rows"] = len(raw) - len(clean)
    row["duplicate_time"] = int(clean["Time"].dropna().duplicated().sum())
    row["wall_start"] = clean["wallTimeCentiseconds"].min() / 100
    row["wall_end"] = clean["wallTimeCentiseconds"].max() / 100
    row["span_s"] = clean["Time"].max() - clean["Time"].min()
    row["invalid_label_rows"] = int(raw["LKA_error"].isna().sum())
    # The course defines the target as the mean over the source CAN record.
    row["lka_error_mean"] = raw["LKA_error"].mean()
    row["status"] = "ok"
    if not row["has_frame"]:
        row["status"] = "no_frame"
    elif pd.isna(row["lka_error_mean"]):
        row["status"] = "no_valid_label"
    row.update(clean[SIGNALS].mean().to_dict())
    row["torque_sd"] = clean["op_state_steer_torque"].std(ddof=0)
    row["confidence_gap"] = abs(row["op_ll_left_prob"] - row["op_ll_right_prob"])
    row["curvature_gap"] = (clean["op_curvature_desired"] - clean["op_curvature_actual"]).abs().mean()
    return row


def overlap_flags(reference, query):
    windows = reference.dropna(subset=["wall_start", "wall_end"]).sort_values("wall_start")
    starts = windows["wall_start"].to_numpy()
    ends = np.maximum.accumulate(windows["wall_end"].to_numpy())
    position = np.searchsorted(starts, query["wall_end"], side="right") - 1
    overlap = np.zeros(len(query), dtype=bool)
    valid = (position >= 0) & query["wall_start"].notna().to_numpy()
    overlap[valid] = ends[position[valid]] >= query["wall_start"].to_numpy()[valid]
    return overlap


def main(data, out):
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for split in ("Train", "Test"):
        for folder in sorted((data / split).glob("record*")):
            by_index = {}
            for path in folder.iterdir():
                kind, _, index = path.stem.partition("_")
                if kind in {"can", "frame", "binary", "instance"} and index.isdigit():
                    by_index.setdefault(int(index), {})[kind] = path
            rows.extend(summarize(split, folder.name, index, paths)
                        for index, paths in sorted(by_index.items()))
        print(f"{split}: {len(rows)} cumulative frame keys", flush=True)
    audit = pd.DataFrame(rows)
    audit.to_csv(out / "frame_audit.csv", index=False)
    frames = audit.loc[audit["status"].eq("ok")].copy()
    frames["label"] = np.where(frames["lka_error_mean"].ge(0.25), 2, 1)
    train = frames.loc[frames["split"].eq("Train")]
    test = frames.loc[frames["split"].eq("Test")]
    assert len(train) and len(test) and set(train["record"]).isdisjoint(test["record"])
    train_overlap = overlap_flags(test, train) | train["can_sha256"].isin(test["can_sha256"]).to_numpy()
    # Drop the whole training clip when any of its windows overlaps the held-out set.
    drop_records = set(train.loc[train_overlap, "record"])
    frames["use_in_training"] = frames["split"].eq("Train") & ~frames["record"].isin(drop_records)
    frames.to_csv(out / "features.csv", index=False)
    stats = {
        "excluded_train_records": sorted(drop_records),
        "directly_overlapping_train_frames": int(train_overlap.sum()),
        "splits": {},
    }
    for split in ("Train", "Test"):
        part = audit.loc[audit["split"].eq(split)]
        valid = frames.loc[frames["split"].eq(split)]
        stats["splits"][split] = {
            "records": part["record"].nunique(),
            "counts": {key: int(part[f"has_{key}"].sum()) for key in ("can", "frame", "binary", "instance")},
            "status": part["status"].value_counts().to_dict(),
            "labels": valid["label"].value_counts().to_dict(),
            "raw_can_rows": int(part["raw_rows"].sum()),
            "duplicate_rows": int(part["duplicate_rows"].sum()),
            "duplicate_time": int(part["duplicate_time"].sum()),
            "out_of_order": int(part["out_of_order"].sum()),
            "invalid_label_rows": int(part["invalid_label_rows"].sum()),
            "missing_clock_frames": int(valid[["wall_start", "wall_end"]].isna().any(axis=1).sum()),
            "missing_features": valid[FEATURES["C"]].isna().sum().to_dict(),
            "window_span_s": valid["span_s"].quantile([0, 0.5, 1]).to_dict(),
            "within_record_overlapping_pairs": int(sum(
                g["wall_start"].iloc[1:].le(g["wall_end"].iloc[:-1].to_numpy()).sum()
                for _, g in valid.groupby("record")
            )),
        }
    (out / "data_audit.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    main(args.data, args.out)
