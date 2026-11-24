"""Summarize errors and inspect representative held-out camera/CAN pairs."""
import json

import pandas as pd
from PIL import Image

from experiment import metrics, plt
from preprocess import DATA, OUT, read_can


def plot_cases(cases, path):
    fig, axes = plt.subplots(len(cases), 3, figsize=(12, 2.8 * len(cases)),
                             squeeze=False, layout="constrained")
    for row, (image_ax, error_ax, signal_ax) in zip(cases.itertuples(), axes):
        folder = DATA / "Test" / row.record
        key = f"{row.frame:04d}"
        with Image.open(folder / f"frame_{key}.jpg") as frame:
            image_ax.imshow(frame)
        image_ax.set_title(f"{row.type} | {row.record} / {key} | score {row.score:.3f}")
        image_ax.axis("off")
        can = read_can(folder / f"can_{key}.csv")
        time = can["Time"] - can["Time"].min()
        error_ax.plot(time, can["LKA_error"], color="tab:purple", marker="." if len(can) == 1 else None)
        error_ax.axhline(0.25, color="gray", linestyle=":", label="Label cutoff")
        error_ax.axhline(row.lka_error_mean, color="tab:green", linestyle="--", label="Window mean")
        error_ax.set(xlabel="Window time (s)", ylabel="LKA_error")
        error_ax.legend(fontsize=8)
        for side in ("left", "right"):
            signal_ax.plot(time, can[f"op_ll_{side}_prob"], label=side,
                           marker="." if len(can) == 1 else None)
        signal_ax.set(xlabel="Window time (s)", ylabel="Lane-line confidence", ylim=(-0.03, 1.03))
        signal_ax.legend(fontsize=8)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    summary = json.loads((OUT / "evaluation.json").read_text(encoding="utf-8"))
    family = summary["primary"].rsplit(" ", 1)[0]
    slug = family.lower().replace(" ", "_")
    threshold = summary["test"][family]["threshold"]
    errors = pd.read_csv(OUT / "primary_errors.csv").assign(
        score=lambda data: data[f"{slug}_score"],
    )
    predictions = pd.read_csv(OUT / "test_predictions.csv")
    frames = pd.read_csv(OUT / "features.csv")
    frames["utc_day"] = pd.to_datetime(frames["wall_start"], unit="s", utc=True).dt.strftime("%Y-%m-%d")
    merged = predictions.merge(frames.loc[frames["split"].eq("Test"),
                                          ["record", "frame", "utc_day", "raw_rows"]],
                               on=["record", "frame"], validate="one_to_one")
    merged["short_window"] = merged["raw_rows"].eq(1)
    strata = []
    for dimension in ("utc_day", "short_window"):
        for value, part in merged.groupby(dimension):
            y = part["label"].eq(2)
            if y.nunique() == 2:
                strata.append({
                    "dimension": dimension, "value": str(value), "frames": len(part),
                    "positives": int(y.sum()), "clips": part["record"].nunique(),
                    **metrics(y, part[f"{slug}_score"], threshold),
                })
    pd.DataFrame(strata).to_csv(OUT / "test_subgroups.csv", index=False)
    cols = ["lka_error_mean", "label_margin", "score", "op_ll_left_prob", "op_ll_right_prob",
            "op_curvature_desired", "curvature_gap", "torque_sd"]
    errors.groupby("type")[cols].median().to_csv(OUT / "error_medians.csv")
    errors.groupby(["type", "record"]).size().rename("count").to_csv(OUT / "errors_by_clip.csv")
    picks = []
    for kind, part in errors.groupby("type"):
        close = part.nsmallest(1, "label_margin")
        wrong = part.loc[part["record"].ne(close.iloc[0]["record"])].assign(
            score_distance=lambda data: (data["score"] - threshold).abs(),
        ).nlargest(1, "score_distance")
        picks.extend([close.assign(example="near label cutoff"), wrong.assign(example="confident error")])
    if picks:
        cases = pd.concat(picks, ignore_index=True)
        cases[["type", "example", "record", "frame", *cols]].to_csv(OUT / "error_examples.csv", index=False)
        plot_cases(cases, OUT / "error_examples.png")
    print(pd.crosstab([frames.split, frames.utc_day], frames.label).to_string())
    print(pd.DataFrame(strata).to_string(index=False))
    print(errors.groupby("type")[cols].median().to_string())
    print("Label margin at most 0.05:")
    print(errors.assign(near=errors.label_margin.le(0.05)).groupby("type").near.agg(["sum", "count"]))


if __name__ == "__main__":
    main()
