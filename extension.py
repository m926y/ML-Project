"""Compare ratio and concentration features with the original grouped validation."""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve
from sklearn.model_selection import GridSearchCV, cross_val_predict
from sklearn.pipeline import make_pipeline
from threadpoolctl import threadpool_limits
from xgboost import XGBClassifier

from experiment import SEED, alert_threshold, candidates, metrics, plt, time_groups
from preprocess import DATA, FEATURES, OUT


SOURCE_COLUMNS = [
    "Time", "op_ll_left_prob", "op_ll_right_prob", "op_curvature_desired",
    "op_curvature_actual", "op_state_steer_torque",
]
RATIOS = ["lane_left_share", "relative_curvature_gap", "relative_curvature_bias"]
CONCENTRATIONS = ["lane_confidence_hhi", "torque_concentration", "curvature_gap_concentration"]
COLUMNS = {
    "C": FEATURES["C"],
    "C+R": FEATURES["C"] + RATIOS,
    "C+K": FEATURES["C"] + CONCENTRATIONS,
    "C+R+K": FEATURES["C"] + RATIOS + CONCENTRATIONS,
}
OUTPUT = OUT / "extension"


def temporal_concentration(amplitude):
    # Equivalent to 1 minus the fraction of effective active observations.
    amplitude = amplitude.dropna().abs()
    if len(amplitude) < 4:
        return np.nan
    second_moment = amplitude.pow(2).mean()
    return float(np.clip(1 - amplitude.mean() ** 2 / second_moment, 0, 1)) if second_moment else 0.0


def extra_features(path):
    raw = pd.read_csv(path, usecols=SOURCE_COLUMNS, dtype=float).replace([np.inf, -np.inf], np.nan)
    raw = raw.drop_duplicates()
    left = raw["op_ll_left_prob"].between(0, 1)
    right = raw["op_ll_right_prob"].between(0, 1)
    confidence = raw.loc[left & right, ["op_ll_left_prob", "op_ll_right_prob"]]
    total = confidence.sum(axis=1)
    share = confidence["op_ll_left_prob"].div(total.where(total.gt(1e-6)))
    desired = raw["op_curvature_desired"]
    actual = raw["op_curvature_actual"]
    residual = desired - actual
    scale = (desired.abs() + actual.abs()).clip(lower=1e-5)
    scaled = residual / scale
    return {
        "lane_left_share": share.mean(),
        "relative_curvature_gap": scaled.abs().mean(),
        "relative_curvature_bias": scaled.mean(),
        "lane_confidence_hhi": (share.pow(2) + (1 - share).pow(2)).mean(),
        "torque_concentration": temporal_concentration(raw["op_state_steer_torque"]),
        "curvature_gap_concentration": temporal_concentration(residual),
    }


def extended_frames():
    frames = pd.read_csv(OUT / "features.csv")
    extra = []
    for row in frames.itertuples():
        path = DATA / row.split / row.record / f"can_{row.frame:04d}.csv"
        extra.append(extra_features(path))
    frames = frames.join(pd.DataFrame(extra))
    assert not np.isinf(frames[RATIOS + CONCENTRATIONS].to_numpy()).any()
    return frames


def model_grids():
    original = pd.read_csv(OUT / "validation.csv")
    rf_params = json.loads(original.query("model == 'Random forest' and features == 'C'").iloc[0]["params"])
    forest, _ = candidates("Random forest", "C")
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    xgb = make_pipeline(
        clone(imputer),
        XGBClassifier(tree_method="hist", eval_metric="logloss", random_state=SEED, n_jobs=1,
                      learning_rate=0.05, subsample=0.85, colsample_bytree=0.85),
    )
    hist = make_pipeline(
        clone(imputer),
        HistGradientBoostingClassifier(learning_rate=0.05, l2_regularization=1,
                                       early_stopping=False, random_state=SEED),
    )
    return {
        "Random forest": (forest, {key: [value] for key, value in rf_params.items()}),
        "XGBoost": (xgb, {
            "xgbclassifier__max_depth": [2, 4, 6],
            "xgbclassifier__n_estimators": [150, 300],
            "xgbclassifier__min_child_weight": [2, 8],
        }),
        "HistGradientBoosting": (hist, {
            "histgradientboostingclassifier__max_iter": [150, 300],
            "histgradientboostingclassifier__max_leaf_nodes": [7, 15],
            "histgradientboostingclassifier__min_samples_leaf": [15, 35],
        }),
    }


def paired_intervals(groups, y, baseline, baseline_threshold, scores, threshold):
    rng = np.random.default_rng(SEED)
    differences = []
    for _ in range(1000):
        idx = np.concatenate([groups[i] for i in rng.integers(len(groups), size=len(groups))])
        labels = y[idx]
        if np.unique(labels).size == 2:
            differences.append([
                average_precision_score(labels, scores[idx]) - average_precision_score(labels, baseline[idx]),
                f1_score(labels, scores[idx] >= threshold) - f1_score(labels, baseline[idx] >= baseline_threshold),
            ])
    return np.quantile(differences, [0.025, 0.975], axis=0).T.ravel()


def plot_comparison(validation, test, predictions):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    for model, rows in validation.groupby("model", sort=False):
        axes[0].errorbar(rows["features"], rows["cv_ap"], yerr=rows["cv_sd"],
                         marker="o", capsize=4, label=model)
    axes[0].set(xlabel="Feature set", ylabel="Average precision",
                title="Grouped validation (mean ± fold SD)")
    axes[0].legend(fontsize=8)
    y = predictions["label"].eq(2)
    for row in test.itertuples():
        precision, recall, _ = precision_recall_curve(y, predictions[row.score_column])
        axes[1].step(recall, precision, where="post", alpha=0.85,
                     label=f"{row.model} {row.features} ({row.ap:.3f})")
    axes[1].axhline(y.mean(), linestyle=":", color="gray", label=f"Prevalence ({y.mean():.3f})")
    axes[1].set(xlabel="Anomaly recall", ylabel="Anomaly precision", title="Original test set")
    axes[1].legend(fontsize=7)
    fig.savefig(OUTPUT / "comparison.png", dpi=175)
    plt.close(fig)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frames = extended_frames()
    frames.to_csv(OUTPUT / "features.csv", index=False)
    train = frames.loc[frames["use_in_training"]].reset_index(drop=True)
    assignment = pd.read_csv(OUT / "cv_assignments.csv")
    keys = ["record", "frame", "label"]
    train = train.merge(assignment[keys + ["fold"]], on=keys, validate="one_to_one")
    assert len(train) == len(assignment)
    folds = [(np.flatnonzero(train["fold"].ne(fold)), np.flatnonzero(train["fold"].eq(fold)))
             for fold in sorted(train["fold"].unique())]
    y = train["label"].eq(2).astype(int).to_numpy()
    summary = []
    grids = []
    fitted = {}
    for family, (estimator, grid) in model_grids().items():
        c_params = None
        for group, columns in COLUMNS.items():
            search = GridSearchCV(estimator, grid, scoring="average_precision", cv=folds,
                                  n_jobs=4, error_score="raise")
            search.fit(train[columns], y)
            result = pd.DataFrame(search.cv_results_)
            grids.append(result.assign(model=family, features=group))
            if group == "C":
                c_params = search.best_params_
            matched = result.loc[result["params"].map(lambda params: params == c_params)].iloc[0]
            row = {
                "model": family, "features": group, "cv_ap": search.best_score_,
                "cv_sd": result.loc[search.best_index_, "std_test_score"],
                "cv_ap_fixed_c_params": matched["mean_test_score"],
                **{f"fold_{i + 1}_ap": result.loc[search.best_index_, f"split{i}_test_score"]
                   for i in range(len(folds))},
                "params": json.dumps(search.best_params_),
            }
            summary.append(row)
            fitted[family, group] = search.best_estimator_
            print(f"{family} {group}: grouped AP {row['cv_ap']:.4f}", flush=True)
    validation = pd.DataFrame(summary)
    validation.to_csv(OUTPUT / "validation.csv", index=False)
    pd.concat(grids, ignore_index=True).to_csv(OUTPUT / "grid_search.csv", index=False)

    # Compare each C control and the expansion chosen exclusively on training CV.
    chosen = validation.loc[validation["features"].eq("C")].copy()
    expanded = validation.loc[validation["features"].ne("C")]
    chosen = pd.concat([chosen, expanded.loc[expanded.groupby("model", sort=False)["cv_ap"].idxmax()]])
    test = frames.loc[frames["split"].eq("Test")].reset_index(drop=True)
    test_y = test["label"].eq(2).astype(int).to_numpy()
    predictions = test[["record", "frame", "label"]].copy()
    intervals = list(test.groupby(time_groups(test)).indices.values())
    rows = []
    for row in chosen.itertuples():
        columns = COLUMNS[row.features]
        model = fitted[row.model, row.features]
        oof = cross_val_predict(model, train[columns], y, cv=folds, method="predict_proba", n_jobs=4)[:, 1]
        threshold = alert_threshold(y, oof)
        probability = model.predict_proba(test[columns])[:, 1]
        slug = f"{row.model.lower().replace(' ', '_')}_{row.features.lower().replace('+', '_')}"
        score_column = f"{slug}_score"
        predictions[score_column] = probability
        predictions[f"{slug}_label"] = np.where(probability >= threshold, 2, 1)
        rows.append({"model": row.model, "features": row.features, "cv_ap": row.cv_ap,
                     "score_column": score_column, "threshold": threshold,
                     **metrics(test_y, probability, threshold)})
        joblib.dump({"pipeline": model, "columns": columns, "threshold": threshold},
                    OUTPUT / f"{slug}.joblib", compress=3)
    comparison = pd.DataFrame(rows)
    reference = comparison.iloc[0]
    old = pd.read_csv(OUT / "test_predictions.csv")
    assert predictions[["record", "frame", "label"]].equals(old[["record", "frame", "label"]])
    assert np.allclose(predictions[reference.score_column], old["random_forest_score"])
    old_cv = pd.read_csv(OUT / "validation.csv").query("model == 'Random forest' and features == 'C'")
    assert np.isclose(reference.cv_ap, old_cv.iloc[0]["cv_ap"])
    comparison["delta_ap"] = comparison["ap"] - reference.ap
    comparison["delta_f1"] = comparison["f1"] - reference.f1
    ci = [paired_intervals(intervals, test_y, predictions[reference.score_column].to_numpy(),
                           reference.threshold, predictions[row.score_column].to_numpy(), row.threshold)
          for row in comparison.itertuples()]
    comparison[["delta_ap_ci_low", "delta_ap_ci_high", "delta_f1_ci_low", "delta_f1_ci_high"]] = ci
    comparison.to_csv(OUTPUT / "test_comparison.csv", index=False)
    predictions.to_csv(OUTPUT / "test_predictions.csv", index=False)
    frames.groupby("split")[RATIOS + CONCENTRATIONS].agg(lambda values: values.isna().sum()).to_csv(
        OUTPUT / "missing_by_split.csv",
    )
    plot_comparison(validation, comparison, predictions)
    print(comparison.to_string(index=False), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
