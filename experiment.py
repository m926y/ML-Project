"""Tune CAN-only LKA classifiers on grouped training folds and assess held-out clips."""
import argparse
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score, precision_recall_curve,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from preprocess import FEATURES, OUT


SEED = 42


def candidates(name, feature_set):
    steps = [("impute", SimpleImputer(strategy="median", keep_empty_features=True))]
    if feature_set == "D":
        steps.extend([("variance", VarianceThreshold()), ("select", SelectKBest(f_classif, k=8))])
    if name == "Logistic":
        steps.extend([("scale", StandardScaler()), ("model", LogisticRegression(max_iter=2000))])
        grid = {"model__C": [0.01, 0.1, 1, 10], "model__class_weight": [None, "balanced"]}
    else:
        steps.append(("model", RandomForestClassifier(random_state=SEED, n_jobs=1)))
        grid = {
            "model__n_estimators": [100, 250], "model__max_depth": [6, None],
            "model__min_samples_leaf": [2, 8], "model__class_weight": [None, "balanced"],
        }
    return Pipeline(steps), grid


def alert_threshold(y, scores):
    precision, recall, thresholds = precision_recall_curve(y, scores)
    p, r = precision[:-1], recall[:-1]
    f1 = np.divide(2 * p * r, p + r, out=np.zeros_like(p), where=(p + r) > 0)
    # Prefer the higher threshold among ties.
    return float(thresholds[np.flatnonzero(np.isclose(f1, f1.max()))[-1]])


def metrics(y, scores, threshold):
    detected = scores >= threshold
    tn, fp, fn, tp = confusion_matrix(y, detected, labels=[0, 1]).ravel()
    return {
        "ap": average_precision_score(y, scores), "roc_auc": roc_auc_score(y, scores),
        "precision": precision_score(y, detected, zero_division=0),
        "recall": recall_score(y, detected, zero_division=0),
        "f1": f1_score(y, detected, zero_division=0),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def clip_bootstrap(test, y, scores, threshold):
    groups = list(test.groupby(time_groups(test)).indices.values())
    rng = np.random.default_rng(SEED)
    samples = []
    for _ in range(1000):
        idx = np.concatenate([groups[i] for i in rng.integers(len(groups), size=len(groups))])
        if np.unique(y[idx]).size == 2:
            samples.append([
                average_precision_score(y[idx], scores[idx]),
                f1_score(y[idx], scores[idx] >= threshold, zero_division=0),
            ])
    limits = np.quantile(samples, [0.025, 0.975], axis=0).T
    return {key: bounds.tolist() for key, bounds in zip(("ap", "f1"), limits)}


def plot_results(results, scores, y, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
    for family, rows in results.groupby("model", sort=False):
        axes[0].errorbar(rows["features"], rows["cv_ap"], yerr=rows["cv_sd"],
                         marker="o", capsize=4, label=family)
    axes[0].set(xlabel="Feature set", ylabel="Mean validation AP",
                title="Grouped validation (±1 fold SD)", ylim=(0, 1.02))
    axes[0].legend()
    for label, probability in scores.items():
        precision, recall, _ = precision_recall_curve(y, probability)
        ap = average_precision_score(y, probability)
        axes[1].step(recall, precision, where="post", label=f"{label} ({ap:.3f})")
    axes[1].axhline(y.mean(), color="gray", linestyle=":", label=f"Prevalence ({y.mean():.3f})")
    axes[1].set(xlabel="Anomaly recall", ylabel="Anomaly precision",
                title="Held-out precision–recall", xlim=(0, 1.01), ylim=(0, 1.02))
    axes[1].legend()
    fig.savefig(out / "model_comparison.png", dpi=180)
    plt.close(fig)


def time_groups(train):
    """Keep windows that share clock time together even across record boundaries."""
    intervals = train.groupby("record").agg(start=("wall_start", "min"), end=("wall_end", "max"))
    intervals = intervals.sort_values("start")
    components = {}
    end = -np.inf
    for record, window in intervals.iterrows():
        if pd.isna(window.start) or window.start > end:
            group = record
            end = -np.inf
        components[record] = group
        if pd.notna(window.end):
            end = max(end, window.end)
    return train["record"].map(components)


def main(out):
    frames = pd.read_csv(out / "features.csv")
    train = frames.loc[frames["use_in_training"]].reset_index(drop=True)
    test = frames.loc[frames["split"].eq("Test")].reset_index(drop=True)
    y = train["label"].eq(2).astype(int).to_numpy()
    groups = time_groups(train)
    cv = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED).split(train, y, groups))
    frame_folds = train[["record", "frame", "label"]].assign(cv_group=groups, fold=-1)
    fold_summary = []
    for fold, (fit, validate) in enumerate(cv, 1):
        assert set(groups.iloc[fit]).isdisjoint(groups.iloc[validate])
        assert np.unique(y[fit]).size == np.unique(y[validate]).size == 2
        frame_folds.loc[validate, "fold"] = fold
        fold_summary.append({
            "fold": fold, "train_frames": len(fit), "val_frames": len(validate),
            "val_positives": int(y[validate].sum()),
            "val_records": int(train.iloc[validate]["record"].nunique()),
        })
    assert frame_folds["fold"].gt(0).all()
    frame_folds.to_csv(out / "cv_assignments.csv", index=False)
    assert train.assign(fold=frame_folds["fold"]).groupby("can_sha256")["fold"].nunique().max() == 1
    rows, searches, models = [], [], {}
    feature_sets = {**FEATURES, "D": FEATURES["C"]}
    for family in ("Logistic", "Random forest"):
        for feature_set, columns in feature_sets.items():
            estimator, grid = candidates(family, feature_set)
            search = GridSearchCV(estimator, grid, scoring="average_precision", cv=cv,
                                  n_jobs=4, error_score="raise")
            search.fit(train[columns], y)
            best = search.best_index_
            cv_results = pd.DataFrame(search.cv_results_)
            searches.append(cv_results.assign(model=family, features=feature_set))
            scores = cross_val_predict(search.best_estimator_, train[columns], y,
                                       cv=cv, method="predict_proba", n_jobs=4)[:, 1]
            threshold = alert_threshold(y, scores)
            row = {
                "model": family, "features": feature_set,
                "cv_ap": search.best_score_, "cv_sd": cv_results.loc[best, "std_test_score"],
                **{f"fold_{i + 1}_ap": cv_results.loc[best, f"split{i}_test_score"] for i in range(len(cv))},
                "oof_ap": average_precision_score(y, scores),
                "oof_f1": f1_score(y, scores >= threshold), "threshold": threshold,
                "params": json.dumps(search.best_params_),
            }
            rows.append(row)
            models[family, feature_set] = (search.best_estimator_, columns, threshold)
            print(f"{family} {feature_set}: CV AP {row['cv_ap']:.4f}", flush=True)
    results = pd.DataFrame(rows)
    results.to_csv(out / "validation.csv", index=False)
    pd.concat(searches, ignore_index=True).to_csv(out / "grid_search.csv", index=False)
    winners = results.loc[results.groupby("model")["cv_ap"].idxmax()].sort_values("cv_ap", ascending=False)
    y_test = test["label"].eq(2).astype(int).to_numpy()
    dummy = DummyClassifier(strategy="prior").fit(train[["vEgo"]], y)
    baseline = dummy.predict_proba(test[["vEgo"]])[:, 1]
    predictions = test[["record", "frame", "label", "lka_error_mean", *FEATURES["C"]]].copy()
    comparison = {"Prior": {"threshold": 0.5, **metrics(y_test, baseline, 0.5)}}
    curves = {}
    for result in winners.itertuples():
        model, columns, threshold = models[result.model, result.features]
        probability = model.predict_proba(test[columns])[:, 1]
        curves[f"{result.model} {result.features}"] = probability
        comparison[result.model] = {
            "features": result.features, "threshold": threshold,
            **metrics(y_test, probability, threshold),
            "group_bootstrap_95": clip_bootstrap(test, y_test, probability, threshold),
        }
        slug = result.model.lower().replace(" ", "_")
        predictions[f"{slug}_score"] = probability
        predictions[f"{slug}_label"] = np.where(probability >= threshold, 2, 1)
        selected = model.named_steps.get("select")
        used = columns
        if selected is not None:
            nonconstant = model.named_steps["variance"].get_support()
            used = np.array(columns)[nonconstant][selected.get_support()].tolist()
        fitted = model.named_steps["model"]
        strength = fitted.coef_[0] if result.model == "Logistic" else fitted.feature_importances_
        pd.DataFrame({"feature": used, "value": strength}).to_csv(out / f"{slug}_features.csv", index=False)
        joblib.dump({"pipeline": model, "columns": columns, "threshold": threshold,
                     "positive_label": 2}, out / f"{slug}.joblib", compress=3)
    predictions.to_csv(out / "test_predictions.csv", index=False)
    primary = winners.iloc[0]
    slug = primary["model"].lower().replace(" ", "_")
    errors = predictions.loc[predictions["label"].ne(predictions[f"{slug}_label"])].copy()
    errors["type"] = np.where(errors["label"].eq(2), "FN", "FP")
    errors["label_margin"] = (errors["lka_error_mean"] - 0.25).abs()
    errors.to_csv(out / "primary_errors.csv", index=False)
    summary = {
        "train_frames": len(train), "train_records": int(train["record"].nunique()),
        "train_groups": int(groups.nunique()), "train_positives": int(y.sum()),
        "test_frames": len(test), "test_positives": int(y_test.sum()),
        "test_records": int(test["record"].nunique()), "test_groups": int(time_groups(test).nunique()),
        "folds": fold_summary, "primary": f"{primary['model']} {primary['features']}",
        "test": comparison,
    }
    (out / "evaluation.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plot_results(results, curves, y_test, out)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    main(parser.parse_args().out)
