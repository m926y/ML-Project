"""Compare CAN window features and classifiers on the existing grouped folds."""
import argparse
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.model_selection import GridSearchCV, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits
from xgboost import XGBClassifier

from experiment import SEED, alert_threshold, metrics, plt, time_groups
from extension import COLUMNS, paired_intervals
from preprocess import DATA, OUT


OUTPUT = OUT / "optimization"
CONFIDENCE = ["op_ll_left_prob", "op_ll_right_prob"]
CURVATURE = ["op_curvature_desired", "op_curvature_actual"]
VEHICLE = ["vEgo", "op_state_steer_angle", "rcm_yaw_rate", "rcm_lateral_accel"]
TEMPORAL = [
    "weak_lane_mean", "weak_lane_p10", "joint_confidence_mean", "confidence_imbalance",
    "curvature_bias", "curvature_abs_p90", "curvature_residual_sd",
    "curvature_residual_change", "confidence_change",
]
DYNAMICS = [
    "steer_angle_mean", "steer_angle_abs", "steer_angle_sd",
    "yaw_rate_abs", "lateral_acceleration_abs", "desired_lateral_demand",
    "lateral_tracking_gap", "yaw_tracking_gap",
]
CORE = COLUMNS["C+R"]
FEATURE_SETS = {
    "C+R": CORE,
    "C+R+T": CORE + TEMPORAL,
    "C+R+V": CORE + DYNAMICS,
    "C+R+T+V": CORE + TEMPORAL + DYNAMICS,
}


def window_features(path):
    columns = ["Time", *CONFIDENCE, *CURVATURE, *VEHICLE]
    raw = pd.read_csv(path, usecols=columns, dtype=float).replace([np.inf, -np.inf], np.nan)
    raw = raw.drop_duplicates().sort_values("Time", kind="stable")
    left, right = (raw[col].where(raw[col].between(0, 1)) for col in CONFIDENCE)
    weak = np.minimum(left, right)
    desired, actual = (raw[col] for col in CURVATURE)
    residual = desired - actual
    speed, angle, yaw, acceleration = (raw[col] for col in VEHICLE)
    # Ignore steps with unknown or repeated sample time.
    forward = raw["Time"].diff().gt(0)
    return {
        "weak_lane_mean": weak.mean(),
        "weak_lane_p10": weak.quantile(0.10),
        "joint_confidence_mean": (left * right).mean(),
        "confidence_imbalance": (left - right).abs().mean(),
        "curvature_bias": residual.mean(),
        "curvature_abs_p90": residual.abs().quantile(0.90),
        "curvature_residual_sd": residual.std(ddof=0),
        "curvature_residual_change": residual.diff().abs().where(forward).mean(),
        "confidence_change": weak.diff().abs().where(forward).mean(),
        "steer_angle_mean": angle.mean(),
        "steer_angle_abs": angle.abs().mean(),
        "steer_angle_sd": angle.std(ddof=0),
        "yaw_rate_abs": yaw.abs().mean(),
        "lateral_acceleration_abs": acceleration.abs().mean(),
        "desired_lateral_demand": (speed.pow(2) * desired).abs().mean(),
        "lateral_tracking_gap": (speed.pow(2) * residual).abs().mean(),
        "yaw_tracking_gap": (speed * residual).abs().mean(),
    }


def model_grids():
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    return {
        "Random forest": (
            make_pipeline(imputer, RandomForestClassifier(
                n_estimators=250, class_weight="balanced", random_state=SEED, n_jobs=1)),
            {"randomforestclassifier__min_samples_leaf": [2, 5],
             "randomforestclassifier__max_features": ["sqrt", 0.7]},
        ),
        "Extra Trees": (
            make_pipeline(imputer, ExtraTreesClassifier(
                n_estimators=300, class_weight="balanced", random_state=SEED, n_jobs=1)),
            {"extratreesclassifier__min_samples_leaf": [1, 3, 6],
             "extratreesclassifier__max_features": ["sqrt", 0.7]},
        ),
        "RBF-SVM": (
            make_pipeline(imputer, RobustScaler(), SVC(class_weight="balanced")),
            {"svc__C": [0.3, 3, 15], "svc__gamma": ["scale", 0.03, 0.15]},
        ),
        "XGBoost": (
            make_pipeline(imputer, XGBClassifier(
                learning_rate=0.05, n_estimators=200, eval_metric="logloss",
                subsample=0.85, colsample_bytree=0.85, random_state=SEED, n_jobs=1)),
            {"xgbclassifier__max_depth": [3, 5], "xgbclassifier__min_child_weight": [2, 8]},
        ),
    }


def model_scores(model, features, method):
    if method == "decision_function":
        return model.decision_function(features)
    return model.predict_proba(features)[:, 1]


def main(jobs):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frames = pd.read_csv(OUT / "extension" / "features.csv")
    additions = pd.DataFrame([
        window_features(DATA / row.split / row.record / f"can_{row.frame:04d}.csv")
        for row in frames.itertuples()
    ])
    assert not np.isinf(additions.to_numpy()).any()
    frames = frames.join(additions)
    frames.to_csv(OUTPUT / "features.csv", index=False)
    frames.groupby("split")[TEMPORAL + DYNAMICS].agg(
        lambda values: values.isna().sum()).to_csv(OUTPUT / "missing_by_split.csv")
    assignment = pd.read_csv(OUT / "cv_assignments.csv")
    keys = ["record", "frame", "label"]
    train = frames.loc[frames["use_in_training"]].merge(
        assignment[keys + ["fold"]], on=keys, how="left", validate="one_to_one")
    assert len(train) == len(assignment) and train["fold"].notna().all()
    assert train.assign(group=time_groups(train)).groupby("group")["fold"].nunique().eq(1).all()
    assert train.groupby("can_sha256")["fold"].nunique().eq(1).all()
    folds = [(np.flatnonzero(train.fold.ne(fold)), np.flatnonzero(train.fold.eq(fold)))
             for fold in sorted(train.fold.unique())]
    y = train["label"].eq(2).to_numpy()

    models, summaries, grids = {}, [], []
    for family, (estimator, grid) in model_grids().items():
        control_params = None
        for features, columns in FEATURE_SETS.items():
            search = GridSearchCV(estimator, grid, cv=folds, n_jobs=jobs,
                                  scoring="average_precision", error_score="raise")
            search.fit(train[columns], y)
            results = pd.DataFrame(search.cv_results_)
            grids.append(results.assign(model=family, features=features))
            if control_params is None:
                control_params = search.best_params_
            fixed = results.loc[results["params"].map(lambda params: params == control_params)].iloc[0]
            best = results.iloc[search.best_index_]
            summaries.append({"model": family, "features": features, "cv_ap": search.best_score_,
                              "cv_sd": best["std_test_score"],
                              "cv_ap_fixed_core_params": fixed["mean_test_score"],
                              **{f"fold_{i + 1}_ap": best[f"split{i}_test_score"] for i in range(len(folds))},
                              "params": json.dumps(search.best_params_)})
            models[family, features] = search.best_estimator_
            print(f"{family} {features}: grouped AP {search.best_score_:.4f}", flush=True)
    validation = pd.DataFrame(summaries)
    validation.to_csv(OUTPUT / "validation.csv", index=False)
    pd.concat(grids, ignore_index=True).to_csv(OUTPUT / "grid_search.csv", index=False)
    winners = validation.loc[validation.groupby("model")["cv_ap"].idxmax()]
    test = frames.loc[frames["split"].eq("Test")].reset_index(drop=True)
    test_y = test["label"].eq(2).to_numpy()
    baseline_table = pd.read_csv(OUT / "extension" / "test_predictions.csv")
    assert test[keys].equals(baseline_table[keys])
    baseline = baseline_table["random_forest_c_r_score"].to_numpy()
    reference = pd.read_csv(OUT / "extension" / "test_comparison.csv").query(
        "model == 'Random forest' and features == 'C+R'").iloc[0]
    intervals = list(test.groupby(time_groups(test)).indices.values())
    predictions = test[keys].assign(reference_score=baseline)
    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    rows = []
    for row in winners.sort_values("cv_ap", ascending=False).itertuples():
        columns = FEATURE_SETS[row.features]
        model = models[row.model, row.features]
        method = "decision_function" if row.model == "RBF-SVM" else "predict_proba"
        oof = cross_val_predict(model, train[columns], y, cv=folds, method=method, n_jobs=jobs)
        if method == "predict_proba":
            oof = oof[:, 1]
        threshold = alert_threshold(y, oof)
        scores = model_scores(model, test[columns], method)
        slug = row.model.lower().replace(" ", "_").replace("-", "_")
        predictions[f"{slug}_score"] = scores
        predictions[f"{slug}_label"] = np.where(scores >= threshold, 2, 1)
        measured = metrics(test_y, scores, threshold)
        bounds = paired_intervals(intervals, test_y, baseline, reference.threshold, scores, threshold)
        rows.append({"model": row.model, "features": row.features, "cv_ap": row.cv_ap,
                     "oof_ap": average_precision_score(y, oof), "threshold": threshold, **measured,
                     "delta_ap": measured["ap"] - reference.ap, "delta_f1": measured["f1"] - reference.f1,
                     **dict(zip(("delta_ap_ci_low", "delta_ap_ci_high",
                                 "delta_f1_ci_low", "delta_f1_ci_high"), bounds))})
        joblib.dump({"pipeline": model, "columns": columns, "threshold": threshold,
                     "score_method": method}, OUTPUT / f"{slug}.joblib", compress=3)
        precision, recall, _ = precision_recall_curve(test_y, scores)
        ax.step(recall, precision, where="post", label=f"{row.model} {row.features} ({measured['ap']:.3f})")
    precision, recall, _ = precision_recall_curve(test_y, baseline)
    ax.step(recall, precision, where="post", linestyle="--", label=f"RF C+R reference ({reference.ap:.3f})")
    ax.axhline(test_y.mean(), linestyle=":", color="gray", label="Test prevalence")
    ax.set(xlabel="Anomaly recall", ylabel="Anomaly precision", xlim=(0, 1), ylim=(0, 1.02))
    ax.legend(fontsize=8)
    fig.savefig(OUTPUT / "precision_recall.png", dpi=175)
    plt.close(fig)
    comparison = pd.DataFrame(rows)
    comparison.to_csv(OUTPUT / "test_comparison.csv", index=False)
    predictions.to_csv(OUTPUT / "test_predictions.csv", index=False)
    print(comparison.to_string(index=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=4)
    with threadpool_limits(limits=1):
        main(parser.parse_args().jobs)
