"""Step 3: train three models, compare them, pick the best, and save the results."""
import json
import os

import joblib
import matplotlib

matplotlib.use("Agg")  # draw plots without a screen
import matplotlib.pyplot as plt
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_COLUMNS, load_data, time_split, xy

BASELINE = "baseline (always predict pass)"


def make_models():
    return {
        BASELINE: DummyClassifier(strategy="most_frequent"),
        "logistic regression": make_pipeline(
            StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000)),
        "random forest": RandomForestClassifier(
            n_estimators=200, max_depth=6, min_samples_leaf=10,
            class_weight="balanced_subsample", random_state=42),
    }


def feature_importance(name, model):
    if name == "random forest":
        values = model.feature_importances_
    else:  # logistic regression: size of the coefficient (inputs are standardised)
        values = abs(model[-1].coef_[0])
    return sorted(zip(FEATURE_COLUMNS, [float(v) for v in values]), key=lambda t: t[1], reverse=True)


def train_and_evaluate(df, out_dir: str = "."):
    train, test = time_split(df)
    X_train, y_train = xy(train)
    X_test, y_test = xy(test)
    cv = TimeSeriesSplit(n_splits=5)  # validation folds also respect time order

    results, fitted, curves = {}, {}, {}
    for name, model in make_models().items():
        cv_auc = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)
        results[name] = {
            "cv_auc_mean": float(cv_auc.mean()),
            "accuracy": float(accuracy_score(y_test, pred)),
            "precision": float(precision_score(y_test, pred, zero_division=0)),
            "recall": float(recall_score(y_test, pred, zero_division=0)),
            "f1": float(f1_score(y_test, pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, proba)),
            "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        }
        fitted[name] = model
        curves[name] = roc_curve(y_test, proba)

    # Choose the model using the training-time validation score, NOT the test score,
    # so the test set stays an honest final exam.
    candidates = [n for n in results if n != BASELINE]
    best = max(candidates, key=lambda n: results[n]["cv_auc_mean"])
    importance = feature_importance(best, fitted[best])

    info = {"n_train": len(train), "n_test": len(test),
            "failure_rate_train": float(y_train.mean()), "failure_rate_test": float(y_test.mean())}
    summary = {"best_model": best, "data": info, "results": results,
               "feature_importance": importance}

    os.makedirs(os.path.join(out_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "reports"), exist_ok=True)
    joblib.dump({"model": fitted[best], "features": FEATURE_COLUMNS, "name": best},
                os.path.join(out_dir, "models", "best_model.joblib"))
    with open(os.path.join(out_dir, "reports", "metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(out_dir, "reports", "metrics.md"), "w") as f:
        f.write(metrics_markdown(summary))
    save_plots(out_dir, curves, results, importance, best)
    return summary


def metrics_markdown(s):
    d = s["data"]
    lines = ["## Build-failure prediction results", "",
             f"Train builds: {d['n_train']} (failure rate {d['failure_rate_train']:.1%}). "
             f"Test builds: {d['n_test']} (failure rate {d['failure_rate_test']:.1%}).", "",
             "| Model | CV AUC | Test AUC | Accuracy | Precision | Recall | F1 |",
             "|---|---|---|---|---|---|---|"]
    for name, r in s["results"].items():
        lines.append(f"| {name} | {r['cv_auc_mean']:.3f} | {r['roc_auc']:.3f} | {r['accuracy']:.3f} | "
                     f"{r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} |")
    best = s["best_model"]
    (tn, fp), (fn, tp) = s["results"][best]["confusion_matrix"]
    lines += ["", f"**Selected model (best CV AUC): {best}**", "",
              f"Confusion matrix on the test builds: {tn} true pass, {fp} false alarms, "
              f"{fn} missed failures, {tp} caught failures.", "", "Top risk factors:"]
    lines += [f"- {n}: {v:.3f}" for n, v in s["feature_importance"][:5]]
    return "\n".join(lines) + "\n"


def save_plots(out_dir, curves, results, importance, best):
    plt.figure(figsize=(5.5, 4.5))
    for name, (fpr, tpr, _) in curves.items():
        plt.plot(fpr, tpr, label=f"{name} (AUC {results[name]['roc_auc']:.2f})")
    plt.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
    plt.title("ROC curves on the test builds"); plt.legend(fontsize=7); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "reports", "roc_curve.png"), dpi=130); plt.close()

    names = [n for n, _ in importance][::-1]
    vals = [v for _, v in importance][::-1]
    plt.figure(figsize=(6, 4.5))
    plt.barh(names, vals, color="#3b6ea5")
    plt.title(f"Feature importance ({best})"); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "reports", "feature_importance.png"), dpi=130); plt.close()


def main():
    summary = train_and_evaluate(load_data())
    print(metrics_markdown(summary))


if __name__ == "__main__":
    main()
