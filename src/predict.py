"""Step 4: use the saved model to score a new build before it runs."""
import argparse
import json
import os

import joblib
import pandas as pd

from .features import add_engineered

DEFAULT_MODEL = "models/best_model.joblib"
DEFAULT_BUILD = dict(files_changed=3, lines_added=40, lines_deleted=10, tests_changed=1,
                     dependency_changed=0, author_prior_builds=150, hour_of_day=14, day_of_week=1,
                     commits_in_push=1, prev_build_failed=0)
ADVICE = {
    "low": "Normal pipeline is fine.",
    "medium": "Run the full test suite and ask for a quick review.",
    "high": "Risky change: split it into smaller pushes, run all tests locally first, get a review.",
}
DEMO_BUILDS = {
    "small safe change": {},
    "medium change": dict(files_changed=9, lines_added=250, lines_deleted=60, author_prior_builds=30),
    "risky change": dict(files_changed=25, lines_added=900, lines_deleted=300, dependency_changed=1,
                         author_prior_builds=5, hour_of_day=23, day_of_week=4, commits_in_push=6,
                         prev_build_failed=1, tests_changed=0),
}


def risk_level(p: float) -> str:
    return "low" if p < 0.30 else "medium" if p < 0.60 else "high"


def load_model(path: str = DEFAULT_MODEL):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found at {path}. Run: python -m src.train")
    return joblib.load(path)


def predict_build(build: dict, model_path: str = DEFAULT_MODEL, bundle=None) -> dict:
    bundle = bundle or load_model(model_path)
    row = {**DEFAULT_BUILD, **build}
    X = add_engineered(pd.DataFrame([row]))[bundle["features"]]
    p = float(bundle["model"].predict_proba(X)[0, 1])
    level = risk_level(p)
    return {"failure_probability": round(p, 3), "risk": level, "advice": ADVICE[level]}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="predict", description="Predict whether a build will fail")
    parser.add_argument("--demo", action="store_true", help="score three example builds")
    for key in DEFAULT_BUILD:
        parser.add_argument("--" + key.replace("_", "-"), dest=key, type=int, default=None)
    args = parser.parse_args(argv)
    bundle = load_model()
    if args.demo:
        for label, build in DEMO_BUILDS.items():
            r = predict_build(build, bundle=bundle)
            print(f"{label:20s} -> {r['failure_probability']:.0%} failure risk ({r['risk']}). {r['advice']}")
        return 0
    build = {k: getattr(args, k) for k in DEFAULT_BUILD if getattr(args, k) is not None}
    print(json.dumps(predict_build(build, bundle=bundle), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
