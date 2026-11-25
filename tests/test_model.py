import json
import os

import pytest

from src.generate_data import generate
from src.predict import DEMO_BUILDS, load_model, predict_build, risk_level
from src.train import BASELINE, train_and_evaluate


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    out = tmp_path_factory.mktemp("out")
    summary = train_and_evaluate(generate(2000), out_dir=str(out))
    return summary, str(out)


def test_best_model_beats_the_baseline(trained):
    summary, _ = trained
    best = summary["results"][summary["best_model"]]
    assert summary["results"][BASELINE]["roc_auc"] == pytest.approx(0.5)
    assert best["roc_auc"] > 0.65


def test_outputs_are_saved(trained):
    _, out = trained
    for path in ["models/best_model.joblib", "reports/metrics.json", "reports/metrics.md",
                 "reports/roc_curve.png", "reports/feature_importance.png"]:
        assert os.path.exists(os.path.join(out, path)), path
    assert "best_model" in json.load(open(os.path.join(out, "reports/metrics.json")))


def test_risky_build_scores_higher_than_safe_build(trained):
    _, out = trained
    bundle = load_model(os.path.join(out, "models/best_model.joblib"))
    safe = predict_build(DEMO_BUILDS["small safe change"], bundle=bundle)
    risky = predict_build(DEMO_BUILDS["risky change"], bundle=bundle)
    assert 0 <= safe["failure_probability"] <= 1 and 0 <= risky["failure_probability"] <= 1
    assert risky["failure_probability"] > safe["failure_probability"]


def test_risk_levels():
    assert risk_level(0.1) == "low" and risk_level(0.45) == "medium" and risk_level(0.9) == "high"


def test_missing_model_gives_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_model(str(tmp_path / "nope.joblib"))
