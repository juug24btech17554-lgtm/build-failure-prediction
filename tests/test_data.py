from src.features import FEATURE_COLUMNS, TARGET, add_engineered, time_split, xy
from src.generate_data import generate


def test_generation_is_repeatable():
    assert generate(300, seed=1).equals(generate(300, seed=1))
    assert not generate(300, seed=1).equals(generate(300, seed=2))


def test_data_is_clean_and_realistic():
    df = generate(1500)
    assert not df.isna().any().any()
    assert {"build_id", "failed", *FEATURE_COLUMNS[:10]} <= set(df.columns)
    assert 0.15 <= df["failed"].mean() <= 0.45  # some failures, but most builds pass


def test_time_split_never_trains_on_the_future():
    train, test = time_split(generate(1000))
    assert train["build_id"].max() < test["build_id"].min()
    assert len(train) == 800 and len(test) == 200


def test_label_is_not_a_feature():
    assert TARGET not in FEATURE_COLUMNS and "build_id" not in FEATURE_COLUMNS


def test_engineered_features():
    df = add_engineered(generate(200))
    assert (df["total_lines_changed"] == df["lines_added"] + df["lines_deleted"]).all()
    X, y = xy(generate(200))
    assert list(X.columns) == FEATURE_COLUMNS and len(X) == len(y)
