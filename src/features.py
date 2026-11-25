"""Step 2: load the data, add engineered features, and split it by time."""
import pandas as pd

BASE_FEATURES = [
    "files_changed", "lines_added", "lines_deleted", "tests_changed", "dependency_changed",
    "author_prior_builds", "hour_of_day", "day_of_week", "commits_in_push", "prev_build_failed",
]
ENGINEERED = ["total_lines_changed", "is_night", "is_friday"]
FEATURE_COLUMNS = BASE_FEATURES + ENGINEERED   # never includes the label ("failed") or build_id
TARGET = "failed"


def load_data(path: str = "data/builds.csv") -> pd.DataFrame:
    return pd.read_csv(path)


def add_engineered(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["total_lines_changed"] = out["lines_added"] + out["lines_deleted"]
    out["is_night"] = ((out["hour_of_day"] >= 22) | (out["hour_of_day"] < 5)).astype(int)
    out["is_friday"] = (out["day_of_week"] == 4).astype(int)
    return out


def time_split(df: pd.DataFrame, test_frac: float = 0.2):
    """Oldest builds train the model, newest builds test it. A random split would let the model
    'see the future', which is not possible when predicting a real new build."""
    df = df.sort_values("build_id").reset_index(drop=True)
    cut = int(len(df) * (1 - test_frac))
    return df.iloc[:cut], df.iloc[cut:]


def xy(df: pd.DataFrame):
    df = add_engineered(df)
    return df[FEATURE_COLUMNS], df[TARGET]
