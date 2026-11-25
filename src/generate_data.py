"""Step 1: create a synthetic CI build history.

Real build histories (for example TravisTorrent) are large downloads, so this script builds a
realistic *synthetic* dataset instead. Each row is one CI build, in time order (build_id).
The chance of failure is driven by common-sense risk factors from CI research and experience:
big changes, dependency changes, a failed previous build, new authors, late-night and Friday pushes.
Because the data is synthetic, the model results show that the pipeline works, not real-world accuracy.
To use real data, replace data/builds.csv with a CSV that has the same columns.
"""
import os

import numpy as np
import pandas as pd


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def generate(n: int = 3000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    files_changed = rng.poisson(4, n) + 1
    lines_added = rng.lognormal(3.5, 1.0, n).astype(int)
    lines_deleted = (lines_added * rng.uniform(0, 0.8, n)).astype(int)
    tests_changed = rng.binomial(1, 0.45, n)
    dependency_changed = rng.binomial(1, 0.08, n)
    author_prior_builds = rng.integers(0, 400, n)  # experience of the author
    hour_of_day = rng.integers(0, 24, n)
    day_of_week = rng.integers(0, 7, n)  # 0 = Monday
    commits_in_push = rng.poisson(1.5, n) + 1

    night = ((hour_of_day >= 22) | (hour_of_day < 5)).astype(int)
    friday = (day_of_week == 4).astype(int)
    z_base = (
        -3.1
        + 0.38 * np.log1p(lines_added + lines_deleted)
        + 0.07 * files_changed
        + 1.2 * dependency_changed
        - 0.25 * tests_changed
        - 0.9 * (author_prior_builds / 400)
        + 0.5 * night
        + 0.35 * friday
        + 0.12 * commits_in_push
        + rng.normal(0, 0.4, n)  # randomness no feature can explain
    )

    failed = np.zeros(n, dtype=int)
    prev_failed = np.zeros(n, dtype=int)
    prev = 0
    for i in range(n):  # a failed build makes the next one more likely to fail
        prev_failed[i] = prev
        failed[i] = int(rng.random() < _sigmoid(z_base[i] + 1.3 * prev))
        prev = failed[i]

    return pd.DataFrame({
        "build_id": np.arange(1, n + 1),
        "files_changed": files_changed,
        "lines_added": lines_added,
        "lines_deleted": lines_deleted,
        "tests_changed": tests_changed,
        "dependency_changed": dependency_changed,
        "author_prior_builds": author_prior_builds,
        "hour_of_day": hour_of_day,
        "day_of_week": day_of_week,
        "commits_in_push": commits_in_push,
        "prev_build_failed": prev_failed,
        "failed": failed,
    })


def main(path: str = "data/builds.csv") -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df = generate()
    df.to_csv(path, index=False)
    print(f"Wrote {len(df)} builds to {path}. Failure rate: {df['failed'].mean():.1%}")


if __name__ == "__main__":
    main()
