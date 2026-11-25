# build-failure-prediction

Predict whether a CI build will fail **before it runs**, from facts about the change: how big it is,
whether dependencies changed, who pushed it, when, and whether the previous build failed.

## Pipeline
1. `src/generate_data.py` creates a synthetic build history (`data/builds.csv`). Swap in real data (for example TravisTorrent) with the same columns.
2. `src/features.py` adds features (total lines, night push, Friday push) and splits **by time**: old builds train, newest builds test.
3. `src/train.py` compares a baseline, logistic regression and a random forest, picks the best by time-aware cross-validation, and saves the model, metrics and plots.
4. `src/predict.py` scores a new build and returns a failure probability with a low / medium / high risk level.

## Run it
```
pip install -r requirements.txt
python -m src.generate_data
python -m src.train
python -m src.predict --demo
python -m src.predict --files-changed 20 --lines-added 800 --dependency-changed 1 --prev-build-failed 1
pytest
```

## CI
`.github/workflows/ci.yml` runs the tests, trains the model, writes the metrics and example predictions to the job summary page, and uploads the model and reports as an artifact.

Note: the data is synthetic, so the scores show that the pipeline works, not real-world accuracy.
