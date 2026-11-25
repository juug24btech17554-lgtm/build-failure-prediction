## Build-failure prediction results

Train builds: 2400 (failure rate 34.5%). Test builds: 600 (failure rate 30.5%).

| Model | CV AUC | Test AUC | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| baseline (always predict pass) | 0.500 | 0.500 | 0.695 | 0.000 | 0.000 | 0.000 |
| logistic regression | 0.712 | 0.728 | 0.675 | 0.476 | 0.656 | 0.552 |
| random forest | 0.697 | 0.719 | 0.667 | 0.464 | 0.590 | 0.519 |

**Selected model (best CV AUC): logistic regression**

Confusion matrix on the test builds: 285 true pass, 132 false alarms, 63 missed failures, 120 caught failures.

Top risk factors:
- prev_build_failed: 0.663
- dependency_changed: 0.342
- is_night: 0.312
- author_prior_builds: 0.232
- is_friday: 0.190
