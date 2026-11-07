# Baseline Predictive Pipeline -- ETAI

20231714 - Pedro Gomes

Week 2:

LR: LR is not overfitting and the accuracy is 68%.
DT: DT is overfitting and the accuracy is 63%.

Current best model: LR because the accuracy is higher than in the DT and it is not overfitting. 

Week 3:

Added cleaning pipeline


After clean-up:
LR - Accuracy still at 68%, however f1-score increased 1% for 0 and decreased 1% for 1.

DT - Accuracy now at 64%. Still overfitting.

Best model: Still LR for the same reasons as before.


Week 4:

Added Preprocessing + CV

Dummy - Accuracy at 55%. Baseline

LR - Accuracy at 67%. Pretty much the same than before CV.

DT - Accuracy at 61%. Lower accuracy and it is still overfitting. Accuracy is lower than before CV.

RF - Accuracy at 65% but the model is overfitting.

Best Model: LR. Best accuracy and no overfitting.

Week 6:

Added ensembles: Random Forest, Oblique Random Forest and AdaBoost (new model types in src/model.py + search spaces in config.yaml), all tuned with Optuna (30 trials) and evaluated with nested CV.

### Ensembles
| Model | Default: CV mean ± std (gap) | Tuned: nested CV mean ± std (gap) | Tuning score (best trial) | Chosen hyperparameters |
|---|---|---|---|---|
| Dummy (baseline) | 0.549 ± 0.000 (-0.000) | — | — | — |
| Logistic regression | 0.672 ± 0.013 (+0.003) | — | — | — |
| Decision tree | 0.610 ± 0.018 (+0.085) | 0.672 ± 0.017 (+0.013) | 0.679 | max_depth=6, min_samples_leaf=101, criterion=gini |
| Random forest | 0.650 ± 0.019 (+0.082) | 0.680 ± 0.013 (+0.010) | 0.682 | max_depth=5, min_samples_leaf=3, max_features=sqrt |
| Oblique random forest | 0.647 ± 0.011 (+0.114) | 0.679 ± 0.013 (+0.013) | 0.683 | max_depth=6, min_samples_leaf=3, n_components=5 |
| AdaBoost | 0.676 ± 0.012 (+0.006) | 0.682 ± 0.011 (+0.006) | 0.686 | max_depth=2, n_estimators=366, learning_rate=0.31 |

I would choose AdaBoost: it has the best nested CV score (0.682) and the smallest gap (+0.006). However, its lead is not bigger than the fold-to-fold std (~0.011–0.013): it is only 0.002–0.003 above the two forests and 0.010 above logistic regression, so all tuned models are basically tied. The chosen hyperparameters mostly match how each model works: AdaBoost picked very shallow trees (depth 2), i.e. weak learners, because boosting reduces bias by adding many of them. I would report AdaBoost with its nested CV score, 0.682 ± 0.011, not the best-trial score 0.686, which is optimistic.

Best Model: AdaBoost (tuned). Best nested CV accuracy and no overfitting, but the difference to RF/ORF/LR is within the noise.


Week 7 and 8:

Changes: selection metric is now ROC AUC (instead of accuracy), feature selection with VoteSelector (min_votes tuned together with the model), new models Naive Bayes, Gradient Boosting, HistGradientBoosting and XGBoost.

### Boosting + all models (ROC AUC)
| Model | Default: CV mean ± std (gap) | Tuned: nested CV mean ± std (gap) | Tuning score (best trial) | Chosen hyperparameters |
|---|---|---|---|---|
| Dummy (baseline) | 0.500 ± 0.000 (+0.000) | — | — | — |
| Logistic regression | 0.716 ± 0.011 (+0.000) | — | — | — |
| Naive Bayes | 0.704 ± 0.015 (+0.001) | 0.708 ± 0.013 (+0.001) | 0.708 | var_smoothing=0.099, min_votes=2 |
| Decision tree | 0.657 ± 0.035 (+0.117) | 0.718 ± 0.014 (+0.016) | 0.720 | max_depth=5, min_samples_leaf=129, criterion=gini, min_votes=3 |
| Random forest | 0.672 ± 0.031 (+0.102) | 0.722 ± 0.013 (+0.012) | 0.724 | max_depth=6, min_samples_leaf=85, max_features=1.0, min_votes=3 |
| Oblique random forest | 0.672 ± 0.033 (+0.106) | 0.724 ± 0.012 (+0.014) | 0.725 | max_depth=23, min_samples_leaf=169, n_components=10, min_votes=2 |
| AdaBoost | 0.721 ± 0.014 (+0.005) | 0.723 ± 0.012 (+0.008) | 0.725 | max_depth=2, n_estimators=393, learning_rate=0.077, min_votes=1 |
| Gradient boosting | 0.723 ± 0.011 (+0.019) | **0.726 ± 0.013 (+0.005)** | 0.726 | n_estimators=171, learning_rate=0.015, max_depth=3, min_samples_leaf=19, subsample=0.53, min_votes=3 |
| HistGradientBoosting | 0.710 ± 0.015 (+0.051) | 0.725 ± 0.013 (+0.008) | 0.726 | max_iter=82, learning_rate=0.11, max_leaf_nodes=4, min_samples_leaf=7, l2≈0, min_votes=1 |
| XGBoost | 0.721 ± 0.013 (+0.027) | 0.725 ± 0.012 (+0.006) | 0.726 | n_estimators=156, learning_rate=0.18, max_depth=5, min_child_weight=2.3, subsample=0.67, colsample_bytree=0.62, gamma=3.5, min_votes=1 |

Gradient boosting has the best nested CV ROC AUC (0.726) and the smallest gap (+0.005), so it would be my choice. But, as in week 6, the lead is not real: the 7 tuned tree models are all between 0.718 and 0.726, a spread of 0.008, which is smaller than the fold-to-fold std (~0.012–0.013). Even logistic regression without tuning (0.716) is only 0.010 below. The optimism check is ~0 for every model (best-trial score ≈ nested CV score), so with 30 trials the tuning is not overfitting the CV.


### Fairness: false positive rate by race (out-of-fold, threshold 0.5)
| Model | Accuracy | Recall (class 1) | FPR African-American | FPR Caucasian | Difference |
|---|---|---|---|---|---|
| COMPAS score | — | — | 0.45 | 0.23 | 0.22 |
| Logistic regression | 0.667 | 0.66 | 0.42 | 0.24 | 0.18 |
| Naive Bayes | 0.631 | 0.28 | 0.13 | 0.05 | 0.08 |
| Decision tree | 0.670 | 0.59 | 0.34 | 0.20 | 0.14 |
| Random forest | 0.670 | 0.62 | 0.37 | 0.22 | 0.15 |
| Oblique random forest | 0.678 | 0.56 | 0.30 | 0.16 | 0.14 |
| AdaBoost | 0.678 | 0.57 | 0.31 | 0.17 | 0.14 |
| Gradient boosting | 0.681 | 0.56 | 0.29 | 0.16 | 0.13 |
| HistGradientBoosting | 0.673 | 0.66 | 0.41 | 0.25 | 0.16 |
| XGBoost | 0.680 | 0.57 | 0.30 | 0.16 | 0.14 |

All models have a lower FPR gap than COMPAS, but African-Americans who did not reoffend are still flagged about 1.8–2x more often than Caucasians, in every model, even though race is not used as a feature (age and priors_count carry it). The FPR mostly follows how many positives a model predicts: the models with class_weight=balanced (LR, HGB) have the highest recall and the highest FPRs, and Naive Bayes has the lowest FPRs only because it almost never predicts 1 (recall 0.28). So these numbers depend on the 0.5 threshold, which is not tuned yet. Accuracy is ~0.68 for the best models, the same as week 6 (0.682 for AdaBoost), so the new boosting models did not improve on it.

Best Model: Gradient boosting (tuned), 0.726 ± 0.013 ROC AUC (nested CV), no overfitting. The difference to the other boosting models, the forests and even logistic regression is within the noise. 

