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


