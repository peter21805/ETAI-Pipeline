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

Goals: 

1. Add preprocessing to pipeline using HOLDOUT, compare results with week 3

2. Add CV to your pipeline, compare BEFORE VS AFTER CV. We can do only this instead of 1.

3. Change something in your pipeline (Imputer -> KNN, encoder, etc.). Can you get a better modeling pipeline?
Also add the new models (Dummies & RF). 
