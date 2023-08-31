"""
Stacking
========

Stacking (short for "stacked generalization") is an ensemble learning technique that combines multiple models via a
supervised meta model. The base-level models are trained based on a complete training set, then the meta model is
trained on the outputs of the base models as features. The concept of stacked generalization was introduced by:

    Wolpert, David H. "Stacked generalization." Neural networks 5.2 (1992): 241-259.

Let's create a stacking model with scikit-learn.
"""

import numpy as np
from matplotlib import pyplot as plt
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted

from graphtask import Task, step

# %%
# A minimal stacking task
# -----------------------
#
# Let's create a ``Task`` and add some minimal steps for a stacking classifier.

base_estimators = [LogisticRegression(), KNeighborsClassifier(), DecisionTreeClassifier()]
final_estimator = RandomForestClassifier()

task = Task()
task.register(base_estimators=base_estimators, final_estimator=final_estimator)


@task.step(map="base_estimators")
def fit_predict_base(base_estimators, X, y):
    base_estimators.fit(X, y)
    return base_estimators.predict(X)


@task.step()
def fit_predict_final(fit_predict_base, y):
    stacked_features = np.column_stack(fit_predict_base)
    final_estimator.fit(stacked_features, y)
    return final_estimator.predict(stacked_features)


# %%
# Let's run the task on some example data.
X, y = make_blobs(random_state=42)
y_pred = task(X=X, y=y)


# %%
# Do the predictions make sense?
fig, ax = plt.subplots(1, 1)
ax.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.show()
