"""
Boosting
========

In machine learning, boosting is an ensemble meta-algorithm for primarily reducing bias, and also variance, see
`Boosting (Wikipedia) <https://en.wikipedia.org/wiki/Boosting_(machine_learning)>`_. This example implements the
classical AdaBoost algorithm proposed by:

    Freund, Yoav, and Robert E. Schapire. "A decision-theoretic generalization of on-line learning and an application to
    boosting." Journal of computer and system sciences 55.1 (1997): 119-139.

For multiclass extensions of the AdaBoost approach, see:

    Hastie, Trevor, et al. "Multi-class adaboost." Statistics and its Interface 2.3 (2009): 349-360.

Let's start with the imports.
"""

import numpy as np
from matplotlib import pyplot as plt
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.datasets import make_blobs
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted

from graphtask import Task, step
from graphtask.visualize import to_pygraphviz


# %%
# Creating the classifier
# -----------------------
#
# Let us define the classifier, such that it inherits from ``Task`` and we can use ``step`` to build the DAG.

class AdaBoostClassifier(BaseEstimator, ClassifierMixin, Task):
    def __init__(self, base_estimator=None, n_estimators=50):
        super().__init__()

        if base_estimator is None:
            base_estimator = DecisionTreeClassifier(max_depth=1)

        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.register(models=[clone(self.base_estimator) for _ in range(self.n_estimators)])

    def fit(self, X, y):
        check_classification_targets(y)
        self.classes_, y = np.unique(y, return_inverse=True)
        self.n_classes_ = len(self.classes_)
        self.register(x_train=X, y=y, weights=np.repeat(1 / len(y), len(y)))
        self.run("_fit_estimator")
        return self

    def predict(self, X):
        class_idx = np.argmax(self.predict_proba(X), axis=1)
        return self.classes_.take(class_idx, axis=0)

    def predict_proba(self, X):
        check_is_fitted(self)
        self.register(x_test=X)
        return self.run("_predict_proba")

    @step
    def _fit_estimator(self, models, x_train, y, weights):
        # this loop cannot be parallelized, because we iteratively update the weights
        for model in models:
            model.fit(x_train, y, sample_weight=weights)
            predicted_labels = model.predict(x_train)
            incorrect_labels = y != predicted_labels
            mean_error = np.average(incorrect_labels, weights=weights)
            weight_factor = np.log((1 - mean_error) / mean_error)
            weights *= np.exp(weight_factor * incorrect_labels)
            yield weight_factor, model

    @step(map="_fit_estimator")
    def _predict_all(self, _fit_estimator, x_test):
        weight, model = _fit_estimator
        y_pred = model.predict(x_test)
        return [weight * (y_pred == cls) for cls in range(self.n_classes_)]

    @step
    def _predict_proba(self, _predict_all):
        # return the sum of all estimator predictions
        return np.array(_predict_all).sum(axis=0).transpose()

# %%
# Let us now instantiate the classifier and look at the inferred DAG.

model = AdaBoostClassifier()
to_pygraphviz(model).draw("model.png")


# %%
# Let's run the classifier on some example data.
X, y = make_blobs(random_state=42)
y_pred = model.fit(X, y).predict(X)

# %%
# Do the predictions make sense?
fig, ax = plt.subplots(1, 1)
ax.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.show()
