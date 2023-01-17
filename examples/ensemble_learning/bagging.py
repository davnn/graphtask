"""
Bagging
=======

Bootstrap aggregating, also called *bagging*, is a machine learning ensemble meta-algorithm designed to improve the
stability and accuracy of machine learning algorithms, see: `Bootstrap aggregating <https://en.wikipedia.org/wiki/Bootstrap_aggregating>`_.

This example shows how you can create a `scikit-learn <https://scikit-learn.org/stable/index.html>`_ `bagging classifier <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html>`_
using graphtask. Let us first start with the necessary imports.
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import ClassifierMixin, BaseEstimator, clone
from sklearn.datasets import make_blobs
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted

from graphtask import Task, step
from graphtask.visualize import to_pygraphviz


# %%
# Creating the classifier
# -----------------------
#
# Let us define the classifier, such that it inherits from ``Task`` and we can use ``step`` to build the DAG.

class BaggingClassifier(BaseEstimator, ClassifierMixin, Task):
    def __init__(
        self,
        base_estimator=None,
        n_estimators=10,
        *,
        max_samples=1.0,
        bootstrap=True
    ):
        super().__init__()
        if base_estimator is None:
            base_estimator = KNeighborsClassifier()

        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.bootstrap = bootstrap

    def fit(self, X, y):
        check_classification_targets(y)
        self.classes_, y = np.unique(y, return_inverse=True)
        self.n_classes_ = len(self.classes_)
        self.register(X_fit=X, y_fit=y)
        self.run("_fit_estimator")
        return self

    def fit_predict(self, X, y):
        self.fit(X, y)
        return self.predict(X)

    def predict(self, X):
        class_idx = np.argmax(self.predict_proba(X), axis=1)
        return self.classes_.take(class_idx, axis=0)

    def predict_proba(self, X):
        check_is_fitted(self)
        self.register(X_pred=X)
        bootstrap_probabilities = self.run("_predict_estimator")
        return np.mean(bootstrap_probabilities, axis=0)

    @step
    def _bootstrap(self, X_fit, y_fit):
        assert 0 < self.max_samples <= 1
        n_samples, n_features = X_fit.shape
        n_bootstrap_samples = int(n_samples * self.max_samples)
        assert n_bootstrap_samples > 0

        for _ in range(self.n_estimators):
            samples = np.random.choice(range(n_samples), n_bootstrap_samples)
            yield X_fit[samples], y_fit[samples]

    @step(map_arg="_bootstrap")
    def _fit_estimator(self, _bootstrap):
        X, y = _bootstrap
        estimator = clone(self.base_estimator)
        estimator.fit(X, y)
        return estimator

    @step(map_arg="_fit_estimator")
    def _predict_estimator(self, X_pred, _fit_estimator):
        return _fit_estimator.predict_proba(X_pred)


# %%
# Let us now instantiate the classifier with a k-nearest neighbors model and look at the inferred DAG.

model = BaggingClassifier(base_estimator=KNeighborsClassifier())
to_pygraphviz(model).draw("model.png")

# %%
# Let's run the classifier on some example data.
X, y = make_blobs(random_state=42)
y_pred = model.fit_predict(X, y)

# %%
# Do the predictions make sense?
fig, ax = plt.subplots(1, 1)
ax.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.show()
