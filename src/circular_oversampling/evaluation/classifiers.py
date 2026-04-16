"""
Classifier factory for the evaluation pipeline.

Provides a registry of seven classifiers commonly used in imbalanced-learning
benchmarks.  Each entry is a zero-argument factory (lambda) that returns a
*fresh*, unfitted estimator -- important when the same classifier must be
instantiated independently for every cross-validation fold.

Classifiers that are sensitive to feature scale (KNN, SVM, MLP, logistic
regression) are wrapped in a :class:`~sklearn.pipeline.Pipeline` with
:class:`~sklearn.preprocessing.StandardScaler`.
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# ---------------------------------------------------------------------------
# Registry: name -> factory callable (returns a fresh estimator each time)
# ---------------------------------------------------------------------------
CLASSIFIERS = {
    "knn": lambda: make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=5),
    ),
    "decision_tree": lambda: DecisionTreeClassifier(random_state=42),
    "svm_rbf": lambda: make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", probability=True, random_state=42),
    ),
    "naive_bayes": lambda: GaussianNB(),
    "mlp": lambda: make_pipeline(
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42),
    ),
    "logistic_regression": lambda: make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, random_state=42),
    ),
    "random_forest": lambda: RandomForestClassifier(
        n_estimators=100, random_state=42
    ),
}


def get_classifier(name):
    """Return a fresh, unfitted classifier instance by *name*.

    Parameters
    ----------
    name : str
        One of the keys in :data:`CLASSIFIERS`
        (e.g. ``"knn"``, ``"svm_rbf"``, ``"random_forest"``).

    Returns
    -------
    estimator : sklearn estimator or Pipeline
        A new estimator instance ready for ``.fit()``.

    Raises
    ------
    KeyError
        If *name* is not in the registry.
    """
    if name not in CLASSIFIERS:
        raise KeyError(
            f"Unknown classifier '{name}'. "
            f"Available: {list(CLASSIFIERS.keys())}"
        )
    return CLASSIFIERS[name]()


def get_all_classifiers():
    """Return a dict mapping every registered name to a fresh estimator.

    Returns
    -------
    classifiers : dict[str, estimator]
        ``{name: estimator_instance}`` for every entry in the registry.
    """
    return {name: factory() for name, factory in CLASSIFIERS.items()}


def list_classifier_names():
    """Return the list of registered classifier names.

    Returns
    -------
    names : list[str]
    """
    return list(CLASSIFIERS.keys())
