"""The 8 candidate classifiers compared in PathSnap (shared by 03_train_compare.py and demo_models.py).
Each entry: name -> (family, configuration text, constructor, optional training-row subsample)."""
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

from common import SEED

MODELS = {
    "LogReg": ("Linear", "StandardScaler + L2 logistic regression, C=1.0",
               lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=500)), None),
    "GaussianNB": ("Probabilistic", "Gaussian naive Bayes, var_smoothing=1e-9",
                   lambda: GaussianNB(), None),
    "KNN": ("Instance-based", "StandardScaler + distance-weighted KNN, k=25 (fit on 60k-row subsample)",
            lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=25, weights="distance", n_jobs=2)), 60000),
    "DecisionTree": ("Tree", "CART, max_depth=12, min_samples_leaf=20",
                     lambda: DecisionTreeClassifier(max_depth=12, min_samples_leaf=20, random_state=SEED), None),
    "RandomForest": ("Bagging ensemble", "100 trees, max_depth=20, min_samples_leaf=5, max_features=sqrt",
                     lambda: RandomForestClassifier(n_estimators=100, max_depth=20, min_samples_leaf=5, n_jobs=2, random_state=SEED), None),
    "ExtraTrees": ("Randomised ensemble", "150 trees, max_depth=20, min_samples_leaf=5",
                   lambda: ExtraTreesClassifier(n_estimators=150, max_depth=20, min_samples_leaf=5, n_jobs=2, random_state=SEED), None),
    "HistGB": ("Boosting", "HistGradientBoosting lr=0.1, max_iter=200, max_leaf_nodes=31 (untuned defaults)",
               lambda: HistGradientBoostingClassifier(learning_rate=0.1, max_iter=200, max_leaf_nodes=31, random_state=SEED), None),
    "MLP": ("Neural network", "StandardScaler + MLP(64,32) ReLU, Adam, alpha=1e-4, early stopping (fit on 200k-row subsample)",
            lambda: make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-4, max_iter=60,
                                                                  early_stopping=True, random_state=SEED)), 200000),
}

