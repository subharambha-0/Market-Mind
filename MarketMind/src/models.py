from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 7


def make_model(name, task, **p):
    """task: 'clf' or 'reg'. Scaling lives INSIDE the pipeline so it is fitted on training rows only."""
    if name == "logistic":
        return make_pipeline(StandardScaler(), LogisticRegression(C=p["C"], max_iter=1000))
    if name == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=p["alpha"]))
    if name == "rf":
        cls = RandomForestClassifier if task == "clf" else RandomForestRegressor
        return cls(n_estimators=200, min_samples_leaf=p["min_samples_leaf"], max_features="sqrt", random_state=SEED, n_jobs=-1)
    if name == "hgb":
        cls = HistGradientBoostingClassifier if task == "clf" else HistGradientBoostingRegressor
        return cls(learning_rate=p["learning_rate"], max_depth=p["max_depth"], max_iter=150, random_state=SEED)
    raise ValueError(name)


def predict(model, task, X):
    return model.predict_proba(X)[:, 1] if task == "clf" else model.predict(X)
