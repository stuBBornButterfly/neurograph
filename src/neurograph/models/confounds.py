import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class SexResidualizer(BaseEstimator, TransformerMixin):
    # last column = sex; removes its linear effect from every other column (fit on training
    # people only), then drops the sex column so the model never sees it
    def fit(self, X, y=None):
        A = np.column_stack([np.ones(len(X)), X[:, -1]])
        self.coef_ = np.linalg.lstsq(A, X[:, :-1], rcond=None)[0]
        return self

    def transform(self, X):
        A = np.column_stack([np.ones(len(X)), X[:, -1]])
        return X[:, :-1] - A @ self.coef_
