from typing import Any

import numpy as np


class ConstantPredictor:
    def __init__(
            self,
            method: str = 'mean'
    ):
        self.method = method
        if self.method not in {'mean', 'zero'}:
            raise ValueError(f'Expected "method" to be one of {"mean", "zero"}, '
                             f'got {method}')
        self._pred_val = None
        self._fitted = False

    def fit(
            self,
            *,
            y: Any = None
    ):
        """
        :param y: target vector, needed if method is 'mean'
        :return: fitted ConstantPredictor object
        """
        if (self.method == 'mean') and (y is None):
            raise ValueError('"mean" method is not supported if y is None')
        if self.method == 'mean':
            self._pred_val = y.mean()
        elif self.method == 'zero':
            self._pred_val = 0.
        self._fitted = True
        return self

    def predict(self, X):
        """
        :param X: Used to determine the length of prediction vector
        :return: constant-valued prediction vector
        """
        if not self._fitted:
            raise RuntimeError('Estimator is not fitted')
        return np.full(X.shape[0], self._pred_val)