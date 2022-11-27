import threading
from typing import Union, Literal, Any

import numpy as np
from joblib import Parallel, delayed
from scipy.optimize import minimize_scalar
from sklearn.tree import DecisionTreeRegressor


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


class BaseTreeEnsemble:
    def __init__(
            self,
            n_estimators,
            *,
            random_state: int = 42,
            base_tree,
            feature_subsample_size,
            max_samples,
            **trees_parameters
    ):
        if n_estimators <= 0:
            raise ValueError(f'n_trees should be positive integer, got {n_estimators}')
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.base_tree_ = base_tree
        self.trees_parameters = trees_parameters
        self.feature_subsample_size = feature_subsample_size
        if isinstance(feature_subsample_size, float) \
                and ((feature_subsample_size > 1.) or (feature_subsample_size < 0.)):
            raise ValueError(f'Expected "feature_subsample_size" to be in [0., 1.], '
                             f'got {feature_subsample_size}')
        self.max_samples = max_samples

        self._trees = None
        self._bags = None
        self._subspaces = None

    def _set_params_fit(self, X):
        if self.random_state is not None:
            np.random.seed(self.random_state)
        if self.feature_subsample_size == 'auto':
            self.feature_subsample_size = max(
                int(np.floor(X.shape[1] / 3)),
                1
            )
        else:
            self.feature_subsample_size = max(
                int(np.floor(self.feature_subsample_size * X.shape[1])),
                1
            )

        if self.max_samples == 'auto':
            bag_size = int(max(
                (1 - (1 - 1 / X.shape[0]) ** X.shape[0]) * X.shape[0],
                1
            ))
        else:
            bag_size = int(self.max_samples * X.shape[0])

        self._bags = np.random.choice(
            X.shape[0],
            (self.n_estimators, bag_size),
            replace=True
        )

        self._subspaces = [np.random.choice(
            X.shape[1],
            self.feature_subsample_size,
            replace=False
        ) for _ in range(self.n_estimators)]

    def fit(self, X, y):
        """
        X : numpy ndarray
            Array of size n_objects, n_features
        y : numpy ndarray
            Array of size n_objects
        """
        raise NotImplementedError

    def predict(self, X):
        """
        X : numpy ndarray
            Array of size n_objects, n_features
        Returns
        -------
        y : numpy ndarray
            Array of size n_objects
        """
        raise NotImplementedError

    def score(self, X, y, squared=True):
        """Compute MSE on given data
        """
        diff = self.predict(X) - y
        mse = 1 / len(y) * np.inner(diff, diff)
        if squared:
            return mse
        return np.sqrt(mse)


class RandomForestMSE(BaseTreeEnsemble):
    def __init__(
            self,
            n_estimators: int,
            *,
            feature_subsample_size: Union[Literal['auto'], float] = 'auto',
            max_samples: Union[Literal['auto'], float] = 'auto',
            random_state: int = 42,
            n_jobs: Union[None, int] = None,
            **trees_parameters
    ):
        """
        n_estimators : int
            The number of trees in the forest.
        max_depth : int
            The maximum depth of the tree. If None then there is no limits.
        feature_subsample_size : float
            The size of feature set for each tree. If 'auto' then use one-third of all features.
        """

        super().__init__(
            n_estimators=n_estimators,
            random_state=random_state,
            base_tree=DecisionTreeRegressor,
            feature_subsample_size=feature_subsample_size,
            max_samples=max_samples,
            **trees_parameters
        )
        self.n_jobs = n_jobs

    def _fit_tree(self, X, y, idx):
        self._trees[idx].fit(
            X.take(self._bags[idx], 0).take(self._subspaces[idx], 1),
            y[self._bags[idx]]
        )
        return self._trees[idx]

    def fit(self, X, y, X_val=None, y_val=None):
        """
        X : numpy ndarray
            Array of size n_objects, n_features
        y : numpy ndarray
            Array of size n_objects
        X_val : numpy ndarray
            Array of size n_val_objects, n_features
        y_val : numpy ndarray
            Array of size n_val_objects
        """
        self._set_params_fit(X)

        self._trees = [self.base_tree_(
            **self.trees_parameters,
            random_state=np.random.randint(np.iinfo(np.int32).max)
        ) for _ in range(self.n_estimators)]

        self._trees = Parallel(
            n_jobs=self.n_jobs,
            prefer='threads',
        )(delayed(self._fit_tree)(X, y, idx) for idx in range(self.n_estimators))

        return self

    @staticmethod
    def _parallel_predict(pred_fn, X, out, lock):
        pred = pred_fn(X)
        with lock:
            out[0] += pred

    def predict(self, X):
        """
        X : numpy ndarray
            Array of size n_objects, n_features
        Returns
        -------
        y : numpy ndarray
            Array of size n_objects
        """
        pred = np.zeros((X.shape[0]), dtype=np.float64)
        lock = threading.Lock()
        Parallel(
            n_jobs=self.n_jobs,
            prefer='threads',
            require='sharedmem'
        )(delayed(self._parallel_predict)(
            tree.predict,
            X.take(self._subspaces[idx], 1),
            [pred],
            lock
        ) for idx, tree in enumerate(self._trees))

        pred /= self.n_estimators
        return pred


class GradientBoostingMSE(BaseTreeEnsemble):
    def __init__(
            self,
            n_estimators,
            *,
            learning_rate: float = 0.1,
            max_depth: int = 3,
            feature_subsample_size: Union[Literal['auto'], float] = 'auto',
            max_samples: float = 1.,
            random_state: int = 42,
            **trees_parameters
    ):
        """
        n_estimators : int
            The number of trees in the forest.
        learning_rate : float
            Use alpha * learning_rate instead of alpha
        max_depth : int
            The maximum depth of the tree. If None then there is no limits.
        feature_subsample_size : float
            The size of feature set for each tree. If 'auto' then use one-third of all features.
        """

        super().__init__(
            n_estimators=n_estimators,
            random_state=random_state,
            base_tree=DecisionTreeRegressor,
            feature_subsample_size=feature_subsample_size,
            max_samples=max_samples,
            max_depth=max_depth,
            **trees_parameters
        )

        self.learning_rate = learning_rate
        self._weights = None
        self._const_pred = None

    def fit(self, X, y, X_val=None, y_val=None):
        """
        X : numpy ndarray
            Array of size n_objects, n_features
        y : numpy ndarray
            Array of size n_objects
        """

        self._set_params_fit(X)

        self._const_pred = ConstantPredictor('mean').fit(y=y)
        self._trees = [self.base_tree_(**self.trees_parameters)
                       for _ in range(self.n_estimators)]
        self._weights = np.ones(self.n_estimators, dtype=np.float64)

        curr_target = self._const_pred.predict(X)

        for idx in range(self.n_estimators):
            self._trees[idx].fit(
                X.take(self._bags[idx], 0).take(self._subspaces[idx], 1),
                (y - curr_target)[self._bags[idx]]
            )

            pred = self._trees[idx].predict(X.take(self._subspaces[idx], 1))

            self._weights[idx] = minimize_scalar(
                lambda x, prediction, target: np.linalg.norm(
                    target + x * prediction - y
                ),
                bounds=(0,),
                args=(pred, curr_target)
            ).x

            curr_target = curr_target + self.learning_rate * self._weights[idx] * pred

        return self

    def predict(self, X):
        """
        X : numpy ndarray
            Array of size n_objects, n_features
        Returns
        -------
        y : numpy ndarray
            Array of size n_objects
        """

        tree_preds = np.array([tree.predict(
            X.take(subspace, 1)
        ) for subspace, tree in zip(
            self._subspaces,
            self._trees
        )])

        return self._const_pred.predict(X) + self.learning_rate * np.dot(self._weights, tree_preds)
