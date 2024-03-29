import threading
import time
from typing import Union, Literal

import numpy as np
from joblib import Parallel, delayed
from scipy.optimize import minimize_scalar
from sklearn.tree import DecisionTreeRegressor

from .base_estimators import ConstantPredictor, BaseEstimatorUtils


class BootstrappedTrees:
    """Holder for trees and their feature and object subspaces
    """

    def __init__(self):
        self.trees_ = []
        self.bags_ = []
        self.subspaces_ = []

    @property
    def trees(self):
        """Getter for 'trees' field
        """
        return self.trees_

    @trees.setter
    def trees(self, value):
        """Setter for 'trees' field
        """
        self.trees_ = value

    @property
    def bags(self):
        """Getter for 'bags' field
        """
        return self.bags_

    @bags.setter
    def bags(self, value):
        """Setter for 'bags' field
        """
        self.bags_ = value

    @property
    def subspaces(self):
        """Getter for 'subspaces' field
        """
        return self.subspaces_

    @subspaces.setter
    def subspaces(self, value):
        """Setter for 'subspaces' field
        """
        self.subspaces_ = value


class BaseTreeEnsemble(BaseEstimatorUtils):
    def __init__(
            self,
            n_estimators,
            *,
            random_state: int = 42,
            base_tree,
            feature_subsample_size,
            max_samples,
            trace_loss: bool = False,
            trace_time: bool = False,
            **trees_parameters
    ):
        super().__init__(
            random_state=random_state,
            trace_loss=trace_loss,
            trace_time=trace_time
        )
        if n_estimators <= 0:
            raise ValueError(f'n_trees should be positive integer, got {n_estimators}')
        self.n_estimators = n_estimators
        self.base_tree_ = base_tree
        self.trees_parameters = trees_parameters
        self.feature_subsample_size = feature_subsample_size
        if isinstance(feature_subsample_size, float) \
                and ((feature_subsample_size > 1.) or (feature_subsample_size < 0.)):
            raise ValueError(f'Expected "feature_subsample_size" to be in [0., 1.], '
                             f'got {feature_subsample_size}')
        self.max_samples = max_samples

        self.estimators_ = BootstrappedTrees()

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

        self.estimators_.bags = np.random.choice(
            X.shape[0],
            (self.n_estimators, bag_size),
            replace=True
        )

        self.estimators_.subspaces = [np.random.choice(
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
            trace_loss: bool = False,
            trace_time: bool = False,
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
            trace_loss=trace_loss,
            trace_time=trace_time,
            **trees_parameters
        )
        self.n_jobs = n_jobs

    @staticmethod
    def _parallel_predict(pred_fn, X, out, lock, partial=False):
        pred = pred_fn(X)
        with lock:
            if not partial:
                out[0] += pred
            else:
                out.append(pred)

    def _partial_predictions(self, X):
        self._check_if_fitted()

        pred = []
        lock = threading.Lock()
        Parallel(
            n_jobs=self.n_jobs,
            prefer='threads',
            require='sharedmem'
        )(delayed(self._parallel_predict)(
            tree.predict,
            X.take(self.estimators_.subspaces[idx], 1),
            pred,
            lock,
            True
        ) for idx, tree in enumerate(self.estimators_.trees))

        return np.cumsum(pred, axis=0) / np.arange(1, self.n_estimators + 1)[:, None]

    def _fit_tree(self, X, y, idx, lock=None):
        t0 = time.time()
        self.estimators_.trees[idx].fit(
            X.take(self.estimators_.bags[idx], 0).take(self.estimators_.subspaces[idx], 1),
            y[self.estimators_.bags[idx]]
        )
        t0 = time.time() - t0

        if self.trace_time_:
            if lock is not None:
                with lock:
                    self.append_time_(t0)
            else:
                raise ValueError('Expected Lock object to trace time')

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
        self._set_fitted()

        self.estimators_.trees = [self.base_tree_(
            **self.trees_parameters,
            random_state=np.random.randint(np.iinfo(np.int32).max)
        ) for _ in range(self.n_estimators)]

        lock = threading.Lock() if self.trace_time_ else None
        Parallel(
            n_jobs=self.n_jobs,
            prefer='threads',
        )(delayed(self._fit_tree)(X, y, idx, lock) for idx in range(self.n_estimators))

        if self.trace_loss_:
            train_diff = self._partial_predictions(X) - y
            self.append_train_loss(
                np.sqrt(np.mean(train_diff ** 2, axis=1))
            )
            if (X_val is not None) and (y_val is not None):
                val_diff = self._partial_predictions(X_val) - y_val
                self.append_val_loss(
                    np.sqrt(np.mean(val_diff ** 2, axis=1))
                )

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
        self._check_if_fitted()

        pred = np.zeros((X.shape[0]), dtype=np.float64)
        lock = threading.Lock()
        Parallel(
            n_jobs=self.n_jobs,
            prefer='threads',
            require='sharedmem'
        )(delayed(self._parallel_predict)(
            tree.predict,
            X.take(self.estimators_.subspaces[idx], 1),
            [pred],
            lock
        ) for idx, tree in enumerate(self.estimators_.trees))

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
            trace_loss: bool = False,
            trace_time: bool = False,
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
            trace_loss=trace_loss,
            trace_time=trace_time,
            **trees_parameters
        )

        self.learning_rate = learning_rate
        self._weights = []
        self._const_pred = None

    def _partial_predictions(self, X):
        self._check_if_fitted()

        tree_preds = [tree.predict(
            X.take(subspace, 1)
        ) for subspace, tree in zip(
            self.estimators_.subspaces,
            self.estimators_.trees
        )]

        return self._const_pred.predict(X) + self.learning_rate \
            * np.cumsum(np.multiply(np.array(self._weights)[:, None],
                                    tree_preds), axis=0)

    def fit(self, X, y, X_val=None, y_val=None):
        """
        X : numpy ndarray
            Array of size n_objects, n_features
        y : numpy ndarray
            Array of size n_objects
        """
        self._set_params_fit(X)
        self._set_fitted()

        self._const_pred = ConstantPredictor('mean').fit(y=y)

        curr_target = self._const_pred.predict(X)

        for idx in range(self.n_estimators):
            t0 = time.time() if self.trace_time_ else None

            self.estimators_.trees.append(self.base_tree_(**self.trees_parameters))
            self.estimators_.trees[idx].fit(
                X.take(self.estimators_.bags[idx], 0).take(self.estimators_.subspaces[idx], 1),
                (y - curr_target)[self.estimators_.bags[idx]]
            )

            if self.trace_time_:
                t0 = time.time() - t0
                self.append_time_(t0)

            pred = self.estimators_.trees[idx].predict(X.take(self.estimators_.subspaces[idx], 1))

            self._weights.append(minimize_scalar(
                lambda x, prediction, target: np.linalg.norm(
                    target + x * prediction - y
                ),
                bounds=(0,),
                args=(pred, curr_target)
            ).x)

            curr_target = curr_target + self.learning_rate * self._weights[idx] * pred

        if self.trace_loss_:
            train_diff = self._partial_predictions(X) - y
            self.append_train_loss(
                np.sqrt(np.mean(train_diff ** 2, axis=1))
            )
            if (X_val is not None) and (y_val is not None):
                val_diff = self._partial_predictions(X_val) - y_val
                self.append_val_loss(
                    np.sqrt(np.mean(val_diff ** 2, axis=1))
                )

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
        self._check_if_fitted()

        tree_preds = np.array([tree.predict(
            X.take(subspace, 1)
        ) for subspace, tree in zip(
            self.estimators_.subspaces,
            self.estimators_.trees
        )])

        return self._const_pred.predict(X) + self.learning_rate * np.dot(self._weights, tree_preds)
