import logging
import random
from typing import Iterable

import numpy as np
import tensorflow as tf
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)

LOGGER = logging.getLogger(__name__)


class BSVClassifier(ClassifierMixin, BaseEstimator):

    def __init__(self, q: float = 1, random_seed: int = 42, init_bound: float = .1, n_iter: int = 100, penalization: int = 10):
        self.random_seed = random_seed
        self.init_bound = init_bound
        self.q = q
        self.c = 1
        self.n_iter = n_iter
        # Should the optimizer be a parameter?
        self.optimizer = tf.optimizers.Adam(learning_rate=1e-4)
        self.penalization = penalization
        self.X_ = None
        self.y_ = None
        self.betas_ = None
        self.constant_term_ = None
        self.radiuses_ = None
        self.sv_i = None
        self.sv_ = None

    def __getstate__(self) -> dict:
        state = self.__dict__
        if state.get('optimizer'):
            del state['optimizer']
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        self.optimizer = tf.optimizers.Adam(learning_rate=1e-4)

    def fit(self, X, y):

        # TODO input validation

        self.X_ = X
        self.y_ = y

        LOGGER.debug(f'Solving optimization {len(X)} with c {self.c}')

        self.X_train_, self.y_train_ = [], []

        for i, y in enumerate(y):
            if y == 0:
                self.y_train_.append(y)
                self.X_train_.append(X[i])

        self.X_train_ = np.array(self.X_train_)
        self.y_train_ = np.array(self.y_train_)

        self.betas_, self.constant_term_ = BSVClassifier._solve_optimization_tensorflow(
            self.X_train_, self.y_train_, self.random_seed, self.init_bound, self.c, self.q, self.n_iter, self.optimizer, self.penalization)

        self.radiuses_ = [self._compute_r(x) for x in X]

        self.sv_i, self.sv_ = self._best_precision_sv()

        # Return the classifier
        return self

    def predict(self, X):
        # Input validation
        # X = check_array(X)

        if self.betas_ is None or self.radiuses_ is None:
            LOGGER.error('You must call fit before predict!')

        rs = [self._compute_r(x) for x in X]
        prediction = [int(ri > self.radiuses_[self.sv_i]) for ri in rs]

        return np.array(prediction)

    @staticmethod
    def _gaussian_kernel(x_i: Iterable, x_j: Iterable, q: float) -> float:
        squared_norm = np.linalg.norm(np.array(x_i) - np.array(x_j)) ** 2
        return np.float64(np.exp(-1 * q * squared_norm))

    @staticmethod
    def _solve_optimization_tensorflow(xs, y, random_seed, init_bound, c, q, n_iter, optimizer, penalization):

        if abs(init_bound) > abs(c):
            LOGGER.warning(
                f'init_bound cannot be bigger than c. Defaulting to init_bound=c')
            init_bound = c

        betas = []

        LOGGER.debug(f'Optimizing with q {q}')

        def gaussian_kernel(
            x1, x2): return BSVClassifier._gaussian_kernel(x1, x2, q)

        # Inits beta values randomly
        np.random.seed(random_seed)
        init_values = np.random.uniform(0, init_bound, len(xs))
        for i in range(len(xs)):
            # Ho fissato le variabili legate agli outliers. Sono che hanno beta=C. Perchè farle cambiare?
            init = c if y[i] == 1 else init_values[i]
            beta = tf.Variable(
                init, name=f'beta_{i}', trainable=True, dtype=tf.float64)
            betas.append(beta)

        self_kernels = [gaussian_kernel(x_i, x_i) for x_i in xs]
        kernel_values = np.array(
            [[gaussian_kernel(x1, x2) for x1 in xs] for x2 in xs])
        kernels = tf.constant(kernel_values, dtype='float64')

        def objective_function():
            constant_term = tf.tensordot(
                tf.linalg.matvec(kernels, betas), betas, axes=1)

            # The wolfe equation is the dual problem for the lagrangian.
            # This means we have to maximize the function.
            # Here i multiply everything for -1 so that I can still minimize
            v = constant_term
            v += tf.tensordot(betas, self_kernels, axes=1)

            # Penalize if sum of betas is not close to 1 from the bottom
            v += penalization * tf.math.maximum(0, 1 - sum(betas))
            # Penalize if the sum of the betas goes above 1
            v += penalization * tf.math.maximum(0, sum(betas) - 1)

            # Penalize for each beta below zero
            v += -penalization * sum([x for x in betas if x < 0])

            # Penalize for each beta that goes above C
            error_c = sum([(b-c) for b in betas if b - c > 0])
            # error_c = sum([(b-c) / c for b in betas if b - c > 0])
            v += penalization * error_c

            return v

        best_v = objective_function()
        best_betas = [b.numpy() for b in betas]
        i = 0
        tot_iter = 0
        last_update = 0
        emergency_limit = 3000

        while i < n_iter and tot_iter < emergency_limit:
            optimizer.minimize(objective_function, betas)
            v = objective_function()

            i += 1
            tot_iter += 1

            if v < best_v:
                best_betas = [b.numpy() for b in betas]
                i = 0
                last_update = best_v - v
                best_v = v

        LOGGER.info(
            f'Optimized in {tot_iter} iterations. Latest update size: {last_update}')

        constant_term = tf.tensordot(tf.linalg.matvec(
            kernels, best_betas), best_betas, axes=1)

        return best_betas, constant_term

    def _compute_r(self, x) -> float:
        v = self.constant_term_
        v += BSVClassifier._gaussian_kernel(x, x, self.q)
        v += -2 * tf.tensordot(self.betas_, [self._gaussian_kernel(x_i, x, self.q) for x_i in self.X_train_], axes=1)
        v = np.sqrt(v)
        return v

    def _best_precision_sv(self):
        score = 0
        best_i = 0
        sv = None

        metric = f1_score

        if sum(self.y_) == 0:
            metric = BSVClassifier.true_negative_count

        for i, x in enumerate(self.X_):
            r = self.radiuses_[i]

            prediction = [int(ri > r) for ri in self.radiuses_]
            s = metric(self.y_, prediction)

            if s > score:
                score = s
                best_i = i
                sv = x

        return best_i, sv

    @ staticmethod
    def true_negative_count(y_test, y_pred):
        tn, _, _, _ = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()

        return tn
