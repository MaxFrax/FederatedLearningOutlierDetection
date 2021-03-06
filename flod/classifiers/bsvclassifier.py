import logging
import random
from typing import Iterable

import gurobipy as gp
import numpy as np
from gurobipy import GRB
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)

LOGGER = logging.getLogger(__name__)


class BSVClassifier(ClassifierMixin, BaseEstimator):

    def __init__(self, c: float = 1, q: float = 1):
        self.q = q
        self.c = c
        self.X_ = None
        self.y_ = None
        self.betas_ = None
        self.constant_term_ = None
        self.radiuses_ = None
        self.sv_i = None
        self.sv_ = None

    def __getstate__(self) -> dict:
        state = self.__dict__
        return state

    def __setstate__(self, state):
        self.__dict__ = state

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

        self.betas_, self.constant_term_ = BSVClassifier._solve_optimization_gurobi(
            self.X_train_, self.y_train_, self.c, self.q)

        self.radiuses_ = np.array([self._compute_r(x) for x in X])

        self.radius_ = self._best_radius()

        # Return the classifier
        return self

    def predict(self, X):
        # Input validation
        # X = check_array(X)

        if self.betas_ is None or self.radiuses_ is None:
            LOGGER.error('You must call fit before predict!')

        rs = [self._compute_r(x) for x in X]
        prediction = [int(ri > self.radius_) for ri in rs]

        return np.array(prediction)

    @staticmethod
    def _gaussian_kernel(x_i: Iterable, x_j: Iterable, q: float) -> float:
        squared_norm = np.linalg.norm(np.array(x_i) - np.array(x_j)) ** 2
        return np.float64(np.exp(-1 * q * squared_norm))

    @staticmethod
    def _solve_optimization_gurobi(xs, y, c, q):

        def gaussian_kernel(
            x1, x2): return BSVClassifier._gaussian_kernel(x1, x2, q)

        self_kernels = np.array([gaussian_kernel(x_i, x_i) for x_i in xs])
        kernels = np.empty((len(xs), len(xs)), dtype=np.float64)

        for i, xi in enumerate(xs):
            for j, xj in enumerate(xs):
                if j > i:
                    break

                kern = gaussian_kernel(xi, xj)
                kernels[i][j] = kern
                kernels[j][i] = kern

        model = gp.Model('WolfeDual')

        betas = model.addMVar(len(xs), name="betas", ub=c, lb=0)

        sum_betas = model.addConstr(sum(betas) == 1, name="sum_betas")

        model.ModelSense = GRB.MAXIMIZE

        model.setParam('ObjScale', -0.5)

        model.setParam('NumericFocus', 3)

        model.setObjective(self_kernels @ betas - betas @
                           kernels @ betas)


        model.optimize()

        best_betas = np.array([v.x for v in model.getVars()], dtype=np.float64)

        return best_betas, best_betas @ kernels @ best_betas

    def _compute_r(self, x) -> float:
        v = self.constant_term_
        v += BSVClassifier._gaussian_kernel(x, x, self.q)
        v += -2 * self.betas_ @ [self._gaussian_kernel(
            x_i, x, self.q) for x_i in self.X_train_]
        v = np.sqrt(v)
        return v

    def _best_radius(self) -> float:        
        return np.average([self._compute_r(x) for x in self.X_train_], weights=[b / self.c for b in self.betas_])

    @ staticmethod
    def true_negative_count(y_test, y_pred):
        tn, _, _, _ = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()

        return tn
