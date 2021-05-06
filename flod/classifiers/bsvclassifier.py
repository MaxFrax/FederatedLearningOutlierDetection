import logging
import random
from typing import Iterable

import gurobipy as gp
import numpy as np
import tensorflow as tf
from gurobipy import GRB
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)

LOGGER = logging.getLogger(__name__)


class BSVClassifier(ClassifierMixin, BaseEstimator):

    def __init__(self, c: float = 1, q: float = 1, optimizer: str = 'tensorflow', p1: float = 1, p2: float = 1, p3: float = 1, p4: float = 1):
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.p4 = p4
        self.optimizer = optimizer
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

        if self.optimizer == 'gurobi':
            self.betas_, self.constant_term_ = BSVClassifier._solve_optimization_gurobi(
                self.X_train_, self.y_train_, self.c, self.q)
        else:
            #dataset = tf.data.Dataset.from_tensor_slices(self.X_train_)
            dataset = self.X_train_
            self.betas_, self.constant_term_ = BSVClassifier._solve_optimization_tensorflow(
                dataset, self.c, self.q, self.p1, self.p2, self.p3, self.p4)


        self.radiuses_ = np.array([self._compute_r(x, self.optimizer) for x in X])
        self.radius_ = self._best_radius(self.optimizer)

        # Return the classifier
        return self

    def predict(self, X):
        # Input validation
        # X = check_array(X)

        if self.betas_ is None or self.radiuses_ is None:
            LOGGER.error('You must call fit before predict!')

        rs = [self._compute_r(x, self.optimizer) for x in X]
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

    #@tf.function
    def _solve_optimization_tensorflow(dataset, c, q, p1, p2, p3, p4):

        optimizer = tf.keras.optimizers.Adam(learning_rate=0.005)

        # Init betas
        betas = []

        for _ in dataset:
            beta = tf.Variable(
                initial_value=np.random.sample()*c, trainable=True)
            betas.append(beta)

        self_kernels = np.array(
            [BSVClassifier._gaussian_kernel(x_i, x_i, q) for x_i in dataset])

        kernel_values = np.empty(
            (len(dataset), len(dataset)), dtype=np.float64)

        for i, xi in enumerate(dataset):
            for j, xj in enumerate(dataset):
                if j > i:
                    break

                kern = BSVClassifier._gaussian_kernel(xi, xj, q)
                kernel_values[i][j] = kern
                kernel_values[j][i] = kern

        kernels = tf.constant(kernel_values, dtype=tf.float64)
        zero = tf.constant(0, dtype=tf.float64)

        @tf.function
        def objective_function():
            constant_term = tf.tensordot(
                tf.linalg.matvec(kernels, betas), betas, axes=1)

            # The wolfe equation is the dual problem for the lagrangian.
            # This means we have to maximize the function.
            # Here i multiply everything for -1 so that I can still minimize
            v = constant_term
            v += tf.tensordot(betas, self_kernels, axes=1)

            # Penalize if sum of betas is not close to 1 from the bottom
            v += p1 * tf.math.maximum(zero, 1 - sum(betas))
            # Penalize if the sum of the betas goes above 1
            v += p2 * tf.math.maximum(zero, sum(betas) - 1)

            # Penalize for each beta below zero
            for b in betas:
                if b > 0:
                    v += -p3 * b

            # Penalize for each beta that goes above C
                if b - c > 0:
                    v += p4 * (b - c)

            return v

        # Gradient descent
        best_v = objective_function()
        best_betas = [tf.Variable(b) for b in betas]
        i = 0
        tot_iter = 0
        last_update = 0
        emergency_limit = 500

        while i < 10 and tot_iter < emergency_limit:
            optimizer.minimize(objective_function, betas)
            v = objective_function()

            i += 1
            tot_iter += 1

            if v < best_v:
                best_betas = [tf.Variable(b) for b in betas]
                i = 0
                last_update = best_v - v
                best_v = v

        tf.print(
            f'Optimized in {tot_iter} iterations. Latest update size: {last_update}')

        constant_term = tf.tensordot(tf.linalg.matvec(
            kernels, best_betas), best_betas, axes=1)

        return best_betas, constant_term

    def _compute_r(self, x, optimizer) -> float:
        v = self.constant_term_
        v += BSVClassifier._gaussian_kernel(x, x, self.q)

        if optimizer == 'gurobi':
            v += -2 * self.betas_ @ [self._gaussian_kernel(
                x_i, x, self.q) for x_i in self.X_train_]
        else:
            v += -2 * tf.tensordot(self.betas_, [self._gaussian_kernel(
                x_i, x, self.q) for x_i in self.X_train_], axes=1)

        v = np.sqrt(v)
        return v

    def _best_radius(self, optimizer) -> float:
        return np.average([self._compute_r(x, optimizer) for x in self.X_train_], weights=[b / self.c for b in self.betas_])

    @ staticmethod
    def true_negative_count(y_test, y_pred):
        tn, _, _, _ = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()

        return tn
