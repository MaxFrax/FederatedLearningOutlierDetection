import logging
from typing import Iterable
from datetime import datetime

import gurobipy as gp
import numpy as np
from gurobipy import GRB
from sklearn.base import BaseEstimator, ClassifierMixin

LOGGER = logging.getLogger(__name__)


class BSVClassifier(ClassifierMixin, BaseEstimator):

    def __init__(self, c: float = 1, q: float = 1, normal_class_label:int=0, outlier_class_label:int=1):
        self.q = q
        self.c = c
        self.X_ = None
        self.y_ = None
        self.betas_ = None
        self.constant_term_ = None
        self.radiuses_ = None
        self.normal_class_label = normal_class_label
        self.outlier_class_label = outlier_class_label

        self.classes_ = [self.outlier_class_label, self.normal_class_label]

        assert (self.c > 0 and self.c <= 1), f"0 < {self.c} <= 1"

    def __getstate__(self) -> dict:
        state = self.__dict__
        return state

    def __setstate__(self, state):
        self.__dict__ = state

    def fit(self, X, y):
        # Input validation
        assert len(X) > 0, f'You cannot fit with X of length 0'
        assert len(y) > 0, f'You cannot fit with y of length 0'
        assert self.c >= 1 / len(X), f'c must be at least {1 / len(X)} to satisfy the constraint on the betas'

        self.X_ = X
        self.y_ = y

        LOGGER.debug(f'Solving optimization {len(X)} with c {self.c}')

        self.X_train_, self.y_train_ = [], []

        for i, y in enumerate(y):
            if y == self.normal_class_label:
                self.y_train_.append(y)
                self.X_train_.append(X[i])

        self.X_train_ = np.array(self.X_train_)
        self.y_train_ = np.array(self.y_train_)

        assert len(self.X_train_) > 0, f"There is no normal data for training among the provided {len(X)}"

        try:
            self.betas_, self.constant_term_ = BSVClassifier._solve_optimization_gurobi(
                self.X_train_, self.c, self.q)
        except gp.GurobiError:
            raise
        except:
            LOGGER.error(f'c: {self.c}')
            LOGGER.error(f'q: {self.q}')
            LOGGER.error(f'normale: {self.normal_class_label}')
            LOGGER.error(f'outlier: {self.outlier_class_label}')
            LOGGER.error(f'X len {self.X_train_.shape}')
            LOGGER.error(f'y len {self.y_train_.shape}')
            self.X_train_.dump('failedX')
            self.y_train_.dump('failedY')
            raise

        self.radiuses_ = [self._compute_r(x) for x in X]

        self.radius_ = self._best_radius()

        # Return the classifier
        return self

    def predict(self, X):
        rs = [self._compute_r(x) for x in X]
        prediction = [self.outlier_class_label if ri > self.radius_ else self.normal_class_label for ri in rs]

        return np.array(prediction)

    def score_samples(self, X):
        return np.array([self.radius_ - self._compute_r(x) for x in X])

    @staticmethod
    def _gaussian_kernel(x_i: Iterable, x_j: Iterable, q: float) -> float:
        squared_norm = np.linalg.norm(np.array(x_i) - np.array(x_j)) ** 2
        return np.float64(np.exp(-1 * q * squared_norm))

    @staticmethod
    def _solve_optimization_gurobi(xs, c, q):

        def gaussian_kernel(
            x1, x2): return BSVClassifier._gaussian_kernel(x1, x2, q)

        self_kernels = np.array([gaussian_kernel(x_i, x_i) for x_i in xs])
        # TODO compress memory of symmetric matrix. Use a smart class or decopose the matrix
        kernels = np.empty((len(xs), len(xs)), dtype=np.float64)

        for i, xi in enumerate(xs):
            for j, xj in enumerate(xs):
                if j > i:
                    break

                kern = gaussian_kernel(xi, xj)
                kernels[i][j] = kern
                kernels[j][i] = kern

        model = gp.Model('WolfeDual')

        model.setParam('OutputFlag', 0)
        model.setParam('TimeLimit', 120)

        model.setParam('TimeLimit', 120)
        # Params suggested by auto tuner
        model.setParam('Method', 0)
        model.setParam('ScaleFlag', 0)
        model.setParam('SimplexPricing', 3)
        model.setParam('NumericFocus', 1)
        model.setParam('Presolve', 0)

        betas = model.addMVar(len(xs), name="betas", ub=c, lb=0)

        sum_betas = model.addConstr(sum(betas) == 1, name="sum_betas")

        model.ModelSense = GRB.MINIMIZE

        model.setObjective(betas @ kernels @ betas)

        model.optimize()
        # To enable when debugging. If not debugging it just spams
        #model.write(f'{now}_BSVClassifier.mps')

        if model.status == GRB.INFEASIBLE:
            now = datetime.now()
            model.computeIIS()
            model.write(f'{now}_BSVClassifier IIS.ilp')
            raise

        best_betas = np.array([v.x for v in model.getVars()], dtype=np.float64)

        return best_betas, best_betas @ kernels @ best_betas

    def _compute_r(self, x) -> float:
        if self.constant_term_ is not None:
            v = 1 + self.constant_term_
            v += -2 * self.betas_ @ [self._gaussian_kernel(
                x_i, x, self.q) for x_i in self.X_train_]
            v = np.sqrt(v)
            return v
        else:
            raise Exception('You must call fit before computing the radius')

    def _best_radius(self) -> float:

        if len(self.X_train_) == 1:
            return 0

        sv = self.get_support_vectors()
        assert len(sv) > 0, f'Cannot compute best radius. Missing support vectors among {len(self.X_train_)} datapoints. Maybe something went wrong during training?'
        return np.average([self._compute_r(x) for x in sv])

    def decision_function(self, X):
        #Like sklearn OneClassSVM "Signed distance is positive for an inlier and negative for an outlier.""
        return self.score_samples(X)
    
    def get_support_vectors(self):
        return [x for b, x in zip(self.betas_, self.X_train_) if not np.isclose(b, self.c) and not np.isclose(b, 0)]