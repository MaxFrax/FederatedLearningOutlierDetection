import logging
from typing import Iterable
from datetime import datetime
from torch.distributions import constraints

import gurobipy as gp
import numpy as np
from gurobipy import GRB
from sklearn.base import BaseEstimator, ClassifierMixin

import torch

LOGGER = logging.getLogger(__name__)


class BSVClassifier(ClassifierMixin, BaseEstimator):

    def __init__(self, c: float = 1, q: float = 1, normal_class_label:int=0, outlier_class_label:int=1, initial_lr=0.1, lr_decay=0.9, max_iter=1000):
        self.q = q
        self.c = c
        self.X_ = None
        self.y_ = None
        self.betas_ = None
        self.constant_term_ = None
        self.radiuses_ = None
        self.normal_class_label = normal_class_label
        self.outlier_class_label = outlier_class_label
        self.initial_lr = initial_lr
        self.lr_decay = lr_decay
        self.max_iter = max_iter

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
            self.betas_, self.constant_term_ = self._solve_optimization_pytorch(
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

        if self.betas_ is None or self.radiuses_ is None:
            LOGGER.error('You must call fit before predict!')

        rs = [self._compute_r(x) for x in X]
        prediction = [self.outlier_class_label if ri > self.radius_ else self.normal_class_label for ri in rs]

        return np.array(prediction)

    def score_samples(self, X):
        if self.betas_ is None:
            LOGGER.error('You must call fit before score_samples!')

        return np.array([self.radius_ - self._compute_r(x) for x in X])

    @staticmethod
    def _gaussian_kernel(x_i: Iterable, x_j: Iterable, q: float) -> float:
        squared_norm = np.linalg.norm(np.array(x_i) - np.array(x_j)) ** 2
        return np.float64(np.exp(-1 * q * squared_norm))

    @staticmethod
    def _solve_optimization_gurobi(xs, c, q):

        def gaussian_kernel(
            x1, x2): return BSVClassifier._gaussian_kernel(x1, x2, q)

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
        #now = datetime.now()
        # To enable when debugging. If not debugging it just spams
        #model.write(f'{now}_BSVClassifier.mps')

        if model.status == GRB.INFEASIBLE:
            raise
            # model.computeIIS()
            # model.write(f'{now}_BSVClassifier IIS.ilp')

        best_betas = np.array([v.x for v in model.getVars()], dtype=np.float64)

        return best_betas, best_betas @ kernels @ best_betas

    def _solve_optimization_pytorch(self, xs, c, q):

        def gaussian_kernel(
            x1, x2): return BSVClassifier._gaussian_kernel(x1, x2, q)

        kernels = torch.tensor([[gaussian_kernel(xi, xj) for xj in xs] for xi in xs], dtype=torch.float64)

        def objective(betas):
            first_prod = torch.inner(betas, kernels)
            return torch.inner(first_prod, betas)

        # Initialize x
        betas = torch.rand((len(xs),), dtype=torch.float64) * c
        betas = betas.detach().requires_grad_(True)

        # Define the optimizer
        optimizer = torch.optim.SGD([betas], lr=self.initial_lr)

        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.lr_decay)
        self.objectives = []

        # Run the optimization
        for i in range(self.max_iter):
            optimizer.zero_grad()
            output = objective(betas)
            self.objectives.append(output.item())
            output.backward()

            with torch.no_grad():
                betas.data /= betas.data.sum()
                betas.data = torch.clamp(betas.data, 0, c)
                betas.data /= betas.data.sum()

            optimizer.step()
            scheduler.step()  # Exponentially decay the learning rate

            betas.data /= betas.data.sum()
            betas.data = torch.clamp(betas.data, 0, c)
            betas.data /= betas.data.sum()

        return betas, objective(betas)

    
    def _compute_r(self, x) -> float:
        self_kernels = torch.tensor([self._gaussian_kernel(x_i, x, self.q) for x_i in self.X_train_])
        v = 1.0 + self.constant_term_
        v += -2.0 * torch.inner(self.betas_ , self_kernels)
        v = torch.math.sqrt(v)
        return v

    def _best_radius(self) -> float:

        if len(self.X_train_) == 1:
            return 0

        sv = [x for b, x in zip(self.betas_, self.X_train_) if not torch.isclose(b, torch.tensor(self.c, dtype=torch.float64)) and not torch.isclose(b, torch.tensor(0.0, dtype=torch.float64))]
        assert len(sv) > 0, f'Cannot compute best radius. Missing support vectors among {len(self.X_train_)} datapoints. Maybe something went wrong during training?'
        return torch.mean(torch.tensor([self._compute_r(x) for x in sv], dtype=torch.float64))

    def decision_function(self, X):
        # Like sklearn OneClassSVM "Signed distance is positive for an inlier and negative for an outlier.""
        return self.score_samples(X)