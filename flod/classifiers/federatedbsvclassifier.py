from flod.classifiers.bsvclassifier import BSVClassifier
import numpy as np
import logging
import gurobipy as gp
from sklearn.base import BaseEstimator, ClassifierMixin
from collections import defaultdict
from flod.utils import error_code_to_string

LOGGER = logging.getLogger(__name__)

class FederatedBSVClassifier(ClassifierMixin, BaseEstimator):

    def __init__(self, method, max_rounds: int = 10, normal_class_label:int=0, outlier_class_label:int=1, q:float=1, C:float=1):
        self.normal_class_label = normal_class_label
        self.outlier_class_label = outlier_class_label
        self.client_fraction = 1
        self.total_clients = 2
        self.max_rounds = max_rounds
        self.q = q
        self.C = C
        self.method = method

        self.classes_ = [self.outlier_class_label, self.normal_class_label]

    def __getstate__(self) -> dict:
        state = self.__dict__
        return state

    def __setstate__(self, state):
        self.__dict__ = state

    def init_server_model(self):
        return {
            'sum_betas': np.full(self.total_clients, 1/self.total_clients),
            'f_norms': np.zeros(self.total_clients),
            'Ws': np.zeros(self.total_clients)
    }

    def _compute_gamma(self, X, y, client_assignment):
        gamma = 0

        clf = BSVClassifier(c=self.C, q=self.q, normal_class_label=self.normal_class_label, outlier_class_label=self.outlier_class_label).fit(X, y)

        # Determine which points belong to which client
        client_betas = [[] for _ in range(self.total_clients)]

        for i, beta in enumerate(clf.betas_):
            client_betas[client_assignment[i]].append((i, beta))
        
        # Compute norm of client 0
        norm0 = 0
        sum_beta0 = 0
        for i, b1 in client_betas[0]:
            sum_beta0 += b1
            for j, b2 in client_betas[0]:
                norm0 += b1 * b2 * BSVClassifier._gaussian_kernel(X[i], X[j], self.q)
        norm0 = np.sqrt(norm0)
        # Compute norm of client 1
        norm1 = 0
        sum_beta1 = 0
        for i, b1 in client_betas[1]:
            sum_beta1 += b1
            for j, b2 in client_betas[1]:
                norm1 += b1 * b2 * BSVClassifier._gaussian_kernel(X[i], X[j], self.q)
        norm1 = np.sqrt(norm1)
        # Compute inner product of client 0 and client 1
        inner_product = 0
        for i, b1 in client_betas[0]:
            for j, b2 in client_betas[1]:
                inner_product += b1 * b2 * BSVClassifier._gaussian_kernel(X[i], X[j], self.q)

        # Compute ideal iper parameters
        gamma = (norm0 * norm1) - inner_product
        cos = inner_product / (norm0 * norm1)

        return cos, gamma, [sum_beta0, sum_beta1], [norm0, norm1], clf

    def fit(self, X, y, client_assignment, round_callback):
        if sum(y) != len(y)*self.normal_class_label:
            LOGGER.warning('FederatedBSVClassifier is not designed to train with outliers. All outliers will be ignored')
        self.X_train_, self.y_train_, self.client_assignment_train = [], [], []

        for i, y in enumerate(y):
            if y == self.normal_class_label:
                self.y_train_.append(y)
                self.X_train_.append(X[i])
                self.client_assignment_train.append(client_assignment[i])

        self.debug = []

        clients_x = defaultdict(list)
        clients_y = defaultdict(list)

        # Divides data among clients
        for i, x in enumerate(self.X_train_):
            assignment = self.client_assignment_train[i]
            clients_x[assignment].append(x)
            clients_y[assignment].append(self.y_train_[i])

        model = self.init_server_model()
        self.cos, self.gamma, self.opt_betas, self.opt_norms, _ = self._compute_gamma(self.X_train_, self.y_train_, self.client_assignment_train)

        # DEBUG uses fixed optimal values to optmize
        model['sum_betas'] = np.array(self.opt_betas)
        model['f_norms'] = np.array(self.opt_norms)

        r = 0
        converged = False
        while r < self.max_rounds and not converged:
            LOGGER.info('\n-----------------------------------')
            LOGGER.info(f'Round {r} of {self.max_rounds-1}. Ws: {model["Ws"]}')
            
            r+=1

            # Selects clients to participate in this round
            # Its a feature for the future. As of today clients are always 0 and 1 at each round.
            selected_clients_count = max(1, self.total_clients * self.client_fraction)
            clients_ix = np.random.choice(range(self.total_clients), int(selected_clients_count), replace=False)

            LOGGER.debug(f'Selected clients {clients_ix}')

            updates = []

            for c_ix in clients_ix:
                updates.append(self.client_compute_update(c_ix, model, clients_x[c_ix], clients_y[c_ix]))

            model, converged = self.global_combine(r, model, updates)

            if callable(round_callback):
                round_callback(self)

        # TODO train estimator for each client

        return self

    def client_compute_update(self, index, global_model, client_data_x, client_data_y):

        assert len(client_data_x) >= 0, f'Client {index} does not have any data to compute its update'

        kernels = np.empty((len(client_data_x), len(client_data_x)), dtype=np.float64)

        # TODO compress memory of symmetric matrix. Use a smart class or decopose the matrix
        # tril, triu
        for i, xi in enumerate(client_data_x):
            for j, xj in enumerate(client_data_x):
                if j > i:
                    break

                kern = BSVClassifier._gaussian_kernel(xi, xj, self.q)
                kernels[i][j] = kern
                kernels[j][i] = kern

        with gp.Model(f'Client_{index}') as opt:
            LOGGER.info(f'\nClient_{index}')
            # Parameters configuration
            opt.setParam('OutputFlag', 0)
            opt.setParam('TimeLimit', 600)
            # Suggested by parameter tuning
            opt.setParam('Method', 2)
            opt.setParam('PrePasses', 1)
            opt.setParam('NonConvex', 2)
            
            opt.ModelSense = gp.GRB.MINIMIZE

            # Variables
            betas = opt.addMVar(shape=len(client_data_x), name='betas', ub=self.C, lb=0)
            root = opt.addVar(name='root')
            inner_product = opt.addVar(name='inner_product', lb=0)

            # Constraints
            sum_betas = opt.addConstr(betas.sum() == global_model['sum_betas'][index], name='sum_betas')
            opt.addConstr(inner_product == betas @ kernels @ betas)
            opt.addGenConstrPow(root, inner_product, .5, "square_root", options="FuncPieces=640000")

            other_ix = (index+1)%2
            other_client_norm = global_model['f_norms'][other_ix]
            assert other_ix != index, f'Client {index} is trying to optimize with itself'

            objective = None
            
            if self.method =='gamma':
                objective = inner_product + 2 * other_client_norm * root -2*self.gamma + other_client_norm**2
            elif self.method == 'cos': 
                objective = inner_product + 2 * other_client_norm * root * self.cos + other_client_norm**2
            else:
                raise ValueError(f'Invalid method {self.method}. Only "gamma" and "cos" are valid methods')

            opt.setObjective(objective)
            opt.optimize()

            try:
                W = opt.objVal
                norm = root.x

                if not np.isclose(root.x**2,inner_product.x):
                    LOGGER.error(f'Root and its square are not close. Improve gurobi model. Root^2: {root.x**2:.4f}, square: {inner_product.x:.4f}, error: {root.x**2 - inner_product.x}')

                if index == 0:
                    self.betas0 = [b for b in betas.x]
                    self.fc0 = lambda x: sum([b *  BSVClassifier._gaussian_kernel(x, xj, self.q) for xj, b in zip(client_data_x, self.betas0)])
                    self.fc0 = lambda x: sum([b *  BSVClassifier._gaussian_kernel(x, xj, self.q) for xj, b in zip(client_data_x, self.betas0) if b > 0 and b < self.C])
                    self.radiuses0 = [self.fc0(s) for s, b in zip(client_data_x, self.betas0) if b > 0 and b < self.C]
                elif index == 1:
                    self.betas1 = [b for b in betas.x]
                    self.fc1 = lambda x: sum([b *  BSVClassifier._gaussian_kernel(x, xj, self.q) for xj, b in zip(client_data_x, self.betas1)])
                    self.fc1 = lambda x: sum([b *  BSVClassifier._gaussian_kernel(x, xj, self.q) for xj, b in zip(client_data_x, self.betas1) if b > 0 and b < self.C])
                    self.radiuses1 = [self.fc1(s) for s, b in zip(client_data_x, self.betas1) if b > 0 and b < self.C]

            except:
                error = error_code_to_string(opt.Status)
                LOGGER.error(f'Client {index} failed to optimize. Status: {error}')
                W = np.nan
                norm = np.nan
                raise

        return W, norm

    def global_combine(self, round, global_model, client_updates):
        model = global_model
        model['Ws'] = np.array([u[0] for u in client_updates])
        # model['f_norms'] = np.array([u[1] for u in client_updates])
        converged = np.isclose(model['Ws'][0] - model['Ws'][1], 0)

        # Stores the parameters that produced the Ws
        self.debug.append({
            'round': round,
            'W0': model['Ws'][0],
            'W1': model['Ws'][1],
            'f_norm0': global_model['f_norms'][0],
            'f_norm1': global_model['f_norms'][1],
            'sum_beta0': global_model['sum_betas'][0],
            'sum_beta1':global_model['sum_betas'][1]
        })

        if not converged:
            model['sum_betas'] = np.array(self.opt_betas)
            model['f_norms'] = np.array(self.opt_norms)

            if not np.isclose(model['sum_betas'].sum(), 1.0):
                model['sum_betas'] /= model['sum_betas'].sum()
        
        assert np.isclose(model['sum_betas'].sum(), 1.0), f'betas sum is not 1, but {model["sum_betas"].sum()}.Sums {model["sum_betas"]}'
        assert model['sum_betas'].min() >= 0.0, f'betas min is not 0, but {model["sum_betas"].min()}'
        assert model['sum_betas'].max() <= 1.0, f'betas max is not 1, but {model["sum_betas"].max()}'

        return model, converged
    
    def predict(self, X):
        radius = np.average(self.radiuses0) + np.average(self.radiuses1)
        return [self.outlier_class_label if self.fc0(x) + self.fc1(x) < radius else self.normal_class_label for x in X]

    def decision_function(self, X):
        return self.score_samples(X)

    def score_samples(self, X):
        radius = np.average(self.radiuses0) + np.average(self.radiuses1)

        # Score samples needs to return the biggest value for the most normal sample
        # and the lowest value to the most outlier sample
        # Given that an anomaly is a sample with a prediction lower than the prediction of the support vecotors, here's the formula:
        return np.array([self.fc0(x) + self.fc1(x) - radius for x in X])