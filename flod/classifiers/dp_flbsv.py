from flod.classifiers.bsvclassifier import BSVClassifier
import numpy as np
import logging
import gurobipy as gp
from sklearn.base import BaseEstimator, ClassifierMixin
from collections import defaultdict

LOGGER = logging.getLogger(__name__)

class DPFLBSV(ClassifierMixin, BaseEstimator):

    def __init__(self, noise:float, tol:float, max_rounds: int = 1, normal_class_label:int=0, outlier_class_label:int=1, q:float=1, C:float=1):
        self.normal_class_label = normal_class_label
        self.outlier_class_label = outlier_class_label
        self.client_fraction = 1
        self.total_clients = 2
        self.max_rounds = max_rounds
        self.q = q
        self.C = C
        self.noise = noise
        self.tol = tol

        self.classes_ = [self.outlier_class_label, self.normal_class_label]

    def __getstate__(self) -> dict:
        state = self.__dict__
        return state

    def __setstate__(self, state):
        self.__dict__ = state

    def init_server_model(self, dimensions_count):
        return {
            'q': 1,
            'c': 1,
            'betas': np.empty(shape=(0, )),
            'xs': np.empty(shape=(0, dimensions_count))
    }

    def fit(self, X, y, client_assignment, round_callback):
        if sum(y) != len(y)*self.normal_class_label:
            LOGGER.warning('FederatedBSVClassifier is not designed to train with outliers. All outliers will be ignored')
        self.X_train_, self.y_train_, self.client_assignment_train = [], [], []

        for i, y in enumerate(y):
            if y == self.normal_class_label:
                self.y_train_.append(y)
                self.X_train_.append(X[i])
                self.client_assignment_train.append(client_assignment[i])        

        clients_x = defaultdict(list)
        clients_y = defaultdict(list)

        # Divides data among clients
        for i, x in enumerate(self.X_train_):
            assignment = self.client_assignment_train[i]
            clients_x[assignment].append(x)
            clients_y[assignment].append(self.y_train_[i])

        model = self.init_server_model(X.shape[1])
        
        for r in range(self.max_rounds):
            LOGGER.info(f'Round {r}')
            # Selects clients to participate in this round
            # Its a feature for the future. As of today clients are always 0 and 1 at each round.
            selected_clients_count = max(1, self.total_clients * self.client_fraction)
            clients_ix = np.random.choice(range(self.total_clients), int(selected_clients_count), replace=False)

            LOGGER.debug(f'Selected clients {clients_ix}')

            updates = []

            for c_ix in clients_ix:
                LOGGER.info(f'Client {c_ix} update')
                updates.append(self.client_compute_update(c_ix, model, clients_x[c_ix], clients_y[c_ix]))

            model = self.global_combine(r, model, updates)

            if callable(round_callback):
                round_callback(self)

        return self

    def client_compute_update(self, index, global_model, client_data_x, client_data_y):

        assert len(client_data_x) >= 0, f'Client {index} does not have any data to compute its update'

        # Filter out anomalies from the training data
        train_x, train_y = [], []

        for i, x in enumerate(client_data_x):
            if client_data_y[i] == self.normal_class_label:
                train_x.append(x)
                train_y.append(client_data_y[i])

        # If there is no normal data, we can't train the model
        if len(train_x) == 0:
            LOGGER.error(f'Client {index} does not have any normal data to train the model')
            return None

        combined_x = np.concatenate([global_model['xs'], np.array(train_x)])
        combined_y = np.concatenate([[self.normal_class_label] * len(global_model['xs']), np.array(train_y)])
        clf = BSVClassifier(q=self.q, c=self.C, normal_class_label=self.normal_class_label, outlier_class_label=self.outlier_class_label)
        clf.fit(combined_x, combined_y)

        support_vectors = []
        for b, x in zip(clf.betas_, clf.X_train_):
            if not np.isclose(b, self.C) and not np.isclose(b, 0):
                support_vectors.append(x)

        if self.noise > 0:
            for i in range(len(support_vectors)):
                new_p = self.anonimize_point(support_vectors[i], clf)
                support_vectors[i]  = new_p

        return support_vectors
    
    def anonimize_point(self, point, clf):
        candidate = point + np.random.normal(0, self.noise, point.shape)

        while abs(clf._compute_r(point) - clf._compute_r(candidate)) > self.tol:
            candidate -= (candidate - point) * 0.1

        if np.isclose(np.linalg.norm(candidate - point), 0.0):
            LOGGER.warning('Anonymization failed')

        return candidate

    def global_combine(self, round, global_model, client_updates):
        model = global_model.copy()

        all_support_vectors = np.concatenate([global_model['xs'], *client_updates])
        y = np.array([self.normal_class_label] * len(all_support_vectors))

        clf = BSVClassifier(c=self.C, q=self.q, normal_class_label=self.normal_class_label, outlier_class_label=self.outlier_class_label).fit(all_support_vectors, y)
        clf.fit(all_support_vectors, y)

        support_vectors, betas = [], []
        for b, x in zip(clf.betas_, clf.X_train_):
            if not np.isclose(b, self.C) and not np.isclose(b, 0):
                support_vectors.append(x)
                betas.append(b)

        model['xs'] = np.array(support_vectors)
        model['betas'] = np.array(betas)

        self.clf = clf

        return model
    
    def predict(self, X):
        return self.clf.predict(X)

    def decision_function(self, X):
        return self.clf.decision_function(X)

    def score_samples(self, X):
        return self.clf.score_samples(X)