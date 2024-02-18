from flod.classifiers.bsvclassifier import BSVClassifier
import numpy as np
import logging
import gurobipy as gp
from sklearn.base import BaseEstimator, ClassifierMixin
from collections import defaultdict

LOGGER = logging.getLogger(__name__)

class DPFLBSV(ClassifierMixin, BaseEstimator):

    def __init__(self, noise:float, tol:float, max_rounds: int = 1, normal_class_label:int=0, outlier_class_label:int=1, q:float=1, C:float=1, total_clients:int=2, client_fraction:float=1):
        self.normal_class_label = normal_class_label
        self.outlier_class_label = outlier_class_label
        self.client_fraction = client_fraction
        self.total_clients = total_clients
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
        self.dimensions_count = dimensions_count
        return {
            'q': 1,
            'c': 1,
            'betas': np.empty(shape=(0, )),
            'xs': np.empty(shape=(0, dimensions_count))
    }

    def fit(self, X, y=None, client_assignment=None, round_callback=None):
        assert X.shape[0] == client_assignment.shape[0], 'X and client_assignment must have the same number of rows'

        if len(np.unique(client_assignment)) != self.total_clients:
            LOGGER.warning(f'It seems like some clients do not have any data. Expected {self.total_clients} clients, but found {len(np.unique(client_assignment))}')  

        clients_x = defaultdict(list)

        # Divides data among clients
        for i, x in enumerate(X):
            assignment = client_assignment[i]
            clients_x[assignment].append(x)

        self.model = self.init_server_model(X.shape[1])
        
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
                updates.append(self.client_compute_update(c_ix, self.model, clients_x[c_ix]))

            self.model = self.global_combine(r, self.model, updates)

            if callable(round_callback):
                round_callback(self)

        return self

    def client_compute_update(self, index, global_model, client_data_x):

        if len(client_data_x) <= 0:
            LOGGER.warning(f'Client {index} does not have any data to compute its update')
            return np.empty(shape=(0, self.dimensions_count))

        combined_x = np.concatenate([global_model['xs'], np.array(client_data_x)])
        clf = BSVClassifier(q=self.q, c=self.C, normal_class_label=self.normal_class_label, outlier_class_label=self.outlier_class_label)

        try:
            clf.fit(combined_x)
        except:
            LOGGER.warning(f'Client {index} failed to train the model over {combined_x.shape[0]} samples')
            return np.empty(shape=(0, self.dimensions_count))

        support_vectors = []
        for b, x in zip(clf.betas_, clf.X_):
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

        self.all_support_vectors = np.concatenate([global_model['xs'], *client_updates])
        y = np.array([self.normal_class_label] * len(self.all_support_vectors))

        clf = BSVClassifier(c=self.C, q=self.q, normal_class_label=self.normal_class_label, outlier_class_label=self.outlier_class_label).fit(self.all_support_vectors, y)
        clf.fit(self.all_support_vectors, y)

        support_vectors, betas = [], []
        for b, x in zip(clf.betas_, clf.X_):
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