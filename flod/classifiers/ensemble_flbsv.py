from flod.classifiers.bsvclassifier import BSVClassifier
import numpy as np
import logging
import gurobipy as gp
from sklearn.base import BaseEstimator, ClassifierMixin
from collections import defaultdict

LOGGER = logging.getLogger(__name__)

class EnsembleFLBSV(ClassifierMixin, BaseEstimator):

    def __init__(self, normal_class_label:int=0, outlier_class_label:int=1, q:float=1, C:float=1):
        self.normal_class_label = normal_class_label
        self.outlier_class_label = outlier_class_label
        self.client_fraction = 1
        self.total_clients = 2
        self.q = q
        self.C = C

        self.classes_ = [self.outlier_class_label, self.normal_class_label]

    def __getstate__(self) -> dict:
        state = self.__dict__
        return state

    def __setstate__(self, state):
        self.__dict__ = state

    def init_server_model(self):
        return []

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

        self.model = self.init_server_model()
        r = 0
        
        # Selects clients to participate in this round
        # Its a feature for the future. As of today clients are always 0 and 1 at each round.
        selected_clients_count = max(1, self.total_clients * self.client_fraction)
        clients_ix = np.random.choice(range(self.total_clients), int(selected_clients_count), replace=False)

        LOGGER.debug(f'Selected clients {clients_ix}')

        updates = []

        for c_ix in clients_ix:
            LOGGER.info(f'Client {c_ix} update')
            updates.append(self.client_compute_update(c_ix, self.model, clients_x[c_ix], clients_y[c_ix]))

        self.model = self.global_combine(r, self.model, updates)

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

        clf = BSVClassifier(q=self.q, c=self.C, normal_class_label=self.normal_class_label, outlier_class_label=self.outlier_class_label)
        clf.fit(train_x, train_y)

        return clf

    def global_combine(self, round, global_model, client_updates):
        model = global_model.copy()

        model = client_updates

        return model
    
    def _predict_one(self, x):
        votes = [clf.predict([x]) for clf in self.model]
        if all(vote == self.outlier_class_label for vote in votes):
            return self.outlier_class_label
        else:
            return self.normal_class_label
    
    def predict(self, X):
        return [self._predict_one(x) for x in X]

    def decision_function(self, X):
        return self.score_samples(X)

    def score_samples(self, X):
        scores = np.full(shape=(len(X), ), fill_value=1.0)
        
        evaluations = np.zeros(shape=(len(X), len(self.model)))
        for i, clf in enumerate(self.model):
            evaluations[:, i] = clf.score_samples(X)

        max_scores = np.max(evaluations, axis=1)
        sum_scores = np.sum(evaluations, axis=1)

        # The intuition is:
        # If a point is an inlier, it's "inlierness" score depends only on the client voting for it with the highest score
        # If a point is an outlier, it's "outlierness" score depends on the sum of the scores of all clients. 
        # So that if a point is a global outlier it will probably be more penalized than a local one
        for i in range(len(scores)):
            if max_scores[i] >= 0:
                scores[i] = max_scores[i]
            else:
                scores[i] = sum_scores[i]

        assert np.array_equal([self.outlier_class_label if s < 0 else self.normal_class_label for s in scores], self.predict(X)), 'Scores and predictions are not equal'

        return scores
