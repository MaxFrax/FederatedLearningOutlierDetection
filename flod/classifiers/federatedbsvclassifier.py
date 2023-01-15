import numpy as np
import logging
import sys
from ..classifiers.bsvclassifier import BSVClassifier
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.base import BaseEstimator, ClassifierMixin
from collections import defaultdict
from scipy.stats import uniform

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(level=logging.DEBUG)

fileHandler = logging.FileHandler(filename=f'{__name__}.log')
fileHandler.setLevel(level=logging.DEBUG)

streamHandler = logging.StreamHandler(stream=sys.stdout)
streamHandler.setLevel(level=logging.DEBUG)

LOGGER.addHandler(fileHandler)
LOGGER.addHandler(streamHandler)

# Reference paper: Communication-Efficient Learning of Deep Network from Decentralized Data

class FederatedBSVClassifier(ClassifierMixin, BaseEstimator):

    def __init__(self, client_fraction:float = 1, total_clients:int = 10, max_rounds: int = 10, normal_class_label:int=0, outlier_class_label:int=1, B:int = 10):
        self.normal_class_label = normal_class_label
        self.outlier_class_label = outlier_class_label
        self.client_fraction = client_fraction # Known as C in the reference paper
        self.total_clients = total_clients # Known as K in the paper
        self.max_rounds = max_rounds
        self.B = B

    def __getstate__(self) -> dict:
        state = self.__dict__
        return state

    def __setstate__(self, state):
        self.__dict__ = state

    def init_server_model(self, dimensions_count):
        return {
            'q': 0.01,
            'c': 1,
            'betas': np.empty(shape=(0, )),
            'xs': np.empty(shape=(0, dimensions_count))
    }

    def fit(self, X, Y, client_assignment):
        # Debug
        self.sv_count = []

        clients_x = defaultdict(list)
        clients_y = defaultdict(list)
        clf = None

        for i, x in enumerate(X):
            assignment = client_assignment[i]
            clients_x[assignment].append(x)
            clients_y[assignment].append(Y[i])

        model = self.init_server_model(X.shape[1])

        for r in range(self.max_rounds):
            LOGGER.debug(f'Round {r} of {self.max_rounds}')
            selected_clients_count = max(1, self.total_clients * self.client_fraction)
            clients_ix = np.random.choice(range(self.total_clients), int(selected_clients_count), replace=False)

            LOGGER.debug(f'Selected clients {clients_ix}')

            updates = []

            for c_ix in clients_ix:
                round_client_x = []
                round_client_y = []
                
                if r*self.B < len(clients_x[c_ix]):
                    lowerbound = min(r * self.B, len(clients_x[c_ix]))
                    upperbound = max((r+1) * self.B, len(clients_x[c_ix]))
                    round_client_x = clients_x[c_ix][lowerbound:upperbound]
                    round_client_y = clients_y[c_ix][lowerbound:upperbound]

                    updates.append(self.client_compute_update(c_ix, model, round_client_x , round_client_y))
                else:
                    LOGGER.warning(f'Client run {c_ix} ran out of data')

            model, clf = self.global_combine(model, updates)

            # Debug
            if clf:
                self.sv_count.append(np.count_nonzero(clf.betas_))
            else:
                self.sv_count.append(0)

        self.clf = clf

        return self

    def predict(self, X):
        return self.clf.predict(X)

    def client_compute_update(self, index, global_model, client_data_x, client_data_y):

        if len(client_data_x) == 0:
            LOGGER.warning(f'Client run {index} ran out of data')
            return np.empty(shape=(0, )), np.empty(shape=(0, ))

        # Concat points from server and from client
        X = np.concatenate((global_model['xs'], client_data_x))
        y = np.array([self.outlier_class_label if np.isclose(b, global_model['c']) else self.normal_class_label for b in global_model['betas']])
        y = np.concatenate((y, client_data_y))

        if self.normal_class_label not in y:
            LOGGER.warning(f'Client {index} does not have normal class datapoints among the {len(client_data_x)} points')
            return np.empty(shape=(0, )), np.empty(shape=(0, ))

        # Init the classifier with q and C from server
        # TODO probably should look for q and c in the client, send them to the server and average them server wise. Or grid search among them
        clf = BSVClassifier(q=global_model['q'], c=global_model['c'], normal_class_label=1, outlier_class_label=-1)


        # Train locally
        clf.fit(X, y)

        # Select only the positive betas related from client_data
        client_betas = clf.betas_[len(global_model['xs']):]
        client_xs = clf.X_train_[len(global_model['xs']):]
        assert(len(client_betas) == len(client_xs))

        xs = []
        betas = []
        
        for _, t in enumerate(zip(client_betas, client_xs)):
            b, x = t
            if not np.isclose(b, 0):
                xs.append(x)
                betas.append(b)

        if len(xs) == 0:
            LOGGER.warning(f'There is no client {index} update. No betas far from zero among all the {len(client_xs)} points')
        
        return np.array(xs), np.array(betas)

    def global_combine(self, global_model, client_updates):
        # Concatenates server points and clients candidate points.
        X = global_model['xs']
        betas = global_model['betas']
        for update in client_updates:
            xs, bs = update
            if xs.shape[0] == 0:
                break
            X = np.concatenate((X, xs))
            betas = np.concatenate((betas, bs))

        y = np.array([self.outlier_class_label if np.isclose(b, global_model['c']) else self.normal_class_label for b in betas])

        # Let's try also old hyper parameters. Notice: C is missing
        search_params = {
            'q': [global_model['q']],
            'c': [global_model['c']]
        }

        if len(X) == 0:
            LOGGER.warning("No datapoints after combining global model and client update. Is everything okay?")
            return global_model, None

        # TODO q should go from minimum proposed in 4.2 of svc paper to current +- 1
        search_params['q'].extend(uniform(loc=0, scale=1).rvs(size=3))
        search_params['c'].extend(uniform(loc=1/len(X), scale=1).rvs(size=3))
        
        # Performs model selection over this new dataset
        test_fold = [0 if v < len(X) else 1 for v in range(len(X) * 2)]
        clf = GridSearchCV(BSVClassifier(outlier_class_label=self.outlier_class_label, normal_class_label=self.normal_class_label), search_params, cv=PredefinedSplit(test_fold=test_fold), n_jobs=-1, scoring='average_precision', refit=True, return_train_score=False, error_score='raise')
        clf.fit(np.concatenate((X,X)), np.concatenate((y,y)))
        
        # Filter and keep only the support vectors
        xs = []
        betas = []
        for _, t in enumerate(zip(clf.best_estimator_.betas_, X)):
            b, x = t
            if not np.isclose(b, 0):
                xs.append(x)
                betas.append(b)

        return {
            'q': clf.best_estimator_.q,
            'c': clf.best_estimator_.c,
            'betas': np.array(betas),
            'xs': np.array(xs)
        }, clf.best_estimator_

    def decision_function(self, X):
        if self.clf:
            return self.clf.decision_function(X)
        else:
            return np.random.choice([self.normal_class_label, self.outlier_class_label], len(X))

    def score_samples(self, X):
        return self.clf.score_samples(X)