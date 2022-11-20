import numpy as np
from ..classifiers.bsvclassifier import BSVClassifier
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.base import BaseEstimator, ClassifierMixin
from collections import defaultdict
from scipy.stats import uniform

# Reference paper: Communication-Efficient Learning of Deep Network from Decentralized Data

class FederatedBSVClassifier(ClassifierMixin, BaseEstimator):

    def __init__(self, client_fraction:float = 1, total_clients:int = 10, max_rounds: int = 10, normal_class_label:int=0, outlier_class_label:int=1):
        self.normal_class_label = normal_class_label
        self.outlier_class_label = outlier_class_label
        self.client_fraction = client_fraction # Known as C in the reference paper
        self.total_clients = total_clients # Known as K in the paper
        self.max_rounds = max_rounds

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

        clients_x = defaultdict(list)
        clients_y = defaultdict(list)
        clf = None

        for i, x in enumerate(X):
            assignment = client_assignment[i]
            clients_x[assignment].append(x)
            clients_y[assignment].append(Y[i])

        model = self.init_server_model(X.shape[1])

        for _ in range(self.max_rounds):
            selected_clients_count = max(1, self.total_clients * self.client_fraction)
            clients_ix = np.random.choice(range(self.total_clients), int(selected_clients_count))

            updates = []

            for c_ix in clients_ix:
                updates.append(self.client_compute_update(model, clients_x[c_ix], clients_y[c_ix]))

            model, clf = self.global_combine(model, updates)

        self.clf = clf

        return self

    def predict(self, X):
        return np.random.choice([self.normal_class_label, self.outlier_class_label], len(X))

    def client_compute_update(self, global_model, client_data_x, client_data_y):
        # Concat points from server and from client
        X = np.concatenate((global_model['xs'], client_data_x))
        y = np.array([self.outlier_class_label if np.isclose(b, global_model['c']) else self.normal_class_label for b in global_model['betas']])
        y = np.concatenate((y, client_data_y))

        # Init the classifier with q and C from server
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
            print('There is no client update. No betas far from zero')
        
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
        search_params['q'].extend(uniform(loc=0, scale=1).rvs(size=3))
        search_params['c'].extend(uniform(loc=.2, scale=.8).rvs(size=3))
        
        # Performs model selection over this new dataset
        test_fold = [0 if v < len(X) else 1 for v in range(len(X) * 2)]
        clf = GridSearchCV(BSVClassifier(outlier_class_label=self.outlier_class_label, normal_class_label=self.normal_class_label), search_params, cv=PredefinedSplit(test_fold=test_fold), n_jobs=4, scoring='average_precision', refit=True, return_train_score=False, error_score='raise')
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
        return self.clf.decision_function(X)

    def score_samples(self, X):
        return self.clf.score_samples(X)