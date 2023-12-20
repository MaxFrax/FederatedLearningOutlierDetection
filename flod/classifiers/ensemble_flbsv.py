from flod.classifiers.bsvclassifier import BSVClassifier
import numpy as np
import logging
from sklearn.base import BaseEstimator, ClassifierMixin
from collections import defaultdict
from collections import Counter

LOGGER = logging.getLogger(__name__)

class EnsembleFLBSV(ClassifierMixin, BaseEstimator):

    def __init__(self, privacy=False, normal_class_label:int=0, outlier_class_label:int=1, q:float=1, C:float=1, total_clients:int=2, client_fraction:float=1):
        self.normal_class_label = normal_class_label
        self.outlier_class_label = outlier_class_label
        self.client_fraction = client_fraction
        self.total_clients = total_clients
        self.q = q
        self.C = C
        self.privacy = privacy

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

        if len(np.unique(client_assignment)) != self.total_clients:
            LOGGER.warning(f'It seems like some clients do not have any data. Expected {self.total_clients} clients, but found {len(np.unique(client_assignment))}')  

        self.model = self.init_server_model()
        self.dimensions_count = X.shape[1]
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

        if len(client_data_x) <= 0:
            LOGGER.warning(f'Client {index} does not have any data to compute its update')
            return None

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
        
        train_x = np.array(train_x)

        clf = BSVClassifier(q=self.q, c=self.C, normal_class_label=self.normal_class_label, outlier_class_label=self.outlier_class_label)
        try:
            clf.fit(train_x, train_y)
        except:
            LOGGER.warning(F'Client {index} failed to train the model over {len(train_x)} points')
            return None

        if self.privacy:
            succeeded = False
            multiplier = 2
            while not succeeded and multiplier < 10:
                dataset = self.generate_synthetic_dataset(clf, train_x, multiplier)

                derived = BSVClassifier(q=self.q, c=self.C, normal_class_label=self.normal_class_label, outlier_class_label=self.outlier_class_label)
                try:
                    derived.fit(dataset, clf.predict(dataset))
                    succeeded = True
                except:
                    multiplier +=1

            LOGGER.debug(f'Derived had {len(derived.X_train_)} training points. Original had {len(clf.X_train_)}')
            LOGGER.debug(f'Derived had {len(derived.get_support_vectors())} support vectors. Original had {len(clf.get_support_vectors())}')

            clf = derived

        return clf
    
    def generate_synthetic_dataset(self, clf, train_x, multiplier):
        attempts = 0
        dataset, count_normal = self.sample_classifier(clf, train_x, multiplier)

        while count_normal is None or count_normal < 3:
            if attempts > 100:
                raise Exception(f'Could not generate a dataset with at least 3 normal points in 100 attempts. I found just {count_normal} normal points')

            # Keeps the normal points already generated
            new_dataset, _ = self.sample_classifier(clf, train_x, multiplier)
            for i, y in enumerate(clf.predict(dataset)):
                if y == self.outlier_class_label:
                    dataset[i] = new_dataset[i]
            
            count_normal = Counter(clf.predict(dataset)).get(self.normal_class_label)
            attempts += 1

        return dataset
    
    def sample_classifier(self, clf, train_x, multiplier = 2):
        inside_points = clf.get_inside_points()
        dataset = np.empty(shape=(0, train_x.shape[1]))

        if inside_points.shape[0] > 0:
            avg_x = inside_points.mean(axis=0)
            std_x = inside_points.std(axis=0)

            # Generates a similar dataset to the one used for training
            dataset = np.random.normal(loc=avg_x, scale=std_x, size=(inside_points.shape[0], train_x.shape[1]))
            # The goal is to have a synthetic dataset about twice the size of the original one.
            multiplier -= 1

        # Ensures that the interesting area around support vectors is properly sampled
        # .01 is the 1% of the domain if the dataset spans from 0 to 1
        for sv in clf.get_support_vectors():
            dataset = np.append(dataset, np.random.normal(loc=sv, scale=.01, size=(int(train_x.shape[0] * multiplier/len(clf.get_support_vectors())), train_x.shape[1])), axis=0)

        count_normal = Counter(clf.predict(dataset)).get(self.normal_class_label)

        return dataset, count_normal

    def global_combine(self, round, global_model, client_updates):
        model = [update for update in client_updates if update is not None]
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
