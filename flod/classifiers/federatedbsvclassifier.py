import numpy as np
from ..classifiers.bsvclassifier import BSVClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

def count_zeros(y, y_pred, **kwargs):
    return len(y)-sum(y_pred)

scoring = {
    'zeros_scorer': make_scorer(count_zeros)
}

class FederatedBSVClassifier():

    def __init__(self, grid_search_parameters):
        self.grid_search_parameters = grid_search_parameters

    def init_server_model(self):
        return {
            'q': 0.01,
            'C': 1,
            'betas': np.empty(shape=(0, 2)),
            'xs': np.empty(shape=(0, 2))
    }


    def client_compute_update(self, global_model, client_data):
        # Concat points from server and from client
        X = np.concatenate((global_model['xs'], client_data))

        # Init the classifier with q and C from server
        clf = BSVClassifier(q=global_model['q'], c=global_model['C'])

        # Train locally
        clf.fit(X, [0]*len(X))

        # Select only the positive betas related from client_data
        client_betas = clf.betas_[len(global_model['xs']):]
        assert(len(client_betas) == len(client_data))
        
        for _, t in enumerate(zip(client_betas, client_data)):
            b, x = t
            if not np.isclose(b, 0):
                yield x

    def global_combine(self, global_model, client_updates):
        # Concatenates server points and clients candidate points.
        X = np.concatenate((global_model['xs'], *client_updates))
        
        # Let's try also old hyper parameters. Notice: C is missing
        search_params = dict(self.grid_search_parameters)
        search_params['q'] = np.append(search_params['q'], global_model['q'])
        
        # Performs model selection over this new dataset
        # Cross validation is low because I want to fit exactly the data I have got
        clf = GridSearchCV(BSVClassifier(), search_params, cv=2, n_jobs=1, scoring=scoring, refit='zeros_scorer', return_train_score=False, verbose=10)
        clf.fit(X, [0] * len(X))
        
        # Filter and keep only the support vectors
        xs = []
        betas = []
        for i, t in enumerate(zip(clf.best_estimator_.betas_, X)):
            b, x = t
            if not np.isclose(b, 0):
                xs.append(x)
                betas.append(b)
                
        return {
            'q': clf.best_estimator_.q,
            'C': clf.best_estimator_.c,
            'betas': betas,
            'xs': xs
        }, clf.best_estimator_