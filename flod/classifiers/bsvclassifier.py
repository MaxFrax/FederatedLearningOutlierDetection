from sklearn.base import BaseEstimator, ClassifierMixin
import random

class BSVClassifier(ClassifierMixin, BaseEstimator):

    def __init__(self, random_state:int):
        self.random_state = random_state

    def fit(self, X, y):

        # TODO input validation
        
        self.X_ = X
        self.y_ = y
        # Return the classifier
        return self

    def predict(self, X):
        
        # TODO verificare che la funzione fit sia stata chiamata, altrimenti dare errore

        # Input validation
        #X = check_array(X)

        random.seed(a=self.random_state, version=2)
        return [random.choice([0, 1]) for _ in range(len(X))]