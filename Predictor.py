from abc import abstractmethod, ABCMeta
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import numpy as np
import pandas as pd
from FeatureEngineering import *
from sklearn.model_selection import GridSearchCV

"""
An abstract class modeling our notion of a predictor.
Concrete implementations should follow the predictors
interface
"""
class Predictor(object):
    __metaclass__ = ABCMeta

    def __init__(self, features, labels, params={}, name=None):
        """
        Base constructor

        :param features: a pd.DataFrame of IDs and raw features
        :param labels: a pd.DataFrame of IDs and logerror labels
        :param params: a dictionary of named model parameters
        """
        self.features = features
        self.labels = labels
        self.params = params
        self.name = name
        self.model = None

    def set_params(self, params):
        """Override parameters set in the constructor. Dictionary expected"""
        self.params = params

    def split(self, test_size=0.3):
        """
        Splits the merged input (features + labels) into a training and a validation set

        :return: a tuple of training and test sets with the given size ratio
        """
        merged = self.labels.merge(self.features, how='left', on='parcelid')
        train, test = train_test_split(merged, test_size=test_size, random_state=42)
        return (train, test)

    @abstractmethod
    def train(self):
        """
        A function that trains the predictor on the given dataset.
        """

    @abstractmethod
    def predict(self, x_test):
        """
        Predicts the label for the given input
        :param x_test: a pd.DataFrame of features to be used for predictions
        :return: The predicted labels
        """

    def persist_tuning(self, score, params, write_to):
        """
        Persists a set of parameters as well as their achieved score to a file.
        :param params: Parameters used
        :param score: Score achieved on the test set using params
        :param write_to: If passed, the optimal parameters found will be written to a file
        :return: Void
        """
        with open(write_to, "a") as f:
            f.write("------------------------------------------------\n")
            f.write("Model\t{}\n".format(self.name))
            f.write("Best MAE\t{}\nparams: {}\n\n".format(score, params))


    def tune(self, params, nfolds=3):
        """
        Exhaustively searches over the grid of parameters for the best combination
        :param params: Grid of parameters to be explored
        :param nfolds: Number of folds to be used by cross-validation.

        :return: Dict of best parameters found.
        """
        self.preprocess()
        train, _ = self.split()
        y_train = train['logerror'].values
        x_train = train.drop_unchecked(['logerror','transactiondate'])

        grid = GridSearchCV(self.model, params, cv=nfolds)
        grid.fit(x_train, y_train)
        return grid.best_params_

    def evaluate(self, metric='mae'):
        _, test = self.split()
        y_val = test['logerror'].values
        x_val = drop_unchecked(test, ['logerror', 'transactiondate'])
        prediction = self.predict(x_val)

        if metric == 'mae':
            return mean_absolute_error(y_val, prediction)

        raise NotImplementedError("Only mean absolute error metric is currently supported.")

class BasePredictor(Predictor):
    """
    A dummy predictor, always outputing the median. Used for benchmarking models.
    """
    def __init__(self, features, labels, params={}, name=None):
        super().__init__(features, labels, params, name='Naive')

    def train(self, params=None):
        """
        A dummy predictor does not require training. We only need the median
        """
        train, _ = self.split()
        y_train = train['logerror'].values
        self.params['median'] = np.median(y_train)

    def predict(self, x_val):
        return [self.params['median']] * len(x_val)


if __name__ == "__main__":

    print("Reading training data...")
    features = pd.read_csv('data/train_features.csv')
    labels = pd.read_csv('data/train_label.csv')

    print("\nSetting up data for Base Predictor ...")
    model = BasePredictor(features, labels)

    # Train the model using the best set of parameters found by the gridsearch
    print("\nTraining Base Predictor ...")
    model.train()

    print("\nEvaluating model...")
    mae = model.evaluate()

    print("\n##########")
    print("Mean Absolute Error is: ", mae)
    print("##########")