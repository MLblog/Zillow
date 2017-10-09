from abc import abstractmethod, ABCMeta
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import numpy as np
import pandas as pd

"""
An abstract class modeling our notion of a predictor.
Concrete implementations should follow the predictors
interface
"""
class Predictor(object):
    __metaclass__ = ABCMeta

    def __init__(self, features, labels, params={}):
        """
        Base constructor

        :param features: a pd.DataFrame of IDs and raw features
        :param labels: a pd.DataFrame of IDs and logerror labels
        :param params: a dictionary of named model parameters
        """
        self.features = features
        self.labels = labels
        self.params = params
        self.model = None
        print("Features shape: {} \nLabels shape: {}".format(features.shape, labels.shape))

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

    def evaluate(self, metric='mae'):
        _, test = self.split()
        y_val = test['logerror'].values
        x_val = test.drop_unchecked(['logerror', 'transactiondate'])
        prediction = self.predict(x_val)

        if metric == 'mae':
            return mean_absolute_error(y_val, prediction)

        raise NotImplementedError("Only mean absolute error metric is currently supported.")

class BasePredictor(Predictor):
    """
    A dummy predictor, always outputing the median. Used for benchmarking models.
    """
    def train(self, params=None):
        """
        A dummy predictor does not require training. We only need the median
        """
        train, _ = self.split()
        y_train = train['logerror'].values
        self.params['median'] = np.median(y_train)

    def predict(self):
        _, test = self.split()
        return [self.params['median']] * len(test)


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