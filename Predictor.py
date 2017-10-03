from abc import abstractmethod, ABCMeta
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

"""
An abstract class modeling our notion of a predictor.
Concrete implementations should follow the predictors
interface
"""
class Predictor(object):
    __metaclass__ = ABCMeta

    def __init__(self, features, labels, params=None):
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
    def predict(self):
        """
        Predicts the label for the given input
        :return: The predicted labels
        """

    def evaluate(self, metric='mae'):
        _, test = self.split()
        y_val = test['logerror'].values
        x_val = test.drop_unchecked("logerror")
        prediction = self.predict()
        if metric == 'mae':
            return mean_absolute_error(y_val, prediction)
        raise NotImplementedError("Only mean absolute error metric is currently supported.")

