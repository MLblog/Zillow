from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from Predictor import Predictor
from FeatureEngineering import *

# Enable OOP usage: df.drop_unchecked(cols) instead of drop_unchecked(df, cols)
pd.DataFrame.drop_unchecked = drop_unchecked

class NNPredictor(Predictor):

    def __init__(self, features, labels, params={}, name=None):
        super().__init__(features, labels, params, name='Neural Network')
        self.model = MLPRegressor()

    def preprocess(self):
        """
        A function that, given the raw dataset creates a feature vector.
        Feature Engineering, cleaning and imputation goes here
        """
        categories = ['airconditioningtypeid', 'architecturalstyletypeid', 'buildingclasstypeid', 'decktypeid', 'fips',
                      'heatingorsystemtypeid', 'propertycountylandusecode', 'propertylandusetypeid',
                      'propertyzoningdesc', 'storytypeid', 'typeconstructiontypeid', 'regionidcounty']

        self.features = dummy_conversion(self.features, 30, categories)
        self.features.fillna(-1, inplace=True)
        self.labels = parse_date(self.labels)

    def train(self, params=None):
        """
        A function that trains the predictor on the given dataset. Optionally accepts a set of parameters
        """

        # If no parameters are supplied use the default ones
        if not params:
            params = self.params

        self.preprocess()
        train, _ = self.split()
        y_train = train['logerror'].values
        x_train = train.drop_unchecked(['logerror','transactiondate']).values
        self.model = MLPRegressor(**params).fit(x_train, y_train)

    def predict(self, x_val):
        if not self.model:
            raise ValueError("The predictor has not been trained yet")

        prediction = self.model.predict(x_val.values)
        return prediction

if __name__ == "__main__":

    # Test that the classifier works
    features = pd.read_csv('data/train_features.csv')
    labels = pd.read_csv('data/train_label.csv')

    print("\nSetting up data for Neural Network ...")

    # Play with the following params (manually or via a gridsearchCV tuner to optimize)
    # hidden_layer_sizes=(100, ), activation=’relu’, solver=’adam’, alpha=0.0001, batch_size=’auto’,
    # learning_rate=’constant’, learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True,
    # random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
    # early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08
    params = {
        'hidden_layer_sizes': (100, 200, 100),
        'solver': 'sgd',
        'activation': 'logistic', # relu sucks. only god knows why
        'verbose': False,
        'warm_start': True,
        'alpha': 0.001,
        'learning_rate': 'invscaling',
        'power_t': 0.1,
        'learning_rate_init': 0.01,
        'max_iter': 125,
        'epsilon': 1e-08,
        'tol': 0.0000001
    }

    model = NNPredictor(features, labels, params)

    # Tune Model
    print("Tuning neural network")
    tuning_params = {
        'hidden_layer_sizes': [(100, 200, 100)],
        'solver': ['sgd', 'adam'],
        'activation': ['logistic'],
        'learning_rate': ['invscaling', 'constant'],
        'learning_rate_init': [0.01, 0.001],
        'power_t': [0.1, 0.5],
        'tol': [0.0001]
    }
    optimal_params, optimal_score = model.tune(tuning_params)
    model.persist_tuning(score=optimal_score, params=optimal_params, write_to='tuning.txt')

    # Train the model using the best set of parameters found by the gridsearch
    print("\nTraining NN ...")
    model.train(optimal_params)

    print("\nEvaluating NN...")
    mae = model.evaluate()
    model.persist_tuning(score=mae, params=optimal_params, write_to='tuning.txt')
    print("\n##########")
    print("Mean Absolute Error is: ", mae)
    print("##########")