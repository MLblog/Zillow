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
        'hidden_layer_sizes': (1000, 500),
        'activation': 'logistic', # relu sucks. only god knows why
        'verbose': True,
        'alpha': 0.001,
        'learning_rate_init': 0.0001,
        'max_iter': 15,
        'epsilon': 1e-08,
        'tol': 0.000001
    }

    model = NNPredictor(features, labels, params)

    # Train the model using the best set of parameters found by the gridsearch
    print("\nTraining NN ...")
    model.train()

    print("\nEvaluating NN...")
    mae = model.evaluate()

    print("\n##########")
    print("Mean Absolute Error is: ", mae)
    print("##########")