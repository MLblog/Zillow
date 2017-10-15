from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
import xgboost as xgb
from Predictor import Predictor
from FeatureEngineering import *

# Enable OOP usage: df.drop_unchecked(cols) instead of drop_unchecked(df, cols)
pd.DataFrame.drop_unchecked = drop_unchecked


class XGBoostPredictor(Predictor):

    def __init__(self, features, labels, params={}, name='XGBoost'):
        self.model = xgb.XGBRegressor()
        super().__init__(features, labels, params, name=name)


    def preprocess_test(self):
        categories = ['airconditioningtypeid', 'architecturalstyletypeid', 'buildingclasstypeid', 'decktypeid', 'fips',
                      'heatingorsystemtypeid', 'propertycountylandusecode', 'propertylandusetypeid',
                      'propertyzoningdesc', 'storytypeid', 'typeconstructiontypeid', 'regionidcounty']

        self.features = dummy_conversion(self.features, 30, categories)
        self.features.fillna(-1, inplace=True)
        self.labels = parse_date(self.labels)

    def preprocess(self):
        """
        A function that, given the raw dataset creates a feature vector.
        Feature Engineering, cleaning and imputation goes here
        """
        self.features.fillna(-1, inplace=True)
        self.features = label_encode(self.features)

        # Drop some useless or extremely rare features
        bad_cols = ['propertyzoningdesc', 'propertycountylandusecode', 'fireplacecnt', 'fireplaceflag']
        self.features = drop_unchecked(self.features, bad_cols)

        # Create additional features.
        self.labels = parse_date(self.labels)

    def train(self, params=None, num_boost_rounds=242):
        """
        A function that trains the predictor on the given dataset. Optionally accepts a set of parameters
        """

        # If no parameters are supplied use the default ones
        if not params:
            params = self.params

        self.preprocess()
        train, _ = self.split()
        y_train = train['logerror'].values
        x_train = train.drop_unchecked(['logerror','transactiondate'])

        self.params['base_score'] = np.median(y_train)

        dtrain = xgb.DMatrix(x_train, y_train)
        self.model = xgb.train(params, dtrain, num_boost_round=num_boost_rounds)


    def predict(self, x_val):
        if not self.model:
            raise ValueError("The predictor has not been trained yet")

        dtest = xgb.DMatrix(x_val)
        prediction = self.model.predict(dtest)
        return prediction


if __name__ == "__main__":

    # Test that the classifier works
    features = pd.read_csv('data/train_features.csv')
    labels = pd.read_csv('data/train_label.csv')

    ##### RUN XGBOOST
    print("\nSetting up data for XGBoost ...")

    initial_params = {
        'max_depth': 5,
        'subsample': 0.80,
        'learning_rate': 0.037,
        'reg_lambda': 0.8,
        'silent': 1,
    }

    xgb_params = {
        'eta': 0.037,
        'max_depth': 5,
        'subsample': 0.80,
        'objective': 'reg:linear',
        'eval_metric': 'mae',
        'lambda': 0.8,
        'alpha': 0.4,
        'silent': 1
    }

    model = XGBoostPredictor(features, labels, xgb_params)

    # Tune Model
    print("Tuning XGBoost...")
    tuning_params = {
        'learning_rate': [0.037, 0.05],
        'silent': [1],
        'max_depth': [5, 7],
        'subsample': [0.8, 1],
        'reg_lambda': [0.5, 0.8, 1],
        'n_jobs': [8],
        'n_estimators': [50, 100, 200],
        'missing': [-1]
    }
    optimal_params, optimal_score = model.tune(tuning_params)
    model.persist_tuning(score=optimal_score, params=optimal_params, write_to='tuning.txt')

    # Train the model using the best set of parameters found by the gridsearch
    print("\nTraining XGBoost ...")
    model.train(optimal_params)

    print("\nEvaluating model...")
    mae = model.evaluate()

    print("\n##########")
    print("Mean Absolute Error is: ", mae)
    print("##########")