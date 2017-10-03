from sklearn.preprocessing import LabelEncoder
from abc import abstractmethod, ABCMeta
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import pandas as pd
import numpy as np
import xgboost as xgb

def drop_unchecked(df, cols):
    """
    An unchecked version of pandas.DataFrame.drop(cols, axis=1). This will not raise
    an error in case of non existing column. Be careful though as this might hide spelling errors
    """
    for col in (set(cols) & set(df.columns)):
        df = df.drop([col], axis=1)
    return df

# Enable OOP usage: df.drop_unchecked(cols) instead of drop_unchecked(df, cols)
pd.DataFrame.drop_unchecked = drop_unchecked

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


class XGBoostPredictor(Predictor):

    def preprocess(self):
        """
        A function that, given the raw dataset creates a feature vector.
        Feature Engineering, cleaning and imputation goes here
        """

        for c in self.features.columns:
            # Replace NaNs
            self.features[c] = self.features[c].fillna(-1)
            if self.features[c].dtype == 'object':
                # Encode categorical features
                lbl = LabelEncoder()
                lbl.fit(list(self.features[c].values))
                self.features[c] = lbl.transform(list(self.features[c].values))

        # Drop some useless or extremely rare features
        bad_cols = ['propertyzoningdesc', 'propertycountylandusecode', 'fireplacecnt', 'fireplaceflag']
        self.features = drop_unchecked(self.features, bad_cols)

        # Create additional features. Month will be used to split into training and validation
        self.labels["transactiondate"] = pd.to_datetime(self.labels["transactiondate"])
        self.labels["Month"] = self.labels["transactiondate"].dt.month
        self.labels["Year"] = self.labels["transactiondate"].dt.year
        self.labels = self.labels.drop_unchecked("transactiondate")

    def split(self):
        merged = self.labels.merge(self.features, how='left', on='parcelid')
        train = merged[merged["Month"] < 10]
        train = train.query('logerror > -0.4 and logerror < 0.4')
        test = merged[merged["Month"] >= 10]
        return (train, test)

    def train(self):
        """
        A function that trains the predictor on the given dataset.
        :return:
        """
        self.preprocess()
        train, _ = self.split()
        y_train = train['logerror'].values
        x_train = train.drop_unchecked(['logerror','transactiondate'])

        dtrain = xgb.DMatrix(x_train, y_train)
        self.params['base_metric'] = np.median(y_train)
        self.model = xgb.train(dict(self.params, silent=1), dtrain, num_boost_round=self.params['num_boost_rounds'])

    def predict(self):
        if not self.model:
            raise ValueError("The predictor has not been trained yet")

        _, test = self.split()
        x_val = test.drop_unchecked(['logerror', 'transactiondate'])
        dtest = xgb.DMatrix(x_val)
        return self.model.predict(dtest)


if __name__ == "__main__":

    # Test that the classifier works
    features = pd.read_csv('data/train_features.csv')
    labels = pd.read_csv('data/train_label.csv')

    ##### RUN XGBOOST
    print("\nSetting up data for XGBoost ...")
    # xgboost params
    xgb_params = {
        'eta': 0.037,
        'max_depth': 5,
        'subsample': 0.80,
        'objective': 'reg:linear',
        'eval_metric': 'mae',
        'lambda': 0.8,
        'alpha': 0.4,
        'silent': 1,
        'num_boost_rounds': 500
    }


    # Train the model. Expect your machine to overheat here.
    print("\nTraining XGBoost ...")
    model = XGBoostPredictor(features, labels, xgb_params)
    model.train()

    print("\n Evaluating model...")
    mae = model.evaluate()

    print("\n##########")
    print("Mean Absolute Error is: ", mae)
    print("##########")