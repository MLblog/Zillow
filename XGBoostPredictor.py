from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
import xgboost as xgb
from Predictor import Predictor

def drop_unchecked(df, cols):
    """
    An unchecked version of pandas.DataFrame.drop(cols, axis=1). This will not raise
    an error in case of non existing column. Be careful though, as this might hide spelling errors.
    """
    for col in (set(cols) & set(df.columns)):
        df = df.drop([col], axis=1)
    return df

# Enable OOP usage: df.drop_unchecked(cols) instead of drop_unchecked(df, cols)
pd.DataFrame.drop_unchecked = drop_unchecked

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

    def tune(self, params, nfolds=3):
        """
        Exhaustively searches over the grid of parameters for the best combination
        :param params: Grid of parameters to be explored
        :return: Dict of best parameters found.
        """
        self.preprocess()
        train, _ = self.split()
        y_train = train['logerror'].values
        x_train = train.drop_unchecked(['logerror','transactiondate'])

        dtrain = xgb.DMatrix(x_train, y_train)
        self.params['base_score'] = np.median(y_train)
        model = xgb.XGBRegressor()

        grid = GridSearchCV(model, params, cv=nfolds)
        grid.fit(x_train, y_train)
        return grid.best_params_


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
        x_train = train.drop_unchecked(['logerror','transactiondate'])

        self.params['base_score'] = np.median(y_train)
        self.model = xgb.XGBRegressor(**params).fit(x_train, y_train)

    def predict(self):
        if not self.model:
            raise ValueError("The predictor has not been trained yet")

        _, test = self.split()
        x_val = test.drop_unchecked(['logerror', 'transactiondate'])
        prediction = self.model.predict(x_val)
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

    tuning_params = {
        'learning_rate': [0.037],
        'silent': [1],
        'max_depth': [5, 6],
        'subsample': [0.8],
        'reg_lambda': [0.8]
    }

    model = XGBoostPredictor(features, labels, initial_params)

    print("Tuning XGBoost...")
    best_params = model.tune(tuning_params)

    # Train the model using the best set of parameters found by the gridsearch
    print("\nTraining XGBoost ...")
    model.train(params=best_params)

    print("\nEvaluating model...")
    mae = model.evaluate()

    print("\n##########")
    print("Mean Absolute Error is: ", mae)
    print("##########")