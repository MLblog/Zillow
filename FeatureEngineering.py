import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def drop_unchecked(df, cols):
    """
    An unchecked version of pandas.DataFrame.drop(cols, axis=1). This will not raise
    an error in case of non existing column. Be careful though, as this might hide spelling errors.
    """
    for col in (set(cols) & set(df.columns)):
        df = df.drop([col], axis=1)
    return df

def parse_date(df, date_col='transactiondate'):
    """
    Replaces the date time field by a month and year column
    :param df: input pd.DataFrame containing the column <date_col>
    :param date_col: the column name containg the date object
    :return: output pd.DataFrame with appended columns for month and year
    """
    df["transactiondate"] = pd.to_datetime(df["transactiondate"])
    df["Month"] = df["transactiondate"].dt.month
    df["Year"] = df["transactiondate"].dt.year
    return df.drop_unchecked("transactiondate")

def drop_columns(df, threshold):
    """
    Drop the coluns with more than H % of missing values. Based on Manos algorithm.
    """
    nan_count = df.isnull().mean()
    nan_count = nan_count[nan_count <= threshold]
    df = df[nan_count.index.tolist()]
    return df

def label_encode(df):
    for c in df.columns:
        if df[c].dtype == 'object':
            # Encode categorical features
            lbl = LabelEncoder()
            lbl.fit(list(df[c].values))
            df[c] = lbl.transform(list(df[c].values))
    return df

def dummy_conversion(df, threshold, categories=[]):
    """
    Transform the columns with strings to features only if the 
    """
    list_names = []
    for c in df.columns:
        if (df[c].dtype == 'object') | (c in categories):
            if c in categories:
                df[c] = df[c].astype('category')
                n = len(df[c].cat.categories)
            else:
                n = len(set(df[c]))
                
            if n <= threshold:
                list_names.append(c)
            else:
                print("Dropping variable " + str(c))
                del df[c]

    print('The features that will be transformed are:')
    print(list_names)
    df = pd.get_dummies(df, columns=list_names)
    return df


if __name__ == '__main__':
    features = pd.read_csv('data/train_features.csv')
    categories=['airconditioningtypeid','architecturalstyletypeid','buildingclasstypeid','decktypeid','fips',
                'heatingorsystemtypeid','propertycountylandusecode','propertylandusetypeid','propertyzoningdesc',
                'storytypeid','typeconstructiontypeid']

    df = drop_col(features, 0.8)
    df = dummy_conversion(df, 30, categories)
