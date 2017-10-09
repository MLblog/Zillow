import numpy as np
import pandas as pd


def drop_columns(df, threshold):
    """
    Drop the coluns with more than H % of missing values. Based on Manos algorithm.
    """
    nan_count = df.isnull().mean()
    nan_count = nan_count[nan_count <= threshold]
    df = df[nan_count.index.tolist()]
    return df


def dummy_conversion(df, threshold, categories=[]):
    """
    Transform the columns with strings to features only if the 
    """
    list_names=[]
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
