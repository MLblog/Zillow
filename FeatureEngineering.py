import numpy as np
import pandas as pd
#from fancyimpute import BiScaler, KNN, NuclearNormMinimization, SoftImpute


def dropCol(df,H):
    """
    Drop the coluns with more than H % of missing values. Based on Manos algorithm.
    """
    nan_count = df.isnull().mean()
    nan_count=nan_count[nan_count<=H]
    df=df[nan_count.index.tolist()]
    return df



def ColumnConvertion(df, H, categories=[]):
    """
    Transform the columns with strings to features only if the 
    """
    
    listNames=[]
    for c in df.columns:
        if ((df[c].dtype == 'object') | (c in categories)):
            if c in categories:
                df[c]=df[c].astype('category')
                n=len(df[c].cat.categories)
            else:
                n=len(set(df[c]))
                
            if n<=H:
                listNames.append(c)
            else:
                del df[c]
    print('The features that will be transformed are:')
    print(listNames)      
    df=pd.get_dummies(df,columns=listNames)
    return df





categories=['airconditioningtypeid','architecturalstyletypeid','buildingclasstypeid','decktypeid','fips','heatingorsystemtypeid','propertycountylandusecode','propertylandusetypeid','propertyzoningdesc','storytypeid','typeconstructiontypeid']
#features = pd.read_csv('data/train_features.csv')
df=dropCol(features,0.8)
df=ColumnConvertion(df,30,categories)
#df=df.interpolate()



#Some algorithms to impute missing values:
    #https://pypi.python.org/pypi/fancyimpute