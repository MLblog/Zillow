from sklearn.svm.libsvm import predict
import pandas as pd
from FeatureEngineering import dummy_conversion
from NNPredictor import *

def create_submission(model,x_eval,train_names):
    #Be sure that the features names are included in the names are the same in features and in the prediction
    x_eval_names=list(x_eval)
    months=[10,11,12]
    years=[2016,2017]
    
    for i in train_names:
        if i not in x_eval_names:
            print('The feature '+i+' is not included in the test set. Accion: create the feature with value=0.')
            x_eval[i]=0
            
    for i in x_eval_names:
        if i not in train_names:
            print('The feature '+i+' is not included in the training set. Accion: delete the feature')
            del x_eval[i]
    
    
    x_eval=x_eval[train_names]#The same order as the model was trained
    
    #index=pd.DataFrame(x_eval['parcelid'],columns=['ParcelId'])
    index=x_eval['parcelid']
    del x_eval['parcelid']
    del x_eval['logerror']      
    #---------------------------------------------------------------
    y_val={}
    for j in years:
        for i in months:
            x_eval['Year']==j
            x_eval['Month']==i 
            y_val[str(str(j)+str(i))]= model.predict(x_eval)
            
    y_val=pd.DataFrame.from_dict(y_val)
    
    y_val = y_val.loc[~y_val.index.duplicated(keep='first')]
    index = index.loc[~index.index.duplicated(keep='first')]
    y_val=pd.concat([index,y_val],axis=1)
    y_names=list(y_val)
    y_names[0]='ParcelId'
    y_val.columns=y_names
    y_val.dropna().to_csv('Prediction.csv',index=False)
            

        
        
    



# =============================================================================
# def test():
#     features = pd.read_csv('data/train_features.csv')
#     labels = pd.read_csv('data/train_label.csv')
# 
#     print("\nSetting up data for Neural Network ...")
# 
#     # Play with the following params (manually or via a gridsearchCV tuner to optimize)
#     # hidden_layer_sizes=(100, ), activation=’relu’, solver=’adam’, alpha=0.0001, batch_size=’auto’,
#     # learning_rate=’constant’, learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True,
#     # random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
#     # early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08
# 
#     
#     model = NNPredictor(features, labels)
#     model.preprocess()
#     train, _ = model.split()
#     x_train = train.drop_unchecked(['logerror','transactiondate'])
# 
#     return x_train
# =============================================================================


if __name__ == "__main__":
    #train_x = test()

    features = pd.read_csv('data/train_features.csv')
    labels = pd.read_csv('data/train_label.csv')

    print("\nSetting up data for Neural Network ...")
    model = NNPredictor(features, labels)
    model.preprocess()
    train, _ = model.split()
    x_train = train.drop_unchecked(['logerror','transactiondate'])
    params = {
        'hidden_layer_sizes': [(100, 200, 100)],
        'solver': ['sgd', "adam"],
        'activation': ['logistic'],
        'learning_rate': ['invscaling', 'constant'],
        'learning_rate_init': [0.01],
        'power_t': [0.1],
        'tol': [0.0001]
    }
    optimal_params = model.tune(params)
    model.train(optimal_params)
    
    
    #Train the model
    categories = ['airconditioningtypeid', 'architecturalstyletypeid', 'buildingclasstypeid', 'decktypeid', 'fips',
                        'heatingorsystemtypeid', 'propertycountylandusecode', 'propertylandusetypeid',
                        'propertyzoningdesc', 'storytypeid', 'typeconstructiontypeid']
      
    x_eval2017=pd.read_csv('data/properties_2017.csv')
    x_eval2016=pd.read_csv('data/properties_2016.csv')
    x_eval2017["Year"]=2017
    x_eval2016["Year"]=2016 
    x_eval=pd.concat([x_eval2017,x_eval2016])
    x_eval.fillna(-1, inplace=True)
    x_eval=dummy_conversion(x_eval, 30, categories)
    #x_eval.isnull().sum()
    create_submission(model,x_eval,list(train))

