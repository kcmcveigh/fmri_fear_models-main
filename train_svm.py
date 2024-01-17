import pickle
from joblib import dump, load
import helper
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn import svm

def main():
    
    # Load configuration
    config = helper.load_config("config.yaml")
    np.random.seed(config["random_state"])
    # Data preparation
    data_path = config["data_path"].format(input_dim=config["input_dim"])
    data_paths = np.array(glob.glob(data_path))
    
    #just get xcols from one example file
    par_df = pd.read_csv(data_paths[0])
    x_cols = [col for col in par_df.columns if '7Networks' in col]
    
    for kernel in ['linear','rbf']:
        folder = KFold(n_splits=5)
        svm_acc_list = []
        fold=0
        for train_idx, test_idx in folder.split(data_paths):
            fold+=1
            print(f'---------------------- fold {fold} ---------------------------')
            
            #load data
            X_train,y_train = helper.create_average_representations(
                data_paths[list(train_idx)],
                x_cols
            )
            X_test,y_test = helper.create_average_representations(
                data_paths[list(test_idx)],
                x_cols
            )
            
            #fit model
            svm_classifier = svm.SVR(kernel=kernel)
            svm_classifier.fit(X_train,y_train)
            
            #calc acc
            y_test_pred = svm_classifier.predict(X_test)
            svm_accuracy = np.corrcoef(y_test,y_test_pred)[0,1]
            svm_acc_list.append(svm_accuracy)
            
            #save
            dump(svm_classifier,f'saved_models/svm/svr_kernel_{kernel}_fold-{fold}.joblib')
            print(svm_accuracy)
        print(np.mean(svm_acc_list))

if __name__ == "__main__":
    main()