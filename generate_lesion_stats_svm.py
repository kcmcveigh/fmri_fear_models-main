import glob
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold

import helper
import joblib

def main():
    
    saved_model_path = 'SavedModels/svm/svr_kernel_{kernel}_fold-{fold}.joblib'
    networks = ['Vis','SomMot','DorsAttn','SalVent','Limbic','Cont','Default','None']
    kernels = ['rbf','linear']
    
    # Load configuration
    config = helper.load_config("config.yaml")

    torch.manual_seed(config["random_state"])
    np.random.seed(config["random_state"])
    
    # Data preparation
    data_path = config["data_path"].format(input_dim=config["input_dim"])
    data_paths = np.array(glob.glob(data_path))
    par_df = pd.read_csv(data_paths[0])
    x_cols = [col for col in par_df.columns if '7Networks' in col]
    for kernel in kernels:
        network_acc_list = []
        for network in networks:

            folder = KFold(n_splits=5)

            zero_cols = [col for col in x_cols if (network in col)]

            acc_list = []
            for fold, (train_idx, test_idx) in enumerate(folder.split(data_paths), start=1):

                X_test,y_test = helper.create_average_representations(
                    data_paths[list(test_idx)],
                    x_cols,
                    zero_cols=zero_cols
                )

                model_path = saved_model_path.format(
                    kernel=kernel,
                    fold=fold
                )
                svm_regressor= joblib.load(model_path)
                y_test_pred = svm_regressor.predict(X_test)
                svm_accuracy = np.corrcoef(y_test,y_test_pred)[0,1]

                acc_list.append(svm_accuracy)

            across_folds_mean_acc = np.mean(acc_list)
            network_acc_list.append(across_folds_mean_acc)

            print(f'{network} mean acc',across_folds_mean_acc)

        save_df = pd.DataFrame(network_acc_list)
        save_df = save_df.T
        save_df.columns = networks
        save_df.to_csv(f'results/svr_{kernel}_acc_stats.csv')

if __name__ == "__main__":
    main()



