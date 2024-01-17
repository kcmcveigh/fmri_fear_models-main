import glob
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

import helper
from models import Lstm_Regressor

def main():
    
    saved_model_path = 'SavedModels/lstm/Lstm_Regressor_fold-{fold}.pt'
    networks = ['Vis','SomMot','DorsAttn','SalVent','Limbic','Cont','Default','None']
    
    # Load configuration
    config = helper.load_config("config.yaml")

    torch.manual_seed(config["random_state"])
    np.random.seed(config["random_state"])
    
    # Data preparation
    data_path = config["data_path"].format(input_dim=config["input_dim"])
    data_paths = np.array(glob.glob(data_path))
    par_df = pd.read_csv(data_paths[0])
    x_cols = [col for col in par_df.columns if '7Networks' in col]
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    network_acc_list = []
    for network in networks:
        
        folder = KFold(n_splits=5, shuffle=False, random_state=config["random_state"])
        
        zero_cols = [col for col in x_cols if (network in col)]
        
        acc_list = []
        for fold, (train_idx, test_idx) in enumerate(folder.split(data_paths), start=1):

            X_test,y_test = helper.create_dataset_tensors(
                data_paths[list(test_idx)],
                x_cols,
                zero_cols=zero_cols
            )
            test_dataset = helper.NeuralTimeSeriesDataset(y_test.float(),X_test.float())
            test_dataloader = DataLoader(test_dataset,batch_size=len(y_test))

            lstm = Lstm_Regressor(
                config["input_dim"],
                config["hidden_dim"],
                config["out_dim"],
                config["n_hidden"],
                dropout=config["dropout"]
            )

            model_path = saved_model_path.format(fold=fold)
            model_state_dict = torch.load(model_path)
            lstm.load_state_dict(model_state_dict) 

            loss_fn = nn.MSELoss()
            acc, _, _ = helper.calc_val_loss_and_accuracy(
                lstm, 
                loss_fn, 
                test_dataloader, 
                verbose=False,
                device=device
            )
            acc_list.append(acc)
            
        across_folds_mean_acc = np.mean(acc_list)
        network_acc_list.append(across_folds_mean_acc)
        print(f'\n\n {network} mean acc',across_folds_mean_acc)
        
    save_df = pd.DataFrame(network_acc_list)
    save_df = save_df.T
    print(save_df)
    save_df.columns = networks
    save_df.to_csv('results/LSTM_acc_stats.csv')

if __name__ == "__main__":
    main()



