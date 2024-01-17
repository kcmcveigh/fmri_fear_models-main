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
    
    folder = KFold(n_splits=5)
    max_acc_list = []
    for fold, (train_idx, test_idx) in enumerate(folder.split(data_paths), start=1):
        print('---------------------- fold {} ---------------------------'.format(fold))
        X_train,y_train = helper.create_dataset_tensors(data_paths[list(train_idx)],x_cols)
        train_dataset = helper.NeuralTimeSeriesDataset(y_train.float(),X_train.float())
        train_dataloader = DataLoader(train_dataset,batch_size=config["batch_size"])

        X_test,y_test = helper.create_dataset_tensors(data_paths[list(test_idx)],x_cols)
        test_dataset = helper.NeuralTimeSeriesDataset(y_test.float(),X_test.float())
        test_dataloader = DataLoader(test_dataset,batch_size=len(y_test))

        loss_fn = nn.MSELoss()

        lstm = Lstm_Regressor(
            config["input_dim"],
            config["hidden_dim"],
            config["out_dim"],
            config["n_hidden"],
            dropout=config["dropout"]
        )
        optimizer = Adam(
            lstm.parameters(), 
            lr=config["learning_rate"],
            weight_decay=.01
        )

        max_acc_fold, state_dict, y_preds, train_metric_df = helper.train_model(
            lstm, 
            loss_fn,
            optimizer,
            train_dataloader,
            test_dataloader,
            epochs=config["epochs"],
            device=device)
        max_acc_list.append(max_acc_fold)
        train_metric_df.to_csv(f'saved_models/lstm/train_metrics_fold-{fold}.csv')
        torch.save(state_dict,f'saved_models/lstm/Lstm_Regressor_fold-{fold}.pt')
    print('\n\n mean acc',np.mean(max_acc_list))

if __name__ == "__main__":
    main()

