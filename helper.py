import numpy as np
import pandas as pd
import yaml
import torch
from torch.utils.data import Dataset

from tqdm import tqdm
from copy import deepcopy


def load_config(config_path):
    """
    Load a configuration file from the given path.
    
    Args:
        config_path (str): The path to the configuration file.
        
    Returns:
        dict: The loaded configuration as a dictionary.
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def create_dataset_tensors(
    data_paths,
    x_cols, 
    zero_cols=[],
    y_col='fear'
):
    """
    Create tensors for a given dataset.

    Args:
        data_paths (list): List of paths to CSV data files.
        x_cols (list): List of column names to use as features.
        zero_cols (list): List of column names to zero out for virtual lesion analysis
        y_col (str): Name of the column to use as a target. Default is 'fear'.

    Returns:
        tuple: A tuple containing feature tensors and target tensors.
    """
    
    trial_x_list = []
    trial_y_list = []
    
    for data_path in data_paths:
        par_df = pd.read_csv(data_path)
        par_df = par_df.loc[par_df[y_col]>=0]
        
        if len(zero_cols):
            par_df[zero_cols]=0
        
        mean_fear = par_df[y_col].mean()
        std_fear = par_df[y_col].std()
        
        for _, trial_df in par_df.groupby('video_name'):
            trial_y = trial_df.fear.unique()[0]
            if trial_y > 0 and not np.isnan(trial_y):
                trial_x = trial_df.loc[:, x_cols].astype(float).values
                trial_x_list.append(trial_x)
                trial_y_list.append((trial_y - mean_fear) / std_fear)

    X_tensor = torch.tensor(np.array(trial_x_list))
    y_tensor = torch.tensor(trial_y_list)
    return X_tensor, y_tensor


def create_average_representations(
    data_paths,
    x_cols, 
    zero_cols=[],
    y_col='fear'
):
    """
    Compute average representations of the data.

    Args:
        data_paths (list): List of paths to CSV data files.
        x_cols (list): List of column names to use as features.
        zero_cols (list): List of column names to zero out for virtual lesion analysis
        y_col (str): Name of the column to use as a target. Default is 'fear'.

    Returns:
        list: A list of feature and target values.
    """
    
    trial_x_list = []
    trial_y_list = []
    
    for data_path in data_paths:
        par_df = pd.read_csv(data_path)
        par_df = par_df.loc[par_df[y_col]>=0]
        
        if len(zero_cols):
            par_df[zero_cols]=0
        
        mean_fear = par_df[y_col].mean()
        std_fear = par_df[y_col].std()
        
        for _, trial_df in par_df.groupby('video_name'):
            trial_y = trial_df.fear.unique()[0]
            if trial_y > 0 and not np.isnan(trial_y):
                trial_x = trial_df.loc[:, x_cols].astype(float).mean().values
                trial_x_list.append(trial_x)
                trial_y_list.append((trial_y - mean_fear) / std_fear)

    return trial_x_list, trial_y_list


class NeuralTimeSeriesDataset(Dataset):
    """
    PyTorch Dataset for time series data.
    """
    
    def __init__(self, labels, features, transform=None, target_transform=None):
        self.labels = labels
        self.X_list = features
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        embedded_list = self.X_list[idx, :, :]
        label = self.labels[idx]
        return embedded_list, label


def calc_val_loss_and_accuracy(
    model, 
    loss_fn, 
    val_loader,
    verbose=True,
    device='cpu'
):
    """
    Calculate validation loss and accuracy.

    Args:
        model: Trained PyTorch model.
        loss_fn: Loss function to use for the model.
        val_loader: PyTorch DataLoader containing validation data.
        device (str): Device to run the model on. Default is 'cpu'.

    Returns:
        tuple: A tuple containing accuracy score and predictions.
    """
    
    with torch.no_grad():
        Y_shuffled, Y_preds, losses = [], [], []
        for X, Y in val_loader:
            X, Y = X.to(device), Y.to(device)
            preds = model(X)
            preds = preds.squeeze()
            loss = loss_fn(preds, Y)
            losses.append(loss.item())
            Y_shuffled.append(Y)
            Y_preds.append(preds)
        
        Y_shuffled = torch.cat(Y_shuffled)
        Y_preds = torch.cat(Y_preds).detach().squeeze().numpy()
        Y_shuffled = Y_shuffled.detach().numpy()
        
        #use pearsons correlation as our accuracy metric by convention of fMRI literature
        accuracy_scored = np.corrcoef(Y_shuffled, Y_preds)[0, 1]
        mean_loss = torch.tensor(losses).mean().item()
        if verbose:
            print(f"Valid Loss : {mean_loss:.3f}")
            print(f"Valid Acc  : {accuracy_scored:.3f}")
        
        return accuracy_scored, Y_preds, mean_loss



def train_model(model, loss_fn, optimizer, train_loader, val_loader, device='cpu', epochs=10):
    """
    Train a PyTorch model.

    Args:
        model: PyTorch model to train.
        loss_fn: Loss function to use for the model.
        optimizer: PyTorch optimizer to use for model training.
        train_loader: PyTorch DataLoader containing training data.
        val_loader: PyTorch DataLoader containing validation data.
        device (str): Device to run the model on. Default is 'cpu'.
        epochs (int): Number of epochs to train the model. Default is 10.

    Returns:
        tuple: A tuple containing the best accuracy score, model state and predictions.
    """
    #save overall stats
    model_acc_list = []
    best_acc = 0
    best_y_preds = []
    
    #save epoch stats
    all_metrics = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    for i in range(1, epochs+1):
        train_losses = []
        train_outputs = []
        train_labels = []
        losses = []
        model.train()
        
        for X, Y in tqdm(train_loader):
            X, Y = X.to(device), Y.to(device)
            
            Y_preds = model(X)
            Y_preds = Y_preds.squeeze()
            loss = loss_fn(Y_preds, Y)
            losses.append(loss.item())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
            train_outputs.extend(Y_preds.detach().numpy())
            train_labels.extend(Y.detach().numpy())
            
        # Compute training metrics after the epoch
        model.eval()
        
        train_loss = np.mean(train_losses)
        train_acc = np.corrcoef(train_labels, train_outputs)[0, 1]
        all_metrics['train_loss'].append(train_loss)
        all_metrics['train_acc'].append(train_acc)
        
        # Compute validation metrics after the epoch
        
        val_acc, y_preds_val, val_loss = calc_val_loss_and_accuracy(model, loss_fn, val_loader, device=device)
        all_metrics['val_loss'].append(val_loss)
        all_metrics['val_acc'].append(val_acc)
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_y_preds = y_preds_val
            model_state = deepcopy(model.state_dict())
            
        model_acc_list.append(val_acc)
        
        df_metrics = pd.DataFrame(all_metrics)
        
    return np.max(model_acc_list), model_state, best_y_preds, df_metrics
