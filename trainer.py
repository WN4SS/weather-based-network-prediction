import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd

import torch.multiprocessing as mp

from sklearn.preprocessing import MinMaxScaler

import wandb
import os
import pickle
import numpy as np

from numpy import ndarray
from tqdm import tqdm

from typing import Tuple

DEVICE = "cpu"
DTYPE = torch.float32

# os.environ["WANDB_LOG_MODEL"] = "true"
# os.environ["WANDB_SILENT"] = "true"

SAVE_PATH = "data_processed/"

class regressionModel(nn.Module):
    def __init__(self,
                 n_inputs: int,
                 n_nodes_layer: list,
                 layer_activations: list):
        super(regressionModel, self).__init__()

        assert len(layer_activations) == len(n_nodes_layer)

        self.layers = nn.ModuleList()
        
        n_in = n_inputs

        for n_out, activation in zip(n_nodes_layer, layer_activations):
            self.layers.append(nn.Linear(n_in, n_out))
            
            if activation == "relu":
                self.layers.append(nn.ReLU())
            elif activation == "lrelu":
                self.layers.append(nn.LeakyReLU())
            elif activation == "sigmoid":
                self.layers.append(nn.Sigmoid())
            elif activation == "tanh":
                self.layers.append(nn.Tanh())

            n_in = n_out
        
        self.layer_activations = layer_activations
    
    def forward(self,
                x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)

        return x

def model_pipeline(project_name: str,
                   run_name: str,
                   config: dict,
                   shared_data: dict,
                   targets_scaler: MinMaxScaler,
                   queue: mp.Queue) -> None:
    X_train = shared_data['X_train']
    y_train = shared_data['y_train']
    X_val   = shared_data["X_val"]
    y_val   = shared_data["y_val"]
    X_test  = shared_data['X_test']
    y_test  = shared_data['y_test']

    test_datetime   = shared_data["test_datetime"]
    test_serial     = shared_data["test_serial"]
    val_datetime    = shared_data["val_datetime"]
    val_serial      = shared_data["val_serial"]
    sensor_means    = shared_data["sensor_means"]

    wandb.init(project=project_name,
               config=config,
               name=run_name)
    config = wandb.config
    train_loader = make_loader(X_train,
                               y_train,
                               config.BATCH_SIZE)
    model, train_criterion, optimizer = make(config,
                                             X_train.shape[1])
    train(model,
          train_criterion,
          train_loader,
          optimizer,
          targets_scaler,
          config,
          X_val,
          y_val,
          X_test,
          y_test,
          test_datetime,
          test_serial,
          val_serial,
          run_name,
          queue)
    test_final(model,
               X_test,
               y_test,
               targets_scaler,
               test_datetime,
               test_serial,
               sensor_means)
    validate_final(model,
                   X_val,
                   y_val,
                   targets_scaler,
                   val_datetime,
                   val_serial,
                   sensor_means)
    save_model(model)

    wandb.finish()

def make(config: dict,
         n_inputs: int) -> Tuple[nn.Module, nn.Module, torch.optim.Optimizer]:
    model = regressionModel(n_inputs=n_inputs,
                            n_nodes_layer=config.N_NODES_LAYER,
                            layer_activations=config.LAYER_ACTIVATIONS)
    model.to(device=DEVICE,
             dtype=DTYPE)
    train_criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=config.LR)
    return model, train_criterion, optimizer

def make_loader(X: ndarray,
                y: ndarray,
                batch_size: int = 64) -> DataLoader:
    if batch_size == 0:
        batch_size = X.shape[0]
    loader = DataLoader(TensorDataset(torch.from_numpy(X), torch.from_numpy(y)),
                        batch_size=batch_size,
                        shuffle=True)
    return loader

def train(model: regressionModel,
          train_criterion: nn.Module,
          train_loader: DataLoader,
          optimizer: torch.optim.Optimizer,
          targets_scaler: MinMaxScaler,
          config: dict,
          X_val: np.ndarray,
          y_val: np.ndarray,
          X_test: np.ndarray,
          y_test: np.ndarray,
          test_datetime: pd.Series,
          test_serial: pd.Series,
          val_serial: pd.Series,
          run_name: str,
          queue: mp.Queue) -> None:
    data_pt_ct = 0
    batch_ct = 0

    val_loss_mae_batch_previous = 0
    train_loss_mae_batch_previous = 0

    for _ in range(config.N_EPOCHS):
        stop_early = False
        
        for train_features, train_targets in train_loader:
                train_loss_mse_batch, train_loss_mae_batch = train_batch(train_features,
                                                                         train_targets,
                                                                         model,
                                                                         train_criterion,
                                                                         optimizer,
                                                                         targets_scaler)
                data_pt_ct += len(train_features)
                batch_ct += 1

                wandb.log({
                    "train/loss(MSE)": train_loss_mse_batch,
                    "train/loss(MAE)": train_loss_mae_batch,
                }, step=data_pt_ct)
                val_loss_mae_batch = validate(model,
                                              X_val,
                                              y_val,
                                              targets_scaler,
                                              val_serial,
                                              data_pt_ct)
                test(model,
                     X_test,
                     y_test,
                     targets_scaler,
                     test_serial,
                     data_pt_ct)

                # # Early stopping
                # if val_loss_mae_batch_previous != 0 and val_loss_mae_batch_previous < val_loss_mae_batch:
                #     print("stopping early")
                #     stop_early = True
                #     break
                # if train_loss_mae_batch_previous != 0 and train_loss_mae_batch_previous == train_loss_mae_batch:
                #     print("stopping early")
                #     stop_early = True
                #     break
                # else:
                #     save_model(model)
                #     val_loss_mae_batch_previous = val_loss_mae_batch
                #     train_loss_mae_batch_previous = train_loss_mae_batch

        # Update progress bar
        queue.put((run_name, 1))

        if stop_early:
            break
    
    wandb.log({f"data_pt_ct": data_pt_ct})

def train_batch(features: torch.Tensor,
                targets: torch.Tensor,
                model: regressionModel,
                train_criterion: nn.Module,
                optimizer: torch.optim.Optimizer,
                targets_scaler: MinMaxScaler) -> Tuple[float, float]:
    # Make predictions
    features, targets = features.to(DEVICE), targets.to(DEVICE)
    preds = model.forward(features)

    # Calculate train criterion and update model weights
    train_loss_mse_batch = train_criterion(preds, targets)
    optimizer.zero_grad()
    train_loss_mse_batch.backward()
    optimizer.step()

    # Calculate test criterion
    targets = targets_scaler.inverse_transform(targets.to("cpu",
                                                          non_blocking=True))
    preds = targets_scaler.inverse_transform(preds.detach().to("cpu",
                                                               non_blocking=True))
    
    train_loss_mae_batch = np.abs(targets - preds).mean()

    return train_loss_mse_batch.item(), train_loss_mae_batch
    
def validate(model: regressionModel,
             X_val: ndarray,
             y_val: ndarray,
             targets_scaler: MinMaxScaler,
             val_serial: pd.Series,
             data_pt_ct: int) -> float:
    X_val = torch.from_numpy(X_val).to(DEVICE)
    with torch.no_grad():
        val_preds = model.forward(X_val).to("cpu")

        val_preds = targets_scaler.inverse_transform(val_preds)
        y_val = targets_scaler.inverse_transform(y_val)

        val_loss_ae = np.abs(val_preds - y_val)
        wandb.log({"val/loss(MAE)": val_loss_ae.mean()},
                  step=data_pt_ct)

        error_df = pd.DataFrame({'serial': val_serial,
                                 'loss(AE)': val_loss_ae.squeeze()})
        for serial, data in error_df.groupby("serial", observed=False):
            wandb.log({f"val/{serial}/loss(MAE)": data["loss(AE)"].mean()},
                      step=data_pt_ct)
        
    return val_loss_ae.mean()

def validate_final(model: regressionModel,
                   X_val: ndarray,
                   y_val: ndarray,
                   targets_scaler: MinMaxScaler,
                   val_datetime: pd.Series,
                   val_serial: pd.Series,
                   sensor_means: dict) -> None:
    X_val = torch.from_numpy(X_val).to(DEVICE)
    with torch.no_grad():
        val_preds = model.forward(X_val).to("cpu")

        val_preds = targets_scaler.inverse_transform(val_preds)
        y_val = targets_scaler.inverse_transform(y_val)

        val_loss_ae = np.abs(val_preds - y_val)
        error_df = pd.DataFrame({'datetime': val_datetime,
                                 'serial': val_serial,
                                 'preds': val_preds.squeeze(),
                                 'y_test': y_val.squeeze(),
                                 'loss(AE)': val_loss_ae.squeeze()}).set_index('datetime')
        for serial, data in error_df.groupby("serial", observed=False):
            data["y_test"] += sensor_means[serial]
            data["preds"] += sensor_means[serial]
            wandb.log({f"val/{serial}/loss(MAE)": data["loss(AE)"].mean()})
            wandb.log({f"val/{serial}/loss(": wandb.Table(dataframe=data.reset_index())})

def test(model: regressionModel,
         X_test: ndarray,
         y_test: ndarray,
         targets_scaler: MinMaxScaler,
         test_serial: pd.Series,
         data_pt_ct: int) -> None:
    X_test = torch.from_numpy(X_test).to(DEVICE)
    with torch.no_grad():
        test_preds = model.forward(X_test).to("cpu")

        test_preds = targets_scaler.inverse_transform(test_preds)
        y_test = targets_scaler.inverse_transform(y_test)

        test_loss_ae = np.abs(test_preds - y_test)
        wandb.log({"test/loss(MAE)": test_loss_ae.mean()},
                  step=data_pt_ct)

        error_df = pd.DataFrame({'serial': test_serial,
                                 'loss(AE)': test_loss_ae.squeeze()})
        for serial, data in error_df.groupby("serial", observed=False):
            wandb.log({f"test/{serial}/loss(MAE)": data["loss(AE)"].mean()},
                      step=data_pt_ct)

def test_final(model: regressionModel,
               X_test: ndarray,
               y_test: ndarray,
               targets_scaler: MinMaxScaler,
               test_datetime: pd.Series,
               test_serial: pd.Series,
               sensor_means: dict) -> None:
    X_test = torch.from_numpy(X_test).to(DEVICE)
    with torch.no_grad():
        test_preds = model.forward(X_test).to("cpu")

        test_preds = targets_scaler.inverse_transform(test_preds)
        y_test = targets_scaler.inverse_transform(y_test)

        test_loss_ae = np.abs(test_preds - y_test)
        error_df = pd.DataFrame({'datetime': test_datetime,
                                 'serial': test_serial,
                                 'preds': test_preds.squeeze(),
                                 'y_test': y_test.squeeze(),
                                 'loss(AE)': test_loss_ae.squeeze()}).set_index('datetime')
        for serial, data in error_df.groupby("serial", observed=False):
            data["y_test"] += sensor_means[serial]
            data["preds"] += sensor_means[serial]
            wandb.log({f"test/{serial}/loss(MAE)": data["loss(AE)"].mean()})
            wandb.log({f"test/{serial}/loss(": wandb.Table(dataframe=data.reset_index())})

def save_model(model: regressionModel) -> None:
    state_dict_torch = model.state_dict()

    state_dict_python = {}

    for key in state_dict_torch:
        state_dict_python[key] = state_dict_torch[key].cpu().numpy()
    
    with open(wandb.run.dir + '/model_weights.pkl', 'wb') as f:
        pickle.dump(state_dict_python, f)
    wandb.save("model_weights.pkl")