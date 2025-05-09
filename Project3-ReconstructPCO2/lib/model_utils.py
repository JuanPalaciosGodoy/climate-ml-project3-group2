#===============================================
# Imports
#===============================================
import gc
import torch
import torch.nn as nn
from tqdm import tqdm
import pickle
import pandas as pd
import numpy as np
import gcsfs
import datetime
import csv
from xgboost import XGBRegressor
import xgboost as xgb
from collections import defaultdict
from enum import Enum
import lib.residual_utils as supporting_functions
    
from torch.utils.data import TensorDataset, DataLoader


#===============================================
# Enums
#===============================================

class Models(Enum):
    NEURAL_NETWORK = "nn"
    XGBOOST = "xgb"

class ColumnFields(Enum):
    PCO2_RECON_FULL = 'pCO2_recon_full'
    PCO2_RECON_UNSEEN = 'pCO2_recon_unseen'
    PCO2_TRUTH = 'pCO2_truth'
    SOCAT_MASK = 'socat_mask'
    NET_MASK = 'net_mask'
    YEAR_MONTH = 'year_month'
    MONTH = 'mon'
    YEAR = 'year'
    TIME = 'time'
    

#===============================================
# Objects
#===============================================

class KappaLayers(nn.Module):
    def __init__(self, input_nodes, hidden_nodes, output_nodes):
        super(KappaLayers, self).__init__()
        self.linear1 = nn.Linear(input_nodes, hidden_nodes)  # First layer: Input to hidden
        self.linear2 = nn.Linear(hidden_nodes, hidden_nodes) # Second layer: Hidden to hidden
        self.linear3 = nn.Linear(hidden_nodes, hidden_nodes) # Second layer: Hidden to hidden
        self.linear4 = nn.Linear(hidden_nodes, hidden_nodes) # Third layer: Hidden to output
        self.linear5 = nn.Linear(hidden_nodes, output_nodes) # Third layer: Hidden to output
        self.dropout = nn.Dropout(0.25) # Dropout for regularization

    def forward(self, x):
        x = self.linear1(x)
        x = torch.relu(x)  # ReLU activation for layer 1
        x = self.dropout(x) # Apply dropout

        x = self.linear2(x)
        x = torch.relu(x) # ReLU activation for layer 2
        x = self.dropout(x) # Apply dropout

        x = self.linear3(x)
        x = torch.relu(x) # ReLU activation for layer 2
        x = self.dropout(x) # Apply dropout

        x = self.linear4(x)
        x = torch.relu(x) # ReLU activation for layer 2
        x = self.dropout(x) # Apply dropout

        x = self.linear5(x) # Final output layer
        return x


    def save_model(self, path):
        # Save the model's state_dict
        torch.save(self.state_dict(), path)

class Model(object):
    def __init__(self, model):
        self.model = model

    def predict(self, x):
        """
        model prediction implementation

        Parameters
        ----------
            x (np.array): features used to make prediction
        """
        raise NotImplementedError("predict method not implemented for model")

    def predict_by_batch(self, x):
        return self.predict(x)

    def train(self, data):
        """
        model training implementation

        Parameters
        ----------
            data: data used to train the model
        """
        raise NotImplementedError("train method not implemented for model")

    def performance(self, x, y):

        # evaluate model performance
        y_pred_test = self.predict_by_batch(x)

        return supporting_functions.evaluate_test(
                _as_numpy(y),
                _as_numpy(y_pred_test)
            )
        

    def save_performance(self, fs, path, data, performance, ens, member):
        
        # format dictionary
        fields = performance[ens][member].keys()
        test_row_dict = dict()
        test_row_dict['model'] = ens
        test_row_dict['member'] = member
            
        for field in fields:
            test_row_dict[field] = performance[ens][member][field]

        # save performance
        file_exists = fs.exists(path)
        with fs.open(path, 'a') as f_object:
            writer = csv.DictWriter(f_object, fieldnames=test_row_dict.keys())
            if not file_exists:
                writer.writeheader() 
            writer.writerow(test_row_dict)

    def save_reconstruction(self, ens:str, member:str, df:pd.DataFrame, x_seen:np.array, x_unseen:np.array, seen_mask:np.array, unseen_mask:np.array, dates, path:str, target:str, model_type:str):

        if Models(model_type) == Models.NEURAL_NETWORK:
            # calculate predictions
            print("Len(x_seen):", len(x_seen))
            print("Len(x_unseen):", len(x_unseen))
            self.model.eval()
            with torch.no_grad():
                y_pred_seen = _as_numpy(self.predict_by_batch(x_seen))
                gc.collect()
                torch.cuda.empty_cache()
                y_pred_unseen = _as_numpy(self.predict_by_batch(x_unseen, batch_size=256))
        else:
            # calculate predictions
            y_pred_seen = _as_numpy(self.predict(x_seen))
            y_pred_unseen = _as_numpy(self.predict(x_unseen))

        # save full reconstruction
        df[ColumnFields.PCO2_RECON_FULL.value] = np.nan
        df.loc[unseen_mask, ColumnFields.PCO2_RECON_FULL.value] = y_pred_unseen 
        df.loc[seen_mask, ColumnFields.PCO2_RECON_FULL.value] = y_pred_seen

        # save unseen reconstruction
        df[ColumnFields.PCO2_RECON_UNSEEN.value] = np.nan
        df.loc[unseen_mask, ColumnFields.PCO2_RECON_UNSEEN.value] = y_pred_unseen
        df.loc[seen_mask, ColumnFields.PCO2_RECON_UNSEEN.value] = np.nan
    
        df[ColumnFields.PCO2_TRUTH.value] = df.loc[:,target]

        # to dataset
        columns = [
            ColumnFields.NET_MASK.value,
            ColumnFields.SOCAT_MASK.value,
            ColumnFields.PCO2_RECON_FULL.value,
            ColumnFields.PCO2_RECON_UNSEEN.value,
            ColumnFields.PCO2_TRUTH.value
        ]
        ds_recon = df[columns].to_xarray()

        # save reconstruction
        supporting_functions.save_recon(
            ds_recon,
            dates,
            path,
            ens,
            member
        )
  

class XGBoostModel(Model):
    def __init__(self, random_seeds, seed_loc, **kwargs):
        model = self._get_model(
            random_seeds=random_seeds,
            seed_loc=seed_loc,
            **kwargs
        )
        super().__init__(model=model)

    def _get_model(self, random_seeds, seed_loc, **kwargs):
        """
        initialize xgboost model
        """
        
        return XGBRegressor(
            random_state=random_seeds[5, seed_loc],
            **kwargs,
        )
    
    def predict(self, x):
        """
        make prediction using xgboost model
        """
        return self.model.predict(x)

    def train(self, data):
        """
        train xgboost model
        """
        eval_set = [(data.x_val, data.y_val)] 
        self.model.fit(
            data.x_train_val,
            data.y_train_val, 
            eval_set=eval_set, 
            verbose=False
        )

    def performance(self, x, y):

        # evaluate model performance
        y_pred_test = self.predict(x)

        return supporting_functions.evaluate_test(
                _as_numpy(y),
                _as_numpy(y_pred_test)
            )
        
    

class NeuralNetworkModel(Model):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, epochs:int=3000, patience:int=400, lr:float=1e-03, **kwargs):
        model = self._get_model(
            input_nodes,
            hidden_nodes,
            output_nodes,
        )
        super().__init__(model=model)
        self.epochs = epochs
        self.loss_array = torch.zeros([epochs, 3])  # Array to store epoch, train, and validation losses
        self.patience = patience
        self.best_loss = float('inf')  # Initialize the best validation loss as infinity
        self.no_improvement = 0  # Counter for epochs without improvement
        self.best_model_state = None  # Placeholder for the best model state

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr)  # Adam optimizer
        self.loss_fn = torch.nn.L1Loss(reduction='mean')  # L1 loss for gradient computation

    def _get_model(self, input_nodes, hidden_nodes, output_nodes):
        """
        initialize neural network model
        """
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        return KappaLayers(
            input_nodes,
            hidden_nodes,
            output_nodes
        ).to(device)
    
    def predict_by_batch(self, x, batch_size=1024):
        preds = []

        for i in range(0, len(x), batch_size):
            batch_x = x[i:i+batch_size]
            batch_x = self.maybe_torch(batch_x)
            batch_preds = self.model(batch_x).T[0].cpu()
            preds.append(batch_preds)
            del batch_x

        return torch.cat(preds)

    def predict(self, x):
        x = self.maybe_torch(x)
        return self.model(x).T[0]

    def train_batched(self, data, batch_size=1024):
        dataset = TensorDataset(torch.FloatTensor(data.x_train_val), torch.FloatTensor(data.y_train_val))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        val_dataset = TensorDataset(torch.FloatTensor(data.x_val), torch.FloatTensor(data.y_val))
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        with tqdm(total=self.epochs, desc="Training Progress", unit="epoch") as pbar:
            for epoch in range(self.epochs):
                self.model.train()
                train_loss_epoch = 0.0
                for x_batch, y_batch in dataloader:
                    self.optimizer.zero_grad()

                    y_pred = self.predict(x_batch)
                    loss = self.loss_fn(y_pred, y_batch)
                    loss.backward()
                    self.optimizer.step()

                    train_loss_epoch += loss.item() * x_batch.size(0)

                train_loss_epoch /= len(dataset)

                # Validation loss calculation
                self.model.eval()
                val_loss_epoch = 0.0
                with torch.no_grad():
                    for x_val_batch, y_val_batch in val_loader:
                        y_val_pred = self.predict(x_val_batch)
                        val_loss_epoch += self.loss_fn(y_val_pred, y_val_batch).item() * x_val_batch.size(0)

                val_loss_epoch /= len(val_dataset)

                # Update results
                self.update_results(
                    k=epoch,
                    train_loss=train_loss_epoch,
                    valid_loss=val_loss_epoch,
                    model_state=self.model.state_dict()
                )

                # Progress bar update
                pbar = update_progress_bar(
                    pbar=pbar,
                    train_loss=train_loss_epoch,
                    valid_loss=val_loss_epoch,
                    no_improvement=self.no_improvement
                )

                if self.exceeded_patience():
                    print(f"\nEarly stopping at epoch {epoch+1}.")
                    break

        self.restore_best_model(epoch)

    def train(self, data):
        return self.train_nobatch(data)

    def train_nobatch(self, data):
        """
        train neural network
        """

        x_train_val = self.maybe_torch(data.x_train_val)
        x_val = self.maybe_torch(data.x_val)
        y_train_val = self.maybe_torch(data.y_train_val)
        y_val = self.maybe_torch(data.y_val)
        
        # Add a progress bar
        with tqdm(total=self.epochs, desc="Training Progress", unit="epoch") as pbar:
            for k in range(self.epochs):
                
                self.optimizer.zero_grad()  # Clear gradients from the previous step
                
                y_pred = self.predict(x_train_val)  # Forward pass for training data
                valid_pred = self.predict(x_val)  # Forward pass for validation data
                
                # Loss used for gradient calculation
                loss = self.loss_fn(y_pred, y_train_val)
                
                train_loss = calculate_loss(actual=y_train_val, prediction=y_pred)
                valid_loss = calculate_loss(actual=y_val, prediction=valid_pred)
                
                loss.backward()  # Backpropagate the gradient
                self.optimizer.step()  # Update model parameter
    
                # update progress bar
                pbar = update_progress_bar(
                    pbar=pbar,
                    train_loss=train_loss,
                    valid_loss=valid_loss,
                    no_improvement=self.no_improvement
                )
    
                # update results
                self.update_results(
                    k=k,
                    train_loss=train_loss,
                    valid_loss=valid_loss,
                    model_state=self.model.state_dict()
                )
    
                # If no improvement for 'patience' epochs, stop training
                if self.exceeded_patience():
                    print(f"\nEarly stopping at epoch {k+1}. Validation loss has not improved for {self.patience} epochs.")
                    break
    
                # Free memory by deleting intermediate variables
                del loss, y_pred
                
        # Restore the best model state after training
        self.restore_best_model(k)

    def update_results(self, k:int, train_loss:float, valid_loss:float, model_state:dict) -> None:

        # Record the losses for this epoch
        self.loss_array[k, 0] = k  
        self.loss_array[k, 1] = train_loss 
        self.loss_array[k, 2] = valid_loss

        # Early stopping: Check if validation loss improves
        if valid_loss < self.best_loss:
            self.best_loss = valid_loss  # Update best loss
            self.no_improvement = 0
            self.best_model_state = model_state 
        else:
            self.no_improvement = self.no_improvement + 1  # Increment no improvement counter

    def exceeded_patience(self) -> bool:
        return self.no_improvement >= self.patience

    def restore_best_model(self, k:int):
        """
        Restore the best model state after training
        """
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        self.loss_array = self.loss_array[:k,:]

    @staticmethod
    def maybe_torch(x):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if isinstance(x, torch.Tensor):
            return x if x.device.type == device else x.to(device)
        if isinstance(x, pd.DataFrame):
            return torch.FloatTensor(x.to_numpy()).to(device)
        elif isinstance(x, np.ndarray):
            return torch.FloatTensor(x).to(device)
        
    def performance(self, x, y):
        self.model.eval()
        with torch.no_grad():
            y_pred_test = self.predict_by_batch(x)

        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y)

        return supporting_functions.evaluate_test_torch(y, y_pred_test)




class DataParameters(object):
    def __init__(
        self,
        x_val:pd.DataFrame,
        y_val:pd.DataFrame,
        x_train_val:pd.DataFrame,
        y_train_val:pd.DataFrame,
        x_test:pd.DataFrame,
        y_test:pd.DataFrame,
        x_unseen:pd.DataFrame,
        y_unseen:pd.DataFrame,
        x_seen:pd.DataFrame,
        y_seen:pd.DataFrame,
        seen_val_mask:np.array,
        unseen_val_mask:np.array,
        df:pd.DataFrame,
    ):
        self.x_val = x_val
        self.y_val = y_val
        self.x_train_val = x_train_val
        self.y_train_val = y_train_val
        self.x_test = x_test
        self.y_test = y_test
        self.x_unseen = x_unseen
        self.y_unseen = y_unseen
        self.x_seen = x_seen
        self.y_seen = y_seen
        self.seen_val_mask = seen_val_mask
        self.unseen_val_mask = unseen_val_mask
        self.df = df

    def save(self, path):
        """Save this object to disk as a pickle file."""
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        """Load this object from a pickle file."""
        with open(path, 'rb') as f:
            return pickle.load(f)


#===============================================
# Functions
#===============================================

def calculate_loss(actual:np.array, prediction:np.array) -> float:
    """
    calculate loss
    """
    return torch.mean(torch.abs(actual - prediction)).item()
    
def update_progress_bar(pbar, train_loss, valid_loss, no_improvement):
    """
    Update the progress bar with the current epoch and losses
    """
    pbar.set_postfix(
        train_loss=train_loss, 
        valid_loss=valid_loss, 
        patience_count=no_improvement
    )
    pbar.update(1)  # Increment the progress bar
    
    return pbar

def _add_year_month(df:pd.DataFrame) -> pd.DataFrame:
    """
    add year, month, and year_month columns to pandas dataframe from index `time`
    
    Parameters
    ----------
        df (pd.DataFrame): pandas dataframe with index `time`

    Returns
    -------
        (pd.DataFrame): pandas dataframe with new columns: year, month, month_year
    """

    df[ColumnFields.YEAR.value] = df.index.get_level_values(ColumnFields.TIME.value).year
    df[ColumnFields.MONTH.value] = df.index.get_level_values(ColumnFields.TIME.value).month
    df[ColumnFields.YEAR_MONTH.value] = df[ColumnFields.YEAR.value].astype(str) + "-" + df[ColumnFields.MONTH.value].astype(str)

    return df

def _as_numpy(x):
    if isinstance(x, np.ndarray):
        return x
        
    try:
        return x.cpu().detach().numpy()
        
    except:
        raise ValueError(f"unable to convert to numpy array instance of type {type(x)}")

def _get_endogenous_exogenous_data(df:pd.DataFrame, index_mask:np.array, features:list, target:list) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    get endogenous and exogenous variables

    Parameters
    ----------
        df (pd.DataFrame): dataframe with both endogenous and exogenous variables
        index_mask (np.array): numpy array with list of indices that will be returned
        features (list): column names of features (X) in df
        target (list): column names of target variable (y) in df

    Returns
    -------
        (pd.DataFrame, pd.DataFrame): tuple of pandas dataframes with the filtered data
    """

    x = df.loc[index_mask, features].to_numpy()
    y = df.loc[index_mask, target].to_numpy().ravel()

    return x, y

def _get_clean_socat_mask(df:pd.DataFrame, features:list, target:list) -> np.array:
    """
    get cleansed socat mask
    """
    columns = features + target + [ColumnFields.NET_MASK.value]

    # remove values below -250 and above 250
    anomaly_mask = ((df[target] < 250) & (df[target] > -250)).to_numpy().ravel()

    # remove nan values
    nan_mask = ~df[columns].isna().any(axis=1)

    # combine both conditions
    recon_sel = ((nan_mask) & (anomaly_mask))

    # create socat mask
    socat_mask_1 = df[ColumnFields.SOCAT_MASK.value] == 1
    socat_mask_0 = df[ColumnFields.SOCAT_MASK.value] == 0
    
    return (recon_sel & socat_mask_1), (recon_sel & socat_mask_0)


def _extract_features_from_file(
    fs,
    saving_paths,
    ens:str,
    member:str,
    features:str,
    target:str,
    train_year_mon:list,
    test_year_mon:list, 
    random_seeds,
    seed_loc,
    test_proportion:float=0.0,
    validation_proportion:float=0.2
) -> tuple:
    
    """
    extract train and test features from file
    """
    file_path = saving_paths.inputs_ens_member_path(ens=ens, member=member)
    
    with fs.open(file_path, 'rb') as filee:
        df = pd.read_parquet(filee)
        df = _add_year_month(df=df)

        # get clean socat mask
        seen_mask, unseen_mask = _get_clean_socat_mask(
            df=df,
            features=features,
            target=target
        )

        # filter train data
        train_mask = pd.Series(df[ColumnFields.YEAR_MONTH.value]).isin(train_year_mon)
        socat_train_mask = (seen_mask & train_mask).to_numpy().ravel()
        x_train, y_train = _get_endogenous_exogenous_data(
            df=df,
            index_mask=socat_train_mask,
            features=features,
            target=target
        )

        # filter test data
        test_mask = pd.Series(df[ColumnFields.YEAR_MONTH.value]).isin(test_year_mon)
        socat_test_mask  = (seen_mask & test_mask).to_numpy().ravel()
        x_test, y_test = _get_endogenous_exogenous_data(
            df=df,
            index_mask=socat_test_mask,
            features=features,
            target=target
        )

        # unseen validation data (only for validation and not for training)
        unseen_val_mask = unseen_mask.to_numpy().ravel()
        x_unseen, y_unseen = _get_endogenous_exogenous_data(
            df=df,
            index_mask=unseen_val_mask,
            features=features,
            target=target
        )

        # seen validation data (only for validation and not for training)
        seen_val_mask = seen_mask.to_numpy().ravel()
        x_seen, y_seen = _get_endogenous_exogenous_data(
            df=df,
            index_mask=seen_val_mask,
            features=features,
            target=target
        )
                
        n = x_train.shape[0]

        # split train and validation data
        train_val_idx, train_idx, val_idx, test_idx = \
            supporting_functions.train_val_test_split(
                N=n,
                test_prop=test_proportion,
                val_prop=validation_proportion,
                random_seeds=random_seeds,
                ens_count=seed_loc
            )
        x_train_val, x_train, x_val, x_test_tmp, y_train_val, y_train, y_val, y_test_tmp = \
            supporting_functions.apply_splits(
                x_train,
                y_train,
                train_val_idx,
                train_idx,
                val_idx,
                test_idx
            ) 

    return DataParameters(
        x_val=x_val,
        y_val=y_val,
        x_train_val=x_train_val,
        y_train_val=y_train_val,
        x_test=x_test,
        y_test=y_test,
        x_unseen=x_unseen, 
        y_unseen=y_unseen,
        x_seen=x_seen,
        y_seen=y_seen,
        seen_val_mask=seen_val_mask,
        unseen_val_mask=unseen_val_mask,
        df=df,
    )

def _load_model(model_type:str, ens:str, member:str, saving_paths, random_seeds, seed_loc, **kwargs):
    """
    load locally saved model
    """

    print("loading model...")
    extension = _model_parameters(model_type=model_type)       
    
    # get model path
    model_path = saving_paths.load_model_local_path(
        ens=ens,
        member=member,
        extension=extension
    )
    
    if Models(model_type) == Models.XGBOOST:

        model = XGBoostModel(
            random_seeds=random_seeds,
            seed_loc=seed_loc,
            **kwargs)

        booster = xgb.Booster()
        booster.load_model(model_path)
            
        xgb_model = xgb.XGBRegressor()
        xgb_model._Booster = booster
        xgb_model._le = None 

        model.model = xgb_model

        return model

    elif Models(model_type) == Models.NEURAL_NETWORK:
        # Create a new instance of the network
        model = NeuralNetworkModel(**kwargs)
        
        # Load the saved state_dict
        checkpoint = torch.load(model_path)
        
        # Load the model's state_dict
        model.model.load_state_dict(checkpoint)

        model.model.eval()

        return model
            
    else:
        raise ValueError(f"model {model_type} not supported! The only models supported are: [`{Models.XGBOOST.value}`, `{Models.NEURAL_NETWORK.value}`]")

def _model_parameters(model_type:str):
    """
    get model specific parameters
    """
    if Models(model_type) == Models.XGBOOST:
        extension = "json"
        
        return extension
            
    elif Models(model_type) == Models.NEURAL_NETWORK:
        extension = "pth"

        return extension
            
    else:
        raise ValueError(f"model {model_type} not supported! The only models supported are: [`{Models.XGBOOST.value}`, `{Models.NEURAL_NETWORK.value}`]")

def _get_new_model(model_type:str, random_seeds, seed_loc, **kwargs):
    """
    define model
    """
    if Models(model_type) == Models.XGBOOST:
        model = XGBoostModel(
            random_seeds=random_seeds,
            seed_loc=seed_loc,
            **kwargs)
            
    elif Models(model_type) == Models.NEURAL_NETWORK:
        model = NeuralNetworkModel(**kwargs)
    else:
        raise ValueError(f"model {model_type} not supported! The only models supported are: [`{Models.XGBOOST.value}`, `{Models.NEURAL_NETWORK.value}`]")

    return model
    
def train_member_models(
    saving_paths,
    features,
    target,
    train_year_mon,
    test_year_mon,
    run_selected_mems_dict,
    seed_loc_dict,
    dates,
    model_type:str,
    is_training:bool,
    test_proportion:float=0.0,
    validation_proportion:float=0.2,
    **kwargs
):
    fs = gcsfs.GCSFileSystem()
    random_seeds = np.load(fs.open(saving_paths.path_seeds))   
    print(datetime.datetime.now())
    
    for ens, mem_list in run_selected_mems_dict.items():
        for member in mem_list:
            print(ens, member)

            seed_loc = seed_loc_dict[ens][member]
            
            # fetch data
            data = _extract_features_from_file(
                fs=fs,
                saving_paths=saving_paths,
                ens=ens,
                member=member,
                features=features,
                target=target,
                train_year_mon=train_year_mon,
                test_year_mon=test_year_mon,
                random_seeds=random_seeds,
                seed_loc=seed_loc,
                test_proportion=test_proportion,
                validation_proportion=validation_proportion
            )

            if is_training:
                # get model
                model = _get_new_model(
                    model_type=model_type,
                    random_seeds=random_seeds,
                    seed_loc=seed_loc,
                    **kwargs
                )
                model.train(data=data)
            else:
                model = _load_model(
                    model_type=model_type,
                    ens=ens,
                    member=member,
                    saving_paths=saving_paths,
                    random_seeds=random_seeds,
                    seed_loc=seed_loc,
                    **kwargs
                )

            # test performance
            performance = defaultdict(dict)
            if is_training:
                performance[ens][member] = model.performance(x=data.x_test, y=data.y_test)
                performance_path = saving_paths.test_perform_fname
                label = "test"
            else:
                performance[ens][member] = model.performance(x=data.x_unseen, y=data.y_unseen)
                performance_path = saving_paths.unseen_perform_fname
                label = "unseen"

                # save reconstruction
                model.save_reconstruction(
                    ens=ens,
                    member=member,
                    df=data.df,
                    x_seen=data.x_seen,
                    x_unseen=data.x_unseen,
                    seen_mask=data.seen_val_mask,
                    unseen_mask=data.unseen_val_mask, 
                    dates=dates,
                    path=saving_paths.recon_output_dir,
                    target=target,
                    model_type=model_type
                )
            
            # save performance
            model.save_performance(
                fs=fs,
                path=performance_path,
                data=data,
                performance=performance,
                ens=ens,
                member=member)

            if is_training:
                # save model locally
                extension = _model_parameters(model_type=model_type)
                supporting_functions.save_model_locally(
                    model=model.model, 
                    dates=dates, 
                    local_output_dir=saving_paths.model_local_save_dir, 
                    ens=ens, 
                    member=member,
                    extension=extension
                )

            print(f'{label} performance metrics:', performance[ens][member])

            del data, model
            
    print('end of all members', datetime.datetime.now())




def get_first_member_predictions(
    saving_paths,
    features,
    target,
    train_year_mon,
    test_year_mon,
    run_selected_mems_dict,
    seed_loc_dict,
    test_proportion:float=0.0,
    validation_proportion:float=0.2,
    **kwargs
):
    fs = gcsfs.GCSFileSystem()
    random_seeds = np.load(fs.open(saving_paths.path_seeds))   
    print(datetime.datetime.now())
    
    for ens, mem_list in run_selected_mems_dict.items():
        for member in mem_list:
            print(ens, member)

            seed_loc = seed_loc_dict[ens][member]
            
            # fetch data
            data = _extract_features_from_file(
                fs=fs,
                saving_paths=saving_paths,
                ens=ens,
                member=member,
                features=features,
                target=target,
                train_year_mon=train_year_mon,
                test_year_mon=test_year_mon,
                random_seeds=random_seeds,
                seed_loc=seed_loc,
                test_proportion=test_proportion,
                validation_proportion=validation_proportion
            )
            nn_params = {
                'input_nodes':13,
                'hidden_nodes':50,
                'output_nodes':1,
                'epochs':3000,
                'lr': 1e-03
            }
            xgb_params = {
                'n_estimators': 500,  # Number of boosting rounds
                'max_depth': 6,  # Maximum depth of each tree to control model complexity
                'learning_rate': 0.05,  # Step size shrinkage to prevent overfitting
                'subsample': 0.8,  # Fraction of samples used for training each tree
                'colsample_bytree': 0.8,  # Fraction of features used per tree
                'gamma': 0.1,  # Minimum loss reduction required for further partitioning
                'min_child_weight': 5,  # Minimum sum of instance weight in a leaf node
                'reg_alpha': 0.1,  # L1 regularization to reduce model complexity
                'reg_lambda': 1.0,  # L2 regularization for preventing overfitting
                'objective': 'reg:squarederror',  # Loss function for regression tasks
                'n_jobs': 30,  # Number of parallel threads to use for training
                'eval_metric': 'rmse',
                'early_stopping_rounds': 50  # Stop training if performance doesn't improve for 50 rounds
            }
            xgb_model = _load_model(
                model_type=Models.XGBOOST.value,
                ens=ens,
                member=member,
                saving_paths=saving_paths,
                random_seeds=random_seeds,
                seed_loc=seed_loc,
                **xgb_params
            )
            nn_model = _load_model(
                model_type=Models.NEURAL_NETWORK.value,
                ens=ens,
                member=member,
                saving_paths=saving_paths,
                random_seeds=random_seeds,
                seed_loc=seed_loc,
                **nn_params
            )

            xgb_output = xgb_model.predict(data.x_unseen)
            nn_model.model.eval()
            with torch.no_grad():
                nn_output = nn_model.predict_by_batch(data.x_unseen)

            return xgb_output, nn_output, data

            
    print('end of all members', datetime.datetime.now())
