import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
import os
import datetime
from collections import defaultdict
import csv

# libraries
class FeedForwardNN:
    def __init__(self, input_dim, hidden_layers=(128, 64, 32), dropout_rate=0.2, 
                 learning_rate=0.001, random_state=42):
        """
        Initialize a feedforward neural network for pCO2 residual prediction.
        
        Parameters:
        -----------
        input_dim : int
            Number of input features
        hidden_layers : tuple
            Number of neurons in each hidden layer
        dropout_rate : float
            Dropout rate for regularization
        learning_rate : float
            Learning rate for Adam optimizer
        random_state : int
            Random seed for reproducibility
        """
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.model = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        
        # Set random seeds for reproducibility
        tf.random.set_seed(random_state)
        np.random.seed(random_state)
        
        # Build the model
        self._build_model()
    
    def _build_model(self):
        """Build the neural network architecture"""
        model = Sequential()
        
        # Input layer
        model.add(Dense(self.hidden_layers[0], activation='relu', 
                        input_dim=self.input_dim,
                        kernel_initializer='he_normal'))
        model.add(BatchNormalization())
        model.add(Dropout(self.dropout_rate))
        
        # Hidden layers
        for units in self.hidden_layers[1:]:
            model.add(Dense(units, activation='relu', kernel_initializer='he_normal'))
            model.add(BatchNormalization())
            model.add(Dropout(self.dropout_rate))
        
        # Output layer
        model.add(Dense(1, activation='linear'))
        
        # Compile model
        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(loss='mse', optimizer=optimizer, metrics=['mae'])
        
        self.model = model
        return model
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, batch_size=64, 
            epochs=100, patience=20, verbose=1, save_path=None):
        """
        Train the neural network
        
        Parameters:
        -----------
        X_train : numpy.ndarray
            Training features
        y_train : numpy.ndarray
            Training target
        X_val : numpy.ndarray, optional
            Validation features
        y_val : numpy.ndarray, optional
            Validation target
        batch_size : int
            Batch size for training
        epochs : int
            Maximum number of epochs
        patience : int
            Patience for early stopping
        verbose : int
            Verbosity level
        save_path : str, optional
            Path to save the best model
            
        Returns:
        --------
        history : tf.keras.callbacks.History
            Training history
        """
        # Scale the input features and target
        X_train_scaled = self.scaler_X.fit_transform(X_train)
        y_train_scaled = self.scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
        
        callbacks = [EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)]
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            callbacks.append(ModelCheckpoint(filepath=save_path, 
                                             monitor='val_loss',
                                             save_best_only=True))
        
        # Prepare validation data if provided
        validation_data = None
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler_X.transform(X_val)
            y_val_scaled = self.scaler_y.transform(y_val.reshape(-1, 1)).ravel()
            validation_data = (X_val_scaled, y_val_scaled)
        
        # Train the model
        history = self.model.fit(
            X_train_scaled, y_train_scaled,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        
        return history
    
    def predict(self, X):
        """
        Make predictions with the trained model
        
        Parameters:
        -----------
        X : numpy.ndarray
            Input features
            
        Returns:
        --------
        y_pred : numpy.ndarray
            Predicted values
        """
        X_scaled = self.scaler_X.transform(X)
        y_pred_scaled = self.model.predict(X_scaled, verbose=0)
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled)
        return y_pred.ravel()

    def save_model(self, model_path, scaler_X_path=None, scaler_y_path=None):
        """Save the trained model and scalers"""
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save only the weights
        self.model.save_weights(model_path)
        
        # Save model configuration separately
        model_config = self.model.to_json()
        with open(f"{os.path.splitext(model_path)[0]}_config.json", 'w') as f:
            f.write(model_config)
        
        # Save scalers if paths are provided
        if scaler_X_path:
            import joblib
            joblib.dump(self.scaler_X, scaler_X_path)
        
        if scaler_y_path:
            import joblib
            joblib.dump(self.scaler_y, scaler_y_path)    
    
    @classmethod
    def load_model(cls, model_path, scaler_X_path=None, scaler_y_path=None):
        """Load a saved model and scalers"""
        import tensorflow as tf
        
        # Create an instance without initializing
        instance = cls.__new__(cls)
        
        # Define custom objects with the correct loss function for TF 2.17.0
        custom_objects = {
            'mse': tf.keras.losses.MeanSquaredError(),
            'mean_squared_error': tf.keras.losses.MeanSquaredError(),
            'MeanSquaredError': tf.keras.losses.MeanSquaredError
        }
        
        # Attempt to load model with proper custom objects
        print(f"Attempting to load model from {model_path}")
        instance.model = tf.keras.models.load_model(
            filepath=model_path, 
            custom_objects=custom_objects,
            compile=False  # Load without compiling first
        )
        
        # Set input dimension
        instance.input_dim = 13  # Default to number of features
        
        # Recompile manually
        instance.model.compile(
            optimizer='adam',
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=['mae']
        )
        
        print("Model loaded successfully")
        
        # Load scalers if paths are provided
        if scaler_X_path and scaler_y_path:
            import joblib
            instance.scaler_X = joblib.load(scaler_X_path)
            instance.scaler_y = joblib.load(scaler_y_path)
            print("Scalers loaded successfully")
        else:
            from sklearn.preprocessing import StandardScaler
            instance.scaler_X = StandardScaler()
            instance.scaler_y = StandardScaler()
        
        return instance

def save_model_locally(model, dates, output_dir, ensemble, member):
    """
    Save a trained FFNN model to local directory
    
    Parameters:
    -----------
    model : FeedForwardNN
        Trained model to save
    dates : pandas.DatetimeIndex
        Date range used for training
    output_dir : str
        Directory to save model
    ensemble : str
        Ensemble name
    member : str
        Member identifier
    """
    init_date = str(dates[0].year) + format(dates[0].month, '02d')
    fin_date = str(dates[-1].year) + format(dates[-1].month, '02d')
    
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate paths
    model_path = f"{output_dir}/model_pCO2_2D_{ensemble}_{member.split('_')[-1]}_mon_1x1_{init_date}_{fin_date}"
    
    # Save model components
    model.save_model(
        f"{model_path}.h5", 
        f"{model_path}_scaler_X.joblib", 
        f"{model_path}_scaler_y.joblib"
    )


def evaluate_test(y_true, y_pred):
    """
    Calculate performance metrics for model evaluation
    
    Parameters:
    -----------
    y_true : numpy.ndarray
        True values
    y_pred : numpy.ndarray
        Predicted values
        
    Returns:
    --------
    metrics : dict
        Dictionary of performance metrics
    """
    metrics = {}
    
    # Mean metrics
    metrics['bias'] = np.mean(y_pred - y_true)
    metrics['mae'] = np.mean(np.abs(y_pred - y_true))
    metrics['rmse'] = np.sqrt(np.mean((y_pred - y_true)**2))
    
    # Calculate R^2
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    ss_res = np.sum((y_true - y_pred)**2)
    metrics['r2'] = 1 - (ss_res / ss_tot)
    
    return metrics


def apply_splits(X, y, train_val_idx, train_idx, val_idx, test_idx):
    """
    Apply train/validation/test splits
    
    Parameters:
    -----------
    X : numpy.ndarray
        Feature matrix
    y : numpy.ndarray
        Target vector
    train_val_idx, train_idx, val_idx, test_idx : numpy.ndarray
        Indices for different splits
        
    Returns:
    --------
    Splits of X and y for training, validation, and testing
    """
    return (
        X[train_val_idx], X[train_idx], X[val_idx], X[test_idx],
        y[train_val_idx], y[train_idx], y[val_idx], y[test_idx]
    )


def train_val_test_split(N, test_prop, val_prop, random_seeds, seed_loc):
    """
    Generate indices for train/validation/test split
    
    Parameters:
    -----------
    N : int
        Total number of samples
    test_prop : float
        Proportion for test set
    val_prop : float
        Proportion for validation set
    random_seeds : numpy.ndarray
        Array of random seeds
    seed_loc : int
        Index in random_seeds to use
        
    Returns:
    --------
    Indices for different data splits
    """
    np.random.seed(random_seeds[0, seed_loc])
    
    # Generate random indices
    idx = np.random.permutation(N)
    
    # Calculate sizes
    n_test = int(N * test_prop)
    n_val = int(N * val_prop)
    n_train = N - n_test - n_val
    
    # Split indices
    train_val_idx = idx[:]  # All data for train+val
    train_idx = idx[:n_train]  # Train set
    val_idx = idx[n_train:n_train+n_val]  # Validation set
    test_idx = idx[n_train+n_val:]  # Test set
    
    return train_val_idx, train_idx, val_idx, test_idx