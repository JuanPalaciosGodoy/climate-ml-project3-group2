import matplotlib.pyplot as plt
import pandas as pd

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_residuals_side_by_side(xgb_output, nn_output, data, features_sel, sample_size=10000, random_state=42):
    """
    Plot XGBoost and Neural Network residuals side by side for each feature,
    using a random sample to avoid overplotting.

    Parameters:
    - xgb_output: Predictions from XGBoost model
    - nn_output: Predictions from Neural Network model
    - data: Data object containing x_unseen and y_unseen
    - features_sel: List of selected feature names
    - sample_size: Number of points to sample for plotting (default 5000)
    - random_state: Seed for reproducibility (default 42)
    """

    # Compute residuals
    residuals_xgb = xgb_output - data.y_unseen
    residuals_nn = nn_output - data.y_unseen

    # Reconstruct test DataFrame
    df_test = pd.DataFrame(data.x_unseen, columns=features_sel)
    df_test["residual_xgb"] = residuals_xgb
    df_test["residual_nn"] = residuals_nn

    # Sample the data
    if len(df_test) > sample_size:
        df_sampled = df_test.sample(n=sample_size, random_state=random_state)
    else:
        df_sampled = df_test

    # Plot residuals vs each feature side by side
    for feature in features_sel:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

        axes[0].scatter(df_sampled[feature], df_sampled["residual_xgb"], alpha=0.3, s=10)
        axes[0].axhline(0, color='red', linestyle='--')
        axes[0].set_title(f"XGB Residual vs {feature}")
        axes[0].set_xlabel(feature)
        axes[0].set_ylabel("Residual")
        axes[0].grid(True)

        axes[1].scatter(df_sampled[feature], df_sampled["residual_nn"], alpha=0.3, s=10)
        axes[1].axhline(0, color='red', linestyle='--')
        axes[1].set_title(f"NN Residual vs {feature}")
        axes[1].set_xlabel(feature)
        axes[1].grid(True)

        plt.tight_layout()
        plt.show()

