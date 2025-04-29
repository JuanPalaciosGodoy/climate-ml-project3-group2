# **Project 3: Machine Learning Reconstruction of Surface Ocean pCO₂**

## Project Assignment

+ Team 2
+ Team members
    + Primanta Bangun
    + Samuel Kortchmar
    + Sarika de Bruyn
    + Juan Palacios

+ Project Summary: In this project, we use machine learning techniques (**XGBoost** and **Neural Networks**) to reconstruct surface ocean partial pressure of carbon dioxide (pCO₂) fields based on environmental predictors like sea surface temperature, salinity, mixed layer depth, and atmospheric CO₂. Due to the sparse observational coverage (e.g., SOCAT), we benchmark model skill using the Large Ensemble Testbed (LET), which provides full-coverage synthetic pCO₂ fields from Earth System Models (ESMs). By decomposing pCO₂-Residual into seasonal+trend and stochastic components, we improve ML reconstructions by matching model selection to the underlying variability patterns.

**CONTRIBUTION STATEMENT**: 

+ All team members contributed substantially to the project:
+ Samuel Kortchmar experimented with different visualizations of XGBoost decision trees to understand feature importance evolution, but ultimately abandoned this approach due to limited insights. Later contributed to optimizing the Neural Networks for GPU use.
+ Sarika de Bruyn tried to add ENSO as a feature in attempts to improve the Baseline model; tried to help fix toggling issues between NGBoost and Neural Network models; edited and cleaned the notebook to improve the data story.
+ Juan Palacios implemented the seasonal decomposition of pCO₂ residuals; developed separate XGBoost and Neural Network models to predict the seasonal+trend and deseasonal components independently; aggregated both outputs into final pCO₂ residual predictions.
+ Primanta Bangun experimented with developing the notebook using only a Neural Network (NN), but later abandoned this approach because the results were worse than the original notebook. Afterwards, do Model Performance Evaluation across Earth System Models and Ocean Regions, including calculating bias, RMSE, and correlation by ensemble and region; visualizing model performance through bar plots and regional comparisons; interpreting performance differences between ESMs.
+ All team members designed the study, contributed to the GitHub repository, and prepared the final presentation.
+ All team members approve the work presented in the GitHub repository, including this contribution statement.

## **Folder Structure**

To reduce complexity in the main notebook, several helper functions and figures are modularized into the `lib/` directory.

```bash
Project3/
├── lib/                       # Helper scripts
│   ├── __init__.py
│   ├── bias_figure2.py        # Code for bias calculation and visualization
│   ├── corr_figure3.py        # Code for correlation calculation and visualization
│   ├── residual_utils.py      # Prepares data for ML, tools for dataset splitting, model evaluation, and saving files.
│   └── visualization.py       # Custom plotting class SpatialMap2 for creating high-quality spatial visualizations with colorbars and map features using Cartopy and Matplotlib.
├── notebooks/
│   ├── Project3_Starter.ipynb # Original notebook containing full analysis & data story
|   ├── leappersistent_file_management.ipynb # check the size of files and clean up
|   ├── Project3_Data.ipynb  # Used for preprocessing data, if more than the 20 preprocessed ESM members are required.
|   └── OceanpCO2_Group2.ipynb # Main file for implementing group 2's project

```
## **Instructions for Reviewer to Run:**
1. Open Project3_Data.ipynb --> change to reviewer's username --> run
2. Open OceanpCO2_Group2.ipynb --> change to reviewer's username
   a. Set __MODEL_TYPE__ = "nn" --> run until section 2.3
   b. Set __MODEL_TYPE__ = "xgb" --> run entire notebook

## **References**

- **Gloege et al. (2020)**  
  *Quantifying Errors in Observationally Based Estimates of Ocean Carbon Sink Variability.*  
  [DOI: 10.1029/2020GB006788](https://doi.org/10.1029/2020GB006788)

- **Bennington et al. (2022)**
  *Explicit Physical Knowledge in Machine Learning for Ocean Carbon Flux Reconstruction: The pCO2-Residual Method*
   [DOI: 10.1029/2021ms002960](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2021MS002960)

- **Heimdal et al. (2024)**  
  *Assessing improvements in global ocean pCO₂ machine learning reconstructions with Southern Ocean autonomous sampling.*  
  [DOI: 10.5194/bg-21-2159-2024](https://doi.org/10.5194/bg-21-2159-2024)


## **Contributions and Collaboration**

We encourage collaborative version control using GitHub. If working in a team, please follow contribution guidelines (e.g., commit messages, branches).

You may find [this GitHub tutorial](https://github.com/leap-stc/LEAPCourse-Climate-Pred-Challenges/blob/main/Tutorials/Github-Tutorial.md) helpful for getting started.


