# Generalized SoC Consumption Estimation Using Machine Learning

This repository contains the code and data for the research paper "Generalized SoC Consumption Estimation Using Machine Learning," submitted to the Journal of Energy Storage. The goal of this project is to estimate the State of Charge (SoC) consumption in electric vehicles using various machine learning models.

## Table of Contents
- [Introduction](#introduction)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
  - [Configuration](#configuration)
  - [Running the Code](#running-the-code)
- [Models and Features](#models-and-features)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Introduction

The increasing adoption of electric vehicles (EVs) necessitates accurate and efficient estimation of battery SoC consumption. This project leverages machine learning techniques to create models that predict SoC consumption based on various input features. The study compares different regression models, including SVR, LinearSVR, DecisionTreeRegressor, XGBoost, KNN, and others.

## Getting Started

### Prerequisites

To run this project, you need the following libraries and tools:
- Python 3.7 or higher
- pandas
- numpy
- matplotlib
- scikit-learn
- optuna
- xgboost
- pickle

### Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/Vaav-ai/Generalized-EV-SoC-Consumption-Estimation-using-Machine-Learning.git
    cd SoC-Consumption-Estimation
    ```

2. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

### Configuration

The configuration for the model, including the type of model to use, number of trials for hyperparameter tuning, feature selection method, and other parameters, is specified in a JSON file. An example configuration file (`config.json`) is provided:

```json
{
    "data_path": "path/to/your/data.csv",
    "model_path": "path/to/store/model/",
    "model": "XGBoost",
    "n_trials": 100,
    "n_input_params": 5,
    "n_features": 5,
    "feature_selection_method": "permutation_importance",
    "selected_features_count": 2
}
```

### Running the Code

1. Place your dataset at the specified path in the `config.json` file.
2. Run the main script:
    ```sh
    python main.py
    ```
3. The script will preprocess the data, perform feature selection, optimize the model hyperparameters using Optuna, train the best model, and save the trained model in a pickled format (`trained_model.pkl`).
4. Ensure that the name of the target column in your CSV file is "SoC Consumed".

## Models and Features

This project supports several machine learning models and feature selection methods:

### Models
- Support Vector Regression (SVR)
- Linear Support Vector Regression (LinearSVR)
- Decision Tree Regressor
- XGBoost Regressor
- K-Nearest Neighbors Regressor (KNN)
- Voting Regressor (ensemble of KNN models)
- Bagging Regressor with KNN
- Linear Regression

### Feature Selection Methods
- Permutation Importance
- F-Regression
- Mutual Information Regression

## Results

The results of the trained models, including plots for actual vs. predicted SoC consumption and residuals, are generated and displayed. The best hyperparameters for the selected model are also outputted.

## Credits

Code contribution by:
1. [Arav Jain](https://github.com/AravJain007) (Primary contributor)
2. [Vatsal Jha](https://github.com/Vatsal-Jha256)

## Acknowledgments

- The project is part of a study submitted to the Journal of Energy Storage. If you find this code useful please cite us.
`
@article{JAIN2024112884,
title = {Machine learning framework using on-road realtime data for battery SoC level prediction in electric two-wheelers},
journal = {Journal of Energy Storage},
volume = {97},
pages = {112884},
year = {2024},
issn = {2352-152X},
doi = {https://doi.org/10.1016/j.est.2024.112884},
url = {https://www.sciencedirect.com/science/article/pii/S2352152X24024708},
author = {Arav Jain and Vatsal Jha and Faisal Alsaif and B. Ashok and Indragandhi Vairavasundaram and C. Kavitha},
}
`
