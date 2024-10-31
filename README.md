This repository contains the code used to run the experiments described in the paper **`Effective Net Load Forecasting in Solar-Integrated Homes with Battery Systems`**

### Files and Folders
- `notebooks/`: Contains Jupyter notebooks used for data pre-processing and post-processing.
- `NetLoadForecasting.py`: The main Python script where the core experiments are implemented.


## Usage

1. The following Python packages are required to run the experiments: `scikit-learn`, `pandas`, `numpy`, `catboost`, `xgboost`

1. Make sure that your dataset is included in a `data/` folder and formatted in a comma-separated value format.

2. To run the Python script, simply execute:
    ```bash
    python NetLoadForecasting.py
    ```


