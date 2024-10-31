import argparse, os, time, torch
import pandas as pd
import numpy as np
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import warnings
warnings.filterwarnings("ignore")


class Model():

    def __init__(self, name):
        self.name = name
        self.mtype = None
        self.param_grid = []  
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        ##initialize model based on its name:
        if self.name == "MLP":
            self.mtype=  MLPRegressor(random_state=42)
            self.param_grid = {'hidden_layer_sizes': [(50, 50), (50, 100), (100, 100)],\
                                'activation': ['relu', 'tanh'], \
                                'learning_rate_init': [0.01, 0.1]
                               }
        elif self.name == "XGBoost":
            self.mtype = xgb.XGBRegressor(device=self.device, booster="gblinear", nthread=32, random_state=42)
            self.param_grid = { 'max_depth': [3, 5, 7], \
                                'learning_rate': [0.1, 0.5, 1], \
                                'n_estimators': [50, 100, 200], \
                                'gamma': [0, 0.1, 0.5], \
                                'subsample': [0.5, 0.8, 1], \
                               'colsample_bytree': [0.5, 0.8, 1]
                              }
        else:
            self.mtype =  CatBoostRegressor(loss_function='RMSE',
                              per_float_feature_quantization="0:border_count=128", bagging_temperature=1,
                              bootstrap_type="Bayesian", random_strength=0.5, grow_policy='Depthwise', border_count=128,
                              has_time=True, feature_border_type="Uniform")
            
            self.param_grid = {'learning_rate': [0.01, 0.1, 1], \
                                'depth': [3, 5, 7], \
                                'l2_leaf_reg': [0.1, 1, 10], \
                                'iterations': [100, 500, 1000]
                               }

    def __repr__(self):
        return f"Model info: {self.mtype} \n Parameters to optimize: \n {self.param_grid}"
        
            
    def train_validate_test(self, X, y, test_size):
        print("Data splitting:")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
        print("------shapes------")
        print(f"Training:  {X_train.shape} \n Test: {X_test.shape}")

        MinMax = MinMaxScaler(feature_range=(0, 1))
        X_train = MinMax.fit_transform(X_train)
        X_test = MinMax.transform(X_test)

        print("Performing Grid Search...")
        start_time = time.time()
        grid_search = GridSearchCV(self.mtype, self.param_grid, cv=5, scoring='r2')
        grid_search.fit(X_train, y_train)

        total_time = time.time() - start_time
        print(f"Total training time: {total_time:.2f} seconds")

        print("Best parameters found:")
        best_model = grid_search.best_estimator_
        print(best_model)

        y_pred = best_model.predict(X_test)
        
        print("Evaluation results:")
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        eval_report = f"Best R2 score: {r2:.4f} \n Mean Absolute Error: {mae:.4f} \n Mean Squared Error: {mse:.4f} \n Root Mean Squared Error: {rmse:.4f}"
        print(eval_report)
        
        return y_pred, best_model, total_time, eval_report

#---------------- MAIN ---------------------------------                       
def main():

    test_set_portions = {
        "three_days_7_9_10_13.csv": 0.5305, \
        "seven_days_3_9_10_13.csv": 0.3214,\
        "one_month_10-9_9_10_13.csv": 0.0984, \
        "three_months_12-7_9_10_13.csv": 0.0350, \
        "five_months_13-5_9_10_13.csv": 0.0213,
    }

    #modelnames = [ "MLP"]
    modelnames = ["XGBoost","CatBoost"]

    with open("results_info.txt", "w", encoding="utf-8") as f:
        dir_path = "data/"
        abs_dir_path = os.path.abspath(dir_path)
        for modelname in modelnames:
            for filename in os.listdir(dir_path):
                file_path = os.path.join(abs_dir_path, filename)
                if os.path.isfile(file_path) and file_path.endswith(".csv"):
                    message = f"Running experiments with {modelname} for data in {file_path}..."
                    print(message)
                    f.write(" \n -------------------------------------------------------------- \n")
                    f.write(message)
                    
                    df = pd.read_csv(file_path, sep=',')
                    df_wo_date = df.drop(columns=["Datetime"], axis = 1)
                    X = df_wo_date.drop(columns=["Purchased(W)"], axis=1)
                    y = df_wo_date["Purchased(W)"]
                
                    model = Model(modelname)
                    y_pred, best_model, total_time, eval_report = model.train_validate_test(X, y, test_set_portions[filename])
                    res_df = pd.DataFrame({"Predicted Net Load (W)": y_pred})
                    res_file = f"res_{modelname}_{filename}"
                    print(f"Printing results in {res_file}")
                    res_df.to_csv(res_file)
    
                    message = f"Best model configuration: \n {best_model} \n Total training time: {total_time} \n Evaluation results: {eval_report}"
                    f.write(message)
                


#------------------------------------------------------------


if __name__=="__main__": 
    main()


