import argparse
import time
import os

import numpy as np
import pandas as pd
import streamlit as st
from joblib import load
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor


def predict(
    path_ml_score,
    path_columns,
    path_mean,
    path_std,
    path_x_test,
    y_pred_filename,
    path_y_test,
    gpu=False,
):

    """
    Predicting binding affinity based on the trained XGBoost model.

     Parameters:
        path_ml_score (joblib): A ML saved model based on XGBoost in .joblib.
        path_columns (txt): A file contains name of all not discarded columns during
        preprocessing.
        path_mean (csv): A csv file contain all mean for features.
        path_std (csv): A csv file contain all std for features.
        path_x_test (csv): A csv file contains test set features.
        y_pred_filename (csv): Filename for saving prediction with .csv extension.
        path_y_test (csv): Label of the test set in .csv.

     Returns:
        rp (float), rmse (float): Return rp and rmse metrics on the test set.
    """
    
    if gpu:
    
      ml_score = XGBRegressor(
            n_estimators=20000,
            max_depth=8,
            learning_rate=0.005,
            subsample=0.7,
            tree_method="gpu_hist",
            predictor="gpu_predictor",
        )
        
      ml_score.load_model(path_ml_score)
      
    else:
    
      ml_score = XGBRegressor(
            n_estimators=20000,
            max_depth=8,
            learning_rate=0.005,
            subsample=0.7,
            tree_method="hist",
            predictor="cpu_predictor",
        )
        
      ml_score.load_model(path_ml_score)

    with open(path_columns, "r") as file:
        lines = file.readlines()
        columns = list(map(lambda x: x.strip(), lines))

    mean = pd.read_csv(path_mean, index_col=0)
    mean = mean.to_numpy().ravel()

    std = pd.read_csv(path_std, index_col=0)
    std = std.to_numpy().ravel()

    x_test = pd.read_csv(path_x_test, index_col=0)
    x_test = (x_test.loc[:, columns] - mean) / std

    y_pred = ml_score.predict(x_test)
    y_pred_df = pd.DataFrame(y_pred, index=list(x_test.index), columns=["y_pred"])
    y_pred_df = y_pred_df.round(3)
    y_pred_df.to_csv(y_pred_filename)

    if path_y_test != "None":

        y_test = pd.read_csv(path_y_test, index_col=0)
        y_test = y_test.reindex(x_test.index)
        y_test = y_test.to_numpy().ravel()
        rp = pearsonr(y_test, y_pred)[0]
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        st.info(f"*Rp*: {rp:.3f}    *RMSE*: {rmse:.3f}")

    return y_pred_df


if __name__ == "__main__":

    start = time.time()
    print("\n")
    print("Job is started.")
    print("------------------------------")

    parser = argparse.ArgumentParser(
        description="""Predicting binding affinity based on the trained 
            XGBoost model"""
    )

    parser.add_argument("-x", "--path_model", help="saved model path", required=True)
    parser.add_argument(
        "-y", "--path_columns", help="path of columns .txt", required=True
    )
    parser.add_argument("-m", "--path_mean", help="path of mean .csv", required=True)
    parser.add_argument("-s", "--path_std", help="path of std .csv", required=True)
    parser.add_argument(
        "-t", "--path_x_test", help="path of x_test .csv", required=True
    )
    parser.add_argument(
        "-f", "--path_filename", help="path of y_pred filename in .csv", required=True
    )
    parser.add_argument("-l", "--path_y_test", help="path of y_test .csv", default=None)

    args = parser.parse_args()

    print("Inputs")
    print(f"Path saved moodel: {args.path_model}")
    print(f"Path columns: {args.path_columns}")
    print(f"Path mean of features: {args.path_mean}")
    print(f"Path std of features: {args.path_std}")
    print(f"Path x_test: {args.path_x_test}")
    print(f"Path y_test: {args.path_y_test}")
    print(f"Path y_pred filename: {args.path_filename}")
    print("------------------------------")

    _ = predict(
        args.path_model,
        args.path_columns,
        args.path_mean,
        args.path_std,
        args.path_x_test,
        args.path_filename,
        path_y_test=args.path_y_test,
    )

    end = time.time()

    seconds = end - start
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)

    print("------------------------------")
    print(f"Job is done at {h} hours, {m} minutes and {s:.2f} seconds!")
    print(f"{args.path_filename} is created.")
