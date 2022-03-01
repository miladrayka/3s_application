import argparse
import time

import numpy as np
import pandas as pd
import streamlit as st
from joblib import dump
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor


def input_reader(path_input, path_target):

    """
    Read csv files of features and target values and return pd.DataFrame
    for both.

     Parameters:
        path_input (csv): Features file in .csv. Pdbid of complexes has to located in column 0.
        path_target (csv): Target file in .csv. This file has to contain a column with
        'pdbid' which determines pdbid of complexes.

     Returns:
        X, Y(tuple): X (features) and Y (target) pd.DataFrame
    """

    X = pd.read_csv(path_input, index_col=0)

    Y = pd.read_csv(path_target, index_col="pdbid")
    Y = Y.drop(labels=["Unnamed: 0"], axis=1)
    Y = Y.reindex(X.index)

    return X, Y


def preprocessing(data, var_threshold=0.01, corr_threshold=0.95):

    """
    Preprocess features input pd.DataFrame and drop static, quasi-static and
    correlated features. Return normalized and processed data in
    pd.DataFrame, mean and std of data.

     Parameters:
        data (pd.DataFrame): Features file in pd.DataFrame.
        var_threshold (float): Variance threshold. Features below this
        threshold are discarded.
        corr_threshold (float): Correlated features are discarded.

     Returns:
        data, mean, std (tuple): Return processed features and mean and std of all features.
    """

    data = data.loc[:, data.var(axis=0) > var_threshold]

    corr_matrix = data.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [
        column for column in upper.columns if any(upper[column] > corr_threshold)
    ]
    data = data.drop(to_drop, axis=1)

    mean = data.mean()
    std = data.std()
    data = (data - mean) / std

    return data, mean, std


def data_spliter(data, core_set_id, val_set_size=0):

    """
    Using "core set id" to splitted data to train and test set.
    return dictionary contains train and test set (and val set) features and targets.

     Parameters:
        data (pd.DataFrame): Features file in pd.DataFrame.
        core_set_id (csv): A csv file contain all pdbid of the core set (PDBbind 2016)
        val_set_size (float): If not zero, indicate the percentage of data in validation set.
        Validation set is created randomly with the stratified split method.

     Returns:
        sets (dict): Return dictionary contains train and test set (and val set) features and targets.
    """

    test_set = data.loc[core_set_id, :]
    train_set = data.drop(core_set_id, axis=0)

    if val_set_size:

        train_set["ba_cat"] = np.ceil(train_set["binding_affinity"] / 1.5)
        train_set["ba_cat"].where(train_set["ba_cat"] < 8, 8, inplace=True)

        split = StratifiedShuffleSplit(
            n_splits=10, test_size=val_set_size, random_state=42
        )

        for train_index, val_index in split.split(train_set, train_set["ba_cat"]):

            strat_train_set = train_set.iloc[list(train_index), :]
            strat_val_set = train_set.iloc[list(val_index), :]

        strat_train_set.drop(["ba_cat"], axis=1, inplace=True)
        strat_val_set.drop(["ba_cat"], axis=1, inplace=True)

        x_train = strat_train_set.iloc[:, :-1]
        y_train = strat_train_set.iloc[:, -1]
        x_val = strat_val_set.iloc[:, :-1]
        y_val = strat_val_set.iloc[:, -1]
        x_test = test_set.iloc[:, :-1]
        y_test = test_set.iloc[:, -1]

        sets = {
            "x_train": x_train,
            "y_train": y_train,
            "x_val": x_val,
            "y_val": y_val,
            "x_test": x_test,
            "y_test": y_test,
        }
    else:

        x_train = train_set.iloc[:, :-1]
        y_train = train_set.iloc[:, -1]
        x_test = test_set.iloc[:, :-1]
        y_test = test_set.iloc[:, -1]

        sets = {
            "x_train": x_train,
            "y_train": y_train,
            "x_test": x_test,
            "y_test": y_test,
        }

    return sets


def data_creator(
    path_x,
    path_y,
    path_test_id,
    var_threshold=0.01,
    corr_threshold=0.95,
    val_set_size=0,
):

    """
    Straightforwarding reading, preprocessing and splitting data.
    return the preprocessed and splited data.

     Parameters:
        path_x (csv): Features file in .csv. Pdbid of complexes has to located in column 0.
        path_y (csv): Target file in .csv. This file has to contain a column with
        'pdbid' which determines pdbid of complexes.
        path_test_id (csv): A csv file contain all pdbid of the core set (PDBbind 2016)
        var_threshold (float): Variance threshold. Features below this threshold are discarded.
        corr_threshold (float): Correlated features are discarded.
        val_set_size (float): If not zero, indicate the percentage of data in validation set.
        Validation set is created randomly with the stratified split method.

     Returns:
        splited (dict), mean (float), std (float): Return dictionary contains preprocessed train
        and test set (and val set) features and targets. Also returns mean and std of data.
    """

    X, Y = input_reader(path_x, path_y)
    X, mean, std = preprocessing(X, var_threshold=0.01, corr_threshold=0.95)

    data = pd.concat([X, Y], axis=1)

    index_id = list(pd.read_csv(path_test_id)["pdbid"])

    splited_data = data_spliter(data, index_id, val_set_size=0)

    return splited_data, mean, std


def train_pipline(
    path_x,
    path_y,
    path_test_id,
    var_threshold=0.01,
    corr_threshold=0.95,
    val_set_size=0,
    gpu=False,
    filename="saved_model.joblib",
):

    """
    Straightforwarding reading, preprocessing ,splitting data and training.
    return rp, rmse and the trained model.

     Parameters:
        path_x (csv): Features file in .csv. Pdbid of complexes has to located in column 0.
        path_y (csv): Target file in .csv. This file has to contain a column with
        'pdbid' which determines pdbid of complexes.
        path_test_id (csv): A csv file contain all pdbid of the core set (PDBbind 2016)
        var_threshold (float): Variance threshold. Features below this threshold are discarded.
        corr_threshold (float): Correlated features are discarded.
        val_set_size (float): If not zero, indicate the percentage of data in validation set.
        Validation set is created randomly with the stratified split method.
        gpu (bool): If GPU is available, XGBoost uses it as an accelerator for the training.
        filename (str): Filename of the trained model for save in .joblib.

     Returns:
        rp (float), rmse (float), xgb_reg (sklearn): Return rp and rmse metrics on the test set and
        the trained model.
    """

    dataset, mean, std = data_creator(
        path_x,
        path_y,
        path_test_id,
        var_threshold=0.01,
        corr_threshold=0.95,
        val_set_size=0,
    )

    with open("columns.txt", "w") as f:
        for item in list(dataset["x_train"].columns):
            f.write(item + "\n")
    st.info("mean.csv ,std.csv and columns.txt files are generated.")
    pd.DataFrame(mean).to_csv("mean.csv")
    pd.DataFrame(std).to_csv("std.csv")

    if gpu:
        xgb_reg = XGBRegressor(
            n_estimators=20000,
            max_depth=8,
            learning_rate=0.005,
            subsample=0.7,
            tree_method="gpu_hist",
            predictor="gpu_predictor",
        )

    else:
        xgb_reg = XGBRegressor(
            n_estimators=20000,
            max_depth=8,
            learning_rate=0.005,
            subsample=0.7,
            tree_method="hist",
            predictor="cpu_predictor",
        )

    print("Training is in progressing...\n")

    xgb_reg.fit(dataset["x_train"], dataset["y_train"])

    y_pred = xgb_reg.predict(dataset["x_test"])

    rp = pearsonr(dataset["y_test"], y_pred)[0]
    rmse = np.sqrt(mean_squared_error(dataset["y_test"], y_pred))

    print(f"Rp: {rp:.3f} RMSE: {rmse:.3f}\n")

    dump(xgb_reg, filename)

    return rp, rmse, xgb_reg, mean, std


if __name__ == "__main__":

    start = time.time()
    print("\n")
    print("Job is started.")
    print("------------------------------")

    parser = argparse.ArgumentParser(
        description="""Reading, preprocessing data and training 
            a XGBoost model"""
    )

    parser.add_argument("-x", "--path_x", help="path of data features", required=True)
    parser.add_argument("-y", "--path_y", help="path of data target", required=True)
    parser.add_argument(
        "-id", "--path_test_id", help="path of test set pdbid", required=True
    )
    parser.add_argument(
        "-v",
        "--var_threshold",
        type=float,
        default=0.01,
        help="Variance threshold is used for discarding static and quasi-static features",
    )
    parser.add_argument(
        "-c",
        "--corr_threshold",
        type=float,
        default=0.95,
        help="Correlation threshold is used for discarding the correlated features",
    )

    # parser.add_argument(
    #   "-s",
    #    "--val_set_size",
    #   type=float,
    #    default=0,
    #    help="Determine the percentage of validation set",
    # )

    parser.add_argument(
        "-g",
        "--gpu",
        type=bool,
        default=False,
        help="GPU as an accelerator of training",
    )
    parser.add_argument(
        "-f",
        "--filename",
        type=str,
        default="saved_model.joblib",
        help="Filename of the trained model for saving.",
    )

    args = parser.parse_args()

    print("Inputs")
    print(f"Path data: {args.path_x}")
    print(f"Path target: {args.path_y}")
    print(f"Path test id: {args.path_test_id}")
    print(f"Variance threshold: {args.var_threshold}")
    print(f"Correlation threshold: {args.corr_threshold}")
    print(f"Validation set size: {args.val_set_size}")
    print(f"GPU: {args.gpu}")
    print(f"Filename: {args.filename}")
    print(f"------------------------------")

    train_pipline(
        args.path_x,
        args.path_y,
        args.path_test_id,
        args.var_threshold,
        args.corr_threshold,
        0,
        args.gpu,
        args.filename,
    )

    seconds = end - start
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)

    print("------------------------------")
    print(f"Job is done at {h} hours, {m} minutes and {s:.2f} seconds!")
    print(f"{args.filename} is created.")
