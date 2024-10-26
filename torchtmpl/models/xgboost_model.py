import numpy as np
import pandas as pd
import logging
import pickle
import time
from torchtmpl.data_rf import generate_submission_file, top_k_error_rate_from_sets, predict_top_30_set
import argparse
import os
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.datasets import dump_svmlight_file
from sklearn.datasets import load_svmlight_file

if __name__ =="__main__":

    parser = argparse.ArgumentParser(description="Your script description.")
    parser.add_argument("n_samples", default=None) # use 0 for all samples
    parser.add_argument("max_depth", default=10, type=int)

    args = parser.parse_args()

    num_classes = 17037

    booster_params = {
    "tree_method": "hist",
    "device": "cpu",
    "objective": "multi:softprob",
    "num_class": num_classes,
    # "subsample": 1,
    # "sampling_method": "gradient_based",
    "max_depth":args.max_depth,
    }

    args = parser.parse_args()
    batch_size_main = 100
    ### PATHS
    dataDir = "/mounts/Datasets4/GeoLifeCLEF2022"
     # Generate a unique timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    logDir = "/usr/users/cei2023_2024_sondra_cself/coscoy_rem/Documents/deepchallenge4-team4/xgboost_results"
    log2Dir = os.path.join(logDir, f"xg_{timestamp}")
    submitdir=os.path.join(log2Dir, "submission")
    
    if not os.path.exists(logDir):
        os.mkdir(logDir)
    if not os.path.exists(log2Dir):
        os.mkdir(log2Dir)
    if not os.path.exists(submitdir):
        os.mkdir(submitdir)
    
    # Configure logging
    logging.basicConfig(filename=os.path.join(log2Dir, "xgboost_results.txt"), level=logging.DEBUG, filemode="w")
    logger = logging.getLogger()

    # Log script arguments
    logger.info("Script arguments:")
    logger.info(args)

    #### OBSERVATIONS
    logger.info("Loading observation data...")
    df_obs_fr = pd.read_csv(os.path.join(dataDir, "observations", "observations_fr_train.csv"), sep=";", index_col="observation_id")
    df_obs_us = pd.read_csv(os.path.join(dataDir, "observations", "observations_us_train.csv"), sep=";", index_col="observation_id")
    df_obs = pd.concat((df_obs_fr, df_obs_us))

    df_obs_fr_test = pd.read_csv(os.path.join(dataDir, "observations", "observations_fr_test.csv"), sep=";", index_col="observation_id")
    df_obs_us_test = pd.read_csv(os.path.join(dataDir, "observations", "observations_us_test.csv"), sep=";", index_col="observation_id")
    df_obs_test = pd.concat((df_obs_fr_test, df_obs_us_test))

    obs_id_train = df_obs.index[df_obs["subset"] == "train"].values
    obs_id_val = df_obs.index[df_obs["subset"] == "val"].values
    obs_id_test = df_obs_test.index.values

    y_train = df_obs.loc[obs_id_train]["species_id"].values
    y_val = df_obs.loc[obs_id_val]["species_id"].values
    le = LabelEncoder()

    #### ENVIRONMENTAL VECTOR
    logger.info("Loading environmental vector data...")
    df_env = pd.read_csv(os.path.join(dataDir,"pre-extracted", "environmental_vectors.csv"), sep=";", index_col="observation_id")

    X_train = df_env.loc[obs_id_train].values
    X_val = df_env.loc[obs_id_val].values
    X_test = df_env.loc[obs_id_test].values

    ## Handling missing values
    logger.info("Handling missing values...")
    imp = SimpleImputer(
        missing_values=np.nan,
        strategy="constant",
        fill_value=np.finfo(np.float32).min,
    )
    imp.fit(X_train)

    X_train = imp.transform(X_train)
    X_val = imp.transform(X_val)
    X_test = imp.transform(X_test)

    ## Limited number of training samples 
    logger.info("Limiting number of training samples...")
    if int(args.n_samples):
        X_train = X_train[:int(args.n_samples)]
        y_train = y_train[:int(args.n_samples)]

    ## Encoding target labels
    logger.info("Encoding target labels...")
    y_train = le.fit_transform(y_train)
    # Convert data to SVMLight format and save to files
    dump_svmlight_file(X_train, y_train, 'train.svm')
    dump_svmlight_file(X_val, y_val, 'val.svm')
    dump_svmlight_file(X_test, [0]*len(X_test), 'test.svm')  # Labels for test set are not available

    logger.info("Data preparation completed.")

    class Iterator(xgb.DataIter):
        def __init__(self, svm_file_paths):
            self._file_paths = svm_file_paths
            self._it = 0
            super().__init__(cache_prefix=os.path.join(".", "cache"))

        def next(self, input_data):
            if self._it == len(self._file_paths):
                return 0

            X, y = load_svmlight_file(self._file_paths[self._it])
            input_data(data=X, label=y)
            self._it += 1
            return 1

        def reset(self):
            self._it = 0
    
    # Create Iterator instance and DMatrix
    it = Iterator(["train.svm", "val.svm", "test.svm"])
    dtest = xgb.DMatrix(X_test)
    dval = xgb.DMatrix(X_val)
    Xy = xgb.DMatrix(it)

    logger.info("Training XGBoost model...")
    booster = xgb.train(booster_params, Xy)
    y_score_test = booster.predict(dtest)
    y_score_val = booster.predict(dval)

    s_pred = predict_top_30_set(y_score_test)
    s_val = predict_top_30_set(y_score_val)

    print("Shape of predictions on test set:", s_pred.shape)
    print("First prediction on test set:", s_pred[0])
    print("Shape of predictions on validation set:", s_val.shape)
    print("First prediction on validation set:", s_val[0])

    score_val = top_k_error_rate_from_sets(y_val, s_val)
    logging.info("Top-30 error rate on validation set: {:.1%}".format(score_val))
    print("Top-30 error rate on validation set: {:.1%}".format(score_val))

    # Generate the submission file  
    logging.info("Saving the submission ...")
    generate_submission_file(os.path.join(submitdir ,"submission.csv"), obs_id_test, s_pred)

    logging.info("Submission saved.")
