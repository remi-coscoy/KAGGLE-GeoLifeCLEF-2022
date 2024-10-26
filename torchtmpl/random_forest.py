# Import necessary libraries
import numpy as np
import pandas as pd
import logging
import pickle
# import datetime
import time
# import torchtmpl.utils
import argparse
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
import multiprocessing

from . import data_rf
from . import utils

def train_test_rf(args):
    """ Trains a random forest model on tabular data; and saves a submission with the model trained."""

    ### PATHS
    #dataDir = "/mounts/Datasets4/GeoLifeCLEF2022"
    dataDir = "/mounts/Datasets3/2023-SONDRA/mounts/Datasets4/GeoLifeCLEF2022/"
    # dataDir = os.path.join(os.getenv("TMPDIR"), "mounts", "Datasets4", "GeoLifeCLEF2022")
    base_logDir = "/usr/users/sdi-labworks-2023-2024/sdi-labworks-2023-2024_17/DeepLearning/deepchallenge4-team4/random_forest"
    other_features_path= "/usr/users/sdi-labworks-2023-2024/sdi-labworks-2023-2024_17/DeepLearning/deepchallenge4-team4/feature_creation"

    logDir=utils.generate_unique_logpath(base_logDir,f"Rf__{args.n_samples}__{args.n_trees}__{args.max_depth}")
    if not os.path.isdir(logDir):
        os.makedirs(logDir)

    submitdir=os.path.join(logDir, "submission")
    if not os.path.exists(submitdir):
        os.mkdir(submitdir)

    ## LOGGER
    logger = logging.basicConfig(filename=os.path.join(logDir, "rf_results.txt"), level=logging.INFO, filemode="w")

    #### LOADING OBSERVATIONS
    df_obs= data_rf.get_obs(dataDir=dataDir, test=False)
    df_obs_test= data_rf.get_obs(dataDir=dataDir, test=True)

    obs_id_train = df_obs.index[df_obs["subset"] == "train"].values
    obs_id_val = df_obs.index[df_obs["subset"] == "val"].values
    obs_id_test = df_obs_test.index.values

    y_train = df_obs.loc[obs_id_train]["species_id"].values
    y_val = df_obs.loc[obs_id_val]["species_id"].values

    if int(args.n_samples):
        obs_id_train=obs_id_train[:int(args.n_samples)]
        y_train = y_train[:int(args.n_samples)]
        ## only for testing
        obs_id_val=obs_id_val[:int(args.n_samples)]
        y_val = y_val[:int(args.n_samples)]

        obs_id_test=obs_id_test[:int(args.n_samples)]
        
    #### ENVIRONMENTAL VECTOR
    logging.info("Building features vector ...")
    df_env = data_rf.get_environmental_vector(dataDir)

    ## dataframes
    other_features = pd.read_parquet(os.path.join(other_features_path, "all_other_features.parquet"))

    X_train = pd.concat([df_env.loc[obs_id_train], other_features.loc[obs_id_train]], axis=1)
    X_train=pd.concat([X_train, df_obs.loc[obs_id_train][["latitude", "longitude", "country"]]], axis=1) ## adding lat long
    logging.info(f"Features used : {X_train.columns}")

    X_val = pd.concat([df_env.loc[obs_id_val], other_features.loc[obs_id_val]], axis=1)
    X_val=pd.concat([X_val, df_obs.loc[obs_id_val][["latitude", "longitude", "country"]]], axis=1) ## adding lat long
    
    X_test = pd.concat([df_env.loc[obs_id_test], other_features.loc[obs_id_test]], axis=1)
    X_test=pd.concat([X_test, df_obs_test.loc[obs_id_test][["latitude", "longitude", "country"]]], axis=1) ## adding lat long
    
    ## to arrays
    logging.info(f"Preparing features ...")
    X_train = np.apply_along_axis(data_rf.flatten_arrays_in_row, 1, X_train.values)
    X_val = np.apply_along_axis(data_rf.flatten_arrays_in_row, 1, X_val.values)
    X_test = np.apply_along_axis(data_rf.flatten_arrays_in_row, 1, X_test.values)

    ## Handling missing values
    imp = SimpleImputer(
        missing_values=np.nan,
        strategy="constant",
        fill_value=np.finfo(np.float32).min,
    )
    imp.fit(X_train)

    X_train = imp.transform(X_train)
    X_val = imp.transform(X_val)
    X_test = imp.transform(X_test)

    logging.info("Number of observations for training: {}".format(len(X_train)))

    # Initialize the Random Forest model
    logging.info(f"Training estimator with n_estimators = {args.n_trees} and max_depth={args.max_depth} ...")
    start_time = time.time()
    rf_model = RandomForestClassifier(n_estimators=int(args.n_trees), max_depth=args.max_depth, criterion="entropy", verbose=2, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    end_time=time.time()
    time_difference = (end_time - start_time)/60
    logging.info(f"Estimator fitted; took {time_difference} minutes.")

    #Saving Random Forest model
    with open(os.path.join(logDir, "model.pickle"), "wb") as f:
        pickle.dump(rf_model, f)
    logging.info("Estimator saved.")

    s_val = data_rf.batch_predict(lambda x: data_rf.predict_func(x,rf_model), X_val, batch_size=1024)
    score_val = data_rf.top_k_error_rate_from_sets(y_val, s_val)
    logging.info("Top-30 error rate on validation set: {:.1%}".format(score_val))

    # Compute on the test set
    s_pred = data_rf.batch_predict(lambda x: data_rf.predict_func(x,rf_model), X_test, batch_size=1024)
    
    # Generate the submission file  
    logging.info("Saving the submission ...")
    data_rf.generate_submission_file(os.path.join(submitdir ,f"submission_rf_est{args.n_trees}_depth{args.max_depth}.csv"), obs_id_test, s_pred)
    logging.info("Submission saved.")


if __name__ =="__main__":

    ## Argument parsing
    parser = argparse.ArgumentParser(description="Random forest")
    parser.add_argument("n_samples", default=None) # use 0 for all samples
    parser.add_argument("n_trees", default=16, type=int)
    parser.add_argument("max_depth", default=10, type=int)
    args = parser.parse_args()
    
    train_test_rf(args)