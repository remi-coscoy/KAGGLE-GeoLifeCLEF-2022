import rasterio
from PIL import Image
import pandas as pd
import numpy as np
import os
from tqdm import tqdm

def get_environmental_vector(dataDir:str)->pd.DataFrame:
    return pd.read_csv(os.path.join(dataDir,"pre-extracted", "environmental_vectors.csv"), sep=";", index_col="observation_id")


def get_obs(dataDir:str, test=False)->pd.DataFrame:
    """ Returns the dataframe for the observations, concatenated for FR (0) and US (1) with a column showing country.
    test (bool) : use test data """ 

    if test:
        df_obs_fr_test = pd.read_csv(os.path.join(dataDir, "observations", "observations_fr_test.csv"), sep=";", index_col="observation_id")
        df_obs_fr_test["country"]=0
        df_obs_us_test = pd.read_csv(os.path.join(dataDir, "observations", "observations_us_test.csv"), sep=";", index_col="observation_id")
        df_obs_fr_test["country"]=1
        df_obs_test = pd.concat((df_obs_fr_test, df_obs_us_test))
        return df_obs_test
    else:
        df_obs_fr = pd.read_csv(os.path.join(dataDir, "observations", "observations_fr_train.csv"), sep=";", index_col="observation_id")
        df_obs_fr["country"]=0
        df_obs_us = pd.read_csv(os.path.join(dataDir, "observations", "observations_us_train.csv"), sep=";", index_col="observation_id")
        df_obs_fr["country"]=1
        df_obs = pd.concat((df_obs_fr, df_obs_us))
        return df_obs


def get_altitude_landcover_rgb_nir(obs_id,dataDir:str="/mounts/Datasets4/GeoLifeCLEF2022")->pd.DataFrame:
    """ Returns a dataframe for the other features : altitude, landcover etc ..."""


    mean_altitudes=[]
    min_max_alt=[]
    landcovers=[]
    r_50=[]
    g_50=[]
    b_50=[]
    r_25=[]
    g_25=[]
    b_25=[]
    r_75=[]
    g_75=[]
    b_75=[]

    nir_50=[]
    nir_25=[]
    nir_75=[]

    # norma_factor= 1/(256*256)

    for id in tqdm(obs_id):
        id=str(id)
        country = id[0]
        contry_folder = "patches-fr" if country == "1" else "patches-us"
        first_subfolder = id[6:8]
        second_subfolder = id[4:6]

        ## altitude
        altitude_path = os.path.join(dataDir, contry_folder, first_subfolder, second_subfolder, id + "_altitude.tif")
        altitude = (rasterio.open(altitude_path).read(1))
        mean_altitudes.append(altitude.mean())
        min_max_alt.append((altitude.max() - altitude.min()))

        #landcover
        landcover_path = os.path.join(dataDir, contry_folder, first_subfolder, second_subfolder, id + "_landcover.tif")
        landcover = (rasterio.open(landcover_path).read(1))
        landcovers.append(np.bincount(landcover.flatten(), minlength=34)[1:]) ## we ignore 0 values

        ## attention, aux valeurs manquantes ?
        ## aligner entre US et FR ?

        ## rgb nir percentiles
        rgb_path=os.path.join(dataDir, contry_folder, first_subfolder, second_subfolder, id + "_rgb.jpg")
        rgb_image=np.asarray(Image.open(rgb_path))
        R = rgb_image[:,:,0]
        G = rgb_image[:,:,1]
        B = rgb_image[:,:,2]

        r_25.append(np.percentile(R, 25))
        g_25.append(np.percentile(G, 25))
        b_25.append(np.percentile(B, 25))

        r_50.append(np.percentile(R, 50))
        g_50.append(np.percentile(G, 50))
        b_50.append(np.percentile(B, 50))
        
        r_75.append(np.percentile(R, 75))
        g_75.append(np.percentile(G, 75))
        b_75.append(np.percentile(B, 75))

        near_ir_path = os.path.join(dataDir, contry_folder, first_subfolder, second_subfolder, id + "_near_ir.jpg")
        near_ir_image=np.asarray(Image.open(near_ir_path))

        nir_25.append(np.percentile(near_ir_image, 25))
        nir_50.append(np.percentile(near_ir_image, 50))
        nir_75.append(np.percentile(near_ir_image, 75))

    return pd.DataFrame({"observation_id": obs_id, "mean_altitude": mean_altitudes, "min_max_alt":min_max_alt , "landcover": landcovers, 
                         "r_25":r_25, "g_25":g_25, "b_25":b_25, "nir_25":nir_25,
                         "r_50":r_50, "g_50":g_50, "b_50":b_50, "nir_50":nir_50,
                         "r_75":r_75, "g_75":g_75, "b_75":b_75, "nir_75":nir_75}).set_index("observation_id")


def predict_top_30_set(y_score):
    """ Returns the top 30 set (indexes of top30 classes) for a given output prediction score.
        Arguments
        -------
        y_score : 2d array-like, [n_samples, n_classes]"""
    
    n_classes = y_score.shape[1]
    #s_pred = np.argpartition(y_score, n_classes - 30, axis=1)[:, -30:]
    s_pred=np.argsort(y_score, axis=1)[:, ::-1][:, :30]
    return s_pred

def predict_func(X, rf_model):
    """ Computes the top30 set for an input batch.
        
        Arguments
        -------
        X : 2d array-like [n_samples, n_features] 
        rf_model : scikit-learn estimator """

    y_score = rf_model.predict_proba(X)
    s_pred = predict_top_30_set(y_score)
    return s_pred

def batch_predict(predict_func, X, batch_size=1024):
    """ Predicts top 30 set by batches (for memory efficiency). 
        
        Arguments
        -------
        X : 2d array-like [n_samples, n_features]  
    """

    res = predict_func(X[:1])
    n_samples, n_outputs, dtype = X.shape[0], res.shape[1], res.dtype
    
    preds = np.empty((n_samples, n_outputs), dtype=dtype)
    
    for i in range(0, len(X), batch_size):
        X_batch = X[i:i+batch_size]
        preds[i:i+batch_size] = predict_func(X_batch)
            
    return preds

def top_k_error_rate_from_sets(y_true, s_pred) -> float:
    r"""Computes the top-k error rate from predicted sets.

    Arguments
    -------
    y_true: 1d array-like, [n_samples]
        True labels.
    s_pred: 2d array-like, [n_samples, k]
        Previously computed top-k sets for each sample.

    Returns
    -------
    float:
        Error rate value.
    """

    pointwise_accuracy = np.sum(s_pred == y_true[:, None], axis=1)
    return 1 - np.mean(pointwise_accuracy)


def generate_submission_file(filename:str, observation_ids, s_pred:list):
    """ Writes a submission file. 
        
        Arguments
        -------
        observation_ids : 1d array-like 
        s_pred : 2d array-like [n_samples, k]
            Previously computed top-k sets for each sample.

    """
    s_pred = [" ".join(map(str, pred_set)) for pred_set in s_pred]

    df = pd.DataFrame(
        {
            "Id": observation_ids,
            "Predicted": s_pred,
        }
    )
    df.to_csv(filename, index=False)


# Function to flatten arrays within a row
def flatten_arrays_in_row(row):
    flattened_row = []
    for item in row:
        if isinstance(item, np.ndarray):
            flattened_row.extend(item.flatten())
        else:
            flattened_row.append(item)
    return np.array(flattened_row)



    

