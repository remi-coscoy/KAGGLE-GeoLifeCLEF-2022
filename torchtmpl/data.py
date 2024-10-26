# coding: utf-8

# Standard imports
import logging
import random

# External imports
import torch
import torch.nn as nn
import torch.utils.data
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision.transforms import Grayscale

import rasterio
import pandas as pd
import matplotlib.pyplot as plt
import os
from torchvision.io import read_image

import numpy as np
from sklearn.impute import SimpleImputer

import warnings
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

def show_image(X):
    num_c = X.shape[0]
    plt.figure()
    plt.imshow(X[0] if num_c == 1 else X.permute(1, 2, 0))
    plt.show()

def get_dataset(dataset_type,data_config,transform):
    dataDir = os.path.join(os.getenv("TMPDIR"), "mounts", "Datasets4", "GeoLifeCLEF2022")
    logging.info(f"***** Using path : {dataDir}")
    if "tab_data" in data_config:
        isTab = bool(data_config["tab_data"])
    else:
        isTab = True
    if dataset_type == "test":
        df_obs_fr = pd.read_csv(os.path.join(dataDir, "observations", "observations_fr_test.csv"), sep=";", index_col="observation_id")
        df_obs_us = pd.read_csv(os.path.join(dataDir, "observations", "observations_us_test.csv"), sep=";", index_col="observation_id")
        df_obs = pd.concat((df_obs_fr, df_obs_us))
        print("testing")
        return AllDatasets(dataDir,df_obs,isTab,transform,isTest=True)
    else:
        df_obs_fr = pd.read_csv(os.path.join(dataDir, "observations", "observations_fr_train.csv"), sep=";", index_col="observation_id")
        df_obs_us = pd.read_csv(os.path.join(dataDir, "observations", "observations_us_train.csv"), sep=";", index_col="observation_id")
        df_obs = pd.concat((df_obs_fr, df_obs_us))
        base_dataset = AllDatasets(dataDir,df_obs,isTab,transform,isTest=False)
        if dataset_type == "stats":
            num_classes = len(df_obs["species_id"].unique())
            input_img_size = tuple(base_dataset[0][0].shape)
            input_tab_size = tuple(base_dataset[0][1].shape)
            return num_classes,input_img_size,input_tab_size
        elif dataset_type == "train":
            train_ratio = data_config["train_ratio"]
            train_df = df_obs[df_obs["subset"] == "train"].sample(frac=train_ratio, random_state=42)
            return AllDatasets(dataDir,train_df,isTab,transform,isTest=False)
        elif dataset_type == "valid":
            val_df = df_obs[df_obs["subset"] == "val"]
            return AllDatasets(dataDir,val_df,isTab,transform, isTest=False)
        


class AllDatasets(Dataset):
    def __init__(self, dataDir,df_obs,isTab = True, transform=None, isTest=False):
        self.dataDir = dataDir
        self.isTab = isTab
        self.transform = transform
        self.normalize_ = torchvision.transforms.Compose([torchvision.transforms.Normalize(0, 1)])
        self.df_obs = df_obs
        self.isTest = isTest
        df_suggested_landcover_alignment = pd.read_csv(os.path.join(dataDir, "metadata", "landcover_suggested_alignment.csv"), sep=";")
        self.landcover_mapping = df_suggested_landcover_alignment["suggested_landcover_code"].values
        self.df_env = pd.read_csv(os.path.join(dataDir, "pre-extracted", "environmental_vectors.csv"), sep=";", index_col="observation_id")
        self.imp = SimpleImputer(
            missing_values=np.nan,
            strategy="constant",
            fill_value=0.0,
        )
        
        self.env_values = np.array(self.df_env.values)
        self.imp.fit(self.env_values)
        self.profile = {
            "driver": "GTiff",
            "dtype": "float32",
            "count": 1,
            "width": 256,
            "height": 256,
        }  

    def __len__(self):
        return len(self.df_obs)
    
    def get_ids(self):
        res = []
        for idx in range(len(self.df_obs)):
            res.append(str(self.df_obs.iloc[idx].name))
        return res
    
    def load_image(self, path):
        image = read_image(path)
        return image

    def load_altitude_landcover(self, altitude_path, landcover_path):
        altitude = self.normalize_(torch.tensor(rasterio.open(altitude_path, "r",**self.profile).read(1), dtype=torch.float32).unsqueeze(0))
        landcover = torch.tensor(rasterio.open(landcover_path, "r",**self.profile).read(1), dtype=torch.float32).unsqueeze(0)/34
        return altitude, landcover
        
    def prepare_images(self, near_ir_path, rgb_path, altitude_path, landcover_path):
        rgb_tensor = self.load_image(rgb_path)/ 255
        if self.transform:
            return self.transform(rgb_tensor.float())             
        altitude_tensor, landcover_tensor = self.load_altitude_landcover(altitude_path, landcover_path)
        near_ir_tensor = Grayscale()(self.load_image(near_ir_path)/255)
        return torch.cat((near_ir_tensor, rgb_tensor, altitude_tensor, landcover_tensor), dim=0)


    def get_env_vector(self, patch_nb):
        env_vector = self.df_env.loc[int(patch_nb)].values
        env_vector = self.imp.transform([env_vector]).flatten()
        env_vector = torch.tensor(env_vector, dtype=torch.float32)
        return env_vector
    
    def __getitem__(self, idx):
        patch_nb = str(self.df_obs.iloc[idx].name)
        country = patch_nb[0]
        contry_folder = "patches-fr" if country == "1" else "patches-us"
        first_subfolder = patch_nb[6:8]
        second_subfolder = patch_nb[4:6]
        altitude_path = os.path.join(self.dataDir, contry_folder, first_subfolder, second_subfolder, patch_nb + "_altitude.tif")
        landcover_path = os.path.join(self.dataDir, contry_folder, first_subfolder, second_subfolder, patch_nb + "_landcover.tif")
        near_ir_path = os.path.join(self.dataDir, contry_folder, first_subfolder, second_subfolder, patch_nb + "_near_ir.jpg")
        rgb_path = os.path.join(self.dataDir, contry_folder, first_subfolder, second_subfolder, patch_nb + "_rgb.jpg")
        image = self.prepare_images(near_ir_path, rgb_path, altitude_path, landcover_path)
        if not self.isTest:
            # species_id for a given observation_id
            label = self.df_obs.loc[int(patch_nb), "species_id"] 
            if self.isTab:
                env_vector = self.get_env_vector(patch_nb)    
                return image, env_vector, label
            return image,label
        else:
            if self.isTab:
                env_vector = self.get_env_vector(patch_nb)    
                return image, env_vector
            return image
        
def get_dataloaders(data_config, use_cuda):
    batch_size = data_config["batch_size"]
    num_workers = data_config["num_workers"]
    if "resize" in data_config:
        isResize = bool(data_config["resize"])
    else:
        isResize = False
    transform_data = None
    if isResize:
        transform_data = transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    

    logging.info("  - Dataset creation")
    train_dataset = get_dataset("train",data_config,transform=transform_data)
    valid_dataset = get_dataset("valid",data_config,transform=transform_data)
    test_dataset = get_dataset("test",data_config,transform=transform_data)
                 
    logging.info(f"  - I loaded {len(train_dataset)} train samples, {len(valid_dataset)} validation samples and {len(test_dataset)} test samples")

    # Build the dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_cuda,
        prefetch_factor=2
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_cuda,
        prefetch_factor=2,

    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_cuda,
        prefetch_factor=2,
    )

    num_classes,input_img_size, input_tab_size = get_dataset("stats",data_config,
        transform=transform_data)

    return train_loader, valid_loader,test_loader, input_img_size, input_tab_size, num_classes, test_dataset.get_ids()

def get_test_dataloader(data_config, use_cuda):
    batch_size = data_config["batch_size"]
    num_workers = data_config["num_workers"]
    if "resize" in data_config:
        isResize = bool(data_config["resize"])
    else:
        isResize = False

    transform_data = None
    if isResize:
        transform_data = transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        # transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    logging.info("  - Dataset creation")

    
    base_dataset = get_dataset("test",data_config,
        transform=transform_data
                               )

    logging.info(f"  - I loaded {len(base_dataset)} samples")

    # Build the dataloaders
    test_loader = torch.utils.data.DataLoader(
        base_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_cuda,
    )
    num_classes,input_img_size, input_tab_size =get_dataset("stats",data_config,
        transform=transform_data)

    return test_loader, input_img_size, input_tab_size, num_classes, base_dataset.get_ids()