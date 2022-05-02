# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from joblib import dump
from tqdm import tqdm
import os


app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def pipeline(df, region):

    # narrow by region
    if region == "full":
        region_df = df.copy()
    else:
        region_df = df.loc[df["region"] == region, :]
    # create feature set (save to disk)
    not_features = ["1D", "region"]
    X = region_df.drop(not_features, axis=1).values
    dump(X, f"{app_dir}/data/engineered/X_{region}.pkl")

    # save IDs for later
    ids = region_df["1D"].values
    dump(ids, f"{app_dir}/data/engineered/ID_{region}.pkl")

    # scale features (save scaler to disk)
    scaler = StandardScaler()
    scaler.fit(X)
    X_scaled = scaler.transform(X)
    dump(scaler, f"{app_dir}/models/scaler_{region}.pkl")

    # reduce dimensionality to where number of components explains at least
    # 95% of the variance (save reducer to disk)
    reducer = PCA(0.95)
    reducer.fit(X_scaled)
    X_reduced = reducer.transform(X_scaled)
    dump(reducer, f"{app_dir}/models/reducer_{region}.pkl")

    # create nearest neighbors model (save model to disk)
    model = NearestNeighbors()
    model.fit(X_reduced)
    dump(model, f"{app_dir}/models/model_{region}.pkl")


def build_regional_models():
    # read in unscaled features
    df = pd.read_hdf(f"{app_dir}/data/engineered/unscaled_features.h5")

    # get regions
    regions = df["region"].drop_duplicates().tolist()
    regions.append("full")

    # pull each region through the pipeline to build regional models
    for region in tqdm(regions):
        pipeline(df, region)


if __name__ == "__main__":

    build_regional_models()
