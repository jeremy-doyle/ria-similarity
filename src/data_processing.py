# -*- coding: utf-8 -*-


import pandas as pd
from tqdm import tqdm
import os
import re


def create_dataframe_from_raw(path, files):
    df = pd.DataFrame()
    for f in tqdm(files):
        df = df.append(
            pd.read_csv(f"{path}/{f}", encoding="ISO-8859-1", parse_dates=[2]),
            ignore_index=True,
        )
    return df


def select_cols(df, kept_cols):
    return df.loc[:, kept_cols]


def process_data():
    # read in raw data
    app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sec_dir = f"{app_dir}/data/raw/SEC"
    adv_dir = sec_dir + "/" + os.listdir(sec_dir)[0]

    # create dataframe
    files = os.listdir(adv_dir)
    df = create_dataframe_from_raw(adv_dir, files)

    # select columns
    matches = (
        r"^1D|1F1-State|^1A|^1B$|^1B1$|"
        + r"5A-Number|5A$|5B\d-Number|5B\d$|5F2.*|5G\d+$|5H$|DateSubmitted"
    )
    cols = df.columns
    kept_cols = [col for col in cols if len(re.findall(matches, col)) > 0]
    df = select_cols(df, kept_cols)

    # save to disk
    df.to_hdf(f"{app_dir}/data/processed/selected_data.h5", "df")


if __name__ == "__main__":

    process_data()
