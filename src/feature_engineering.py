# -*- coding: utf-8 -*-


import pandas as pd
import re
from datetime import datetime
from dateutil.relativedelta import relativedelta
from tqdm import tqdm
import numpy as np
import os


def engineer_features():
    # read in processed data
    app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    df = pd.read_hdf(f"{app_dir}/data/processed/selected_data.h5")

    # if DBA name is null (1B or 1B1), use legal name (1A)
    df["name"] = df["1B"]
    df.loc[df["name"].isnull(), "name"] = df.loc[df["name"].isnull(), "1B1"]
    df.loc[df["name"].isnull(), "name"] = df.loc[df["name"].isnull(), "1A"]
    df = df.drop(["1A", "1B", "1B1"], axis=1)

    # converting Y/N questions to 1/0
    for col in df.columns:
        if len(re.findall(r"5G", col)) > 0:
            df[col] = df[col].map({"Y": 1, "N": 0})
    # cleaning up employee numbers due to change in data collection
    for col in ["5A", "5B1", "5B2", "5B3"]:
        df.loc[df[col].isnull(), col] = df.loc[df[col].isnull(), f"{col}-Number"]
        df = df.drop(f"{col}-Number", axis=1)
    # setting null asset/client values to 0
    values = dict(zip(["5F2a", "5F2b", "5F2c", "5F2d", "5F2e", "5F2f"], [0] * 6))
    df = df.fillna(value=values)

    # setting null financial planning client values(5H) to 0 and discretizing rest
    disc = {
        "0": 0,
        "1-10": 1,
        "11-25": 11,
        "26-50": 26,
        "51-100": 51,
        "101-250": 101,
        "251-500": 251,
        "More than 500": 501,
    }
    df["5H"] = df["5H"].map(disc)
    df["5H"] = df["5H"].fillna(value=0)

    # remove records with missing ID
    df = df.loc[df["1D"] != "801-", :]

    # setting region based on state (based on Bureau of Economic Analysis regions)
    region = {
        "AL": "Southeast",
        "AK": "FarWest",
        "AR": "Southeast",
        "AZ": "Southwest",
        "CA": "FarWest",
        "CO": "RockyMountain",
        "CT": "NewEngland",
        "DC": "Mideast",
        "DE": "Mideast",
        "FL": "Southeast",
        "GA": "Southeast",
        "GU": "Foreign",
        "HI": "FarWest",
        "IA": "Plains",
        "ID": "RockyMountain",
        "IL": "GreatLakes",
        "IN": "GreatLakes",
        "KS": "Plains",
        "KY": "Southeast",
        "LA": "Southeast",
        "MA": "NewEngland",
        "MD": "Mideast",
        "ME": "NewEngland",
        "MI": "GreatLakes",
        "MN": "Plains",
        "MO": "Plains",
        "MS": "Southeast",
        "MT": "RockyMountain",
        "NC": "Southeast",
        "ND": "Plains",
        "NE": "Plains",
        "NH": "NewEngland",
        "NJ": "Mideast",
        "NM": "Southwest",
        "NV": "FarWest",
        "NY": "Mideast",
        "OH": "GreatLakes",
        "OK": "Southwest",
        "OR": "FarWest",
        "PA": "Mideast",
        "PR": "Foreign",
        "RI": "NewEngland",
        "SC": "Southeast",
        "SD": "Plains",
        "TN": "Southeast",
        "TX": "Southwest",
        "UT": "RockyMountain",
        "VA": "Southeast",
        "VI": "Foreign",
        "VT": "NewEngland",
        "WA": "FarWest",
        "WI": "GreatLakes",
        "WV": "Southeast",
        "WY": "RockyMountain",
    }
    df["region"] = df["1F1-State"].map(region)
    df["region"] = df["region"].fillna(value="Foreign")

    # drop records with no employees or advisors
    df = df.loc[(df["5A"] > 0) & (df["5B3"] > 0), :]

    # drop records with no assets or clients
    df = df.loc[(df["5F2c"] > 0) & (df["5F2f"] > 0), :]

    # create advisory-to-brokerage-employee ratio
    df["advisory_to_brokerage"] = df["5B3"] / df["5B2"]

    # create assets-per-advisor ratio
    df["assets_per_advisor"] = df["5F2c"] / df["5B3"]

    # create clients-per-advisor ratio
    df["clients_per_advisor"] = df["5F2f"] / df["5B3"]

    # create assets-per-client ratio
    df["assets_per_client"] = df["5F2c"] / df["5F2f"]

    # create discretionary ratio
    df["disc_to_total_assets"] = df["5F2a"] / df["5F2c"]

    # get latest submission; only keep advisors with recent filings
    df["DateSubmitted"] = df["DateSubmitted"].dt.floor("d")
    latest = df.loc[:, ["1D", "DateSubmitted"]].groupby("1D").agg("max").reset_index()
    max_date = latest["DateSubmitted"].max()
    max_date = datetime(max_date.year, max_date.month, max_date.day)

    def recent_filing(x):
        return x >= max_date - relativedelta(years=1)

    latest = latest.loc[latest["DateSubmitted"].apply(recent_filing), :]
    latest = latest["1D"].drop_duplicates().tolist()
    df = df.loc[df["1D"].isin(latest), :]

    # narrowing scope to include only retail investment advisors (no institutional)
    # can change this in the future to broaden or change scope
    df = df.loc[(df["5G2"] == 1) & (df["5G3"] == 0) & (df["5G4"] == 0), :]
    for col in df.columns:
        if col.find("5G") > -1:
            df = df.drop(col, axis=1)
    # resample to get values at at yearly checkpoints in the past
    min_date = df["DateSubmitted"].min()
    min_date = datetime(min_date.year, min_date.month, min_date.day)
    span = int((max_date - min_date).days / 365)
    checkpoints = list()
    for i in range(span + 1):
        checkpoints.append(max_date - relativedelta(years=+i))

    def do_resample(advisor):
        adv = df.loc[df["1D"] == advisor, :]
        adv = adv.drop_duplicates(subset="DateSubmitted", keep="last")
        adv = adv.set_index("DateSubmitted")
        adv = adv.append(pd.Series(name=max_date + relativedelta(days=+1)))
        adv = adv.resample("D").ffill()
        adv = adv.loc[adv.index.isin(checkpoints), :]

        # while we're at it, add more features using the time series
        adv["advisor_growth_1y"] = adv["5B3"].pct_change()
        adv["advisor_growth_3y"] = adv["5B3"].pct_change(3)
        adv["asset_growth_1y"] = adv["5F2c"].pct_change()
        adv["asset_growth_3y"] = adv["5F2c"].pct_change(3)
        adv["assets_per_advisor_growth_1y"] = adv["assets_per_advisor"].pct_change()
        adv["assets_per_advisor_growth_3y"] = adv["assets_per_advisor"].pct_change(3)
        adv["clients_per_advisor_growth_1y"] = adv["clients_per_advisor"].pct_change()
        adv["clients_per_advisor_growth_3y"] = adv["clients_per_advisor"].pct_change(3)
        adv["assets_per_client_growth_1y"] = adv["assets_per_client"].pct_change()
        adv["assets_per_client_growth_3y"] = adv["assets_per_client"].pct_change(3)
        adv["disc_to_total_assets_growth_1y"] = adv["disc_to_total_assets"].pct_change()
        adv["disc_to_total_assets_growth_3y"] =adv["disc_to_total_assets"].pct_change(3)

        if len(adv) == 1:
            adv = adv.fillna(0)
        adv = adv.reset_index()
        return adv

    ts = pd.DataFrame()
    advisors = df["1D"].drop_duplicates().tolist()
    for advisor in tqdm(advisors, desc="resampling"):
        ts = ts.append(do_resample(advisor), ignore_index=True)
    # keep the records as of max_date
    engineered = ts.loc[ts["DateSubmitted"] == max_date, :]

    # save pre-adjusted calculations for reporting
    engineered.to_hdf(f"{app_dir}/data/reporting/reporting_data.h5", "df")

    # drop columns not being used as features as they are either only for
    # contextual information or derivatives of them have been calculated and used
    # keep region since regional models will be built as well as the advisor ID
    not_features = [
        "DateSubmitted",
        "1F1-State",
        "name",
        "5A",
        "5B1",
        "5B2",
        "5B5",
        "5F2a",
        "5F2b",
        "5F2d",
        "5F2e",
    ]
    engineered = engineered.drop(not_features, axis=1)

    # fill NAs with 0 and infs with 1
    engineered = engineered.fillna(0)
    engineered = engineered.replace([np.inf, -np.inf], 1)

    # clipping features so extremely large or small values don't have an
    # oversized effect on the similarity algorithm
    for col in engineered.columns:
        if col not in ["region", "1D"]:
            limits = np.quantile(engineered[col].values, [0.01, 0.99])
            engineered[col] = np.clip(engineered[col].values, limits[0], limits[1])
    # save to file
    engineered = engineered.reset_index(drop=True)
    engineered.to_hdf(f"{app_dir}/data/engineered/unscaled_features.h5", "df")


if __name__ == "__main__":

    engineer_features()
