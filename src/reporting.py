# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from joblib import load
from tqdm import tqdm
import os
import sqlite3 as db
import warnings

warnings.filterwarnings("ignore")


app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_neighbors(advisor_ID, region, k):
    X = load(f"{app_dir}/data/engineered/X_{region}.pkl")
    ids = load(f"{app_dir}/data/engineered/ID_{region}.pkl")
    id_list = ids.tolist()
    scaler = load(f"{app_dir}/models/scaler_{region}.pkl")
    reducer = load(f"{app_dir}/models/reducer_{region}.pkl")
    model = load(f"{app_dir}/models/model_{region}.pkl")

    i = id_list.index(advisor_ID)
    X_advisor = X[i].reshape(1, -1)
    X_advisor = scaler.transform(X_advisor)
    X_advisor = reducer.transform(X_advisor)
    neighbors = model.kneighbors(X_advisor, k, return_distance=False)
    neighbors = ids[neighbors]

    neighbors_df = pd.DataFrame()
    neighbors_df["ID"] = [advisor_ID] * k
    neighbors_df["neighbor"] = np.arange(k)
    neighbors_df["neighbor_ID"] = neighbors[0]

    return neighbors_df


def get_id_name_state(df, region):
    if region == "full":
        output = df.loc[:, ["name", "1D", "1F1-State"]]
    else:
        output = df.loc[df["region"] == region, ["name", "1D", "1F1-State"]]
    output = output.replace(np.nan, "Foreign")
    output["name"] = output.apply(lambda x: f"{x['name']} ({x['1F1-State']})", axis=1)
    output = output.drop("1F1-State", axis=1)
    output.columns = ["Name", "ID"]
    output = output.drop_duplicates(subset="Name").sort_values("Name")
    return output


def format_df(df):
    order = [
        "1D",
        "name",
        "1F1-State",
        "5F2a",
        "5F2b",
        "5F2c",
        "5F2d",
        "5F2e",
        "5F2f",
        "5H",
        "5B1",
        "5B3",
        "5B2",
        "5B4",
        "5B5",
        "5B6",
        "assets_per_advisor",
        "clients_per_advisor",
        "assets_per_client",
        "disc_to_total_assets",
        "advisor_growth_1y",
        "advisor_growth_3y",
        "asset_growth_1y",
        "asset_growth_3y",
        "assets_per_advisor_growth_1y",
        "assets_per_advisor_growth_3y",
        "clients_per_advisor_growth_1y",
        "clients_per_advisor_growth_3y",
        "assets_per_client_growth_1y",
        "assets_per_client_growth_3y",
        "disc_to_total_assets_growth_1y",
        "disc_to_total_assets_growth_3y",
    ]
    df = df[order]

    df.iloc[:, 2] = df.iloc[:, 2].replace(np.nan, "Foreign").astype(str)

    plan = {
        0: "0",
        1: "1-10",
        11: "11-25",
        26: "26-50",
        51: "51-100",
        101: "101-250",
        251: "251-500",
        501: "More than 500",
    }
    df["5H"] = df["5H"].map(plan)

    df = df.replace([np.inf, -np.inf], np.nan)

    names = [
        "ID",
        "Name",
        "State",
        "Discretionary Assets",
        "Non-Discretionary Assets",
        "Total Assets",
        "Discretionary Accounts",
        "Non-Discretionary Accounts",
        "Total Accounts",
        "Financial Planning Clients",
        "Advisory Employees",
        "Investment Advisor Reps",
        "Broker-Dealer Reps",
        "Dually-Registered IARs",
        "Insurance Agents",
        "Outside Solicitors",
        "Assets Per Advisor",
        "Clients Per Advisor",
        "Assets Per Client",
        "Discretionary to Total Assets",
        "Advisor Growth (1y)",
        "Advisor Growth (3y)",
        "Assets Growth (1y)",
        "Assets Growth (3y)",
        "Assets Per Advisor Growth (1y)",
        "Assets Per Advisor Growth (3y)",
        "Clients Per Advisor Growth (1y)",
        "Clients Per Advisor Growth (3y)",
        "Assets Per Client Growth (1y)",
        "Assets Per Client Growth (3y)",
        "Discretionary to Total Assets Growth (1y)",
        "Discretionary to Total Assets Growth (3y)",
    ]
    df.columns = names

    return df


def create_dashboard_data():
    con = db.connect(f"{app_dir}/dashboard/data/advisor_similarity.db")

    df = pd.read_hdf(f"{app_dir}/data/reporting/reporting_data.h5")

    format_df(df).to_sql(
        "reporting_data_formatted", con, if_exists="replace", index=False
    )

    regions = df["region"].drop_duplicates().tolist()
    regions.append("full")

    for region in regions:
        region_neighbors = pd.DataFrame()
        ids = load(f"{app_dir}/data/engineered/ID_{region}.pkl")
        for id_ in tqdm(ids, desc=f"{region}: "):
            adv_neighbors = get_neighbors(id_, region, 21)
            region_neighbors = region_neighbors.append(adv_neighbors, ignore_index=True)
        region_neighbors.to_sql(
            f"neighbors_{region}", con, if_exists="replace", index=False
        )
        get_id_name_state(df, region).to_sql(
            f"ID_lookup_{region}", con, if_exists="replace", index=False
        )
    con.close()


if __name__ == "__main__":

    create_dashboard_data()
