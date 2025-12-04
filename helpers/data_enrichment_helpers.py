import pandas as pd


def enrich_with_volume_and_density(df):
    df["Volume_m3"] = (df["Length"] / 1000) * (df["Width"] / 1000) * (df["Height"] / 1000)
    df["Density_kg_m3"] = df["Net Weight"] / df["Volume_m3"]
    return df


def rename_features(df, old_column_name, new_column_name):
    df.rename(columns={old_column_name: new_column_name}, inplace=True)
    return df
