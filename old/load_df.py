import pandas as pd

def load_df(source:str):
    df = pd.read_csv(source)
    return df

def restrict_df(df:pd.DataFrame,
                start:int,
                end:int):
    return df[(start <= df["obs_id"]) & (df["obs_id"]<end)]