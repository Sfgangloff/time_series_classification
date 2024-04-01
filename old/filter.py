import pandas as pd

class Filter():
    def __init__(self):
        pass

    def select(self,df_obs:pd.DataFrame):
        obs_id = df_obs["obs_id"].values[0]
        boolean = (len(df_obs["bid"].unique())==1) | (len(df_obs["ask"].unique())==1)
        df = pd.DataFrame({"obs_id":[obs_id],
                        "value":[boolean]})
        return df

    def filter(self,df:pd.DataFrame):
        grouped = df.groupby("obs_id").apply(lambda df_obs: self.select(df_obs))
        grouped.set_index("obs_id",inplace=True)
        return grouped