import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

class Simplifier():
    def __init__(self,column:str):
        self.model = LinearRegression()
        self.column = column

    def transform(self,df_obs:pd.DataFrame):
        df_obs["venue"] = df_obs["venue"].sort_values().values
        #Y = df_obs[self.column].values.reshape(-1, 1) 
        #X = np.asarray(df_obs.index).reshape(-1, 1)
        #self.model.fit(X, Y)
        #df_obs[self.column] = self.model.predict(X)
        return df_obs

    def apply(self,df:pd.DataFrame):
        grouped = df.groupby("obs_id").apply(lambda df_obs: self.transform(df_obs))
        grouped.set_index("obs_id",inplace=True)
        return grouped