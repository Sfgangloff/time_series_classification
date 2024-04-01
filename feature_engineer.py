import pandas as pd
from typing import List

#todo: Fourier transform + higher order derivatives + autocorellation (correlation with lags)

class FeatureEngineer():
    def __init__(self,scalar_columns:List[str],
                      category_columns:List[str],
                      index_column:str):
        self.scalar_columns = scalar_columns
        self.category_columns = category_columns
        self.index_column = index_column

    def apply(self,df:pd.DataFrame):
        for column in self.scalar_columns:
            df['{}_lag1'.format(column)] = df.groupby(self.index_column)[column].shift(1)
            df['{}_lag1'.format(column)].fillna(df[column].median(), inplace=True)
            df['{}_diff'.format(column)] = df[column] - df['{}_lag1'.format(column)] 
            df['{}_roll_mean3'.format(column)]=df['{}'.format(column)].rolling(window=3).mean()
            df['{}_roll_mean6'.format(column)]=df['{}'.format(column)].rolling(window=6).mean()
            df['{}_roll_mean9'.format(column)]=df['{}'.format(column)].rolling(window=9).mean()
            df['{}_roll_mean3'.format(column)].fillna(df['{}_roll_mean3'.format(column)].median(), inplace=True)
            df['{}_roll_mean6'.format(column)].fillna(df['{}_roll_mean6'.format(column)].median(), inplace=True)
            df['{}_roll_mean9'.format(column)].fillna(df['{}_roll_mean9'.format(column)].median(), inplace=True)
            s_diff='{}_diff'.format(column)
            df['{}_mean'.format(column)] = df.groupby(self.index_column)[column].transform('mean')
            df['{}_diff_mean'.format(column)] = df.groupby(self.index_column)[s_diff].transform('mean')
            df['{}_med'.format(column)] = df.groupby(self.index_column)[column].transform('median')
            df['{}_std'.format(column)] = df.groupby(self.index_column)[column].transform('std')
            df['{}_skew'.format(column)] = df.groupby(self.index_column)[column].transform('skew')
            # df['{}_kurt'.format(column)] = df.groupby(self.index_column)[column].transform("kurt")
            df['{}_min'.format(column)] = df.groupby(self.index_column)[column].transform('min')
            df['{}_max'.format(column)] = df.groupby(self.index_column)[column].transform('max')
        
        for column in self.category_columns:
            df['{}_sum'.format(column)] = df.groupby(self.index_column)[column].sum()

        return df