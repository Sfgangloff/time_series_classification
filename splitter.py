import pandas as pd

class Splitter():
    def __init__(self,
                 df:pd.DataFrame,
                 batch_size:int,
                 index_column:str,
                 test_size:int):
        self.df = df
        self.position = 0
        self.index_column = index_column
        self.batch_size = batch_size
        self.end = False
        self.test_size = test_size

    def shift(self):
        self.position = self.position + 1
        if self.position*self.batch_size*100 > len(self.df):
            self.end = True

    def get_current_slice(self):
        print(self.position)
        return self.df[(self.df[self.index_column] < (self.position +1) * self.batch_size
                              ) & (self.df[self.index_column] >= self.position*self.batch_size
                                   )]
    
    def get_test_df(self):
        return self.df[self.df[self.index_column] > self.df[self.index_column].max()-self.test_size]

    
