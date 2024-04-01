import pandas as pd
from typing import List

class Encoder():
    def __init__(self,category_colums:List[str]):
        self.category_columns = category_colums
        self.new_columns = []

    def apply(self,df:pd.DataFrame):
        df = pd.get_dummies(df,columns=self.category_columns,
                                    dtype=int)
        self.new_columns = [col for col in df.columns if col not in df.columns]
        return df