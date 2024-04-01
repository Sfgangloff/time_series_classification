import pandas as pd
import numpy as np
from typing import List

class Preprocessor():
    def __init__(self):
        self.columns = []

    def input_df_to_array(self,df:pd.DataFrame,
                        category_columns:List[str],
                        scalar_columns:List[str],
                        index_column:str):
        df = df[
                category_columns
                +scalar_columns
                +[index_column]
                ]
        
        df_encoded = pd.get_dummies(df,
                                    columns=category_columns,
                                    dtype=int)
        
        self.columns = list(df_encoded.columns)
        
        #df_encoded[scalar_columns] = (df_encoded[scalar_columns] 
        #                            - df_encoded[scalar_columns].mean()
        #                            )/df[scalar_columns].std()
        
        grouped = df_encoded.groupby(index_column)
        group_arrays = [group.values[:, 1:] for _, group in grouped]
        max_rows = max(group.shape[0] for group in group_arrays)
        max_cols = max(group.shape[1] for group in group_arrays)
        x_array = np.full((len(group_arrays), max_rows, max_cols), np.nan)
        for i, group_arr in enumerate(group_arrays):
            rows, cols = group_arr.shape
            x_array[i, :rows, :cols] = group_arr

        return x_array

    def output_df_to_array(self,df:pd.DataFrame):
        df_encoded = pd.get_dummies(df["eqt_code_cat"],
                                    columns=["eqt_code_cat"],dtype=int)
        y_array = df_encoded.values

        return y_array

    def shuffle_2d_sections(self,array_list:List[np.array]):
        length_list = [len(arr) for arr in array_list]
        assert max(length_list) == min(length_list)
        num_sections, _, _ = array_list[0].shape
        shuffled_indices = np.random.permutation(num_sections)
        new_array_list = [arr[shuffled_indices] for arr in array_list]

        return new_array_list
