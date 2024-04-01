import pandas as pd

class DataExtractor():
    def __init__(self):
        pass

    def venue_statistics(self,df_obs:pd.DataFrame):
        ranking = df_obs["venue"].value_counts().index.to_list()
        str_ranking = [str(rank) for rank in ranking]
        return ''.join(str_ranking)
    
    def order_statistics(self,df_obs:pd.DataFrame):
        information = df_obs["order_id"].value_counts().value_counts().head(2)
        ranking = information.index.to_list()
        str_ranking = [str(rank) for rank in ranking]
        str_ranking = ''.join(str_ranking)
        ratio = information.values[0]/information.values[1]
        return str_ranking, ratio
    
    def extract(self,df_obs:pd.DataFrame):
        obs_id = df_obs["obs_id"].values[0]
        order_statistics = self.order_statistics(df_obs)
        extract_df = pd.DataFrame({"obs_id":[obs_id],
                                   "venue_ranking":[self.venue_statistics(df_obs)],
                                   "order_ranking":[order_statistics[0]],
                                   "order_ratio":[order_statistics[1]]})
        return extract_df
    
    def global_extraction(self,df:pd.DataFrame):
        grouped = df.groupby("obs_id")
        applied = grouped.apply(lambda 
                                df_obs: self.extract(df_obs)
                          )
        applied.set_index("obs_id",inplace=True)
        applied = applied[(applied["venue_ranking"].str.len() == 6)]
        order_rankings = applied["order_ranking"].value_counts().index.to_list()[:2]
        applied = applied[(applied["order_ranking"].isin(order_rankings))]
        return applied