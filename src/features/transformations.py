import pandas as pd
import numpy as np
from src.utils.store import AssignmentStore


def transform_manufacturer(df: pd.DataFrame) -> pd.DataFrame:
    _mf_list = pd.DataFrame(df['Manufacturer'].value_counts(normalize=True).cumsum())
    _mf_list = list(_mf_list[_mf_list['Manufacturer'] <= 0.95].index)
    df = df.assign(Manufacturer=np.where(df['Manufacturer'].isin(_mf_list), df['Manufacturer'], 'Others'))
    return df


def get_model_avg_price(df: pd.DataFrame, load_dict=False) -> pd.DataFrame:
    store = AssignmentStore()
    if load_dict:
        _price_by_model = store.get_processed("price_by_model.csv")
    else:
        _price_by_model = df.groupby('Model').agg({'Price': 'mean'}).reset_index()
        _price_by_model.columns = ['Model', 'Model_avg_price']
        store.put_processed("price_by_model.csv", _price_by_model)
    df = pd.merge(df, _price_by_model, on='Model', how='left')
    return df




