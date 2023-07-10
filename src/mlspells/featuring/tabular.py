from mlspells.base import StringEnum
from typing import Union, List

import pandas as pd
import numpy as np

class SplitMode(StringEnum):
    COLUMNS = 'columns'
    ROWS = 'rows'
    ARRAY = 'array'

class ExplodeStrategy(StringEnum):
    COLUMNS = 'columns'
    ROWS = 'rows'

def split_column(data: pd.DataFrame, column: Union[str, List[str]], split_mode: SplitMode = SplitMode.ARRAY,
                 split_by: str = ' ', new_columns_name:Union[str, List[str]]=None, drop_original: bool = True):
    """
    Splits a given column into multiple column values
    """

    if drop_original:
        assert split_mode == SplitMode.COLUMNS or (new_columns_name != None and new_columns_name != column), "`drop_original needs to be False or `new_columns_name` has to be indicated with a different column name"

    if split_mode == SplitMode.COLUMNS:
        # Calculate number of columns to use
        max_count = data[column].str.count(split_by).max()
        if not new_columns_name:
            new_columns_name = [f'{column}_{index}' for index in range(0, max_count)]
        
        assert isinstance(new_columns_name, list), "`new_columns_name` has to be of type list"
        assert len(new_columns_name) == max_count + 1, f"Split will generate {max_count + 1}, but {len(new_columns_name)} column names were provided."

        data[new_columns_name] = data[column].str.split(split_by, expand=True)

    elif split_mode == SplitMode.ROWS or split_mode == SplitMode.ARRAY:
        if new_columns_name:
            if isinstance(new_columns_name, list):
                assert len(new_columns_name) == 1, f"Only 1 column name can be indicated if split_mode = {split_mode}."
                col_name = new_columns_name[0]
        else:
            col_name = column

        data[col_name] = data[column].apply(lambda x: x.split(split_by))
    
        if split_mode == SplitMode.ROWS:
            data = data.explode(col_name)
    else:
        raise ValueError(f'{split_mode} is not a valid split mode')

    if drop_original:
        data.drop([column], axis=1, inplace=True)


def featurize_date_column(data: pd.DataFrame, column: str, date_time: bool = False, drop_original: bool = False):
    data[column] = pd.to_datetime(data[column])

    data[f'{column}_year'] = data[column].dt.year
    data[f'{column}_month'] = data[column].dt.month
    data[f'{column}_day'] = data[column].dt.day
    data[f'{column}_quarter'] = data[column].dt.quarter
    data[f'{column}_semester'] = np.where(data[f'{column}_quarter'].isin([1,2]),1,2)
    data[f'{column}_dayofweek'] = data[column].dt.day_name()
    data[f'{column}_weekend'] = np.where(data[f'{column}_dayofweek'].isin(['Sunday','Saturday']),1,0)
    data[f'{column}_dayofyear'] = data[column].dt.dayofyear
    data[f'{column}_weekofyear'] = data[column].dt.weekofyear

    if date_time:
        data[f'{column}_hour'] = data[column].dt.hour
        data[f'{column}_minute'] = data[column].dt.minute
        data[f'{column}_ampm'] = np.where(data[f'{column}_hour'] < 12, 'am', 'pm')

    if drop_original:
        data.drop([column], axis=1, inplace=True)


def explode_column_values(data: pd.DataFrame, column: str, mode: ExplodeStrategy = ExplodeStrategy.ROWS, 
    new_columns_name: List[str] = None):

    if mode == ExplodeStrategy.COLUMNS:
        max_count = data[column].map(len).max()

        if new_columns_name:
            assert max_count == len(new_columns_name), f"Explode will generate {max_count}, but {len(new_columns_name)} column names were provided."
        else:
            new_columns_name = [f'{column_name}_{index}' for index in range(0, max_count)]
        
        padded_data = np.array(list(itertools.zip_longest(*data[column_name].tolist(), fillvalue=None))).T
        data[new_columns_name] = pd.DataFrame(padded_data, index=data.index)

    elif explode_mode == ExplodeStrategy.ROWS:
        data = data.explode(column_name)
    else:
        raise ValueError(f'{explode_mode} is not a valid explode mode')
